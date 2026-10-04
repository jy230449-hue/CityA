from vln.drone_agent import AirsimAgent
import json
import sys
import os
import airsim
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "GroundSAM"))

from llm_agent import chat_with_llm
import cv2
from Segment_Image import segment_observation, get_relevance_scores,get_scores_for_class_names,overlay_masks_on_depth
from update_uncertainty_map import generate_uncertainty_map, uncertainty_map_update, visualize_uncertainty_map,is_target_visible, is_target_visible2, cognitive_map_denoising
from update_semantic_map import generate_semantic_map, update_semantic_map, visualize_semantic_map, visualize_cognitive_map, generate_coginitive_map, update_coginitive_map
from CityAVOS_tools import *
from prompts import *
from segment_anything import sam_model_registry, SamPredictor
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap



def load_dino_model(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    model.eval()
    return model.to(device)


def main():

    # initialization
    print("\r Initializing")

    drone = AirsimAgent(None, None, None)
    config_file = r"C:\hjy\project\CityAVOS\GroundSAM\GroundingDINO\groundingdino\config\GroundingDINO_SwinT_OGC.py"
    grounded_checkpoint = "GroundSAM/groundingdino_swint_ogc.pth"
    sam_checkpoint = "GroundSAM/sam_vit_h_4b8939.pth"
		
    sam_version = "vit_h"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dino_model = load_dino_model(config_file, grounded_checkpoint, device)
    sam = sam_model_registry[sam_version](checkpoint=sam_checkpoint)
    sam.to(device=device)
    sam_predictor = SamPredictor(sam)

 # ============================================================
# 原作者/原当前代码
# dataset_path = "./data/target_information_2420.json"
# ============================================================

# ============================================================
# 2026-09-29 修改
# 论文实验由于 API 成本使用 605 个 task 子集
# ============================================================
    dataset_path = "./data/target_information_605.json"
    with open(dataset_path, "r") as file:
        dataset = json.load(file)
# ============================================================
# 原作者代码
# scenes_path = "./data/scenes.json"
# ============================================================

# ============================================================
# 2026-09-29 修改
# 605-task 实验使用补齐 6 scene 的兼容配置
# ============================================================
    scenes_path = "./data/scenes_compat_6.json"
    with open(scenes_path, "r") as file:
        scenes = json.load(file)
   task_start = 0

# ============================================================
# 原作者代码
# task_end = len(dataset)
# ============================================================

# ============================================================
# 2026-09-29 修改
# 调试阶段先运行前 5 个任务
# 正式实验时改成 len(dataset)
# ============================================================
    task_end = 5

 # ============================================================
# 原当前代码
# 每个 Step 都会弹 Cognitive Map 和 Uncertainty Map。
# if_figure_plot = 1
# if_draw = 1
# ============================================================

# ============================================================
# 2026-09-29 修改
# 批量实验关闭可视化，提高运行效率
# ============================================================
    if_figure_plot = 0
    if_draw = 0

    theta_T = 0.1
    mission_logs = {}
# ============================================================
# 2026-09-29 修改
# 将 GitHub 数据中的 3 个 mission_grade
# 映射为 AAAI 最终论文使用的 Easy / Hard
#
# Grade 1 -> Easy
# Grade 2 -> Hard
# Grade 3 -> Hard
#
# 注意：
# 原始 mission_grade 不删除。
# ============================================================

    difficulty_map = {
    "1": "Easy",
    "2": "Hard",
    "3": "Hard"
    }

    # task start
    for idx in range(task_start, task_end):

        # load data
        print("Task - ", idx+1)
        mission_logs[idx] = {}
        task_item = dataset[idx]

#  日志会出现：
# Task - 1
# Mission ID: 1
# Original mission grade: 1
# Paper difficulty: Easy
        mission_id = task_item["mission_id"]
        mission_grade = str(task_item["mission_grade"])

        paper_difficulty = difficulty_map.get(
            mission_grade,
            "Unknown"
        )

        print("Mission ID:", mission_id)
        print("Original mission grade:", mission_grade)
       print("Paper difficulty:", paper_difficulty)

        drone.pos = np.array(task_item["initial_pos"][:3])
        drone.ori = np.array(task_item["initial_pos"][3:])
        drone.MovetoPose(np.concatenate([drone.pos, drone.ori]).tolist())
        scene_id = task_item["scene_id"]
        scene = scenes[int(scene_id)-1]
        target_pos = task_item["target_pos"]
        target_text = task_item["target_text"]
        target_image_path = ("./" + task_item["target_image_path"])
        print("Search object: ", target_text)

        # create map
        uncertainty_map = generate_uncertainty_map(
            scene["x_min"], scene["x_max"], scene["y_min"], scene["y_max"], scene["z_min"], scene["z_max"],
            scene["x_interval"], scene["z_interval"])
        total_uncertainty = np.sum(uncertainty_map[:, 3:])
        semantic_map = generate_semantic_map(
            scene["x_min"], scene["x_max"], scene["y_min"], scene["y_max"], scene["z_min"], scene["z_max"],
            1, 1)
        cognitive_map = generate_coginitive_map(
            scene["x_min"], scene["x_max"], scene["y_min"], scene["y_max"], scene["z_min"], scene["z_max"],
            1, 1)

        drone_pos = []
        drone_ori = []

        step = 0
        drone_pos.append(drone.pos * [1, -1, -1])
        drone_ori.append(drone.ori.copy())
        prompt_sam = chat_with_llm(prompt_rel, "./" + target_image_path)
        scores_rel = get_relevance_scores(prompt_sam, target_text, target_image_path)
        print("Related Objects with Scores:", scores_rel)
        mission_complete = False

        while not mission_complete:

            print("Step: ", step)

            # observation
            print("Get Observation.")
            img1, img2 = drone.get_observation()
            rgb_path = f"./output/rgb_image_{step}.png"
            depth_path = f"./output/depth_image_{step}.png"
            cv2.imwrite(os.path.join(rgb_path), img1)
            cv2.imwrite(os.path.join(depth_path), img2)
            look_direction = np.round(rad_to_deg(drone.ori[2])).astype(int)

            # map update
            print("Update cognitive map and uncertainty map.")
            uncertainty_map = uncertainty_map_update(uncertainty_map, drone.pos * [1, -1, -1], look_direction,
                                                            scene["step_x"], fov=90, max_distance=1000)
            cognitive_map = cognitive_map_denoising(cognitive_map, drone.pos * [1, -1, -1], look_direction,
                                                            scene["step_x"], fov=90, max_distance=1000)

            class_ids, class_names, class_name_to_id, masks = segment_observation(rgb_path, prompt_sam, dino_model, sam_predictor, device=device)
            # scores_rel = get_relevance_scores(class_name_to_id, target_text, target_image_path, rgb_path)
            class_scores = get_scores_for_class_names(class_names, scores_rel)
            depth_plus_id, depth_plus_score = overlay_masks_on_depth(img2, masks, class_ids, class_scores)
            semantic_map, cognitive_map = update_semantic_map(semantic_map, cognitive_map, depth_path, drone.pos * [1, -1, -1], [-90, 0, 270] - drone.ori,
                                               depth_plus_id, class_scores)

            max_value, max_cluster_center = find_max_cluster_center(cognitive_map)
            print("max_value: ", max_value)
            print("max_cluster", max_cluster_center)
# --------------------------202609292050修改-----------------
#作者原
#            adviser_uncertainty_map = action_value_choose(drone, uncertainty_map, scene, total_uncertainty)
#		修改后
            adviser_uncertainty_map = action_value_choose(drone, uncertainty_map, scene, total_uncertainty, theta_T)
#-------------------------------------------完毕----------------------------------
            print("advisier_un: ", adviser_uncertainty_map)
            if max_cluster_center is not None and max_value > 0.5:
                adviser_cognitive_map = action_set[action_to_pos(drone, max_cluster_center, scene)]
            else:
                adviser_cognitive_map = None
                max_value = 0
            attraction_value = max_value
            print("advisier_co: ", adviser_cognitive_map)

            # map visualization
            if if_draw:
                # visualize_semantic_map(semantic_map, step, if_figure_plot)
                visualize_cognitive_map(drone.pos * [1, -1, -1], look_direction, cognitive_map, scores_rel, step,
                                       max_cluster_center, if_figure_plot)
                visualize_uncertainty_map(uncertainty_map, drone.pos * [1, -1, -1], look_direction, step,
                                          if_figure_plot)


            # Agent thinking
            action_label = get_action_from_llm(adviser_cognitive_map, adviser_uncertainty_map, target_text, target_image_path, rgb_path, attraction_value)
            print("agent_choose_action: ", action_label)

            # take action
            if action_label == 7:
                break
            else:
                drone.take_action(action_label, scene["step_x"], scene["step_z"])
            time.sleep(1)
            drone_ori.append(drone.ori.copy())
            drone_pos.append(drone.pos * [1, -1, -1])

            # step detection
            if is_target_visible2(np.array(target_pos) * [1, -1, -1], drone.pos * [1, -1, -1], look_direction, scene["step_x"]):
                if_success = 1
                break

            if step > scene["step_all"]:
                if_success = 0
                break
            step = step + 1

        # save data
        if is_target_visible(np.array(target_pos) * [1, -1, -1], drone.pos * [1, -1, -1], look_direction):
            if_success = 1
            print("Task success!")
        else:
            if_success = 0
            print("Task failure!")
#---------------------------------修改----------------
# -----作者原代码------------------
#        mission_logs[idx] = {
#            "position": [pos.tolist() for pos in drone_pos],
#            "orientation": [ori.tolist() for ori in drone_ori],
#            "if_success": if_success
#       }

# -------------  
        mission_logs[idx] = {

    # ==============================
    # 2026-09-29 新增：实验基本信息
    # ==============================

            "task_index": idx,
            "mission_id": mission_id,

            "scene_id": scene_id,

           "mission_grade": mission_grade,
           "paper_difficulty": paper_difficulty,

           "target_text": target_text,

    # ==============================
    # 算法参数
    # ==============================

           "theta_T": theta_T,
 
           "step_x": scene["step_x"],
           "step_z": scene["step_z"],
           "step_all": scene["step_all"],

    # ==============================
    # 任务结果
    # ==============================

           "steps": step,
           "if_success": int(if_success),

    # ==============================
    # 飞行轨迹
    # ==============================

           "position": [
                pos.tolist()
                for pos in drone_pos
           ],

            "orientation": [
             ori.tolist()
             for ori in drone_ori
          ]
      }
#-------------------完毕----------------
        with open(f"./output/experiemnt_data/ours/data_{idx}.json", "w", encoding="utf-8") as file:
            json.dump(mission_logs[idx], file)
        print("Save data in scene-", idx + 1)


if __name__ == "__main__":
    main()



