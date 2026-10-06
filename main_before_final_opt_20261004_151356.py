from vln.drone_agent import AirsimAgent
import json
import sys
import os
import argparse  # =========================================================
                 # 2026-10-02 新增：支持 python main.py --task 8 单独运行指定 Task
                 # =========================================================
import time  # ============================================================
             # 2026-09-29 修改：显式导入 time，避免依赖 CityAVOS_tools 的 * 导入
             # ============================================================
import airsim
import torch
import numpy as np  # ======================================================
                    # 2026-09-29 修改：显式导入 numpy，避免依赖 CityAVOS_tools 的 * 导入
                    # ======================================================

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "GroundSAM"))

from llm_agent import chat_with_llm
import cv2
from Segment_Image import (
    segment_observation,
    get_relevance_scores,
    get_scores_for_class_names,
    overlay_masks_on_depth,
    prepare_detection_prompt,  # ==================================================
                               # 2026-10-03 新增：把 VLM 返回内容整理成 GroundingDINO 更适合检测的实体类别
                               # ==================================================
)
from update_uncertainty_map import (
    generate_uncertainty_map,
    uncertainty_map_update,
    visualize_uncertainty_map,
    is_target_visible,
    is_target_visible2,
    cognitive_map_denoising,
)
from update_semantic_map import (
    generate_semantic_map,
    update_semantic_map,
    visualize_semantic_map,
    visualize_cognitive_map,
    generate_coginitive_map,
    update_coginitive_map,
)
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
    load_res = model.load_state_dict(
        clean_state_dict(checkpoint["model"]),
        strict=False
    )
    print(load_res)
    model.eval()
    return model.to(device)


# ============================================================
# 2026-09-29 新增
# 计算无人机实际飞行轨迹长度。
# 后续统计 SPL / 路径效率时会用到。
# 注意：这里只记录真实轨迹长度，不宣称它就是论文中的 shortest path。
# ============================================================
def calculate_trajectory_length(positions):
    if positions is None or len(positions) < 2:
        return 0.0

    total_length = 0.0
    for i in range(1, len(positions)):
        p1 = np.asarray(positions[i - 1], dtype=float)
        p2 = np.asarray(positions[i], dtype=float)
        total_length += float(np.linalg.norm(p2 - p1))

    return total_length


# ============================================================
# 2026-10-02 新增：性能测速汇总。
# 作用：统计每个模块平均/总耗时，不改变导航算法。
# ============================================================
def summarize_step_timing(step_timing_history):
    if not step_timing_history:
        return {}

    keys = set()
    for item in step_timing_history:
        keys.update(k for k in item.keys() if k != "step")

    summary = {}
    for key in sorted(keys):
        values = [
            float(item[key])
            for item in step_timing_history
            if key in item
        ]
        if values:
            summary[key] = {
                "total_s": round(sum(values), 4),
                "avg_s": round(sum(values) / len(values), 4),
                "count": len(values),
            }
    return summary


def main():

    # ============================================================
    # 2026-10-02 新增：单 Task 调试参数
    #
    # 用法：
    #   python main.py --task 8
    #   python main.py --task 21
    #
    # --task 使用 mission_id，而不是 Python 的 0-based idx。
    # 不传 --task 时，继续使用下面 task_start / task_end 的默认批量范围。
    # ============================================================
    parser = argparse.ArgumentParser(
        description="CityAVOS reproduction runner"
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Run one mission_id only, e.g. --task 8 or --task 21",
    )

    # ============================================================
    # 2026-10-02 新增：测速步数上限。
    # 作用：例如 --profile-steps 5 只测前5步，不覆盖正式结果。
    # ============================================================
    parser.add_argument(
        "--profile-steps",
        type=int,
        default=None,
        help="Enable profiling and stop after N executed steps",
    )
    args = parser.parse_args()

    if args.profile_steps is not None and args.profile_steps <= 0:
        raise ValueError("--profile-steps 必须大于 0。")

    profile_mode = args.profile_steps is not None

    # ============================================================
    # initialization
    # ============================================================
    program_init_start = time.perf_counter()
    print("\r Initializing")

    drone = AirsimAgent(None, None, None)

    config_file = (
        r"C:\hjy\project\CityAVOS\GroundSAM\GroundingDINO"
        r"\groundingdino\config\GroundingDINO_SwinT_OGC.py"
    )
    grounded_checkpoint = "GroundSAM/groundingdino_swint_ogc.pth"
    sam_checkpoint = "GroundSAM/sam_vit_h_4b8939.pth"

    sam_version = "vit_h"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dino_model = load_dino_model(
        config_file,
        grounded_checkpoint,
        device
    )

    sam = sam_model_registry[sam_version](
        checkpoint=sam_checkpoint
    )
    sam.to(device=device)
    sam_predictor = SamPredictor(sam)

    # ============================================================
    # 2026-10-02 新增：模型初始化耗时。
    # 作用：区分启动固定耗时和每 Step 耗时。
    # ============================================================
    model_init_time = time.perf_counter() - program_init_start
    print(f"[TIME] Model initialization: {model_init_time:.2f}s")

    # ============================================================
    # 原作者 / 原当前代码
    #
    # dataset_path = "./data/target_information_2420.json"
    # ============================================================

    # ============================================================
    # 2026-09-29 修改
    # 论文正式实验使用公开的 605-task 子集。
    # ============================================================
    dataset_path = "./data/target_information_605.json"

    with open(dataset_path, "r", encoding="utf-8") as file:
        dataset = json.load(file)

    # ============================================================
    # 原作者代码
    #
    # scenes_path = "./data/scenes.json"
    # ============================================================

    # ============================================================
    # 2026-09-29 修改
    # 使用补齐到 6 个 scene 的兼容配置。
    # 原始 scenes.json 请继续保留，不要覆盖。
    # ============================================================
    scenes_path = "./data/scenes_compat_6.json"

    with open(scenes_path, "r", encoding="utf-8") as file:
        scenes = json.load(file)

    # ============================================================
    # 2026-09-29 新增
    # 启动前检查，避免后面跑到 Scene 5 / 6 才数组越界。
    # ============================================================
    if len(scenes) < 6:
        raise ValueError(
            f"scenes 配置不足 6 个，目前只有 {len(scenes)} 个。"
            f"请检查 {scenes_path}"
        )

    if len(dataset) != 605:
        print(
            f"[Warning] 当前数据集任务数为 {len(dataset)}，"
            "不是预期的 605。"
        )

    task_start = 0

    # ============================================================
    # 原作者代码
    #
    # task_end = len(dataset)
    # ============================================================

    # ============================================================
    # 2026-09-29 修改
    # 当前先跑前 5 个 Task 做稳定性测试。
    #
    # 测试 5 个：
    # task_end = 5
    #
    # 测试 20 个：
    # task_end = 20
    #
    # 正式跑全部 605 个：
    # task_end = len(dataset)
    # ============================================================
    task_end = 5

    # ============================================================
    # 2026-10-02 新增：支持通过 mission_id 单独测试一个 Task。
    #
    # 例如：
    #   python main.py --task 8
    # 会自动找到 mission_id == "8" 的数据，而不是要求手算 idx=7。
    # ============================================================
    if args.task is not None:
        matched_indices = [
            i
            for i, item in enumerate(dataset)
            if str(item.get("mission_id")) == str(args.task)
        ]

        if not matched_indices:
            raise ValueError(
                f"未找到 mission_id={args.task} 的 Task。"
            )

        task_start = matched_indices[0]
        task_end = task_start + 1

        print(
            f"[Single Task Mode] mission_id={args.task}, "
            f"dataset_index={task_start}"
        )

    task_end = min(task_end, len(dataset))

    # ============================================================
    # 原当前代码
    #
    # if_figure_plot = 1
    # if_draw = 1
    #
    # 每个 Step 都会弹出 Cognitive Map / Uncertainty Map。
    # ============================================================

    # ============================================================
    # 2026-09-29 修改
    # 批量实验关闭地图绘图和弹窗，提高运行速度。
    #
    # 如果之后想重新看图：
    # if_draw = 1
    # if_figure_plot = 1
    # ============================================================
    if_figure_plot = 0
    if_draw = 0

    # ============================================================
    # 论文 / 实验参数
    # ============================================================
    theta_T = 0.1
    cognitive_threshold = 0.5

    # ============================================================
    # 2026-09-29 修改
    # 将 GitHub 数据中的三档 mission_grade 映射为
    # AAAI 最终论文统计时使用的 Easy / Hard。
    #
    # Grade 1 -> Easy
    # Grade 2 -> Hard
    # Grade 3 -> Hard
    #
    # 原始 mission_grade 仍然保留到日志里。
    # ============================================================
    difficulty_map = {
        "1": "Easy",
        "2": "Hard",
        "3": "Hard",
    }

    # ============================================================
    # 2026-09-29 修改
    # 统一创建运行过程中需要的输出目录。
    # 保留作者原来的 experiemnt_data 拼写，避免影响已有目录/脚本。
    # ============================================================
    os.makedirs("./output", exist_ok=True)

    # ============================================================
    # 2026-10-02 新增：测速结果单独保存。
    # 作用：避免 profile 数据覆盖正式 data_*.json。
    # ============================================================
    if profile_mode:
        result_output_dir = "./output/experiemnt_data/profile"
    else:
        result_output_dir = "./output/experiemnt_data/ours"

    os.makedirs(result_output_dir, exist_ok=True)

    mission_logs = {}

    print("=" * 70)
    print("CityAVOS experiment configuration")
    print("Dataset:", dataset_path)
    print("Total dataset tasks:", len(dataset))
    print("Run task range:", task_start, "->", task_end - 1)
    print("Scene config:", scenes_path)
    print("theta_T:", theta_T)
    print("cognitive_threshold:", cognitive_threshold)
    print("if_draw:", if_draw)
    print("if_figure_plot:", if_figure_plot)
    print("profile_mode:", profile_mode)
    print("profile_steps:", args.profile_steps)
    print("=" * 70)

    # ============================================================
    # task start
    # ============================================================
    for idx in range(task_start, task_end):

        task_total_start = time.perf_counter()

        print("\n" + "=" * 70)
        print("Task -", idx + 1)

        mission_logs[idx] = {}
        task_item = dataset[idx]

        # ============================================================
        # 2026-09-29 新增
        # 固定记录 Task ID、原始 grade 和论文难度映射。
        # ============================================================
        mission_id = str(task_item["mission_id"])
        mission_grade = str(task_item["mission_grade"])
        paper_difficulty = difficulty_map.get(
            mission_grade,
            "Unknown"
        )

        print("Mission ID:", mission_id)
        print("Original mission grade:", mission_grade)
        print("Paper difficulty:", paper_difficulty)

        # ============================================================
        # 初始化无人机到该 Task 的起点
        # ============================================================
        drone.pos = np.array(
            task_item["initial_pos"][:3],
            dtype=float
        )

        drone.ori = np.array(
            task_item["initial_pos"][3:],
            dtype=float
        )

        drone.MovetoPose(
            np.concatenate(
                [drone.pos, drone.ori]
            ).tolist()
        )

        scene_id = str(task_item["scene_id"])
        scene_index = int(scene_id) - 1

        # ============================================================
        # 2026-09-29 新增
        # 防止 scene_id 超过配置数量。
        # ============================================================
        if scene_index < 0 or scene_index >= len(scenes):
            raise IndexError(
                f"Task {mission_id} 使用 Scene {scene_id}，"
                f"但 scenes 文件只有 {len(scenes)} 个配置。"
            )

        scene = scenes[scene_index]

        target_pos = task_item["target_pos"]
        target_text = task_item["target_text"]

        # ============================================================
        # 原作者 / 原当前代码
        #
        # target_image_path = ("./" + task_item["target_image_path"])
        #
        # 后面又使用：
        # chat_with_llm(prompt_rel, "./" + target_image_path)
        #
        # 会形成 ././data/...，Windows 通常仍能读取，
        # 但路径写法不够干净。
        # ============================================================

        # ============================================================
        # 2026-09-29 修改
        # 统一规范目标图片路径，避免重复添加 "./"。
        # ============================================================
        target_image_path = os.path.normpath(
            task_item["target_image_path"]
        )

        print("Scene ID:", scene_id)
        print("Search object:", target_text)
        print("Target image:", target_image_path)

        # ============================================================
        # 2026-10-02 新增：地图初始化测速。
        # 作用：确认 Task 启动阶段是否耗时。
        # ============================================================
        map_init_start = time.perf_counter()

        # ============================================================
        # create map
        # ============================================================
        uncertainty_map = generate_uncertainty_map(
            scene["x_min"],
            scene["x_max"],
            scene["y_min"],
            scene["y_max"],
            scene["z_min"],
            scene["z_max"],
            scene["x_interval"],
            scene["z_interval"],
        )

        total_uncertainty = np.sum(
            uncertainty_map[:, 3:]
        )

        semantic_map = generate_semantic_map(
            scene["x_min"],
            scene["x_max"],
            scene["y_min"],
            scene["y_max"],
            scene["z_min"],
            scene["z_max"],
            1,
            1,
        )

        cognitive_map = generate_coginitive_map(
            scene["x_min"],
            scene["x_max"],
            scene["y_min"],
            scene["y_max"],
            scene["z_min"],
            scene["z_max"],
            1,
            1,
        )

        map_init_time = time.perf_counter() - map_init_start
        if profile_mode:
            print(f"[TIME] Map initialization: {map_init_time:.2f}s")

        # ============================================================
        # Task 日志初始化
        # ============================================================
        drone_pos = []
        drone_ori = []

        action_history = []
        adviser_uncertainty_history = []
        adviser_cognitive_history = []
        attraction_value_history = []

        # ============================================================
        # 2026-10-02 新增：边界保护审计日志
        #
        # boundary_reject_count:
        #   LLM 给出会导致越界的动作时，累计拒绝次数。
        # boundary_rejected_actions:
        #   保存被拒绝动作、当前位置、预测位置和当时合法动作。
        # valid_actions_history:
        #   保存每个“实际执行 Step”对应的合法动作集合。
        # ============================================================
        boundary_reject_count = 0
        boundary_rejected_actions = []
        valid_actions_history = []

        # ============================================================
        # 2026-10-02 新增：每 Step 性能日志。
        # 作用：定位耗时最大的模块。
        # ============================================================
        step_timing_history = []
        target_prompt_time = 0.0
        relevance_scores_time = 0.0

        # ============================================================
        # 2026-09-29 修改
        #
        # step 的定义：
        #   “已经实际执行的无人机动作数量”
        #
        # 因此：
        # step = 0 表示还没有执行动作。
        # step_all = 40 表示最多执行 40 个动作。
        # ============================================================
        step = 0
        max_steps = int(scene["step_all"])

        # ============================================================
        # 2026-10-02 新增：测速时单独限制运行步数。
        # 作用：正式 max_steps 保持不变，测速可只跑前 N 步。
        # ============================================================
        run_step_limit = max_steps
        if profile_mode:
            run_step_limit = min(max_steps, args.profile_steps)

        if_success = 0
        termination_reason = "unknown"

        drone_pos.append(
            (drone.pos * [1, -1, -1]).copy()
        )

        drone_ori.append(
            drone.ori.copy()
        )

        # ============================================================
        # 原作者/原当前代码：
        #
        # prompt_sam = chat_with_llm(
        #     prompt_rel,
        #     "./" + target_image_path
        # )
        # ============================================================

        # ============================================================
        # 2026-09-29 修改
        # target_image_path 已经规范化，不再额外添加 "./"。
        # ============================================================
        target_prompt_start = time.perf_counter()

        # ============================================================
        # 2026-10-03 修改：GroundingDINO Prompt 修复
        #
        # 原流程：
        #   VLM 可能返回 text / numbers / yellow background /
        #   professional wording 等“属性词”，这些词很难被 GroundingDINO
        #   作为可定位实体检测，导致 class_names=[] / masks=0 /
        #   cognitive_map 长期为 0。
        #
        # 新流程：
        #   1. 保留云端 VLM 原始返回；
        #   2. prepare_detection_prompt() 只保留/补充可定位的物理实体；
        #   3. 后续 relevance 与 DINO 使用同一个修复后的 prompt，
        #      保证类别名与 relevance score 尽量一致。
        # ============================================================
        prompt_sam_raw = chat_with_llm(
            prompt_rel,
            target_image_path
        )

        prompt_sam = prepare_detection_prompt(
            prompt_sam_raw,
            target_text=target_text
        )

        print("[DINO PROMPT] raw:", prompt_sam_raw)
        print("[DINO PROMPT] prepared:", prompt_sam)

        target_prompt_time = time.perf_counter() - target_prompt_start

        relevance_start = time.perf_counter()
        scores_rel = get_relevance_scores(
            prompt_sam,
            target_text,
            target_image_path
        )
        relevance_scores_time = time.perf_counter() - relevance_start

        if profile_mode:
            print(f"[TIME] Target prompt: {target_prompt_time:.2f}s")
            print(f"[TIME] Relevance scores: {relevance_scores_time:.2f}s")

        print(
            "Related Objects with Scores:",
            scores_rel
        )

        mission_complete = False

        while not mission_complete:

            # ============================================================
            # 原作者代码位于循环末尾：
            #
            # if step > scene["step_all"]:
            #     if_success = 0
            #     break
            # step = step + 1
            #
            # 这会导致 step_all=40 时仍可能看到 Step 41。
            # ============================================================

            # ============================================================
            # 2026-09-29 修改
            # 在开始下一次动作前检查上限。
            # 这样 max_steps=40 时最多执行 40 个实际动作。
            # ============================================================
            if step >= run_step_limit:
                if profile_mode and run_step_limit < max_steps:
                    termination_reason = "profile_step_limit"
                else:
                    termination_reason = "max_steps"
                if_success = 0

                print(
                    f"Reached step limit: "
                    f"{step}/{run_step_limit}"
                )
                break

            step_total_start = time.perf_counter()
            step_timing = {"step": int(step)}

            print(
                f"Step: {step} "
                f"(executed actions: {step}/{run_step_limit})"
            )

            # ============================================================
            # observation
            # ============================================================
            print("Get Observation.")

            t = time.perf_counter()
            img1, img2 = drone.get_observation()
            step_timing["observation_s"] = time.perf_counter() - t

            rgb_path = f"./output/rgb_image_{step}.png"
            depth_path = f"./output/depth_image_{step}.png"

            t = time.perf_counter()
            cv2.imwrite(rgb_path, img1)
            cv2.imwrite(depth_path, img2)
            step_timing["image_save_s"] = time.perf_counter() - t

            look_direction = np.round(
                rad_to_deg(drone.ori[2])
            ).astype(int)

            # ============================================================
            # map update
            # ============================================================
            print(
                "Update cognitive map and uncertainty map."
            )

            t = time.perf_counter()
            uncertainty_map = uncertainty_map_update(
                uncertainty_map,
                drone.pos * [1, -1, -1],
                look_direction,
                scene["step_x"],
                fov=90,
                max_distance=1000,
            )

            cognitive_map = cognitive_map_denoising(
                cognitive_map,
                drone.pos * [1, -1, -1],
                look_direction,
                scene["step_x"],
                fov=90,
                max_distance=1000,
            )
            step_timing["map_preprocess_s"] = time.perf_counter() - t

            # ============================================================
            # 2026-10-02 修改：推理阶段关闭梯度。
            # 作用：减少 DINO/SAM 推理开销，不改变训练参数。
            # ============================================================
            t = time.perf_counter()
            with torch.inference_mode():
                class_ids, class_names, class_name_to_id, masks = (
                    segment_observation(
                        rgb_path,
                        prompt_sam,
                        dino_model,
                        sam_predictor,
                        device=device,
                    )
                )
            step_timing["dino_sam_s"] = time.perf_counter() - t

            # ============================================================
            # 原作者代码保留：
            #
            # scores_rel = get_relevance_scores(
            #     class_name_to_id,
            #     target_text,
            #     target_image_path,
            #     rgb_path
            # )
            # ============================================================

            # ============================================================
            # 2026-10-04 综合测速：
            # 一次拆开 score / overlay / update_semantic_map / cluster。
            # 只增加计时，不改变原计算。
            # ============================================================
            semantic_total_start = time.perf_counter()

            t_sem = time.perf_counter()
            class_scores = get_scores_for_class_names(
                class_names,
                scores_rel
            )
            score_match_time = time.perf_counter() - t_sem

            t_sem = time.perf_counter()
            depth_plus_id, depth_plus_score = (
                overlay_masks_on_depth(
                    img2,
                    masks,
                    class_ids,
                    class_scores,
                )
            )
            overlay_time = time.perf_counter() - t_sem

            t_sem = time.perf_counter()
            semantic_map, cognitive_map = update_semantic_map(
                semantic_map,
                cognitive_map,
                depth_path,
                drone.pos * [1, -1, -1],
                [-90, 0, 270] - drone.ori,
                depth_plus_id,
                class_scores,
            )
            update_semantic_time = time.perf_counter() - t_sem

            t_sem = time.perf_counter()
            max_value, max_cluster_center = (
                find_max_cluster_center(
                    cognitive_map
                )
            )
            cluster_time = time.perf_counter() - t_sem

            semantic_total_time = (
                time.perf_counter() - semantic_total_start
            )

            step_timing["semantic_cognitive_s"] = semantic_total_time

            if profile_mode:
                print(
                    "[SEM PROFILE] "
                    f"score_match={score_match_time:.3f}s | "
                    f"overlay={overlay_time:.3f}s | "
                    f"update_map={update_semantic_time:.3f}s | "
                    f"cluster={cluster_time:.3f}s | "
                    f"TOTAL={semantic_total_time:.3f}s"
                )

            # ============================================================
            # 2026-10-02 新增：Cognitive Map 调试信息。
            # 作用：排查 adviser 长期为 None / attraction=0。
            # ============================================================
            if profile_mode:
                mask_count = 0 if masks is None else len(masks)
                cognitive_nonzero = int(np.count_nonzero(cognitive_map[:, 3]))
                cognitive_max_debug = (
                    float(np.max(cognitive_map[:, 3]))
                    if len(cognitive_map) > 0
                    else 0.0
                )
                print("[DEBUG] class_names:", class_names)
                print("[DEBUG] class_scores:", class_scores)
                print("[DEBUG] mask_count:", mask_count)
                print("[DEBUG] cognitive_max:", cognitive_max_debug)
                print("[DEBUG] cognitive_nonzero:", cognitive_nonzero)

            print("max_value:", max_value)
            print("max_cluster:", max_cluster_center)

            # ============================================================
            # 原作者代码：
            #
            # adviser_uncertainty_map = action_value_choose(
            #     drone,
            #     uncertainty_map,
            #     scene,
            #     total_uncertainty
            # )
            # ============================================================

            # ============================================================
            # 2026-09-29 修改
            # 将 main.py 中的 theta_T 真正传给 action_value_choose。
            #
            # 注意：
            # CityAVOS_tools.py 也必须同步改为：
            #
            # def action_value_choose(
            #     drone,
            #     points,
            #     scene,
            #     total_uncertainty,
            #     theta_T=0.1
            # ):
            #
            # 并把：
            # if temp > 0.1:
            #
            # 改成：
            # if temp > theta_T:
            # ============================================================
            t = time.perf_counter()
            adviser_uncertainty_map = action_value_choose(
                drone,
                uncertainty_map,
                scene,
                total_uncertainty,
                theta_T,
            )
            step_timing["uncertainty_adviser_s"] = time.perf_counter() - t

            print(
                "advisier_un:",
                adviser_uncertainty_map
            )

            # ============================================================
            # 原作者逻辑保留，仅把 0.5 提取成明确实验参数
            # ============================================================
            if (
                max_cluster_center is not None
                and max_value > cognitive_threshold
            ):
                action_index = action_to_pos(
                    drone,
                    max_cluster_center,
                    scene
                )

                # ========================================================
                # 2026-09-29 修改
                # 防止 action_to_pos 在极端情况下返回 None，
                # 进而出现 action_set[None] 报错。
                # ========================================================
                if action_index is not None:
                    adviser_cognitive_map = (
                        action_set[action_index]
                    )
                else:
                    adviser_cognitive_map = None
            else:
                adviser_cognitive_map = None
                max_value = 0

            attraction_value = float(max_value)

            print(
                "advisier_co:",
                adviser_cognitive_map
            )

            # ============================================================
            # map visualization
            # ============================================================
            if if_draw:

                # 原作者保留：
                # visualize_semantic_map(
                #     semantic_map,
                #     step,
                #     if_figure_plot
                # )

                visualize_cognitive_map(
                    drone.pos * [1, -1, -1],
                    look_direction,
                    cognitive_map,
                    scores_rel,
                    step,
                    max_cluster_center,
                    if_figure_plot,
                )

                visualize_uncertainty_map(
                    uncertainty_map,
                    drone.pos * [1, -1, -1],
                    look_direction,
                    step,
                    if_figure_plot,
                )

            # ============================================================
            # 2026-10-02 新增：最终动作边界保护
            #
            # 原问题：
            # action_value_choose() 内部虽然会过滤越界候选动作，
            # 但最终 LLM 选择的 action 过去会直接 drone.take_action()，
            # 因此仍可能飞出 Scene，并在越界后继续浪费几十个 Step。
            #
            # 新流程：
            # 1. 先计算当前所有不越界的动作。
            # 2. 越界的 Adviser 不再传给 LLM，避免 Prompt 冲突。
            # 3. 把合法动作集合告诉 LLM。
            # 4. LLM 输出后再做一次“硬边界检查”。
            # 5. 若仍越界，最多重新请求 3 次；非法尝试不增加 step。
            # 6. 连续 3 次仍越界则以 boundary_stuck 结束，避免无限循环。
            # ============================================================
            t = time.perf_counter()
            valid_action_names = get_valid_action_names(
                drone,
                scene,
                include_stop=False,
            )
            step_timing["boundary_filter_s"] = time.perf_counter() - t

            adviser_cognitive_for_llm = adviser_cognitive_map
            adviser_uncertainty_for_llm = adviser_uncertainty_map

            if (
                adviser_cognitive_for_llm is not None
                and adviser_cognitive_for_llm not in valid_action_names
            ):
                print(
                    "[Boundary Filter Adviser] cognitive:",
                    adviser_cognitive_for_llm,
                    "-> None"
                )
                adviser_cognitive_for_llm = None

            if (
                adviser_uncertainty_for_llm is not None
                and adviser_uncertainty_for_llm not in valid_action_names
            ):
                print(
                    "[Boundary Filter Adviser] uncertainty:",
                    adviser_uncertainty_for_llm,
                    "-> None"
                )
                adviser_uncertainty_for_llm = None

            action_label = None
            next_pos = None
            max_boundary_retries = 3
            qwen_total_start = time.perf_counter()

            for boundary_attempt in range(1, max_boundary_retries + 1):

                # ========================================================
                # Agent thinking
                # ========================================================
                action_label = get_action_from_llm(
                    adviser_cognitive_for_llm,
                    adviser_uncertainty_for_llm,
                    target_text,
                    target_image_path,
                    rgb_path,
                    attraction_value,
                    valid_actions=valid_action_names,

                    # ====================================================
                    # 2026-10-03 新增：导航历史
                    # 作用：
                    # - 把最近动作和最近位置提供给 VLM；
                    # - 检测 Go Left <-> Go Right 等往返循环；
                    # - 避免模型每一步都像“第一次看到场景”。
                    # ====================================================
                    action_history=action_history,
                    position_history=drone_pos,
                )

                print(
                    "agent_choose_action:",
                    action_label
                )

                # LLM 主动 Stop，或者 API 多次失败后函数兜底 Stop。
                if action_label == 7:
                    break

                next_pos = predict_next_position(
                    drone,
                    action_label,
                    scene
                )

                if is_position_in_scene(next_pos, scene):
                    break

                boundary_reject_count += 1

                rejected_action_name = (
                    action_set[action_label]
                    if isinstance(action_label, int)
                    and 0 <= action_label < len(action_set)
                    else str(action_label)
                )

                current_map_pos = np.asarray(
                    drone.pos * [1, -1, -1],
                    dtype=float
                )

                boundary_rejected_actions.append({
                    "step": step,
                    "attempt": boundary_attempt,
                    "action": rejected_action_name,
                    "current_position": current_map_pos.tolist(),
                    "predicted_position": next_pos.tolist(),
                    "valid_actions": list(valid_action_names),
                })

                print(
                    "[Boundary Reject]",
                    f"Step={step}",
                    f"Attempt={boundary_attempt}/{max_boundary_retries}",
                    f"Action={rejected_action_name}",
                    f"Current={current_map_pos.tolist()}",
                    f"Predicted={next_pos.tolist()}",
                    f"Valid={valid_action_names}"
                )

                # 不执行、不增加 step；重新请求 LLM。
                action_label = None

            step_timing["qwen_vlm_s"] = time.perf_counter() - qwen_total_start

            # 连续多次仍没有得到合法动作，直接结束该 Task。
            if action_label is None:
                termination_reason = "boundary_stuck"
                step_timing["total_step_s"] = time.perf_counter() - step_total_start
                step_timing_history.append(step_timing)
                print(
                    "[Boundary Stop] Failed to obtain a legal action "
                    f"after {max_boundary_retries} attempts."
                )
                break

            # Stop 不算实际执行动作。
            if action_label == 7:
                if termination_reason == "unknown":
                    termination_reason = "llm_stop"
                step_timing["total_step_s"] = time.perf_counter() - step_total_start
                step_timing_history.append(step_timing)
                break

            # ============================================================
            # 2026-10-02 修改
            # 只有最终确认合法、准备实际执行的动作才写入 action_history。
            # 这样 len(action_history) 与 steps 更容易保持一致；
            # 被拒绝的越界动作单独保存在 boundary_rejected_actions。
            # ============================================================
            action_name = (
                action_set[action_label]
                if isinstance(action_label, int)
                and 0 <= action_label < len(action_set)
                else str(action_label)
            )

            action_history.append(action_name)
            adviser_uncertainty_history.append(
                adviser_uncertainty_map
            )
            adviser_cognitive_history.append(
                adviser_cognitive_map
            )
            attraction_value_history.append(
                attraction_value
            )
            valid_actions_history.append(
                list(valid_action_names)
            )

            # ============================================================
            # take action
            # ============================================================
            t = time.perf_counter()
            drone.take_action(
                action_label,
                scene["step_x"],
                scene["step_z"],
            )

            time.sleep(1)
            step_timing["action_execution_s"] = time.perf_counter() - t

            # ============================================================
            # 2026-09-29 修改
            # action 真正执行完成后，step 才 +1。
            # 这样 steps 字段就是实际执行动作数。
            # ============================================================
            step += 1

            drone_ori.append(
                drone.ori.copy()
            )

            drone_pos.append(
                (drone.pos * [1, -1, -1]).copy()
            )

            # ============================================================
            # 原作者代码的问题：
            # look_direction 是动作执行前计算的。
            # 如果刚刚 Turn Left / Turn Right，
            # 成功判定会使用旧朝向。
            # ============================================================

            # ============================================================
            # 2026-09-29 修改
            # 动作完成以后重新计算无人机当前朝向。
            # ============================================================
            look_direction = np.round(
                rad_to_deg(drone.ori[2])
            ).astype(int)

            # ============================================================
            # step detection / success detection
            # ============================================================
            t = time.perf_counter()
            target_visible_now = is_target_visible2(
                np.array(target_pos) * [1, -1, -1],
                drone.pos * [1, -1, -1],
                look_direction,
                scene["step_x"],
            )
            step_timing["success_check_s"] = time.perf_counter() - t
            step_timing["total_step_s"] = time.perf_counter() - step_total_start
            step_timing_history.append(step_timing)

            if profile_mode:
                print("[TIME] " + " | ".join([
                    f"obs={step_timing.get('observation_s', 0):.2f}s",
                    f"save={step_timing.get('image_save_s', 0):.2f}s",
                    f"map={step_timing.get('map_preprocess_s', 0):.2f}s",
                    f"DINO+SAM={step_timing.get('dino_sam_s', 0):.2f}s",
                    f"semantic={step_timing.get('semantic_cognitive_s', 0):.2f}s",
                    f"uncertainty={step_timing.get('uncertainty_adviser_s', 0):.2f}s",
                    f"Qwen={step_timing.get('qwen_vlm_s', 0):.2f}s",
                    f"action={step_timing.get('action_execution_s', 0):.2f}s",
                    f"TOTAL={step_timing.get('total_step_s', 0):.2f}s",
                ]))

            if target_visible_now:
                if_success = 1
                termination_reason = "target_visible"
                break

            # ============================================================
            # 原作者代码（已移动到 while 循环顶部，不再执行）
            #
            # if step > scene["step_all"]:
            #     if_success = 0
            #     break
            #
            # step = step + 1
            # ============================================================

        # ============================================================
        # final task evaluation
        # ============================================================

        # 保证 final evaluation 使用最新无人机朝向
        look_direction = np.round(
            rad_to_deg(drone.ori[2])
        ).astype(int)

        final_target_visible = is_target_visible(
            np.array(target_pos) * [1, -1, -1],
            drone.pos * [1, -1, -1],
            look_direction,
        )

        # ============================================================
        # 原作者逻辑：
        #
        # if is_target_visible(...):
        #     if_success = 1
        #     print("Task success!")
        # else:
        #     if_success = 0
        #     print("Task failure!")
        #
        # 下面继续保留这个最终判定思想，但额外记录终止原因。
        # ============================================================
        if final_target_visible:
            if_success = 1

            if termination_reason != "target_visible":
                termination_reason = "final_target_visible"

            print("Task success!")
        else:
            if_success = 0

            if termination_reason == "unknown":
                termination_reason = "final_check_failure"

            print("Task failure!")

        # ============================================================
        # 2026-09-29 新增
        # 为后续 NE / SPL 等指标保留原始距离与轨迹长度。
        # ============================================================
        target_pos_map = (
            np.asarray(target_pos, dtype=float)
            * [1, -1, -1]
        )

        final_drone_pos_map = (
            np.asarray(drone.pos, dtype=float)
            * [1, -1, -1]
        )

        initial_drone_pos_map = np.asarray(
            drone_pos[0],
            dtype=float
        )

        final_navigation_error = float(
            np.linalg.norm(
                target_pos_map
                - final_drone_pos_map
            )
        )

        initial_target_distance = float(
            np.linalg.norm(
                target_pos_map
                - initial_drone_pos_map
            )
        )

        trajectory_length = (
            calculate_trajectory_length(
                drone_pos
            )
        )

        # ============================================================
        # 2026-10-02 新增：Task 性能汇总。
        # 作用：直接查看每个模块平均耗时。
        # ============================================================
        task_total_time = time.perf_counter() - task_total_start
        timing_summary = summarize_step_timing(step_timing_history)

        if profile_mode:
            print("\n[TIME SUMMARY]")
            print(f"Task total: {task_total_time:.2f}s")
            for name, stat in timing_summary.items():
                print(
                    f"{name}: avg={stat['avg_s']:.2f}s, "
                    f"total={stat['total_s']:.2f}s, n={stat['count']}"
                )

        # ============================================================
        # 原作者代码
        #
        # mission_logs[idx] = {
        #     "position": [
        #         pos.tolist()
        #         for pos in drone_pos
        #     ],
        #     "orientation": [
        #         ori.tolist()
        #         for ori in drone_ori
        #     ],
        #     "if_success": if_success
        # }
        # ============================================================

        # ============================================================
        # 2026-09-29 修改
        # 扩展为论文复现实验日志。
        # ============================================================
        mission_logs[idx] = {

            # --------------------------------------------------------
            # Task 基本信息
            # --------------------------------------------------------
            "task_index": idx,
            "mission_id": mission_id,
            "scene_id": scene_id,

            "mission_grade": mission_grade,
            "paper_difficulty": paper_difficulty,

            "target_text": target_text,
            "target_image_path": target_image_path,

            # --------------------------------------------------------
            # 实验参数
            # --------------------------------------------------------
            "theta_T": theta_T,
            "cognitive_threshold": cognitive_threshold,

            "step_x": scene["step_x"],
            "step_z": scene["step_z"],
            "step_all": max_steps,

            # --------------------------------------------------------
            # Task 结果
            # --------------------------------------------------------
            "steps": step,
            "if_success": int(if_success),
            "termination_reason": termination_reason,

            # --------------------------------------------------------
            # 2026-10-02 新增：边界保护日志
            # --------------------------------------------------------
            "boundary_reject_count": boundary_reject_count,
            "boundary_rejected_actions": boundary_rejected_actions,
            "valid_actions_history": valid_actions_history,

            # --------------------------------------------------------
            # 2026-10-02 新增：性能测速日志
            # --------------------------------------------------------
            "profile_mode": profile_mode,
            "profile_step_limit": args.profile_steps,
            "model_init_time_s": round(model_init_time, 4),
            "map_init_time_s": round(map_init_time, 4),
            "target_prompt_time_s": round(target_prompt_time, 4),
            "relevance_scores_time_s": round(relevance_scores_time, 4),
            "task_total_time_s": round(task_total_time, 4),
            "step_timing_history": step_timing_history,
            "timing_summary": timing_summary,

            # --------------------------------------------------------
            # 为后续论文指标计算保留的数据
            # --------------------------------------------------------
            "initial_target_distance": initial_target_distance,
            "final_navigation_error": final_navigation_error,
            "trajectory_length": trajectory_length,

            # --------------------------------------------------------
            # 每一步 Agent 决策
            # --------------------------------------------------------
            "action_history": action_history,
            "adviser_uncertainty_history": (
                adviser_uncertainty_history
            ),
            "adviser_cognitive_history": (
                adviser_cognitive_history
            ),
            "attraction_value_history": (
                attraction_value_history
            ),

            # --------------------------------------------------------
            # 飞行轨迹
            # --------------------------------------------------------
            "position": [
                pos.tolist()
                for pos in drone_pos
            ],

            "orientation": [
                ori.tolist()
                for ori in drone_ori
            ],
        }

        # ============================================================
        # 原作者代码
        #
        # with open(
        #     f"./output/experiemnt_data/ours/data_{idx}.json",
        #     "w",
        #     encoding="utf-8"
        # ) as file:
        #     json.dump(mission_logs[idx], file)
        #
        # print("Save data in scene-", idx + 1)
        # ============================================================

        # ============================================================
        # 2026-09-29 修改
        # 1. 文件名改成真实 mission_id，避免 idx 与任务编号错位。
        # 2. JSON 格式化保存，便于人工检查和后续统计。
        # 3. 输出真实 Scene、Difficulty 和任务结果。
        # ============================================================
        if profile_mode:
            result_filename = f"data_{mission_id}_profile.json"
        else:
            result_filename = f"data_{mission_id}.json"

        result_path = os.path.join(
            result_output_dir,
            result_filename
        )

        with open(
            result_path,
            "w",
            encoding="utf-8"
        ) as file:
            json.dump(
                mission_logs[idx],
                file,
                ensure_ascii=False,
                indent=2,
            )

        print(
            f"Saved Task {mission_id} | "
            f"Scene {scene_id} | "
            f"{paper_difficulty} | "
            f"Success={if_success} | "
            f"Steps={step}/{run_step_limit} | "
            f"Reason={termination_reason}"
        )

        print(
            "Result:",
            result_path
        )


if __name__ == "__main__":
    main()
