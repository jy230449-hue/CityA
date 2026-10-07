from vln.drone_agent import AirsimAgent

import json

import sys

import os
import re



# ============================================================

# 2026-10-04 最终性能优化：

# 减少 CUDA 动态分配时的显存碎片。

# 不改变模型权重、输入、阈值或数值计算公式。

# 必须位于 import torch 之前。

# ============================================================

# os.environ.setdefault(

#    "PYTORCH_CUDA_ALLOC_CONF",

#    "expandable_segments:True",

# )

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





# ============================================================

# 2026-10-04 新增：Task 元数据缓存。

# 作用：断点续跑时复用同一次实验已生成的云端结果。

# ============================================================

def load_or_create_task_metadata(

    task_item,

    cache_dir,

    use_cache=False,

    save_cache=True,

):

    mission_id = str(task_item["mission_id"])

    target_text = task_item["target_text"]

    target_image_path = os.path.normpath(

        task_item["target_image_path"]

    )



    os.makedirs(cache_dir, exist_ok=True)

    cache_path = os.path.join(

        cache_dir,

        f"task_{mission_id}.json",

    )



    if use_cache and os.path.isfile(cache_path):

        try:

            with open(cache_path, "r", encoding="utf-8") as file:

                cached = json.load(file)



            if (

                str(cached.get("mission_id")) == mission_id

                and cached.get("target_text") == target_text

                and os.path.normpath(

                    cached.get("target_image_path", "")

                ) == target_image_path

            ):

                return cached, True

        except Exception as e:

            print(

                f"[CACHE] Task {mission_id} 缓存无效，重新计算: {e}"

            )



    target_prompt_start = time.perf_counter()



    prompt_sam_raw = chat_with_llm(

        prompt_rel,

        target_image_path,

    )



    prompt_sam = prepare_detection_prompt(

        prompt_sam_raw,

        target_text=target_text,

    )



    target_prompt_time = (

        time.perf_counter() - target_prompt_start

    )



    relevance_start = time.perf_counter()



    scores_rel = get_relevance_scores(

        prompt_sam,

        target_text,

        target_image_path,

    )



    relevance_scores_time = (

        time.perf_counter() - relevance_start

    )



    metadata = {

        "mission_id": mission_id,

        "target_text": target_text,

        "target_image_path": target_image_path,

        "prompt_sam_raw": prompt_sam_raw,

        "prompt_sam": prompt_sam,

        "scores_rel": scores_rel,

        "target_prompt_time_s": round(

            target_prompt_time,

            4,

        ),

        "relevance_scores_time_s": round(

            relevance_scores_time,

            4,

        ),

    }



    if save_cache:

        temp_path = cache_path + ".tmp"



        with open(

            temp_path,

            "w",

            encoding="utf-8",

        ) as file:

            json.dump(

                metadata,

                file,

                ensure_ascii=False,

                indent=2,

            )



        os.replace(temp_path, cache_path)



    return metadata, False





# ============================================================

# 2026-10-04 新增：断点结果检查。

# 作用：--resume 时只跳过完整正式结果。

# ============================================================

def is_completed_result(result_path, mission_id):

    if not os.path.isfile(result_path):

        return False



    try:

        with open(result_path, "r", encoding="utf-8") as file:

            data = json.load(file)



        return (

            str(data.get("mission_id")) == str(mission_id)

            and int(data.get("if_success")) in (0, 1)

            and isinstance(data.get("steps"), int)

            and not bool(data.get("profile_mode"))

        )

    except Exception:

        return False





# ============================================================

# 2026-10-04 新增：断点状态。

# 作用：记录最近完成的正式 Task。

# ============================================================

def save_run_state(

    run_state_path,

    mission_id,

    task_index,

    result_path,

):

    state = {

        "last_completed_mission_id": str(mission_id),

        "last_completed_task_index": int(task_index),

        "result_path": result_path,

    }



    temp_path = run_state_path + ".tmp"



    with open(

        temp_path,

        "w",

        encoding="utf-8",

    ) as file:

        json.dump(

            state,

            file,

            ensure_ascii=False,

            indent=2,

        )



    os.replace(temp_path, run_state_path)





# ============================================================
# 2026-10-07 新增：纯统计 Audit
# 作用：只统计 DINO -> relevance 匹配和 Adviser 触发情况。
# 不修改 class_scores、地图、阈值、Prompt 或动作决策。
# ============================================================
def audit_relevance_mapping(class_names, scores_rel):
    def normalize_str(value):
        return re.sub(r'[\s\-_]', '', str(value)).lower()

    result = {
        "detection_count": 0,
        "exact_match_count": 0,
        "fallback_count": 0,
        "matched_classes": [],
        "fallback_classes": [],
    }

    if class_names is None:
        return result

    class_names_list = [str(name) for name in list(class_names)]
    result["detection_count"] = len(class_names_list)

    if not isinstance(scores_rel, dict):
        result["fallback_count"] = len(class_names_list)
        result["fallback_classes"] = class_names_list
        return result

    normalized_scores = {
        normalize_str(key): key
        for key in scores_rel.keys()
    }

    for name in class_names_list:
        clean_name = normalize_str(name)
        if clean_name in normalized_scores:
            score_key = normalized_scores[clean_name]
            result["exact_match_count"] += 1
            result["matched_classes"].append({
                "dino_name": name,
                "score_key": score_key,
                "score": float(scores_rel[score_key]),
            })
        else:
            result["fallback_count"] += 1
            result["fallback_classes"].append(name)

    return result


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



    # ============================================================

    # 2026-10-04 新增：断点续跑。

    # 作用：跳过已完成正式 Task，并复用该实验缓存。

    # ============================================================

    parser.add_argument(

        "--resume",

        action="store_true",

        help="Resume formal run from completed task files",

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

    # 2026-10-04 修改：正式默认运行全部 605 个 Task。

    # 单任务仍使用 --task；测速仍使用 --profile-steps。

    # ============================================================

    task_end = len(dataset)



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

        result_output_dir = "./output/experiemnt_data/audit"



    os.makedirs(result_output_dir, exist_ok=True)



    # ============================================================

    # 2026-10-04 新增：正式运行辅助目录。

    # ============================================================

    metadata_cache_dir = "./output/task_metadata_cache"

    run_state_dir = "./output/run_state"

    logs_dir = "./output/logs"



    os.makedirs(metadata_cache_dir, exist_ok=True)

    os.makedirs(run_state_dir, exist_ok=True)

    os.makedirs(logs_dir, exist_ok=True)



    run_state_path = os.path.join(

        run_state_dir,

        "run_state_audit.json",

    )



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



        # ============================================================

        # 2026-10-04 新增：正式断点续跑。

        # 只跳过可正常解析的完整结果文件。

        # ============================================================

        if args.resume and not profile_mode:

            completed_path = os.path.join(

                result_output_dir,

                f"data_{mission_id}.json",

            )



            if is_completed_result(

                completed_path,

                mission_id,

            ):

                print(

                    f"[RESUME] Skip completed Task {mission_id}"

                )

                continue



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

        # 修改1：记录安全恢复次数，方便后续统计。
        action_recovery_count = 0
        action_recovery_history = []

        # ============================================================
        # Audit：当前 Task 的纯统计计数。
        # 只记录，不参与任何算法计算。
        # ============================================================
        audit_stats = {
            "dino_detection_count": 0,
            "dino_nonempty_step_count": 0,
            "dino_empty_step_count": 0,
            "relevance_exact_match_count": 0,
            "relevance_fallback_count": 0,
            "cognitive_adviser_count": 0,
            "uncertainty_adviser_count": 0,
            "both_adviser_count": 0,
            "no_adviser_count": 0,
        }
        audit_step_history = []



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

        # 2026-10-04 修改：Task 元数据缓存。

        # 首次正式运行仍按原流程计算；--resume 才复用缓存。

        # Profile 不读写正式缓存。

        # ============================================================

        task_metadata, metadata_from_cache = (

            load_or_create_task_metadata(

                task_item,

                metadata_cache_dir,

                use_cache=(

                    args.resume

                    and not profile_mode

                ),

                save_cache=not profile_mode,

            )

        )



        prompt_sam_raw = task_metadata[

            "prompt_sam_raw"

        ]

        prompt_sam = task_metadata[

            "prompt_sam"

        ]

        scores_rel = task_metadata[

            "scores_rel"

        ]



        target_prompt_time = float(

            task_metadata.get(

                "target_prompt_time_s",

                0.0,

            )

        )

        relevance_scores_time = float(

            task_metadata.get(

                "relevance_scores_time_s",

                0.0,

            )

        )



        print("[DINO PROMPT] raw:", prompt_sam_raw)

        print("[DINO PROMPT] prepared:", prompt_sam)



        if metadata_from_cache:

            print(

                f"[CACHE] Reuse Task {mission_id} metadata"

            )



        if profile_mode:

            print(

                f"[TIME] Target prompt: "

                f"{target_prompt_time:.2f}s"

            )

            print(

                f"[TIME] Relevance scores: "

                f"{relevance_scores_time:.2f}s"

            )



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



            t = time.perf_counter()

            class_scores = get_scores_for_class_names(

                class_names,

                scores_rel

            )

            # ============================================================
            # Audit：旁路统计当前 DINO 类别与 scores_rel 的匹配情况。
            # 不修改 class_scores。
            # ============================================================
            relevance_audit = audit_relevance_mapping(
                class_names,
                scores_rel,
            )
            audit_stats["dino_detection_count"] += (
                relevance_audit["detection_count"]
            )
            if relevance_audit["detection_count"] > 0:
                audit_stats["dino_nonempty_step_count"] += 1
            else:
                audit_stats["dino_empty_step_count"] += 1
            audit_stats["relevance_exact_match_count"] += (
                relevance_audit["exact_match_count"]
            )
            audit_stats["relevance_fallback_count"] += (
                relevance_audit["fallback_count"]
            )



            depth_plus_id, depth_plus_score = (

                overlay_masks_on_depth(

                    img2,

                    masks,

                    class_ids,

                    class_scores,

                )

            )



            semantic_map, cognitive_map = update_semantic_map(

                semantic_map,

                cognitive_map,

                depth_path,

                drone.pos * [1, -1, -1],

                [-90, 0, 270] - drone.ori,

                depth_plus_id,

                class_scores,

            )



            max_value, max_cluster_center = (

                find_max_cluster_center(

                    cognitive_map

                )

            )

            step_timing["semantic_cognitive_s"] = time.perf_counter() - t



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
            # Audit：统计两个 Adviser 是否产生建议，并保存当前 Step。
            # ============================================================
            if adviser_cognitive_map is not None:
                audit_stats["cognitive_adviser_count"] += 1
            if adviser_uncertainty_map is not None:
                audit_stats["uncertainty_adviser_count"] += 1
            if (
                adviser_cognitive_map is not None
                and adviser_uncertainty_map is not None
            ):
                audit_stats["both_adviser_count"] += 1
            if (
                adviser_cognitive_map is None
                and adviser_uncertainty_map is None
            ):
                audit_stats["no_adviser_count"] += 1

            audit_step_history.append({
                "step": int(step),
                "dino_detection_count": int(
                    relevance_audit["detection_count"]
                ),
                "relevance_exact_match_count": int(
                    relevance_audit["exact_match_count"]
                ),
                "relevance_fallback_count": int(
                    relevance_audit["fallback_count"]
                ),
                "matched_classes": relevance_audit["matched_classes"],
                "fallback_classes": relevance_audit["fallback_classes"],
                "cognitive_adviser": adviser_cognitive_map,
                "uncertainty_adviser": adviser_uncertainty_map,
                "cognitive_max_value": float(max_value),
            })



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
            # ============================================================
            # 修改2：动作决策 / 边界保护修复
            # 保留物理边界；无有效动作时安全恢复，不再直接结束 Task。
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

            # 越界 Adviser 不再传给 LLM，避免 Prompt 冲突。
            if (
                adviser_cognitive_for_llm is not None
                and adviser_cognitive_for_llm not in valid_action_names
            ):
                print(
                    "[Boundary Filter Adviser] cognitive:",
                    adviser_cognitive_for_llm,
                    "-> None",
                )
                adviser_cognitive_for_llm = None

            if (
                adviser_uncertainty_for_llm is not None
                and adviser_uncertainty_for_llm not in valid_action_names
            ):
                print(
                    "[Boundary Filter Adviser] uncertainty:",
                    adviser_uncertainty_for_llm,
                    "-> None",
                )
                adviser_uncertainty_for_llm = None

            qwen_total_start = time.perf_counter()

            # get_action_from_llm() 内部负责重试。
            # 配套修改要求：3 次仍失败时返回 None，而不是返回 Stop(7)。
            action_label = get_action_from_llm(
                adviser_cognitive_for_llm,
                adviser_uncertainty_for_llm,
                target_text,
                target_image_path,
                rgb_path,
                attraction_value,
                valid_actions=valid_action_names,
                action_history=action_history,
                position_history=drone_pos,
            )

            print("agent_choose_action:", action_label)

            # 修改2.1：None 代表“没有可执行动作”，不是 Stop。
            if action_label is None:
                recovery_action = None

                # 优先使用仍合法的 Adviser。
                if (
                    adviser_cognitive_for_llm is not None
                    and adviser_cognitive_for_llm in valid_action_names
                ):
                    recovery_action = adviser_cognitive_for_llm
                elif (
                    adviser_uncertainty_for_llm is not None
                    and adviser_uncertainty_for_llm in valid_action_names
                ):
                    recovery_action = adviser_uncertainty_for_llm

                # 没有 Adviser 时原地转向，不改变 x/y/z，不会飞出边界。
                elif "Turn Left" in valid_action_names:
                    recovery_action = "Turn Left"
                elif "Turn Right" in valid_action_names:
                    recovery_action = "Turn Right"

                if recovery_action is None:
                    termination_reason = "no_safe_action"
                    step_timing["qwen_vlm_s"] = (
                        time.perf_counter() - qwen_total_start
                    )
                    step_timing["total_step_s"] = (
                        time.perf_counter() - step_total_start
                    )
                    step_timing_history.append(step_timing)
                    print("[ACTION RECOVERY] No safe action is available.")
                    break

                action_label = action_set.index(recovery_action)
                action_recovery_count += 1
                action_recovery_history.append({
                    "step": int(step),
                    "action": recovery_action,
                    "reason": "llm_no_valid_action",
                    "valid_actions": list(valid_action_names),
                })
                print(
                    "[ACTION RECOVERY] "
                    f"Fallback safe action = {recovery_action}"
                )

            # 修改2.2：只有模型明确返回 7 才记为主动 Stop。
            if action_label == 7:
                if termination_reason == "unknown":
                    termination_reason = "llm_explicit_stop"
                step_timing["qwen_vlm_s"] = (
                    time.perf_counter() - qwen_total_start
                )
                step_timing["total_step_s"] = (
                    time.perf_counter() - step_total_start
                )
                step_timing_history.append(step_timing)
                break

            # 修改2.3：执行前保留最后一道硬边界检查。
            next_pos = predict_next_position(
                drone,
                action_label,
                scene,
            )

            if not is_position_in_scene(next_pos, scene):
                boundary_reject_count += 1

                rejected_action_name = (
                    action_set[action_label]
                    if isinstance(action_label, int)
                    and 0 <= action_label < len(action_set)
                    else str(action_label)
                )

                current_map_pos = np.asarray(
                    drone.pos * [1, -1, -1],
                    dtype=float,
                )

                boundary_rejected_actions.append({
                    "step": int(step),
                    "attempt": 1,
                    "action": rejected_action_name,
                    "current_position": current_map_pos.tolist(),
                    "predicted_position": next_pos.tolist(),
                    "valid_actions": list(valid_action_names),
                })

                print(
                    "[Boundary Reject]",
                    f"Step={step}",
                    f"Action={rejected_action_name}",
                    f"Current={current_map_pos.tolist()}",
                    f"Predicted={next_pos.tolist()}",
                    f"Valid={valid_action_names}",
                )

                # 最终保险：越界动作不执行，改为原地安全转向。
                if "Turn Left" in valid_action_names:
                    recovery_action = "Turn Left"
                elif "Turn Right" in valid_action_names:
                    recovery_action = "Turn Right"
                else:
                    recovery_action = None

                if recovery_action is None:
                    termination_reason = "no_safe_action"
                    step_timing["qwen_vlm_s"] = (
                        time.perf_counter() - qwen_total_start
                    )
                    step_timing["total_step_s"] = (
                        time.perf_counter() - step_total_start
                    )
                    step_timing_history.append(step_timing)
                    print("[ACTION RECOVERY] No safe boundary action.")
                    break

                action_label = action_set.index(recovery_action)
                action_recovery_count += 1
                action_recovery_history.append({
                    "step": int(step),
                    "action": recovery_action,
                    "reason": "hard_boundary_recovery",
                    "valid_actions": list(valid_action_names),
                })
                print(
                    "[ACTION RECOVERY] "
                    f"Boundary fallback = {recovery_action}"
                )

            step_timing["qwen_vlm_s"] = (
                time.perf_counter() - qwen_total_start
            )

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



            # 2026-10-04 最终性能优化：

            # take_action() 最终调用 simSetVehiclePose()。

            # 删除额外固定 1s sleep。

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
        # Audit：Task 结束后生成汇总。
        # ============================================================
        relevance_total_count = (
            audit_stats["relevance_exact_match_count"]
            + audit_stats["relevance_fallback_count"]
        )
        if relevance_total_count > 0:
            audit_stats["relevance_match_rate"] = round(
                audit_stats["relevance_exact_match_count"]
                / relevance_total_count,
                6,
            )
            audit_stats["relevance_fallback_rate"] = round(
                audit_stats["relevance_fallback_count"]
                / relevance_total_count,
                6,
            )
        else:
            audit_stats["relevance_match_rate"] = 0.0
            audit_stats["relevance_fallback_rate"] = 0.0

        audit_stats["executed_steps"] = int(step)
        audit_stats["action_recovery_count"] = int(
            action_recovery_count
        )
        audit_stats["boundary_reject_count"] = int(
            boundary_reject_count
        )

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

            # 修改3：记录安全恢复，便于统计。
            "action_recovery_count": action_recovery_count,
            "action_recovery_history": action_recovery_history,

            # --------------------------------------------------------
            # 2026-10-07 新增：纯统计 Audit
            # --------------------------------------------------------
            "audit": audit_stats,
            "audit_step_history": audit_step_history,



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

            # 2026-10-04 新增：四指标复核原始坐标。

            "metric_initial_position": (

                initial_drone_pos_map.tolist()

            ),

            "metric_target_position": (

                target_pos_map.tolist()

            ),

            "metric_final_position": (

                final_drone_pos_map.tolist()

            ),



            "initial_target_distance": initial_target_distance,

            "final_navigation_error": final_navigation_error,

            "trajectory_length": trajectory_length,



            # 2026-10-04 新增：记录元数据是否来自断点缓存。

            "metadata_from_cache": bool(

                metadata_from_cache

            ),



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



        # ============================================================

        # 2026-10-04 新增：正式 Task 完成后记录断点。

        # ============================================================

        if not profile_mode:

            save_run_state(

                run_state_path,

                mission_id,

                idx,

                result_path,

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
            "[AUDIT] "
            f"DINO={audit_stats['dino_detection_count']} | "
            f"Match={audit_stats['relevance_exact_match_count']} | "
            f"Fallback={audit_stats['relevance_fallback_count']} | "
            f"MatchRate={audit_stats['relevance_match_rate']:.2%} | "
            f"Cog={audit_stats['cognitive_adviser_count']} | "
            f"Unc={audit_stats['uncertainty_adviser_count']} | "
            f"Both={audit_stats['both_adviser_count']} | "
            f"None={audit_stats['no_adviser_count']} | "
            f"Recovery={action_recovery_count}"
        )

        print(

            "Result:",

            result_path

        )





if __name__ == "__main__":

    main()
