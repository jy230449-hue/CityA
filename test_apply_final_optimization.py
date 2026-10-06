# apply_final_optimization.py
from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path

ROOT = Path.cwd()
MAIN = ROOT / "main.py"
SEGMENT = ROOT / "Segment_Image.py"
SEMANTIC = ROOT / "update_semantic_map.py"

MAIN_PRE_PROFILE = ROOT / "main_before_full_profile_20261004.py"
SEM_PRE_PROFILE = ROOT / "update_semantic_map_before_full_profile_20261004.py"

STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")


def backup(path: Path):
    dst = path.with_name(f"{path.stem}_before_final_opt_{STAMP}{path.suffix}")
    shutil.copy2(path, dst)
    print(f"[BACKUP] {path.name} -> {dst.name}")


def restore_pre_profile_files():
    if not MAIN_PRE_PROFILE.exists():
        raise FileNotFoundError(f"缺少 {MAIN_PRE_PROFILE.name}")
    if not SEM_PRE_PROFILE.exists():
        raise FileNotFoundError(f"缺少 {SEM_PRE_PROFILE.name}")

    shutil.copy2(MAIN_PRE_PROFILE, MAIN)
    shutil.copy2(SEM_PRE_PROFILE, SEMANTIC)
    print("[RESTORE] main.py <- pre-full-profile backup")
    print("[RESTORE] update_semantic_map.py <- pre-full-profile backup")


def optimize_main():
    text = MAIN.read_text(encoding="utf-8")

    allocator_block = '''import os

# ============================================================
# 2026-10-04 最终性能优化：
# 减少 CUDA 动态分配时的显存碎片。
# 不改变模型权重、输入、阈值或数值计算公式。
# 必须位于 import torch 之前。
# ============================================================
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True",
)
'''

    if "expandable_segments:True" not in text:
        if "import os\n" not in text:
            raise RuntimeError("main.py 找不到 import os")
        text = text.replace("import os\n", allocator_block, 1)
        print("[OPT] main.py: expandable_segments enabled")

    old_action = '''            drone.take_action(
                action_label,
                scene["step_x"],
                scene["step_z"],
            )

            time.sleep(1)
            step_timing["action_execution_s"] = time.perf_counter() - t
'''

    new_action = '''            drone.take_action(
                action_label,
                scene["step_x"],
                scene["step_z"],
            )

            # 2026-10-04 最终性能优化：
            # take_action() 最终调用 simSetVehiclePose()。
            # 删除额外固定 1s sleep。
            step_timing["action_execution_s"] = time.perf_counter() - t
'''

    if old_action in text:
        text = text.replace(old_action, new_action, 1)
        print("[OPT] main.py: removed post-action sleep(1)")
    elif "time.sleep(1)" not in text:
        print("[OPT] main.py: post-action sleep(1) already absent")
    else:
        raise RuntimeError("未能安全删除 take_action 后的 time.sleep(1)")

    MAIN.write_text(text, encoding="utf-8")


def production_segment_observation():
    return '''def segment_observation(image_path, text_prompt, dino_model, sam_predictor, device="cuda"):
    # 生产版：保留模型/阈值/输入/输出逻辑，移除诊断 synchronize/显存查询。
    text_prompt = prepare_detection_prompt(text_prompt)

    image_pil = Image.open(image_path).convert("RGB")
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    image_tensor, _ = transform(image_pil, None)

    threshold_schedule = [
        (0.40, 0.25),
        (0.30, 0.20),
        (0.24, 0.18),
    ]

    boxes_filt = None
    logits_filt = None
    used_box_threshold = None
    used_text_threshold = None

    with torch.no_grad():
        outputs = dino_model(
            image_tensor[None].to(device),
            captions=[text_prompt]
        )

    logits = outputs["pred_logits"].cpu().sigmoid()[0]
    boxes = outputs["pred_boxes"].cpu()[0]

    for box_threshold, text_threshold in threshold_schedule:
        filt_mask = logits.max(dim=1)[0] > box_threshold
        candidate_boxes = boxes[filt_mask]

        print(
            "[DINO] prompt=",
            repr(text_prompt),
            "| box_threshold=",
            box_threshold,
            "| text_threshold=",
            text_threshold,
            "| detections=",
            int(candidate_boxes.shape[0]),
        )

        if candidate_boxes.shape[0] > 0:
            boxes_filt = candidate_boxes
            logits_filt = logits[filt_mask]
            used_box_threshold = box_threshold
            used_text_threshold = text_threshold
            break

    pred_phrases = []

    if boxes_filt is not None and boxes_filt.shape[0] > 0:
        tokenizer = dino_model.tokenizer
        tokenized = tokenizer(text_prompt)

        pred_phrases = [
            get_phrases_from_posmap(
                logit > used_text_threshold,
                tokenized,
                tokenizer
            ).replace('.', '')
            for logit in logits_filt
        ]
    else:
        print(
            f"Warning: No objects found for prompt after adaptive thresholds: "
            f"{text_prompt}"
        )
        return (
            np.array([]),
            np.array([]),
            {},
            torch.empty((0, 0, 0, 0)),
        )

    image_cv = cv2.imread(image_path)
    if image_cv is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)

    sam_predictor.set_image(image_cv)

    size = image_pil.size
    H, W = size[1], size[0]

    boxes_filt = boxes_filt * torch.tensor([W, H, W, H])
    boxes_filt[:, :2] -= boxes_filt[:, 2:] / 2
    boxes_filt[:, 2:] += boxes_filt[:, :2]

    boxes_filt = boxes_filt.to(device)

    transformed_boxes = sam_predictor.transform.apply_boxes_torch(
        boxes_filt,
        image_cv.shape[:2]
    )

    with torch.no_grad():
        masks, _, _ = sam_predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )

    class_ids_list = []
    class_names_list = []
    unique_name_to_id = {}
    next_id = 0

    for phrase in pred_phrases:
        pred_name = phrase.split('(')[0].strip().lower()

        if not pred_name:
            continue

        if pred_name not in unique_name_to_id:
            unique_name_to_id[pred_name] = next_id
            next_id += 1

        class_ids_list.append(unique_name_to_id[pred_name])
        class_names_list.append(pred_name)

    valid_count = min(len(class_ids_list), len(masks))

    if valid_count <= 0:
        print(
            "[DINO WARNING] boxes were found but no valid class phrase "
            "was decoded."
        )
        return (
            np.array([]),
            np.array([]),
            {},
            torch.empty((0, 0, 0, 0)),
        )

    masks = masks[:valid_count]
    class_ids_list = class_ids_list[:valid_count]
    class_names_list = class_names_list[:valid_count]

    masks_np = masks.squeeze(1).cpu().numpy().astype(bool)
    class_ids_np = np.array(class_ids_list, dtype=int)
    class_names_np = np.array(class_names_list)

    class_name_to_id = {
        str(name): int(id)
        for name, id in zip(class_names_np, class_ids_np)
    }

    print(
        "[DINO] accepted classes:",
        class_names_np.tolist(),
        "| masks:",
        len(masks),
        "| threshold:",
        (used_box_threshold, used_text_threshold),
    )

    # 必须保留：Full Profile 实测整卡最低空闲显存只有 277MiB。
    # Qwen 是独立进程，需要把 DINO/SAM 的缓存显存交还给系统。
    torch.cuda.empty_cache()

    return class_ids_np, class_names_np, class_name_to_id, masks
'''


def optimize_segment():
    text = SEGMENT.read_text(encoding="utf-8")

    start = text.find("def segment_observation(")
    if start < 0:
        raise RuntimeError("Segment_Image.py 找不到 segment_observation()")

    end = text.find('\n\nif __name__ == "__main__":', start)
    if end < 0:
        raise RuntimeError("Segment_Image.py 找不到 __main__ 分隔位置")

    text = text[:start] + production_segment_observation() + text[end:]
    SEGMENT.write_text(text, encoding="utf-8")

    print("[OPT] Segment_Image.py: diagnostic profile removed")
    print("[OPT] Segment_Image.py: one required empty_cache kept")


def vectorized_add_label():
    return '''def add_label_to_semantic_map(world_points, labels, semantic_map, cognitive_map, scores_rel):
    # 等价优化：
    # - np.round() 规则不变
    # - dominant label 规则不变
    # - count 相同时仍取原输入中最先出现的 label
    # - 地图索引和 score 公式不变

    if world_points is None or labels is None:
        return semantic_map, cognitive_map

    world_points = np.asarray(world_points)
    labels = np.asarray(labels)

    if len(world_points) == 0 or len(labels) == 0:
        return semantic_map, cognitive_map

    finite_mask = np.all(np.isfinite(world_points), axis=1)

    if not np.any(finite_mask):
        return semantic_map, cognitive_map

    valid_points = world_points[finite_mask]
    valid_labels = labels[finite_mask]

    rounded_points = np.round(valid_points).astype(np.int64)

    unique_points, inverse = np.unique(
        rounded_points,
        axis=0,
        return_inverse=True,
    )

    if len(unique_points) == 0:
        return semantic_map, cognitive_map

    pair_data = np.column_stack((
        inverse.astype(np.int64),
        valid_labels,
    ))

    unique_pairs, first_indices, counts = np.unique(
        pair_data,
        axis=0,
        return_index=True,
        return_counts=True,
    )

    group_ids = unique_pairs[:, 0].astype(np.int64)

    # group 升序、count 降序、first occurrence 升序。
    order = np.lexsort((
        first_indices,
        -counts,
        group_ids,
    ))

    sorted_pairs = unique_pairs[order]
    sorted_group_ids = group_ids[order]

    _, first_for_group = np.unique(
        sorted_group_ids,
        return_index=True,
    )

    winning_pairs = sorted_pairs[first_for_group]

    winning_groups = winning_pairs[:, 0].astype(np.int64)
    dominant_labels = winning_pairs[:, 1]
    dominant_points = unique_points[winning_groups]

    meta = _SEMANTIC_GRID_META.get(id(semantic_map))

    if meta is not None:
        fx = (
            dominant_points[:, 0].astype(np.float64)
            - meta["x0"]
        ) / meta["dx"]

        fy = (
            dominant_points[:, 1].astype(np.float64)
            - meta["y0"]
        ) / meta["dy"]

        fz = (
            dominant_points[:, 2].astype(np.float64)
            - meta["z0"]
        ) / meta["dz"]

        ix = np.rint(fx).astype(np.int64)
        iy = np.rint(fy).astype(np.int64)
        iz = np.rint(fz).astype(np.int64)

        valid_grid = (
            np.isclose(fx, ix, atol=1e-9)
            & np.isclose(fy, iy, atol=1e-9)
            & np.isclose(fz, iz, atol=1e-9)
            & (ix >= 0)
            & (ix < meta["nx"])
            & (iy >= 0)
            & (iy < meta["ny"])
            & (iz >= 0)
            & (iz < meta["nz"])
        )

        if not np.any(valid_grid):
            return semantic_map, cognitive_map

        ix = ix[valid_grid]
        iy = iy[valid_grid]
        iz = iz[valid_grid]
        dominant_labels = dominant_labels[valid_grid]

        map_indices = (
            (ix * meta["ny"] + iy) * meta["nz"] + iz
        ).astype(np.int64)

        semantic_map[map_indices, 3] = dominant_labels

        score_array = np.asarray(scores_rel)
        label_indices = dominant_labels.astype(np.int64) - 1

        cognitive_map[map_indices, 3] = (
            score_array[label_indices]
            * cognitive_map[map_indices, 4]
        )

        return semantic_map, cognitive_map

    # 非规则/外部 semantic_map 保留旧兼容路径。
    for key, dominant_label in zip(
        dominant_points,
        dominant_labels,
    ):
        idx = np.where(
            (semantic_map[:, 0] == key[0])
            & (semantic_map[:, 1] == key[1])
            & (semantic_map[:, 2] == key[2])
        )

        if idx[0].size > 0:
            idx_value = int(idx[0][0])

            semantic_map[idx_value, 3] = dominant_label
            cognitive_map[idx_value, 3] = (
                scores_rel[int(dominant_label) - 1]
                * cognitive_map[idx_value, 4]
            )

    return semantic_map, cognitive_map
'''


def optimize_semantic():
    text = SEMANTIC.read_text(encoding="utf-8")

    start = text.find("def add_label_to_semantic_map(")
    if start < 0:
        raise RuntimeError(
            "update_semantic_map.py 找不到 add_label_to_semantic_map()"
        )

    end = text.find("\ndef visualize_semantic_map", start)
    if end < 0:
        raise RuntimeError(
            "update_semantic_map.py 找不到 visualize_semantic_map()"
        )

    text = text[:start] + vectorized_add_label() + text[end:]
    SEMANTIC.write_text(text, encoding="utf-8")

    print("[OPT] update_semantic_map.py: vectorized add_label")


def syntax_check():
    for p in [MAIN, SEGMENT, SEMANTIC]:
        source = p.read_text(encoding="utf-8")
        compile(source, str(p), "exec")
        print(f"[CHECK] syntax OK: {p.name}")


def main():
    print("=" * 78)
    print("CityAVOS FINAL PERFORMANCE OPTIMIZATION")
    print("=" * 78)
    print("Qwen must stay at GPU_MAX_MEMORY = 8GiB.")
    print("No paper/model/threshold/resolution/navigation parameter will be changed.")
    print()

    for p in [MAIN, SEGMENT, SEMANTIC]:
        if not p.exists():
            raise FileNotFoundError(f"Missing: {p}")
        backup(p)

    restore_pre_profile_files()

    optimize_main()
    optimize_segment()
    optimize_semantic()

    syntax_check()

    print()
    print("=" * 78)
    print("FINAL OPTIMIZATION APPLIED")
    print("=" * 78)
    print("1. Qwen: KEEP 8GiB")
    print("2. Diagnostic synchronize/VRAM queries removed")
    print("3. One required empty_cache kept per Step")
    print("4. Semantic add_label vectorized")
    print("5. Redundant post-action sleep(1) removed")
    print("6. CUDA allocator expandable_segments enabled")
    print()
    print("Acceptance command:")
    print("  python main.py --task 8 --profile-steps 5")
    print("=" * 78)


if __name__ == "__main__":
    main()
