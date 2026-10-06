# ============================================================
# Segment_Image.py
#
# 基础：
# - 保留 CityAVOS 官方当前 Segment_Image.py 的主要结构与函数。
#
# 2026-10-03 修改：
# 1. 新增 prepare_detection_prompt()
# 2. segment_observation() 新增自适应阈值回退
# 3. get_relevance_scores() 增加稳健处理
# 4. get_scores_for_class_names() 增加 scores=None 保护
#
# 2026-10-04 新增：
# 5. segment_observation() 增加 DINO / SAM 细分性能 Profile。
#    - 只增加计时和显存打印，不修改模型、阈值、输入、输出或导航逻辑。
# ============================================================

import argparse
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import re
import time  # 2026-10-04 新增：性能测速

import numpy as np
import json
import torch
from PIL import Image
from llm_agent import chat_with_llm

sys.path.append(os.path.join(os.getcwd(), "GroundSAM/GroundingDINO"))
sys.path.append(os.path.join(os.getcwd(), "GroundSAM/segment_anything"))

# Grounding DINO
import GroundSAM.GroundingDINO.groundingdino.datasets.transforms as T
from GroundSAM.GroundingDINO.groundingdino.models import build_model
from GroundSAM.GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundSAM.GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

# segment anything
from segment_anything import (
    sam_model_registry,
    sam_hq_model_registry,
    SamPredictor
)
import cv2
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


# ============================================================
# 2026-10-04 新增：GPU 显存状态
# 只用于 profile，不改变计算结果。
# ============================================================
def print_gpu_profile(tag):
    if not torch.cuda.is_available():
        print(f"[GPU PROFILE] {tag} | CUDA unavailable")
        return

    try:
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        free_bytes, total_bytes = torch.cuda.mem_get_info(0)
        free_gb = free_bytes / 1024**3
        total_gb = total_bytes / 1024**3

        print(
            f"[GPU PROFILE] {tag} | "
            f"allocated={allocated:.2f}GB | "
            f"reserved={reserved:.2f}GB | "
            f"global_free={free_gb:.2f}/{total_gb:.2f}GB"
        )
    except Exception as e:
        print(
            f"[GPU PROFILE] {tag} | "
            f"memory query failed: {e!r}"
        )


# ============================================================
# 2026-10-04 新增：CUDA 同步
# GPU 运算异步，测速前后同步才能得到真实耗时。
# ============================================================
def cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


# ============================================================
# 2026-10-03 新增：GroundingDINO Prompt 清洗
# ============================================================
def prepare_detection_prompt(text_prompt, target_text=None, max_objects=10):
    raw = str(text_prompt or "").lower()
    parts = re.split(r"[.\n,;|]+", raw)

    non_object_exact = {
        "text", "texts", "number", "numbers", "phone number",
        "phone numbers", "professional wording", "wording",
        "service list", "rectangular shape", "shape", "color",
        "colors", "yellow background", "red background",
        "blue background", "white background", "black background",
        "background",
    }

    weak_attribute_terms = (
        "background", "wording", "number", "text", "shape", "color",
    )

    physical_object_hints = (
        "poster", "sign", "signboard", "board", "notice",
        "building", "wall", "window", "door", "shop", "store",
        "car", "vehicle", "bus", "truck", "van", "taxi",
        "bicycle", "bike", "motorcycle", "scooter",
        "person", "pedestrian", "tree", "pole", "lamp",
        "traffic light", "traffic sign", "road", "sidewalk",
        "bench", "fence", "bridge", "tower", "gate",
        "billboard", "advertisement", "banner",
    )

    objects = []

    def add_object(name):
        name = re.sub(r"[\[\]\(\)\"']", "", str(name)).strip().lower()
        name = re.sub(r"\s+", " ", name)

        if not name:
            return
        if name in non_object_exact:
            return
        if (
            any(term in name for term in weak_attribute_terms)
            and not any(hint in name for hint in physical_object_hints)
        ):
            return
        if len(name.split()) > 6:
            return
        if name not in objects:
            objects.append(name)

    for part in parts:
        add_object(part)

    target_lower = str(target_text or "").lower()

    target_alias_groups = [
        (("poster",), ["poster", "wall poster", "sign", "signboard", "notice board"]),
        (("sign", "signboard"), ["sign", "signboard", "wall sign", "billboard"]),
        (("billboard", "advertisement"), ["billboard", "advertisement board", "signboard"]),
        (("building", "house"), ["building", "building facade", "wall", "window"]),
        (("car", "vehicle", "automobile"), ["car", "vehicle"]),
        (("bus",), ["bus", "vehicle"]),
        (("truck",), ["truck", "vehicle"]),
        (("bicycle", "bike"), ["bicycle", "bike"]),
        (("motorcycle", "motorbike"), ["motorcycle", "motorbike"]),
        (("traffic light",), ["traffic light", "traffic signal"]),
        (("traffic sign",), ["traffic sign", "road sign", "sign"]),
        (("person", "pedestrian"), ["person", "pedestrian"]),
        (("tree",), ["tree"]),
    ]

    for keywords, aliases in target_alias_groups:
        if any(keyword in target_lower for keyword in keywords):
            for alias in aliases:
                add_object(alias)

    generic_urban_anchors = ["building", "wall", "window", "road", "sidewalk"]

    for anchor in generic_urban_anchors:
        if len(objects) >= 3:
            break
        add_object(anchor)

    objects = objects[:max_objects]
    prepared = ". ".join(objects)
    if prepared and not prepared.endswith("."):
        prepared += "."
    if not prepared:
        prepared = "building. wall. window."
    return prepared


def load_image(image_path):
    image_pil = Image.open(image_path).convert("RGB")
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    image, _ = transform(image_pil, None)
    return image_pil, image


def load_model(model_config_path, model_checkpoint_path, bert_base_uncased_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    args.bert_base_uncased_path = bert_base_uncased_path
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model


def get_grounding_output(model, image, caption, box_threshold, text_threshold, with_logits=True, device="cpu"):
    caption = caption.lower().strip()
    if not caption.endswith("."):
        caption = caption + "."
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]
    boxes = outputs["pred_boxes"].cpu()[0]

    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]
    boxes_filt = boxes_filt[filt_mask]

    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    pred_phrases = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        if with_logits:
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        else:
            pred_phrases.append(pred_phrase)
    return boxes_filt, pred_phrases


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax, label):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))
    ax.text(x0, y0, label)


def save_mask_data(output_dir, mask_list, box_list, label_list):
    value = 0
    mask_img = torch.zeros(mask_list.shape[-2:])
    for idx, mask in enumerate(mask_list):
        mask_img[mask.cpu().numpy()[0] == True] = value + idx + 1
    plt.figure(figsize=(10, 10))
    plt.imshow(mask_img.numpy())
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, 'mask.jpg'), bbox_inches="tight", dpi=300, pad_inches=0.0)

    json_data = [{'value': value, 'label': 'background'}]
    for label, box in zip(label_list, box_list):
        value += 1
        name, logit = label.split('(')
        logit = logit[:-1]
        json_data.append({
            'value': value,
            'label': name,
            'logit': float(logit),
            'box': box.numpy().tolist(),
        })
    with open(os.path.join(output_dir, 'mask.json'), 'w') as f:
        json.dump(json_data, f)


def get_relevance_scores(objects, target_text, target_image_path):
    objects = prepare_detection_prompt(objects, target_text=target_text)
    objects_list = [item.strip() for item in objects.split('.') if item.strip()]
    prompt = (
        f"You are looking for the ['{target_text}'] in the image. "
        f"Please analyze the relevance of the following {len(objects_list)} objects('{objects}') "
        "to the search target and give a score between 0 and 1 (rounded to two decimal places)."
        "When analyzing the relevance, consider in sequence whether the search target is likely to "
        "exist in the object/scene to be scored."
        "0 indicates completely impossible, and 1 indicates highly likely."
        "Scores are required to be evenly distributed between 0 and 1."
        "Only return the score numbers, separated by commas, without any other words"
    )

    response = chat_with_llm(prompt, ("./" + target_image_path))

    try:
        number_strings = re.findall(
            r"(?<![\w.])(?:0(?:\.\d+)?|1(?:\.0+)?)(?![\w.])",
            str(response),
        )
        scores = [max(0.0, min(1.0, float(score))) for score in number_strings]

        if len(scores) != len(objects_list):
            print(
                "[REL SCORE WARNING] score count mismatch | "
                f"expected={len(objects_list)} got={len(scores)} | raw={response!r}"
            )
            scores = scores[:len(objects_list)]
            while len(scores) < len(objects_list):
                scores.append(0.5)

        return dict(zip(objects_list, scores))

    except Exception as e:
        print(f"处理LLM响应时出错: {e}")
        return {obj: 0.5 for obj in objects_list}


def overlay_masks_on_depth(depth_image, masks, class_ids, class_scores):
    masks_enhanced_id = np.zeros((480, 640), dtype=np.float32)
    masks_enhanced_score = np.zeros((480, 640), dtype=np.float32)

    for idx, mask in enumerate(masks):
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy().squeeze()
        mask_val = mask.astype(np.float32)
        current_id_map = mask_val * (class_ids[idx] + 1)
        masks_enhanced_id = np.maximum(masks_enhanced_id, current_id_map)
        current_score_map = mask_val * (class_scores[idx] + 1)
        masks_enhanced_score = np.maximum(masks_enhanced_score, current_score_map)

    return masks_enhanced_id, masks_enhanced_score


def get_scores_for_class_names(class_names, scores):
    if not isinstance(scores, dict):
        print(
            "[REL SCORE WARNING] scores is not a dict; "
            "using default 0.5 for detected classes."
        )
        return [0.5 for _ in class_names]

    def normalize_str(s):
        return re.sub(r'[\s\-_]', '', str(s)).lower()

    normalized_scores_map = {}
    for key, value in scores.items():
        normalized_scores_map[normalize_str(key)] = value

    scores_names = []
    for name in class_names:
        clean_name = normalize_str(name)
        if clean_name in normalized_scores_map:
            scores_names.append(normalized_scores_map[clean_name])
        else:
            print(
                f"Warning: Key '{name}' (normalized: '{clean_name}') "
                "not found in scores dict. Using default 0.5."
            )
            scores_names.append(0.5)
    return scores_names


def segment_observation(image_path, text_prompt, dino_model, sam_predictor, device="cuda"):
    """
    2026-10-04：只增加性能/显存 Profile，不修改原推理结果。
    """

    profile_total_start = time.perf_counter()
    profile_times = {
        "dino_preprocess": 0.0,
        "dino_inference": 0.0,
        "dino_postprocess": 0.0,
        "sam_image_read": 0.0,
        "sam_set_image": 0.0,
        "sam_box_transform": 0.0,
        "sam_predict": 0.0,
        "sam_postprocess": 0.0,
    }

    print_gpu_profile("segment start")

    # 2026-10-04 新增：DINO预处理计时，原逻辑不变
    t0 = time.perf_counter()
    text_prompt = prepare_detection_prompt(text_prompt)
    image_pil = Image.open(image_path).convert("RGB")
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    image_tensor, _ = transform(image_pil, None)
    profile_times["dino_preprocess"] = time.perf_counter() - t0
    print(f"[PROFILE] DINO preprocess = {profile_times['dino_preprocess']:.3f}s")

    # 原阈值逻辑不变
    threshold_schedule = [
        (0.40, 0.25),
        (0.30, 0.20),
        (0.24, 0.18),
    ]

    boxes_filt = None
    logits_filt = None
    used_box_threshold = None
    used_text_threshold = None

    # 原代码：with torch.no_grad(): outputs = dino_model(...)
    # 2026-10-04 新增：同步+计时
    print_gpu_profile("before DINO inference")
    cuda_sync()
    t0 = time.perf_counter()
    with torch.no_grad():
        outputs = dino_model(
            image_tensor[None].to(device),
            captions=[text_prompt]
        )
    cuda_sync()
    profile_times["dino_inference"] = time.perf_counter() - t0
    print(f"[PROFILE] DINO inference = {profile_times['dino_inference']:.3f}s")
    print_gpu_profile("after DINO inference")

    # 2026-10-04 新增：DINO后处理计时
    t0 = time.perf_counter()
    logits = outputs["pred_logits"].cpu().sigmoid()[0]
    boxes = outputs["pred_boxes"].cpu()[0]

    for box_threshold, text_threshold in threshold_schedule:
        filt_mask = logits.max(dim=1)[0] > box_threshold
        candidate_boxes = boxes[filt_mask]

        print(
            "[DINO] prompt=", repr(text_prompt),
            "| box_threshold=", box_threshold,
            "| text_threshold=", text_threshold,
            "| detections=", int(candidate_boxes.shape[0]),
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
        profile_times["dino_postprocess"] = time.perf_counter() - t0
        profile_total = time.perf_counter() - profile_total_start
        print(f"[PROFILE] DINO postprocess = {profile_times['dino_postprocess']:.3f}s")
        print_gpu_profile("no detection return")
        print(
            "[PROFILE SUMMARY] "
            f"DINO_pre={profile_times['dino_preprocess']:.3f}s | "
            f"DINO_infer={profile_times['dino_inference']:.3f}s | "
            f"DINO_post={profile_times['dino_postprocess']:.3f}s | "
            f"TOTAL={profile_total:.3f}s"
        )
        print(f"Warning: No objects found for prompt after adaptive thresholds: {text_prompt}")
        return np.array([]), np.array([]), {}, torch.empty((0, 0, 0, 0))

    profile_times["dino_postprocess"] = time.perf_counter() - t0
    print(f"[PROFILE] DINO postprocess = {profile_times['dino_postprocess']:.3f}s")

    # 原图读取逻辑不变
    t0 = time.perf_counter()
    image_cv = cv2.imread(image_path)
    if image_cv is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
    profile_times["sam_image_read"] = time.perf_counter() - t0
    print(f"[PROFILE] SAM image read = {profile_times['sam_image_read']:.3f}s")

    # 原代码：sam_predictor.set_image(image_cv)
    # 2026-10-04 新增：同步+计时
    print_gpu_profile("before SAM set_image")
    cuda_sync()
    t0 = time.perf_counter()
    sam_predictor.set_image(image_cv)
    cuda_sync()
    profile_times["sam_set_image"] = time.perf_counter() - t0
    print(f"[PROFILE] SAM set_image = {profile_times['sam_set_image']:.3f}s")
    print_gpu_profile("after SAM set_image")

    size = image_pil.size
    H, W = size[1], size[0]

    # 原box变换逻辑不变
    t0 = time.perf_counter()
    boxes_filt = boxes_filt * torch.tensor([W, H, W, H])
    boxes_filt[:, :2] -= boxes_filt[:, 2:] / 2
    boxes_filt[:, 2:] += boxes_filt[:, :2]
    boxes_filt = boxes_filt.to(device)
    transformed_boxes = sam_predictor.transform.apply_boxes_torch(
        boxes_filt,
        image_cv.shape[:2]
    )
    cuda_sync()
    profile_times["sam_box_transform"] = time.perf_counter() - t0
    print(f"[PROFILE] SAM box transform = {profile_times['sam_box_transform']:.3f}s")

    # 原代码：sam_predictor.predict_torch(...)
    # 2026-10-04 新增：同步+计时
    print_gpu_profile("before SAM predict")
    cuda_sync()
    t0 = time.perf_counter()
    with torch.no_grad():
        masks, _, _ = sam_predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )
    cuda_sync()
    profile_times["sam_predict"] = time.perf_counter() - t0
    print(f"[PROFILE] SAM predict = {profile_times['sam_predict']:.3f}s")
    print_gpu_profile("after SAM predict")

    # 原结果整理逻辑不变
    t0 = time.perf_counter()
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
        profile_times["sam_postprocess"] = time.perf_counter() - t0
        profile_total = time.perf_counter() - profile_total_start
        print(f"[PROFILE] SAM postprocess = {profile_times['sam_postprocess']:.3f}s")
        print_gpu_profile("invalid phrase return")
        print(
            "[PROFILE SUMMARY] "
            f"DINO_pre={profile_times['dino_preprocess']:.3f}s | "
            f"DINO_infer={profile_times['dino_inference']:.3f}s | "
            f"DINO_post={profile_times['dino_postprocess']:.3f}s | "
            f"SAM_read={profile_times['sam_image_read']:.3f}s | "
            f"SAM_set={profile_times['sam_set_image']:.3f}s | "
            f"SAM_box={profile_times['sam_box_transform']:.3f}s | "
            f"SAM_predict={profile_times['sam_predict']:.3f}s | "
            f"SAM_post={profile_times['sam_postprocess']:.3f}s | "
            f"TOTAL={profile_total:.3f}s"
        )
        print("[DINO WARNING] boxes were found but no valid class phrase was decoded.")
        return np.array([]), np.array([]), {}, torch.empty((0, 0, 0, 0))

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

    cuda_sync()
    profile_times["sam_postprocess"] = time.perf_counter() - t0
    print(f"[PROFILE] SAM postprocess = {profile_times['sam_postprocess']:.3f}s")

    print(
        "[DINO] accepted classes:",
        class_names_np.tolist(),
        "| masks:", len(masks),
        "| threshold:",
        (used_box_threshold, used_text_threshold),
    )

    # 原代码保留：torch.cuda.empty_cache()
    print_gpu_profile("before empty_cache")
    torch.cuda.empty_cache()
    print_gpu_profile("after empty_cache")

    profile_total = time.perf_counter() - profile_total_start
    print(
        "[PROFILE SUMMARY] "
        f"DINO_pre={profile_times['dino_preprocess']:.3f}s | "
        f"DINO_infer={profile_times['dino_inference']:.3f}s | "
        f"DINO_post={profile_times['dino_postprocess']:.3f}s | "
        f"SAM_read={profile_times['sam_image_read']:.3f}s | "
        f"SAM_set={profile_times['sam_set_image']:.3f}s | "
        f"SAM_box={profile_times['sam_box_transform']:.3f}s | "
        f"SAM_predict={profile_times['sam_predict']:.3f}s | "
        f"SAM_post={profile_times['sam_postprocess']:.3f}s | "
        f"TOTAL={profile_total:.3f}s"
    )

    return class_ids_np, class_names_np, class_name_to_id, masks


if __name__ == "__main__":
    image_path = "./GroundSAM/assets/demo7.jpg"
    text_prompt = "Horse. Sky"
    segment_observation(image_path, text_prompt)
