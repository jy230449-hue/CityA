# full_profile_once.py
# CityAVOS 一键综合性能诊断
# 一次 5-Step 同时定位：
# AirSim/Depth、DINO、SAM、CUDA cache、Semantic 内部、Qwen、Action、GPU 显存峰值。
# 不修改论文参数、模型、阈值、分辨率、Step 上限或导航逻辑。
#
# 用法（在 C:\hjy\project\CityAVOS 下，airport 环境）：
#   python full_profile_once.py
#
# 运行前要求：
#   1) EmbodiedCity/TrafficSimulation 已启动
#   2) 本地 Qwen Server 已启动，GPU_MAX_MEMORY 仍保持 "8GiB"

from __future__ import annotations

import csv
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from urllib.request import urlopen

ROOT = Path.cwd()
SEGMENT = ROOT / "Segment_Image.py"
SEMANTIC = ROOT / "update_semantic_map.py"
MAIN = ROOT / "main.py"

MISSION_ID = "8"
PROFILE_STEPS = "5"

# 防止再次出现“卡 30 分钟”
NO_OUTPUT_TIMEOUT_S = 180
TOTAL_TIMEOUT_S = 720

STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_PATH = ROOT / f"full_profile_{STAMP}.log"
GPU_PATH = ROOT / f"gpu_profile_{STAMP}.csv"
SUMMARY_PATH = ROOT / f"full_profile_summary_{STAMP}.txt"


def backup_once(path: Path):
    backup = path.with_name(path.stem + "_before_full_profile_20261004" + path.suffix)
    if not backup.exists():
        shutil.copy2(path, backup)
        print(f"[BACKUP] {path.name} -> {backup.name}")


def patch_segment_image():
    text = SEGMENT.read_text(encoding="utf-8")

    # 必须仍是上一轮已跑通的综合 Profile 版。
    if "def print_gpu_profile(" not in text or "[PROFILE SUMMARY]" not in text:
        raise RuntimeError(
            "当前 Segment_Image.py 不是上一轮已经跑通的综合 Profile 版。"
            "请先使用你刚才成功运行过的 Profile 版 Segment_Image.py。"
        )

    # 已经加过则直接跳过，保证脚本可重复运行。
    if "[PROFILE] CUDA empty_cache" in text:
        print("[PATCH] Segment_Image.py: CUDA empty_cache timing already applied")
        return

    # 2026-10-04 修复：
    # 不再依赖固定的前后注释/空行，只定位 segment_observation() 内最后一次
    # torch.cuda.empty_cache()，然后原位替换为“计时 + 原调用”。
    seg_start = text.find("def segment_observation(")
    seg_end = text.find('\n\nif __name__ == "__main__":', seg_start)

    if seg_start < 0:
        raise RuntimeError("Segment_Image.py 中找不到 segment_observation()。")

    if seg_end < 0:
        seg_end = len(text)

    segment_block = text[seg_start:seg_end]

    matches = list(
        re.finditer(
            r'(?m)^(?P<indent>[ \t]*)torch\.cuda\.empty_cache\(\)\s*$',
            segment_block
        )
    )

    if not matches:
        raise RuntimeError(
            "Segment_Image.py 的 segment_observation() 中找不到 "
            "torch.cuda.empty_cache()。请把当前 Segment_Image.py 发给我。"
        )

    match = matches[-1]
    indent = match.group("indent")

    replacement = (
        f'{indent}# 2026-10-04 综合测速：单独统计 CUDA cache 释放耗时\n'
        f'{indent}cuda_sync()\n'
        f'{indent}empty_cache_start = time.perf_counter()\n\n'
        f'{indent}torch.cuda.empty_cache()\n\n'
        f'{indent}cuda_sync()\n'
        f'{indent}empty_cache_time = time.perf_counter() - empty_cache_start\n\n'
        f'{indent}print(\n'
        f'{indent}    f"[PROFILE] CUDA empty_cache = {{empty_cache_time:.3f}}s"\n'
        f'{indent})'
    )

    new_segment_block = (
        segment_block[:match.start()]
        + replacement
        + segment_block[match.end():]
    )

    text = (
        text[:seg_start]
        + new_segment_block
        + text[seg_end:]
    )

    SEGMENT.write_text(text, encoding="utf-8")
    print("[PATCH] Segment_Image.py: CUDA empty_cache timing applied")


def patch_update_semantic_map():
    text = SEMANTIC.read_text(encoding="utf-8")

    if not re.search(r"(?m)^import time\s*$", text):
        m = re.search(r"(?m)^(import .+)$", text)
        if not m:
            raise RuntimeError("update_semantic_map.py 找不到 import 区域。")
        insert_at = m.end()
        text = text[:insert_at] + "\nimport time" + text[insert_at:]
        print("[PATCH] update_semantic_map.py: import time applied")

    if "[SEM INNER]" in text:
        print("[PATCH] update_semantic_map.py: inner timing already applied")
        SEMANTIC.write_text(text, encoding="utf-8")
        return

    start = text.find("def update_semantic_map(")
    end = text.find("\ndef update_coginitive_map(", start)

    if start < 0 or end < 0:
        raise RuntimeError("无法定位 update_semantic_map() 函数。")

    new_func = '''def update_semantic_map(semantic_map, cognitive_map, depth_image_path, camera_position, euler_angles, depth_enhanced, scores_rel):

    semantic_inner_total_start = time.perf_counter()

    # 1. 相机参数
    t_inner = time.perf_counter()

    rotation_matrix = create_rotation_matrix(*euler_angles)
    extrinsic_matrix = create_extrinsic_matrix(camera_position, rotation_matrix)

    camera_params = {
        'intrinsic_matrix': np.array([
            [320, 0, 320],
            [0, 320, 240],
            [0, 0, 1]
        ]),
        'extrinsic_matrix': extrinsic_matrix,
        'width': 640,
        'height': 480
    }

    camera_time = time.perf_counter() - t_inner

    # 2. 读取深度 PNG
    t_inner = time.perf_counter()

    depth_image = read_depth_image(depth_image_path)

    read_depth_time = time.perf_counter() - t_inner

    # 3. 深度 -> 世界坐标
    t_inner = time.perf_counter()

    world_points, labels = depth_to_world_coordinates(
        depth_image,
        camera_params,
        depth_enhanced
    )

    depth_world_time = time.perf_counter() - t_inner

    # 4. 世界点 -> semantic/cognitive map
    t_inner = time.perf_counter()

    updated_semantic_map, updated_cognitive_map = add_label_to_semantic_map(
        world_points,
        labels,
        semantic_map,
        cognitive_map,
        scores_rel
    )

    add_label_time = time.perf_counter() - t_inner
    total_time = time.perf_counter() - semantic_inner_total_start

    print(
        "[SEM INNER] "
        f"camera={camera_time:.3f}s | "
        f"read_depth={read_depth_time:.3f}s | "
        f"depth_to_world={depth_world_time:.3f}s | "
        f"add_label={add_label_time:.3f}s | "
        f"TOTAL={total_time:.3f}s"
    )

    return updated_semantic_map, updated_cognitive_map
'''

    text = text[:start] + new_func + text[end:]
    SEMANTIC.write_text(text, encoding="utf-8")
    print("[PATCH] update_semantic_map.py: inner timing applied")


def patch_main():
    text = MAIN.read_text(encoding="utf-8")

    if "[SEM PROFILE]" in text:
        print("[PATCH] main.py: semantic breakdown already applied")
        return

    anchor = '''            t = time.perf_counter()
            class_scores = get_scores_for_class_names(
                class_names,
                scores_rel
            )
'''

    start = text.find(anchor)
    if start < 0:
        raise RuntimeError("main.py 找不到 semantic_cognitive 计时起点。")

    end_marker = '''            step_timing["semantic_cognitive_s"] = time.perf_counter() - t
'''

    end = text.find(end_marker, start)
    if end < 0:
        raise RuntimeError("main.py 找不到 semantic_cognitive 计时终点。")
    end += len(end_marker)

    new_block = '''            # ============================================================
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
'''

    text = text[:start] + new_block + text[end:]
    MAIN.write_text(text, encoding="utf-8")
    print("[PATCH] main.py: semantic breakdown applied")


def check_qwen():
    try:
        with urlopen("http://127.0.0.1:8001/v1/models", timeout=3) as r:
            body = r.read().decode("utf-8", errors="replace")

        if "qwen3-vl-4b" not in body:
            raise RuntimeError("8001 已响应，但没有发现 qwen3-vl-4b。")

        print("[CHECK] Local Qwen: OK")

    except Exception as e:
        raise RuntimeError(
            "本地 Qwen Server 未正常启动。"
            "请先启动保持 8GiB 限制的 Qwen Server，再运行本脚本。\n"
            f"Detail: {e!r}"
        )


def gpu_monitor(stop_event: threading.Event):
    header = [
        "local_time",
        "memory_used_mib",
        "memory_free_mib",
        "gpu_util_percent",
        "power_w",
    ]

    with GPU_PATH.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        while not stop_event.is_set():
            try:
                out = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=memory.used,memory.free,utilization.gpu,power.draw",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                    stderr=subprocess.DEVNULL,
                    timeout=5,
                ).strip().splitlines()[0]

                parts = [x.strip() for x in out.split(",")]

                writer.writerow([
                    datetime.now().isoformat(timespec="seconds"),
                    *parts[:4],
                ])
                f.flush()

            except Exception:
                pass

            stop_event.wait(1.0)


def stream_process():
    cmd = [
        sys.executable,
        "main.py",
        "--task",
        MISSION_ID,
        "--profile-steps",
        PROFILE_STEPS,
    ]

    print("[RUN]", " ".join(cmd))
    print(f"[RUN] log -> {LOG_PATH.name}")
    print(f"[RUN] gpu -> {GPU_PATH.name}")
    print(
        f"[WATCHDOG] no-output>{NO_OUTPUT_TIMEOUT_S}s or "
        f"total>{TOTAL_TIMEOUT_S}s will terminate the profile run."
    )

    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )

    q = queue.Queue()
    reader_done = threading.Event()

    def reader():
        assert proc.stdout is not None
        for line in proc.stdout:
            q.put(line)
        reader_done.set()

    threading.Thread(target=reader, daemon=True).start()

    start = time.monotonic()
    last_output = start
    timed_out_reason = None

    with LOG_PATH.open("w", encoding="utf-8") as log:
        while True:
            try:
                line = q.get(timeout=0.5)
                last_output = time.monotonic()
                print(line, end="")
                log.write(line)
                log.flush()
            except queue.Empty:
                pass

            now = time.monotonic()

            if proc.poll() is not None and reader_done.is_set() and q.empty():
                break

            if now - last_output > NO_OUTPUT_TIMEOUT_S:
                timed_out_reason = (
                    f"超过 {NO_OUTPUT_TIMEOUT_S}s 没有新日志，"
                    "判定可能卡死，自动终止。"
                )
                proc.kill()
                break

            if now - start > TOTAL_TIMEOUT_S:
                timed_out_reason = (
                    f"总测试超过 {TOTAL_TIMEOUT_S}s，自动终止。"
                )
                proc.kill()
                break

        time.sleep(0.5)

        while not q.empty():
            line = q.get_nowait()
            print(line, end="")
            log.write(line)

        if timed_out_reason:
            msg = f"\n[PROFILE WATCHDOG] {timed_out_reason}\n"
            print(msg, end="")
            log.write(msg)

    return proc.returncode, timed_out_reason


def values(pattern: str, text: str):
    return [float(x) for x in re.findall(pattern, text)]


def stat_line(name, xs):
    if not xs:
        return f"{name:<24} no data"

    return (
        f"{name:<24} avg={sum(xs)/len(xs):7.3f}s | "
        f"max={max(xs):7.3f}s | "
        f"min={min(xs):7.3f}s | "
        f"n={len(xs)}"
    )


def analyze():
    text = LOG_PATH.read_text(encoding="utf-8", errors="replace")

    metrics = {
        "Observation": values(r"\[TIME\] obs=([0-9.]+)s", text),
        "DINO+SAM total": values(r"DINO\+SAM=([0-9.]+)s", text),
        "Semantic total": values(r"semantic=([0-9.]+)s", text),
        "Qwen": values(r"Qwen=([0-9.]+)s", text),
        "Action": values(r"action=([0-9.]+)s", text),
        "Step total": values(r"TOTAL=([0-9.]+)s", text),

        "DINO preprocess": values(r"\[PROFILE\] DINO preprocess = ([0-9.]+)s", text),
        "DINO inference": values(r"\[PROFILE\] DINO inference = ([0-9.]+)s", text),
        "DINO postprocess": values(r"\[PROFILE\] DINO postprocess = ([0-9.]+)s", text),
        "SAM set_image": values(r"\[PROFILE\] SAM set_image = ([0-9.]+)s", text),
        "SAM predict": values(r"\[PROFILE\] SAM predict = ([0-9.]+)s", text),
        "CUDA empty_cache": values(r"\[PROFILE\] CUDA empty_cache = ([0-9.]+)s", text),

        "SEM score_match": values(r"\[SEM PROFILE\].*?score_match=([0-9.]+)s", text),
        "SEM overlay": values(r"\[SEM PROFILE\].*?overlay=([0-9.]+)s", text),
        "SEM update_map": values(r"\[SEM PROFILE\].*?update_map=([0-9.]+)s", text),
        "SEM cluster": values(r"\[SEM PROFILE\].*?cluster=([0-9.]+)s", text),

        "SEM camera": values(r"\[SEM INNER\].*?camera=([0-9.]+)s", text),
        "SEM read_depth": values(r"\[SEM INNER\].*?read_depth=([0-9.]+)s", text),
        "SEM depth_to_world": values(r"\[SEM INNER\].*?depth_to_world=([0-9.]+)s", text),
        "SEM add_label": values(r"\[SEM INNER\].*?add_label=([0-9.]+)s", text),
    }

    gpu_rows = []

    if GPU_PATH.exists():
        with GPU_PATH.open("r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                try:
                    gpu_rows.append({
                        "used": float(row["memory_used_mib"]),
                        "free": float(row["memory_free_mib"]),
                        "util": float(row["gpu_util_percent"]),
                        "power": float(row["power_w"]),
                    })
                except Exception:
                    pass

    lines = []
    lines.append("=" * 78)
    lines.append("CityAVOS FULL PROFILE SUMMARY")
    lines.append("=" * 78)
    lines.append("")
    lines.append("High-level")

    for name in [
        "Observation",
        "DINO+SAM total",
        "Semantic total",
        "Qwen",
        "Action",
        "Step total",
    ]:
        lines.append(stat_line(name, metrics[name]))

    lines.append("")
    lines.append("DINO / SAM / CUDA")

    for name in [
        "DINO preprocess",
        "DINO inference",
        "DINO postprocess",
        "SAM set_image",
        "SAM predict",
        "CUDA empty_cache",
    ]:
        lines.append(stat_line(name, metrics[name]))

    lines.append("")
    lines.append("Semantic breakdown")

    for name in [
        "SEM score_match",
        "SEM overlay",
        "SEM update_map",
        "SEM cluster",
        "SEM camera",
        "SEM read_depth",
        "SEM depth_to_world",
        "SEM add_label",
    ]:
        lines.append(stat_line(name, metrics[name]))

    lines.append("")

    min_free = None

    if gpu_rows:
        min_free = min(x["free"] for x in gpu_rows)
        max_used = max(x["used"] for x in gpu_rows)
        max_util = max(x["util"] for x in gpu_rows)
        max_power = max(x["power"] for x in gpu_rows)

        lines.append("GPU sampling")
        lines.append(f"min free VRAM          {min_free:.0f} MiB")
        lines.append(f"max used VRAM          {max_used:.0f} MiB")
        lines.append(f"max GPU util           {max_util:.0f}%")
        lines.append(f"max power              {max_power:.1f} W")
    else:
        lines.append("GPU sampling: no data")

    candidates = []

    for key in [
        "Observation",
        "DINO inference",
        "SAM set_image",
        "CUDA empty_cache",
        "SEM update_map",
        "SEM depth_to_world",
        "SEM add_label",
        "SEM cluster",
        "Qwen",
        "Action",
    ]:
        xs = metrics.get(key, [])
        if xs:
            candidates.append((max(xs), sum(xs) / len(xs), key))

    candidates.sort(reverse=True)

    lines.append("")
    lines.append("Bottleneck ranking by worst observed latency")

    for i, (mx, av, name) in enumerate(candidates[:10], 1):
        lines.append(
            f"{i:>2}. {name:<24} "
            f"max={mx:7.3f}s | avg={av:7.3f}s"
        )

    lines.append("")
    lines.append("Automatic safety notes")

    if min_free is not None and min_free < 2048:
        lines.append(
            "- GPU free VRAM dropped below 2 GiB: DO NOT raise Qwen above 8GiB yet."
        )
    elif min_free is not None and min_free >= 4096:
        lines.append(
            "- GPU free VRAM stayed >=4 GiB: a later 9GiB Qwen A/B test may be safe."
        )
    else:
        lines.append(
            "- Keep Qwen at 8GiB until the final optimization pass."
        )

    ec = metrics["CUDA empty_cache"]
    if ec and max(ec) >= 1.0:
        lines.append(
            "- CUDA empty_cache itself is expensive: final version should reduce/relocate cache clears."
        )

    sem = metrics["SEM update_map"]
    if sem and max(sem) >= 2.0:
        lines.append(
            "- update_semantic_map has a real spike: optimize the dominant SEM INNER substage."
        )

    sam = metrics["SAM set_image"]
    if sam and max(sam) >= 5.0:
        lines.append(
            "- SAM image encoder has a spike: preserve SAM model, optimize GPU memory scheduling."
        )

    qwen = metrics["Qwen"]
    if qwen and (sum(qwen) / len(qwen)) >= 5.0:
        lines.append(
            "- Qwen is slowed by CPU offload; VRAM headroom decides whether it can receive more GPU memory."
        )

    summary = "\n".join(lines) + "\n"

    SUMMARY_PATH.write_text(summary, encoding="utf-8")

    print("\n" + summary)
    print(f"[RESULT] Full log: {LOG_PATH}")
    print(f"[RESULT] GPU log:  {GPU_PATH}")
    print(f"[RESULT] Summary:  {SUMMARY_PATH}")


def main():
    print("=" * 78)
    print("CityAVOS one-shot full performance profile")
    print("=" * 78)
    print("No paper/model/threshold/resolution parameters will be changed.")
    print("Qwen GPU limit should remain 8GiB for this run.")
    print()

    for p in [SEGMENT, SEMANTIC, MAIN]:
        if not p.exists():
            raise FileNotFoundError(f"Missing: {p}")

    check_qwen()

    for p in [SEGMENT, SEMANTIC, MAIN]:
        backup_once(p)

    patch_segment_image()
    patch_update_semantic_map()
    patch_main()

    for p in [SEGMENT, SEMANTIC, MAIN]:
        source = p.read_text(encoding="utf-8")
        compile(source, str(p), "exec")
        print(f"[CHECK] syntax OK: {p.name}")

    stop_gpu = threading.Event()
    gpu_thread = threading.Thread(
        target=gpu_monitor,
        args=(stop_gpu,),
        daemon=True,
    )
    gpu_thread.start()

    try:
        rc, timeout_reason = stream_process()
    finally:
        stop_gpu.set()
        gpu_thread.join(timeout=3)

    print(f"\n[RUN] return code = {rc}")

    if timeout_reason:
        print("[RUN] watchdog:", timeout_reason)

    analyze()


if __name__ == "__main__":
    main()
