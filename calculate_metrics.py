import os
import json
import glob
import math

RESULT_DIR = r"./output/experiemnt_data/ours"
MAX_MISSION_ID = 219


def safe_float(value, default=None):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


results = []
invalid_files = []

for path in glob.glob(os.path.join(RESULT_DIR, "data_*.json")):
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        mission_id = int(data["mission_id"])

        # 只统计 1~219
        if mission_id > MAX_MISSION_ID:
            continue

        # 必须至少存在正式实验核心字段
        required = [
            "if_success",
            "steps",
            "final_navigation_error",
        ]

        if not all(k in data for k in required):
            invalid_files.append(path)
            continue

        # 排除 profile 结果
        if data.get("profile_mode", False):
            continue

        results.append(data)

    except Exception as e:
        invalid_files.append((path, str(e)))


# 按 mission_id 排序
results.sort(key=lambda x: int(x["mission_id"]))

N = len(results)

if N == 0:
    print("没有找到可用的正式实验结果。")
    raise SystemExit


# ============================================================
# 1. SR
# ============================================================

success_count = sum(int(x["if_success"]) for x in results)
failure_count = N - success_count

SR = success_count / N * 100.0


# ============================================================
# 2. MSS
# ============================================================

steps = [float(x["steps"]) for x in results]
MSS = sum(steps) / N


# ============================================================
# 3. NE
# ============================================================

ne_values = [
    float(x["final_navigation_error"])
    for x in results
]

NE = sum(ne_values) / N


# ============================================================
# 4. SPL
#
# 注意：
# 当前公开 CityAVOS 605 数据没有提供论文中的
# ground-truth / optimal trajectory length。
#
# 因此这里不能计算“严格论文 SPL”。
#
# 下面只额外给出一个 Euclidean proxy，
# 用 initial_target_distance 作为 L_i。
#
# 这个值只能用于当前实验内部检查，
# 不应直接与论文 SPL=40.57 比较。
# ============================================================

spl_proxy_sum = 0.0
spl_proxy_valid = 0

for x in results:

    success = int(x["if_success"])

    L = safe_float(x.get("initial_target_distance"))
    P = safe_float(x.get("trajectory_length"))

    if L is None or P is None:
        continue

    if L <= 0:
        ratio = 1.0 if success else 0.0
    else:
        # 标准 SPL 写法，保证比例不超过 1
        ratio = L / max(L, P)

    spl_proxy_sum += success * ratio
    spl_proxy_valid += 1


if spl_proxy_valid:
    SPL_EUCLIDEAN_PROXY = spl_proxy_sum / N * 100.0
else:
    SPL_EUCLIDEAN_PROXY = None


# ============================================================
# Easy / Hard 分组
# ============================================================

def calculate_group(group):

    n = len(group)

    if n == 0:
        return None

    success = sum(int(x["if_success"]) for x in group)

    sr = success / n * 100.0

    mss = sum(
        float(x["steps"])
        for x in group
    ) / n

    ne = sum(
        float(x["final_navigation_error"])
        for x in group
    ) / n

    return {
        "N": n,
        "success": success,
        "failure": n - success,
        "SR": sr,
        "MSS": mss,
        "NE": ne,
    }


easy = [
    x for x in results
    if str(x.get("paper_difficulty", "")).lower() == "easy"
]

hard = [
    x for x in results
    if str(x.get("paper_difficulty", "")).lower() == "hard"
]


# ============================================================
# 输出
# ============================================================

mission_ids = [int(x["mission_id"]) for x in results]

missing_ids = [
    i for i in range(1, MAX_MISSION_ID + 1)
    if i not in mission_ids
]


print("=" * 70)
print("CityAVOS Intermediate Evaluation")
print("=" * 70)

print(f"目标统计范围       : 1 ~ {MAX_MISSION_ID}")
print(f"实际完整结果数量   : {N}")
print(f"成功任务           : {success_count}")
print(f"失败任务           : {failure_count}")

if missing_ids:
    print(f"尚未发现完整结果   : {missing_ids}")

print()
print("------ Paper Metrics ------")
print(f"SR  = {SR:.2f}%")
print(f"MSS = {MSS:.2f}")
print(f"NE  = {NE:.2f} m")

print()
print("------ SPL ------")

if SPL_EUCLIDEAN_PROXY is not None:
    print(
        f"SPL Euclidean Proxy = "
        f"{SPL_EUCLIDEAN_PROXY:.2f}%"
    )

print(
    "注意：上面的 SPL 是以 initial_target_distance "
    "代替 optimal trajectory length 的临时值，"
)
print(
    "不是论文严格 SPL，暂时不要直接与论文 40.57 比较。"
)

print()
print("------ Easy ------")
print(calculate_group(easy))

print()
print("------ Hard ------")
print(calculate_group(hard))

print()
print("------ Paper PRPSearcher Reference ------")
print("SR  = 53.50%")
print("MSS = 35.26")
print("SPL = 40.57%")
print("NE  = 64.86 m")

print("=" * 70)