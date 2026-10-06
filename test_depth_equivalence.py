import airsim
import numpy as np
import cv2
import time
import os


client = airsim.MultirotorClient()
client.confirmConnection()

client.enableApiControl(True)
client.armDisarm(True)

temp_path = "./output/_depth_planar_test.png"
os.makedirs("./output", exist_ok=True)

try:
    # ============================================================
    # 2026-10-04 修改：
    # 冻结仿真，确保 DepthPlanar / DepthVis 使用同一位姿
    # ============================================================
    client.simPause(True)

    # Task 8 起点
    pose = airsim.Pose(
        airsim.Vector3r(
            6360,
            -4160,
            -5
        ),
        airsim.to_quaternion(
            0,
            0,
            0
        )
    )

    client.simSetVehiclePose(
        pose,
        True
    )

    # ------------------------------------------------------------
    # 拍图前位置
    # ------------------------------------------------------------
    state_before = client.simGetGroundTruthKinematics()

    print("\n========== Frozen pose ==========")
    print(
        "Before:",
        state_before.position.x_val,
        state_before.position.y_val,
        state_before.position.z_val
    )

    # ============================================================
    # 一次 RPC 同时获取两种深度
    # ============================================================
    print(
        "\n========== Synchronized depth capture =========="
    )

    start = time.perf_counter()

    responses = client.simGetImages([
        airsim.ImageRequest(
            0,
            airsim.ImageType.DepthPlanar,
            True,
            False
        ),

        airsim.ImageRequest(
            0,
            airsim.ImageType.DepthVis,
            False,
            False
        )
    ])

    total_time = time.perf_counter() - start

    print(
        f"Combined request time: "
        f"{total_time:.2f}s"
    )

    # ------------------------------------------------------------
    # 拍图后位置
    # ------------------------------------------------------------
    state_after = client.simGetGroundTruthKinematics()

    print(
        "After:",
        state_after.position.x_val,
        state_after.position.y_val,
        state_after.position.z_val
    )

    position_change = np.sqrt(
        (
            state_after.position.x_val
            - state_before.position.x_val
        ) ** 2
        +
        (
            state_after.position.y_val
            - state_before.position.y_val
        ) ** 2
        +
        (
            state_after.position.z_val
            - state_before.position.z_val
        ) ** 2
    )

    print(
        f"Position change: "
        f"{position_change:.8f} m"
    )

    # ============================================================
    # DepthPlanar
    # ============================================================
    planar_response = responses[0]

    depth_planar_raw = np.array(
        planar_response.image_data_float,
        dtype=np.float32
    ).reshape(
        planar_response.height,
        planar_response.width
    )

    print(
        "\n========== DepthPlanar =========="
    )

    print(
        "shape:",
        depth_planar_raw.shape
    )

    print(
        "raw min:",
        float(
            np.nanmin(
                depth_planar_raw
            )
        )
    )

    print(
        "raw max:",
        float(
            np.nanmax(
                depth_planar_raw
            )
        )
    )

    print(
        "raw mean:",
        float(
            np.nanmean(
                depth_planar_raw
            )
        )
    )

    valid_raw = (
        np.isfinite(depth_planar_raw)
        &
        (depth_planar_raw >= 0)
        &
        (depth_planar_raw < 100)
    )

    print(
        "pixels <100m:",
        int(
            np.count_nonzero(
                valid_raw
            )
        ),
        "/",
        depth_planar_raw.size,
        f"({np.mean(valid_raw) * 100:.2f}%)"
    )

    # 当前 CityAVOS 裁剪逻辑
    depth_planar_clipped = (
        depth_planar_raw.copy()
    )

    depth_planar_clipped[
        depth_planar_clipped > 100
    ] = 100

    # ============================================================
    # 模拟 CityAVOS 当前 PNG 流程
    # ============================================================
    cv2.imwrite(
        temp_path,
        depth_planar_clipped
    )

    current_depth = cv2.imread(
        temp_path,
        cv2.IMREAD_UNCHANGED
    )

    print(
        "\n========== CityAVOS PNG =========="
    )

    print(
        "dtype:",
        current_depth.dtype
    )

    print(
        "min:",
        int(
            np.min(
                current_depth
            )
        )
    )

    print(
        "max:",
        int(
            np.max(
                current_depth
            )
        )
    )

    # ============================================================
    # DepthVis
    # ============================================================
    vis_response = responses[1]

    depth_vis = np.frombuffer(
        vis_response.image_data_uint8,
        dtype=np.uint8
    ).reshape(
        vis_response.height,
        vis_response.width,
        3
    )

    print(
        "\n========== DepthVis =========="
    )

    print(
        "shape:",
        depth_vis.shape
    )

    # 检查是不是标准灰度三通道
    channel_diff_01 = np.max(
        np.abs(
            depth_vis[:, :, 0].astype(np.int16)
            -
            depth_vis[:, :, 1].astype(np.int16)
        )
    )

    channel_diff_02 = np.max(
        np.abs(
            depth_vis[:, :, 0].astype(np.int16)
            -
            depth_vis[:, :, 2].astype(np.int16)
        )
    )

    print(
        "max channel diff 0-1:",
        int(channel_diff_01)
    )

    print(
        "max channel diff 0-2:",
        int(channel_diff_02)
    )

    depth_vis_gray = (
        depth_vis[:, :, 0]
        .astype(np.float32)
    )

    # AirSim DepthVis：
    # 黑=0m，白=100m+
    depth_vis_meters = (
        depth_vis_gray
        / 255.0
        * 100.0
    )

    print(
        "meters min:",
        float(
            np.min(
                depth_vis_meters
            )
        )
    )

    print(
        "meters max:",
        float(
            np.max(
                depth_vis_meters
            )
        )
    )

    # ============================================================
    # Float DepthPlanar vs DepthVis
    # ============================================================
    print(
        "\n========== Float comparison =========="
    )

    valid_mask = (
        np.isfinite(depth_planar_clipped)
        &
        (depth_planar_clipped >= 0)
        &
        (depth_planar_clipped < 100)
    )

    valid_count = int(
        np.count_nonzero(
            valid_mask
        )
    )

    print(
        "Valid pixels:",
        valid_count
    )

    if valid_count > 0:

        diff = np.abs(
            depth_planar_clipped[
                valid_mask
            ]
            -
            depth_vis_meters[
                valid_mask
            ]
        )

        print(
            f"MAE: "
            f"{np.mean(diff):.4f} m"
        )

        print(
            f"RMSE: "
            f"{np.sqrt(np.mean(diff ** 2)):.4f} m"
        )

        print(
            f"Median: "
            f"{np.percentile(diff, 50):.4f} m"
        )

        print(
            f"P90: "
            f"{np.percentile(diff, 90):.4f} m"
        )

        print(
            f"P95: "
            f"{np.percentile(diff, 95):.4f} m"
        )

        print(
            f"P99: "
            f"{np.percentile(diff, 99):.4f} m"
        )

        print(
            f"<=0.5m: "
            f"{np.mean(diff <= 0.5) * 100:.2f}%"
        )

        print(
            f"<=1m: "
            f"{np.mean(diff <= 1.0) * 100:.2f}%"
        )

        print(
            f"<=2m: "
            f"{np.mean(diff <= 2.0) * 100:.2f}%"
        )

    # ============================================================
    # 与当前真正进入 Semantic Map 的 PNG 比较
    # ============================================================
    print(
        "\n========== CityAVOS PNG comparison =========="
    )

    depth_vis_cityavos = np.clip(
        np.rint(
            depth_vis_meters
        ),
        0,
        100
    ).astype(
        np.uint8
    )

    png_diff = np.abs(
        current_depth.astype(
            np.int16
        )
        -
        depth_vis_cityavos.astype(
            np.int16
        )
    )

    print(
        f"MAE: "
        f"{np.mean(png_diff):.4f}"
    )

    print(
        f"Max error: "
        f"{np.max(png_diff)}"
    )

    print(
        f"P95: "
        f"{np.percentile(png_diff, 95):.4f}"
    )

    print(
        f"Exact same: "
        f"{np.mean(png_diff == 0) * 100:.2f}%"
    )

    print(
        f"Difference <=1: "
        f"{np.mean(png_diff <= 1) * 100:.2f}%"
    )

    print(
        f"Difference <=2: "
        f"{np.mean(png_diff <= 2) * 100:.2f}%"
    )

finally:

    # 恢复仿真
    client.simPause(False)

    try:
        os.remove(
            temp_path
        )
    except OSError:
        pass