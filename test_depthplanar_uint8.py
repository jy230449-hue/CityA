import airsim
import numpy as np
import cv2
import time
import os


client = airsim.MultirotorClient()
client.confirmConnection()

client.enableApiControl(True)
client.armDisarm(True)

os.makedirs("./output", exist_ok=True)

temp_path = "./output/_float_planar.png"

try:

    # ============================================================
    # 固定 Task 8 位姿
    # ============================================================
    client.simPause(True)

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

    state = client.simGetGroundTruthKinematics()

    print("\n========== Frozen pose ==========")

    print(
        state.position.x_val,
        state.position.y_val,
        state.position.z_val
    )


    # ============================================================
    # 1. 正式代码：DepthPlanar float
    # ============================================================
    print(
        "\n========== DepthPlanar FLOAT =========="
    )

    start = time.perf_counter()

    float_response = client.simGetImages([
        airsim.ImageRequest(
            0,
            airsim.ImageType.DepthPlanar,
            True,
            False
        )
    ])[0]

    float_time = (
        time.perf_counter() - start
    )

    depth_float = np.array(
        float_response.image_data_float,
        dtype=np.float32
    ).reshape(
        float_response.height,
        float_response.width
    )

    print(
        f"Time: {float_time:.2f}s"
    )

    print(
        "shape:",
        depth_float.shape
    )

    print(
        "min:",
        float(np.nanmin(depth_float))
    )

    print(
        "max:",
        float(np.nanmax(depth_float))
    )


    # ============================================================
    # 当前 CityAVOS 实际进入 PNG 的数据
    # ============================================================
    depth_current = depth_float.copy()

    depth_current[
        depth_current > 100
    ] = 100

    cv2.imwrite(
        temp_path,
        depth_current
    )

    current_png = cv2.imread(
        temp_path,
        cv2.IMREAD_UNCHANGED
    )

    print(
        "\nCurrent CityAVOS PNG:"
    )

    print(
        "dtype:",
        current_png.dtype
    )

    print(
        "min:",
        int(np.min(current_png))
    )

    print(
        "max:",
        int(np.max(current_png))
    )


    # ============================================================
    # 2. DepthPlanar uint8 / RAW
    #    ImageType完全不变，只改 pixels_as_float
    # ============================================================
    print(
        "\n========== DepthPlanar UINT8 RAW =========="
    )

    start = time.perf_counter()

    uint8_response = client.simGetImages([
        airsim.ImageRequest(
            0,
            airsim.ImageType.DepthPlanar,
            False,
            False
        )
    ])[0]

    uint8_time = (
        time.perf_counter() - start
    )

    print(
        f"Time: {uint8_time:.2f}s"
    )

    print(
        "width:",
        uint8_response.width
    )

    print(
        "height:",
        uint8_response.height
    )

    print(
        "byte count:",
        len(
            uint8_response.image_data_uint8
        )
    )

    raw = np.frombuffer(
        uint8_response.image_data_uint8,
        dtype=np.uint8
    )

    pixel_count = (
        uint8_response.width
        * uint8_response.height
    )

    print(
        "pixel count:",
        pixel_count
    )

    print(
        "bytes/pixel:",
        len(raw) / pixel_count
    )

    # 尝试判断通道数
    if len(raw) == pixel_count:

        uint8_image = raw.reshape(
            uint8_response.height,
            uint8_response.width
        )

        print(
            "Detected: 1 channel"
        )

    elif len(raw) == pixel_count * 3:

        uint8_image = raw.reshape(
            uint8_response.height,
            uint8_response.width,
            3
        )

        print(
            "Detected: 3 channels"
        )

        print(
            "channel 0-1 max diff:",
            int(
                np.max(
                    np.abs(
                        uint8_image[:, :, 0].astype(np.int16)
                        -
                        uint8_image[:, :, 1].astype(np.int16)
                    )
                )
            )
        )

        print(
            "channel 0-2 max diff:",
            int(
                np.max(
                    np.abs(
                        uint8_image[:, :, 0].astype(np.int16)
                        -
                        uint8_image[:, :, 2].astype(np.int16)
                    )
                )
            )
        )

        uint8_image = (
            uint8_image[:, :, 0]
        )

    elif len(raw) == pixel_count * 4:

        uint8_image = raw.reshape(
            uint8_response.height,
            uint8_response.width,
            4
        )

        print(
            "Detected: 4 channels"
        )

        uint8_image = (
            uint8_image[:, :, 0]
        )

    else:

        print(
            "Unknown output format."
        )

        uint8_image = None


    if uint8_image is not None:

        print(
            "uint8 min:",
            int(
                np.min(
                    uint8_image
                )
            )
        )

        print(
            "uint8 max:",
            int(
                np.max(
                    uint8_image
                )
            )
        )

        print(
            "uint8 mean:",
            float(
                np.mean(
                    uint8_image
                )
            )
        )

        print(
            "unique count:",
            len(
                np.unique(
                    uint8_image
                )
            )
        )


    # ============================================================
    # 3. DepthPlanar uint8 + compressed
    # ============================================================
    print(
        "\n========== DepthPlanar UINT8 COMPRESSED =========="
    )

    start = time.perf_counter()

    compressed_response = client.simGetImages([
        airsim.ImageRequest(
            0,
            airsim.ImageType.DepthPlanar,
            False,
            True
        )
    ])[0]

    compressed_time = (
        time.perf_counter() - start
    )

    print(
        f"Time: {compressed_time:.2f}s"
    )

    print(
        "compressed bytes:",
        len(
            compressed_response.image_data_uint8
        )
    )


    # ============================================================
    # 性能比较
    # ============================================================
    print(
        "\n========== SPEED =========="
    )

    print(
        f"FLOAT:      "
        f"{float_time:.2f}s"
    )

    print(
        f"UINT8 RAW:  "
        f"{uint8_time:.2f}s"
    )

    print(
        f"UINT8 PNG:  "
        f"{compressed_time:.2f}s"
    )

    if uint8_time > 0:

        print(
            f"RAW speedup: "
            f"{float_time / uint8_time:.2f}x"
        )

    if compressed_time > 0:

        print(
            f"Compressed speedup: "
            f"{float_time / compressed_time:.2f}x"
        )


finally:

    client.simPause(False)

    try:
        os.remove(
            temp_path
        )
    except OSError:
        pass