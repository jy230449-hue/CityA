import time
import airsim


client = airsim.MultirotorClient()
client.confirmConnection()

TEST_COUNT = 3


def benchmark(name, request):
    times = []

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    for i in range(TEST_COUNT):
        start = time.perf_counter()

        responses = client.simGetImages([request])

        elapsed = time.perf_counter() - start
        times.append(elapsed)

        response = responses[0]

        print(
            f"{i + 1}: {elapsed:.2f}s | "
            f"{response.width}x{response.height}"
        )

    print(
        f"{name} AVG: "
        f"{sum(times) / len(times):.2f}s"
    )


# 1. 当前正式代码使用的 DepthPlanar
benchmark(
    "Camera0 DepthPlanar float",
    airsim.ImageRequest(
        0,
        airsim.ImageType.DepthPlanar,
        True,
        False
    )
)

# 2. 同一个 Camera 0，换 DepthPerspective
# 仅测速，不用于正式实验
benchmark(
    "Camera0 DepthPerspective float",
    airsim.ImageRequest(
        0,
        airsim.ImageType.DepthPerspective,
        True,
        False
    )
)

# 3. Camera 0 DepthVis
# 仅测速，不用于正式实验
benchmark(
    "Camera0 DepthVis",
    airsim.ImageRequest(
        0,
        airsim.ImageType.DepthVis,
        False,
        False
    )
)

# 4. Camera 1 DepthPlanar
# 看是否是 Camera 0 本身的问题
benchmark(
    "Camera1 DepthPlanar float",
    airsim.ImageRequest(
        1,
        airsim.ImageType.DepthPlanar,
        True,
        False
    )
)