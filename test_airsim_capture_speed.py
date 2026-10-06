import time
import airsim
import numpy as np


client = airsim.MultirotorClient()
client.confirmConnection()

TEST_COUNT = 5


def test_rgb():
    times = []

    print("\n========== RGB ONLY ==========")

    for i in range(TEST_COUNT):
        start = time.perf_counter()

        responses = client.simGetImages([
            airsim.ImageRequest(
                1,
                airsim.ImageType.Scene,
                False,
                False
            )
        ])

        elapsed = time.perf_counter() - start
        times.append(elapsed)

        response = responses[0]

        print(
            f"RGB {i + 1}: "
            f"{elapsed:.2f}s | "
            f"{response.width}x{response.height}"
        )

    print(
        f"RGB AVG: {sum(times) / len(times):.2f}s"
    )


def test_depth():
    times = []

    print("\n========== DEPTH ONLY ==========")

    for i in range(TEST_COUNT):
        start = time.perf_counter()

        responses = client.simGetImages([
            airsim.ImageRequest(
                0,
                airsim.ImageType.DepthPlanar,
                True,
                False
            )
        ])

        elapsed = time.perf_counter() - start
        times.append(elapsed)

        response = responses[0]

        print(
            f"Depth {i + 1}: "
            f"{elapsed:.2f}s | "
            f"{response.width}x{response.height}"
        )

    print(
        f"DEPTH AVG: {sum(times) / len(times):.2f}s"
    )


def test_rgb_depth():
    times = []

    print("\n========== RGB + DEPTH ==========")

    for i in range(TEST_COUNT):
        start = time.perf_counter()

        responses = client.simGetImages([
            airsim.ImageRequest(
                1,
                airsim.ImageType.Scene,
                False,
                False
            ),
            airsim.ImageRequest(
                0,
                airsim.ImageType.DepthPlanar,
                True,
                False
            ),
        ])

        elapsed = time.perf_counter() - start
        times.append(elapsed)

        print(
            f"RGB+Depth {i + 1}: "
            f"{elapsed:.2f}s"
        )

    print(
        f"RGB+DEPTH AVG: "
        f"{sum(times) / len(times):.2f}s"
    )


if __name__ == "__main__":
    test_rgb()
    test_depth()
    test_rgb_depth()