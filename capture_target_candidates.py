import argparse
import json
import math
import os
import time

import cv2
import numpy as np

from vln.drone_agent import AirsimAgent


def save_rgb(path, image):
    """
    AirSim raw Scene images are RGB-like arrays in this project.
    cv2.imwrite expects BGR, so convert before saving.
    """
    if image is None:
        raise RuntimeError(f"Empty image for {path}")
    if image.ndim == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, image)


def main():
    parser = argparse.ArgumentParser(
        description="Capture candidate target images around a CityAVOS target_pos."
    )
    parser.add_argument("--task", type=int, default=286, help="mission_id, default: 286")
    parser.add_argument(
        "--dataset",
        default=r".\data\target_information_605.json",
        help="dataset json path",
    )
    parser.add_argument(
        "--output",
        default=r".\reconstructed_targets\raw",
        help="output root",
    )
    parser.add_argument(
        "--radii",
        nargs="+",
        type=float,
        default=[8.0, 12.0],
        help="horizontal distances from target in meters",
    )
    parser.add_argument(
        "--height",
        type=float,
        default=5.0,
        help="camera height above target in NED coordinates",
    )
    parser.add_argument(
        "--wait",
        type=float,
        default=0.8,
        help="seconds to wait after teleport before capture",
    )
    args = parser.parse_args()

    with open(args.dataset, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    task = next(
        (x for x in dataset if int(x["mission_id"]) == args.task),
        None,
    )
    if task is None:
        raise ValueError(f"Task {args.task} not found in {args.dataset}")

    tx, ty, tz = [float(v) for v in task["target_pos"]]
    task_dir = os.path.join(args.output, str(args.task))
    os.makedirs(task_dir, exist_ok=True)

    print("=" * 70)
    print(f"Task          : {task['mission_id']}")
    print(f"Scene         : {task['scene_id']}")
    print(f"Target text   : {task['target_text']}")
    print(f"Target pos    : {task['target_pos']}")
    print(f"Expected file : {task['target_image_path']}")
    print(f"Output dir    : {task_dir}")
    print("=" * 70)

    drone = AirsimAgent(None, None, None)

    # 4 directions around target, 2 radii by default.
    angles = [0, 90, 180, 270]
    records = []
    view_id = 0

    for radius in args.radii:
        for angle_deg in angles:
            angle = math.radians(angle_deg)

            cx = tx + radius * math.cos(angle)
            cy = ty + radius * math.sin(angle)

            # AirSim uses NED coordinates: more negative Z means higher.
            cz = tz - abs(args.height)

            # Point the vehicle horizontally toward the target.
            yaw = math.degrees(math.atan2(ty - cy, tx - cx))

            drone.pos = np.array([cx, cy, cz], dtype=float)
            drone.ori = np.array([0.0, 0.0, yaw], dtype=float)

            pose = np.concatenate([drone.pos, drone.ori]).tolist()
            print(
                f"[view {view_id:02d}] radius={radius:.1f} "
                f"angle={angle_deg:3d} pose={pose}"
            )

            drone.MovetoPose(pose)
            time.sleep(args.wait)

            # Save several project camera IDs so we can confirm which view is best.
            saved = {}
            for camera_id in [1, 2, 3]:
                try:
                    img = drone.get_xyg_image(0, camera_id)
                    filename = f"view_{view_id:02d}_r{int(radius)}_a{angle_deg}_cam{camera_id}.png"
                    out_path = os.path.join(task_dir, filename)
                    save_rgb(out_path, img)
                    saved[str(camera_id)] = filename
                except Exception as e:
                    print(f"  camera {camera_id} failed: {e}")

            records.append(
                {
                    "view_id": view_id,
                    "radius": radius,
                    "angle_deg": angle_deg,
                    "capture_pose": pose,
                    "target_pos": task["target_pos"],
                    "yaw_to_target": yaw,
                    "files": saved,
                }
            )
            view_id += 1

    metadata = {
        "source": "reconstructed_from_cityavos_simulator",
        "representative_task": int(task["mission_id"]),
        "scene_id": int(task["scene_id"]),
        "target_text": task["target_text"],
        "target_pos": task["target_pos"],
        "expected_target_image_path": task["target_image_path"],
        "capture_height_above_target": abs(args.height),
        "radii": args.radii,
        "records": records,
    }

    with open(
        os.path.join(task_dir, "metadata.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print()
    print("Capture finished.")
    print(f"Open folder: {os.path.abspath(task_dir)}")
    print("Do NOT copy any image into data/target_image yet.")
    print("First verify that the real target is clearly visible.")


if __name__ == "__main__":
    main()
