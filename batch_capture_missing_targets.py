import argparse
import csv
import json
import math
import os
import re
import time
import zipfile

import cv2
import numpy as np

from vln.drone_agent import AirsimAgent


def safe_name(text):
    text = re.sub(r'[\\/:*?"<>|]+', '_', text)
    text = re.sub(r'\s+', '_', text.strip())
    return text[:120]


def save_rgb(path, image):
    if image is None:
        raise RuntimeError(f"Empty image for {path}")
    if image.ndim == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, image)


def make_contact_sheet(image_paths, labels, out_path, cols=4, thumb_w=240, thumb_h=180):
    if not image_paths:
        return

    tiles = []
    for p, label in zip(image_paths, labels):
        img = cv2.imread(p)
        if img is None:
            continue

        h, w = img.shape[:2]
        scale = min(thumb_w / max(w, 1), (thumb_h - 28) / max(h, 1))
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)

        tile = np.full((thumb_h, thumb_w, 3), 245, dtype=np.uint8)
        x = (thumb_w - nw) // 2
        y = 24 + max(0, (thumb_h - 28 - nh) // 2)
        tile[y:y+nh, x:x+nw] = resized
        cv2.putText(
            tile,
            label,
            (5, 17),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
        tiles.append(tile)

    if not tiles:
        return

    rows = math.ceil(len(tiles) / cols)
    blank = np.full((thumb_h, thumb_w, 3), 245, dtype=np.uint8)
    while len(tiles) < rows * cols:
        tiles.append(blank.copy())

    sheet_rows = []
    for r in range(rows):
        sheet_rows.append(np.hstack(tiles[r*cols:(r+1)*cols]))
    sheet = np.vstack(sheet_rows)
    cv2.imwrite(out_path, sheet)


def collect_missing_unique(dataset):
    seen = set()
    items = []
    for task in dataset:
        path = task["target_image_path"]
        norm = os.path.normpath(path)
        if os.path.exists(norm):
            continue
        key = norm.lower()
        if key in seen:
            continue
        seen.add(key)
        items.append({
            "representative_task": int(task["mission_id"]),
            "scene_id": int(task["scene_id"]),
            "target_text": task["target_text"],
            "target_pos": task["target_pos"],
            "target_image_path": path,
        })
    return items


def main():
    parser = argparse.ArgumentParser(
        description="Batch-capture all currently missing unique CityAVOS target images."
    )
    parser.add_argument("--dataset", default=r".\data\target_information_605.json")
    parser.add_argument("--output", default=r".\reconstructed_targets\batch_raw")
    parser.add_argument("--review-dir", default=r".\reconstructed_targets\review")
    parser.add_argument("--radii", nargs="+", type=float, default=[8.0, 12.0])
    parser.add_argument("--height", type=float, default=5.0)
    parser.add_argument("--wait", type=float, default=0.35)
    parser.add_argument("--force", action="store_true", help="Recapture even if folder already has metadata.json")
    args = parser.parse_args()

    with open(args.dataset, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    missing = collect_missing_unique(dataset)

    print("=" * 78)
    print(f"Unique missing target images to capture: {len(missing)}")
    for x in missing:
        print(
            f"Task {x['representative_task']:>3} | Scene {x['scene_id']} | "
            f"{x['target_image_path']}"
        )
    print("=" * 78)

    if not missing:
        print("Nothing to capture. All target images already exist.")
        return

    os.makedirs(args.output, exist_ok=True)
    os.makedirs(args.review_dir, exist_ok=True)

    # One AirSim connection for the whole batch: much faster than reconnecting per task.
    drone = AirsimAgent(None, None, None)

    manifest_rows = []
    angles = [0, 90, 180, 270]
    camera_ids = [1, 2, 3]

    for idx, item in enumerate(missing, 1):
        task_id = item["representative_task"]
        target_basename = os.path.basename(item["target_image_path"])
        stem = os.path.splitext(target_basename)[0]
        folder_name = f"{task_id}_{safe_name(stem)}"
        task_dir = os.path.join(args.output, folder_name)
        metadata_path = os.path.join(task_dir, "metadata.json")
        contact_path = os.path.join(
            args.review_dir,
            f"{task_id}_{safe_name(stem)}_contact.jpg"
        )

        print()
        print(f"[{idx}/{len(missing)}] Task {task_id} -> {target_basename}")
        print(f"Target text: {item['target_text']}")
        print(f"Target pos : {item['target_pos']}")

        if os.path.exists(metadata_path) and not args.force:
            print("  Existing capture found, skipping recapture.")
        else:
            os.makedirs(task_dir, exist_ok=True)
            tx, ty, tz = [float(v) for v in item["target_pos"]]
            records = []
            view_id = 0

            for radius in args.radii:
                for angle_deg in angles:
                    angle = math.radians(angle_deg)
                    cx = tx + radius * math.cos(angle)
                    cy = ty + radius * math.sin(angle)
                    cz = tz - abs(args.height)
                    yaw = math.degrees(math.atan2(ty - cy, tx - cx))

                    drone.pos = np.array([cx, cy, cz], dtype=float)
                    drone.ori = np.array([0.0, 0.0, yaw], dtype=float)
                    pose = np.concatenate([drone.pos, drone.ori]).tolist()

                    print(
                        f"  view {view_id:02d} | r={radius:.1f} | a={angle_deg:3d} | "
                        f"yaw={yaw:.1f}"
                    )

                    drone.MovetoPose(pose)
                    time.sleep(args.wait)

                    saved = {}
                    for camera_id in camera_ids:
                        try:
                            img = drone.get_xyg_image(0, camera_id)
                            filename = (
                                f"view_{view_id:02d}_r{int(radius)}_"
                                f"a{angle_deg}_cam{camera_id}.png"
                            )
                            out_path = os.path.join(task_dir, filename)
                            save_rgb(out_path, img)
                            saved[str(camera_id)] = filename
                        except Exception as e:
                            print(f"    camera {camera_id} failed: {e}")

                    records.append({
                        "view_id": view_id,
                        "radius": radius,
                        "angle_deg": angle_deg,
                        "capture_pose": pose,
                        "target_pos": item["target_pos"],
                        "yaw_to_target": yaw,
                        "files": saved,
                    })
                    view_id += 1

            metadata = {
                "source": "reconstructed_from_cityavos_simulator",
                **item,
                "capture_height_above_target": abs(args.height),
                "radii": args.radii,
                "camera_ids": camera_ids,
                "records": records,
            }
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)

        # Build/rebuild contact sheet from whatever images exist locally.
        image_paths = []
        labels = []
        if os.path.isdir(task_dir):
            for name in sorted(os.listdir(task_dir)):
                if not name.lower().endswith(".png"):
                    continue
                image_paths.append(os.path.join(task_dir, name))
                labels.append(name.replace(".png", ""))

        make_contact_sheet(image_paths, labels, contact_path)

        manifest_rows.append({
            "representative_task": task_id,
            "scene_id": item["scene_id"],
            "target_text": item["target_text"],
            "target_pos": json.dumps(item["target_pos"]),
            "target_image_path": item["target_image_path"],
            "raw_folder": task_dir,
            "contact_sheet": contact_path,
        })

    csv_path = os.path.join(args.review_dir, "batch_manifest.csv")
    json_path = os.path.join(args.review_dir, "batch_manifest.json")

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=manifest_rows[0].keys())
        writer.writeheader()
        writer.writerows(manifest_rows)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(manifest_rows, f, ensure_ascii=False, indent=2)

    zip_path = os.path.join(r".\reconstructed_targets", "review_pack.zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.write(csv_path, arcname="batch_manifest.csv")
        z.write(json_path, arcname="batch_manifest.json")
        for row in manifest_rows:
            cp = row["contact_sheet"]
            if os.path.exists(cp):
                z.write(cp, arcname=os.path.basename(cp))

    print()
    print("=" * 78)
    print("Batch capture finished.")
    print(f"Raw images : {os.path.abspath(args.output)}")
    print(f"Review dir : {os.path.abspath(args.review_dir)}")
    print(f"Upload this ZIP for review: {os.path.abspath(zip_path)}")
    print("=" * 78)


if __name__ == "__main__":
    main()
