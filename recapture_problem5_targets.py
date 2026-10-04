import json
import math
import os
import time
import zipfile

import cv2
import numpy as np

from vln.drone_agent import AirsimAgent


PROBLEM_TASKS = {
    289: {
        "radii": [3.0, 5.0],
        "heights": [2.5, 4.0],
        "angles": list(range(0, 360, 45)),
        "cameras": [1, 2, 3],
    },
    291: {
        "radii": [2.5, 4.0, 6.0],
        "heights": [2.0, 3.5],
        "angles": list(range(0, 360, 45)),
        "cameras": [1, 2],
    },
    292: {
        "radii": [3.0, 5.0, 7.0],
        "heights": [2.5, 4.0],
        "angles": list(range(0, 360, 45)),
        "cameras": [1, 2, 3],
    },
    308: {
        "radii": [5.0, 8.0, 12.0],
        "heights": [3.5, 6.0],
        "angles": list(range(0, 360, 45)),
        "cameras": [1, 2],
    },
    314: {
        "radii": [6.0, 10.0, 14.0],
        "heights": [4.0, 7.0],
        "angles": list(range(0, 360, 45)),
        "cameras": [1, 2],
    },
}


def save_rgb(path, image):
    if image is None:
        raise RuntimeError(f"Empty image: {path}")
    if image.ndim == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, image)


def make_contact_sheet(image_paths, labels, out_path, cols=5, thumb_w=220, thumb_h=170):
    tiles = []
    for p, label in zip(image_paths, labels):
        img = cv2.imread(p)
        if img is None:
            continue
        h, w = img.shape[:2]
        scale = min(thumb_w / max(w, 1), (thumb_h - 30) / max(h, 1))
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)

        tile = np.full((thumb_h, thumb_w, 3), 245, np.uint8)
        x = (thumb_w - nw) // 2
        y = 25 + max(0, (thumb_h - 30 - nh) // 2)
        tile[y:y+nh, x:x+nw] = resized
        cv2.putText(
            tile, label, (5, 17),
            cv2.FONT_HERSHEY_SIMPLEX, 0.36,
            (0, 0, 0), 1, cv2.LINE_AA
        )
        tiles.append(tile)

    if not tiles:
        return

    rows = math.ceil(len(tiles) / cols)
    blank = np.full((thumb_h, thumb_w, 3), 245, np.uint8)
    while len(tiles) < rows * cols:
        tiles.append(blank.copy())

    sheet_rows = []
    for r in range(rows):
        sheet_rows.append(np.hstack(tiles[r*cols:(r+1)*cols]))
    cv2.imwrite(out_path, np.vstack(sheet_rows))


def main():
    dataset_path = r".\data\target_information_605.json"
    out_root = r".\reconstructed_targets\problem5_raw"
    review_dir = r".\reconstructed_targets\problem5_review"
    os.makedirs(out_root, exist_ok=True)
    os.makedirs(review_dir, exist_ok=True)

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    tasks = {}
    for task_id in PROBLEM_TASKS:
        item = next(x for x in dataset if int(x["mission_id"]) == task_id)
        tasks[task_id] = item

    print("Problem targets:")
    for task_id, item in tasks.items():
        print(
            f"Task {task_id} | {item['target_text']} | "
            f"{item['target_image_path']} | pos={item['target_pos']}"
        )

    drone = AirsimAgent(None, None, None)

    contact_files = []

    for task_id, cfg in PROBLEM_TASKS.items():
        item = tasks[task_id]
        tx, ty, tz = [float(v) for v in item["target_pos"]]
        stem = os.path.splitext(os.path.basename(item["target_image_path"]))[0]
        task_dir = os.path.join(out_root, f"{task_id}_{stem}")
        os.makedirs(task_dir, exist_ok=True)

        print("\n" + "=" * 78)
        print(f"Task {task_id}: {item['target_text']}")
        print(f"Target pos: {item['target_pos']}")
        print("=" * 78)

        records = []
        view = 0

        for height in cfg["heights"]:
            for radius in cfg["radii"]:
                for angle_deg in cfg["angles"]:
                    angle = math.radians(angle_deg)
                    cx = tx + radius * math.cos(angle)
                    cy = ty + radius * math.sin(angle)
                    cz = tz - abs(height)
                    yaw = math.degrees(math.atan2(ty - cy, tx - cx))

                    drone.pos = np.array([cx, cy, cz], dtype=float)
                    drone.ori = np.array([0.0, 0.0, yaw], dtype=float)
                    pose = np.concatenate([drone.pos, drone.ori]).tolist()

                    drone.MovetoPose(pose)
                    time.sleep(0.25)

                    files = {}
                    for cam in cfg["cameras"]:
                        try:
                            img = drone.get_xyg_image(0, cam)
                            name = (
                                f"v{view:03d}_r{radius:g}_h{height:g}_"
                                f"a{angle_deg}_cam{cam}.png"
                            )
                            path = os.path.join(task_dir, name)
                            save_rgb(path, img)
                            files[str(cam)] = name
                        except Exception as e:
                            print(f"camera {cam} failed: {e}")

                    records.append({
                        "view": view,
                        "radius": radius,
                        "height": height,
                        "angle_deg": angle_deg,
                        "capture_pose": pose,
                        "files": files,
                    })
                    view += 1

        metadata = {
            "representative_task": task_id,
            "scene_id": item["scene_id"],
            "target_text": item["target_text"],
            "target_pos": item["target_pos"],
            "target_image_path": item["target_image_path"],
            "capture_config": cfg,
            "records": records,
        }
        with open(os.path.join(task_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        image_paths = []
        labels = []
        for name in sorted(os.listdir(task_dir)):
            if name.lower().endswith(".png"):
                image_paths.append(os.path.join(task_dir, name))
                labels.append(name[:-4])

        contact = os.path.join(review_dir, f"{task_id}_{stem}_contact.jpg")
        make_contact_sheet(image_paths, labels, contact)
        contact_files.append(contact)
        print(f"Saved contact sheet: {contact}")

    zip_path = os.path.join(r".\reconstructed_targets", "problem5_review_pack.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for p in contact_files:
            if os.path.exists(p):
                z.write(p, arcname=os.path.basename(p))

    print("\nDone.")
    print(f"Review ZIP: {os.path.abspath(zip_path)}")
    print("Upload problem5_review_pack.zip for final review.")


if __name__ == "__main__":
    main()
