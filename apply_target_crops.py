import argparse
import csv
import os
import shutil

import cv2


def main():
    parser = argparse.ArgumentParser(
        description="Apply reviewed target-image crops from a CSV manifest."
    )
    parser.add_argument(
        "--manifest",
        default=r".\reconstructed_targets\crop_manifest.csv",
        help="CSV returned after review",
    )
    parser.add_argument(
        "--selected-dir",
        default=r".\reconstructed_targets\selected",
    )
    parser.add_argument(
        "--install",
        action="store_true",
        help="Also copy approved crops into data/target_image paths",
    )
    args = parser.parse_args()

    os.makedirs(args.selected_dir, exist_ok=True)

    with open(args.manifest, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    ok = 0
    skipped = 0

    for row in rows:
        status = row.get("status", "").strip().lower()
        if status not in ("ok", "approved", "1", "yes"):
            print(f"SKIP status={status}: {row.get('target_image_path')}")
            skipped += 1
            continue

        source = row["source_file"]
        target_path = row["target_image_path"]

        x1 = int(row["x1"])
        y1 = int(row["y1"])
        x2 = int(row["x2"])
        y2 = int(row["y2"])

        img = cv2.imread(source)
        if img is None:
            print(f"ERROR cannot read: {source}")
            skipped += 1
            continue

        h, w = img.shape[:2]
        x1 = max(0, min(x1, w - 1))
        x2 = max(x1 + 1, min(x2, w))
        y1 = max(0, min(y1, h - 1))
        y2 = max(y1 + 1, min(y2, h))

        crop = img[y1:y2, x1:x2]
        basename = os.path.basename(target_path)
        selected_path = os.path.join(args.selected_dir, basename)
        cv2.imwrite(selected_path, crop)

        if args.install:
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            shutil.copy2(selected_path, target_path)
            print(f"INSTALLED {target_path}")
        else:
            print(f"CREATED   {selected_path}")

        ok += 1

    print()
    print(f"Approved processed: {ok}")
    print(f"Skipped/not approved: {skipped}")
    if not args.install:
        print("Review selected images, then rerun with --install to copy them into data/target_image.")


if __name__ == "__main__":
    main()
