# src/phase3_color.py

import cv2
import numpy as np
import os
import glob
import json
import argparse

COLOR_RANGES = [
    ((0, 0, 0), (180, 255, 50), "black"),
    ((0, 0, 180), (180, 30, 255), "white"),
    ((0, 0, 50), (180, 30, 180), "silver/gray"),
    ((0, 70, 50), (10, 255, 255), "red"),
    ((170, 70, 50), (180, 255, 255), "red"),
    ((10, 70, 50), (25, 255, 255), "orange"),
    ((25, 70, 50), (35, 255, 255), "yellow"),
    ((35, 40, 40), (85, 255, 255), "green"),
    ((85, 40, 40), (130, 255, 255), "blue"),
    ((130, 40, 40), (170, 255, 255), "purple/pink"),
]


def get_dominant_color(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return "unknown"

    h, w = img.shape[:2]
    y1, y2 = int(h * 0.2), int(h * 0.8)
    x1, x2 = int(w * 0.2), int(w * 0.8)
    center = img[y1:y2, x1:x2]

    hsv = cv2.cvtColor(center, cv2.COLOR_BGR2HSV)

    color_counts = {}
    for lower, upper, name in COLOR_RANGES:
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        count = cv2.countNonZero(mask)
        color_counts[name] = color_counts.get(name, 0) + count

    if not color_counts or max(color_counts.values()) == 0:
        return "unknown"

    return max(color_counts, key=color_counts.get)


def run_phase3(video_path):
    video_id = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = f"outputs/{video_id}"
    crops_dir = f"{output_dir}/crops"
    output_json = f"{output_dir}/colors.json"

    crop_files = glob.glob(f"{crops_dir}/*.jpg")
    results = {}

    for path in crop_files:
        filename = os.path.basename(path)
        parts = filename.replace(".jpg", "").split("_")
        track_id = parts[1]
        color = get_dominant_color(path)
        results[filename] = {"track_id": track_id, "color": color}

    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)

    return {"video_id": video_id, "processed": len(results)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="videos/traffic1.mp4", help="path to input video")
    args = parser.parse_args()

    summary = run_phase3(args.video)
    print(f"Done. {summary['processed']} vehicles processed for {summary['video_id']}")