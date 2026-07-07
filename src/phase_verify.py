# src/phase_verify.py

import json
import os
from vlm_helper import analyze_crop

AMBIGUOUS_CLASSES = {"truck", "bus"}


def verify_truck_bus(video_path):
    video_id = os.path.splitext(os.path.basename(video_path))[0]
    counted_json = f"outputs/{video_id}/phase2_counted.json"

    with open(counted_json, "r") as f:
        counted_log = json.load(f)

    changed = 0
    checked = 0

    for entry in counted_log:
        if entry.get("vehicle_type") not in AMBIGUOUS_CLASSES:
            continue
        crop_path = entry.get("crop_path")
        if not crop_path or not os.path.exists(crop_path):
            continue

        checked += 1
        result = analyze_crop(crop_path)
        vlm_type = result.get("vehicle_type") if result else None

        if vlm_type in AMBIGUOUS_CLASSES and vlm_type != entry["vehicle_type"]:
            entry["vehicle_type"] = vlm_type
            changed += 1

    with open(counted_json, "w") as f:
        json.dump(counted_log, f, indent=2)

    return {"checked": checked, "corrected": changed}