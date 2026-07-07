# src/phase5_db.py

import sqlite3
import json
import os
import argparse

DB_PATH = "outputs/atip.db"


def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS vehicles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id TEXT,
            track_id TEXT,
            vehicle_type TEXT,
            color TEXT,
            crop_path TEXT,
            timestamp_seconds REAL,
            direction TEXT,
            confirmed_by TEXT,
            vlm_analyzed INTEGER DEFAULT 0,
            vlm_data TEXT DEFAULT NULL,
            UNIQUE(video_id, track_id)
        )
    """)
    # migration: add saved column if missing
    cur.execute("PRAGMA table_info(vehicles)")
    cols = [row[1] for row in cur.fetchall()]
    if "saved" not in cols:
        cur.execute("ALTER TABLE vehicles ADD COLUMN saved INTEGER DEFAULT 0")
    conn.commit()
    return conn


def run_phase5(video_path):
    video_id = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = f"outputs/{video_id}"
    counted_json = f"{output_dir}/phase2_counted.json"
    colors_json = f"{output_dir}/colors.json"
    timestamps_json = f"{output_dir}/timestamps.json"

    conn = init_db()

    with open(counted_json, "r") as f:
        counted_log = json.load(f)
    with open(colors_json, "r") as f:
        colors_data = json.load(f)
    with open(timestamps_json, "r") as f:
        ts_data = json.load(f)

    fps = ts_data["fps"]

    cur = conn.cursor()

    # NEW: wipe old rows for this video_id first — track_ids aren't stable across reruns
    cur.execute("DELETE FROM vehicles WHERE video_id = ?", (video_id,))

    inserted = 0

    for entry in counted_log:
        track_id = str(entry["track_id"])
        vehicle_type = entry["vehicle_type"]
        crop_path = entry["crop_path"]
        direction = entry.get("direction")
        confirmed_by = entry.get("confirmed_by")
        first_seen_frame = entry.get("first_seen_frame", 0)
        timestamp_seconds = first_seen_frame / fps if fps else 0

        color = "unknown"
        if crop_path:
            filename = os.path.basename(crop_path)
            color_info = colors_data.get(filename, {})
            color = color_info.get("color", "unknown")

        cur.execute("""
            INSERT INTO vehicles
                (video_id, track_id, vehicle_type, color, crop_path, timestamp_seconds, direction, confirmed_by, vlm_analyzed, vlm_data)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, NULL)
        """, (video_id, track_id, vehicle_type, color, crop_path, timestamp_seconds, direction, confirmed_by))
        inserted += 1

    conn.commit()

    cur.execute("SELECT COUNT(*), vehicle_type FROM vehicles WHERE video_id=? GROUP BY vehicle_type", (video_id,))
    summary = {vtype: count for count, vtype in cur.fetchall()}

    conn.close()
    return {"video_id": video_id, "inserted": inserted, "summary": summary}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="videos/traffic1.mp4", help="path to input video")
    args = parser.parse_args()

    result = run_phase5(args.video)
    print(f"Inserted/updated {result['inserted']} vehicles for video_id={result['video_id']}")
    print(f"\nSummary: {result['summary']}")