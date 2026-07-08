# src/phase2_track.py

import cv2
import os
import json
import glob
import argparse
from ultralytics import YOLO

TARGET_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

# --- single counting line (replaces the old dual-zone system) ---
LINE_POSITION = 0.75       # fraction of frame height where the counting line sits
LINE_MARGIN = 15          # hysteresis band (px) around the line to avoid flicker double-counts
LOST_TRACK_ZONE_MARGIN = 30
LOST_TRACK_BUFFER_FRAMES = 8
LOST_TRACK_MIN_FRAMES_SEEN = 3
TRACK_TIMEOUT_FRAMES = 10
IOU_REATTACH_THRESHOLD = 0.25
IOU_SEARCH_MAX_LOST_FRAMES = LOST_TRACK_BUFFER_FRAMES

MIN_CROP_AREA = 3000      # px^2 — crops smaller than this are junk, never saved/analyzed
CROP_MIN_DIM = 200        # upscale crops smaller than this (helps VLM + UI thumbnail quality)

_model = None


def get_model():
    global _model
    if _model is None:
        _model = YOLO("yolo11n.pt")
    return _model


def iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / float(area_a + area_b - inter)


def upscale_if_small(img):
    h, w = img.shape[:2]
    m = min(h, w)
    if m <= 0 or m >= CROP_MIN_DIM:
        return img
    scale = CROP_MIN_DIM / m
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)


def run_phase2(video_path, progress_callback=None):
    """
    Runs detection + single-line tracking on a video.
    progress_callback(frame_count, total_frames, confirmed_count) is called periodically if provided.
    Returns a summary dict.
    """
    video_id = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = f"outputs/{video_id}"
    output_path = f"{output_dir}/phase2_test.mp4"
    crops_dir = f"{output_dir}/crops"
    timestamps_json = f"{output_dir}/timestamps.json"
    counted_json = f"{output_dir}/phase2_counted.json"

    os.makedirs(crops_dir, exist_ok=True)
    for old_file in glob.glob(f"{crops_dir}/*.jpg"):
        os.remove(old_file)

    model = get_model()

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    LINE_Y = int(h * LINE_POSITION)
    LINE_TOP = LINE_Y - LINE_MARGIN
    LINE_BOTTOM = LINE_Y + LINE_MARGIN

    tracks = {}
    vehicle_count = 0
    counted_log = []
    first_seen_frame = {}

    def new_track(frame_idx):
        return {
            "first_seen": frame_idx, "last_seen": frame_idx,
            "prev_bottom": None, "prev_box": None, "best_crop": None,
            "best_score": -1.0, "best_conf": -1.0, "votes": {}, "final_class": None,
            "counted": False, "direction": None, "lost_frames": 0,
            "side": None,  # "above" or "below" the line, set once track is clearly on one side
        }

    def side_of(bottom_y):
        if bottom_y < LINE_TOP:
            return "above"
        if bottom_y > LINE_BOTTOM:
            return "below"
        return None  # inside the hysteresis band — ambiguous, don't commit yet

    def near_line(bottom_y):
        if bottom_y is None:
            return False
        return (LINE_TOP - LOST_TRACK_ZONE_MARGIN) <= bottom_y <= (LINE_BOTTOM + LOST_TRACK_ZONE_MARGIN)

    def finalize_class(track):
        if not track["votes"]:
            return "unknown"
        return max(track["votes"], key=track["votes"].get)

    def save_crop(track_id, track, frame_idx):
        if track["best_crop"] is None or track["best_crop"].size == 0:
            return None
        area = track["best_crop"].shape[0] * track["best_crop"].shape[1]
        if area < MIN_CROP_AREA:
            return None  # junk crop — too small to be useful, skip saving/analyzing entirely
        cls = track["final_class"] or finalize_class(track)
        filename = f"vehicle_{track_id}_{cls}_{frame_idx}.jpg"
        path = os.path.join(crops_dir, filename)
        cv2.imwrite(path, upscale_if_small(track["best_crop"]))
        return path

    def confirm_counted(track_id, track, frame_idx, direction):
        nonlocal vehicle_count
        if track["counted"]:
            return
        track["counted"] = True
        track["direction"] = direction
        track["final_class"] = finalize_class(track)
        vehicle_count += 1

        crop_path = save_crop(track_id, track, frame_idx)
        first_seen_frame[str(track_id)] = track["first_seen"]

        counted_log.append({
            "video_id": video_id,
            "track_id": track_id,
            "vehicle_type": track["final_class"],
            "crop_path": crop_path,
            "first_seen_frame": track["first_seen"],
            "counted_frame": frame_idx,
            "best_conf": track["best_conf"],
            "direction": direction,
            "confirmed_by": "line",
        })

    def process_detection(track_id, box, conf, cls, frame, frame_count):
        x1, y1, x2, y2 = map(int, box)
        crop = frame[y1:y2, x1:x2]
        class_name = TARGET_CLASSES.get(int(cls), "unknown")
        bottom_y = y2

        if track_id not in tracks:
            tracks[track_id] = new_track(frame_count)

        t = tracks[track_id]
        t["last_seen"] = frame_count
        t["lost_frames"] = 0
        t["votes"][class_name] = t["votes"].get(class_name, 0) + 1

        area = max(0, x2 - x1) * max(0, y2 - y1)
        score = area * float(conf)
        if crop.size > 0 and score > t["best_score"]:
            t["best_score"] = score
            t["best_conf"] = float(conf)
            t["best_crop"] = crop.copy()

        if not t["counted"]:
            current_side = side_of(bottom_y)
            if t["side"] is None:
                if current_side is not None:
                    t["side"] = current_side
            else:
                if current_side is not None and current_side != t["side"]:
                    direction = "down" if t["side"] == "above" else "up"
                    confirm_counted(track_id, t, frame_count, direction)

        t["prev_bottom"] = bottom_y
        t["prev_box"] = (x1, y1, x2, y2)
        return t, class_name, (x1, y1, x2, y2)

    results_gen = model.track(
        source=video_path,
        classes=list(TARGET_CLASSES.keys()),
        tracker="botsort.yaml",
        conf=0.25,
        imgsz=640,
        persist=True,
        stream=True,
        verbose=False
    )

    frame_count = 0

    for result in results_gen:
        frame = result.orig_img
        annotated = frame.copy()

        cv2.line(annotated, (0, LINE_Y), (w, LINE_Y), (0, 255, 255), 2)
        cv2.putText(annotated, "COUNT LINE", (20, LINE_Y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        seen_ids_this_frame = set()
        matched_lost_ids = set()

        if result.boxes.id is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            ids = result.boxes.id.cpu().numpy().astype(int)
            confs = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)

            for box, track_id, conf, cls in zip(boxes, ids, confs, classes):
                track_id = int(track_id)
                seen_ids_this_frame.add(track_id)
                t, class_name, (x1, y1, x2, y2) = process_detection(
                    track_id, box, conf, cls, frame, frame_count
                )
                color = (0, 255, 0) if t["counted"] else (0, 0, 255)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                label = f"ID{track_id} {class_name}"
                cv2.putText(annotated, label, (x1, max(0, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            currently_new_tid_boxes = [
                (int(tid), box) for tid, box in zip(ids, boxes)
                if int(tid) not in tracks or tracks[int(tid)]["first_seen"] == frame_count
            ]

            for lost_tid, lt in list(tracks.items()):
                if lost_tid in seen_ids_this_frame:
                    continue
                if lt["counted"] or lt["lost_frames"] > IOU_SEARCH_MAX_LOST_FRAMES:
                    continue
                if lt["prev_box"] is None:
                    continue

                best_match = None
                best_iou = IOU_REATTACH_THRESHOLD
                for new_tid, box in currently_new_tid_boxes:
                    if new_tid in matched_lost_ids or new_tid == lost_tid:
                        continue
                    score = iou(lt["prev_box"], tuple(map(int, box)))
                    if score > best_iou:
                        best_iou = score
                        best_match = (new_tid, box)

                if best_match is not None:
                    new_tid, box = best_match
                    idx = list(ids).index(new_tid)
                    conf = confs[idx]
                    cls = classes[idx]

                    merged = lt.copy()
                    tracks[new_tid] = merged
                    if lost_tid != new_tid:
                        del tracks[lost_tid]

                    process_detection(new_tid, box, conf, cls, frame, frame_count)
                    matched_lost_ids.add(new_tid)
                    seen_ids_this_frame.add(new_tid)

        lost_ids = [tid for tid, t in tracks.items()
                    if tid not in seen_ids_this_frame and not t["counted"]]

        for tid in lost_ids:
            t = tracks[tid]
            t["lost_frames"] += 1
            frames_observed = t["last_seen"] - t["first_seen"]
            close_to_line = near_line(t["prev_bottom"])

            if (t["side"] is not None and close_to_line and
                    frames_observed >= LOST_TRACK_MIN_FRAMES_SEEN):
                if t["lost_frames"] >= LOST_TRACK_BUFFER_FRAMES:
                    direction = "down" if t["side"] == "above" else "up"
                    confirm_counted(tid, t, frame_count, direction)

        for tid in list(tracks.keys()):
            t = tracks[tid]
            if not t["counted"] and t["lost_frames"] >= TRACK_TIMEOUT_FRAMES:
                del tracks[tid]

        cv2.putText(annotated, f"Tracked: {len(tracks)}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(annotated, f"Confirmed: {vehicle_count}", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        out.write(annotated)
        frame_count += 1

        if frame_count % 15 == 0 and progress_callback:
            progress_callback(frame_count, total_frames, vehicle_count)

    for tid, t in tracks.items():
        if not t["counted"] and t["side"] is not None and near_line(t["prev_bottom"]):
            direction = "down" if t["side"] == "above" else "up"
            confirm_counted(tid, t, frame_count, direction)

    out.release()
    import subprocess

    def reencode_for_browser(input_path):
        """OpenCV's mp4v codec isn't browser-compatible — re-encode to H.264."""
        temp_path = input_path.replace(".mp4", "_h264.mp4")
        try:
            subprocess.run([
                "ffmpeg", "-y", "-i", input_path,
                "-vcodec", "libx264", "-pix_fmt", "yuv420p",
                "-preset", "fast", "-loglevel", "error",
                temp_path
            ], check=True)
            os.replace(temp_path, input_path)
        except Exception as e:
            print(f"Warning: re-encode failed ({e}), video may not play in browser")
    reencode_for_browser(output_path)

    with open(timestamps_json, "w") as f:
        json.dump({"video_id": video_id, "fps": fps, "first_seen_frame": first_seen_frame}, f, indent=2)

    with open(counted_json, "w") as f:
        json.dump(counted_log, f, indent=2)

    if progress_callback:
        progress_callback(frame_count, total_frames, vehicle_count)

    return {
        "video_id": video_id,
        "frames": frame_count,
        "confirmed": vehicle_count,
        "output_video": output_path,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="videos/traffic1.mp4", help="path to input video")
    args = parser.parse_args()

    def cli_progress(frame_count, total_frames, confirmed):
        print(f"Processed {frame_count}/{total_frames} frames | Confirmed: {confirmed}")

    summary = run_phase2(args.video, progress_callback=cli_progress)
    print(f"\nDone. {summary}")