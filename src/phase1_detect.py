import cv2
from ultralytics import YOLO

VIDEO_PATH = "videos/traffic4.mp4"
OUTPUT_PATH = "outputs/phase1_test.mp4"

TARGET_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck", 0: "person"}

model = YOLO("yolo11n.pt")

cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (w, h))

frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, classes=list(TARGET_CLASSES.keys()), verbose=False)
    annotated = results[0].plot()
    out.write(annotated)

    frame_count += 1
    if frame_count % 30 == 0:
        print(f"Processed {frame_count} frames")

cap.release()
out.release()
print(f"Done. Output saved to {OUTPUT_PATH}")