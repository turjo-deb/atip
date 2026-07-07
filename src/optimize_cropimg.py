# add to phase2 or run separately — filters tiny/likely-junk crops
import cv2, glob, os

MIN_AREA = 3000  # pixels, tune based on your crops

for path in glob.glob("outputs/crops/*.jpg"):
    img = cv2.imread(path)
    if img is None or img.shape[0] * img.shape[1] < MIN_AREA:
        os.remove(path)
        print(f"removed {path}")

print(f"Remaining: {len(glob.glob('outputs/crops/*.jpg'))}")