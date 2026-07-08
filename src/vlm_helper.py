# src/phase4_vlm.py

import os
import glob
import json
import time
import base64
import threading
import collections
import cv2
import numpy as np
from dotenv import load_dotenv
from groq import Groq
import itertools

load_dotenv()
print("DEBUG GROQ_API_KEYS:", repr(os.getenv("GROQ_API_KEYS")))

CROPS_DIR = "outputs/crops"
OUTPUT_JSON = "outputs/vlm_descriptions.json"
MIN_DIM_FOR_VLM = 500  # upscale below this so the model has enough detail to work with
MAX_CALLS_PER_MINUTE = 28  # stay a hair under Groq's ~30 RPM cap on this model



API_KEYS = [k.strip() for k in os.getenv("GROQ_API_KEYS", "").split(",") if k.strip()]
if not API_KEYS:
    single = os.getenv("GROQ_API_KEY")
    API_KEYS = [single] if single else []
if not API_KEYS:
    print("WARNING: no GROQ_API_KEYS/GROQ_API_KEY found in environment")
    API_KEYS = ["MISSING_KEY_PLACEHOLDER"]

_key_cycle = itertools.cycle(API_KEYS)
_current_key = next(_key_cycle)
client = Groq(api_key=_current_key)

def _rotate_key():
    global _current_key, client
    _current_key = next(_key_cycle)
    client = Groq(api_key=_current_key)
    print(f"  -> rotated to key ...{_current_key[-4:]}")

VALID_VEHICLE_TYPES = {"car", "bus", "truck", "motorcycle", "van", "other"}


class RateLimiter:
    """Thread-safe sliding-window limiter so concurrent workers collectively
    stay under Groq's RPM cap instead of each pacing independently with a
    fixed sleep (which is what made eager analysis scale linearly with
    vehicle count before)."""

    def __init__(self, max_calls, period=60.0):
        self.max_calls = max_calls
        self.period = period
        self.calls = collections.deque()
        self.lock = threading.Lock()

    def acquire(self):
        while True:
            with self.lock:
                now = time.time()
                while self.calls and now - self.calls[0] > self.period:
                    self.calls.popleft()
                if len(self.calls) < self.max_calls:
                    self.calls.append(now)
                    return
                sleep_time = self.period - (now - self.calls[0])
            time.sleep(max(sleep_time, 0.05))


_rate_limiter = RateLimiter(MAX_CALLS_PER_MINUTE)

PROMPT = """Analyze this vehicle image in detail. Return ONLY a valid JSON object, no markdown, no explanation, no code fences. Format exactly:

{
  "vehicle_type": "car|bus|truck|motorcycle|van|other",
  "dominant_color": "main color covering most of the vehicle body",
  "visible_colors": ["list", "every", "distinct", "color", "visible", "on", "any", "part", "e.g.", "yellow back", "white front", "red stripe"],
  "cargo": ["anything the vehicle is carrying, towing, or transporting - including another vehicle, boxes, furniture, animals, people, containers - empty list if none"],
  "cargo_location": "roof|truck_bed|towed_trailer|flatbed|interior_visible|none",
  "roof_items": ["anything mounted or placed on the roof - rack, cargo box, luggage, ladder, AC unit, satellite dish, antenna - empty list if none/not visible"],
  "front_view_details": ["notable features visible from the front - grille style, headlight condition, bumper condition, taxi light, license plate holder, damage - empty list if front not visible in this crop"],
  "rear_view_details": ["notable features visible from the back - taillight condition, bumper stickers, exhaust, spare tire, damage - empty list if rear not visible in this crop"],
  "side_view_details": ["notable features visible from the side - door dents, decals, number of doors, side ladder rack, mirror condition - empty list if side not visible in this crop"],
  "advertisement_or_text": ["any readable text, logos, brand names, or company names visible anywhere on the vehicle body - transcribe what's actually readable, empty list if none"],
  "special_vehicle": false,
  "special_vehicle_type": "ambulance|police|fire|none",
  "company_logo": false,
  "description": ["tag", "words", "describing", "notable", "overall", "features"]
}

Rules:
- dominant_color: the single color covering the largest visible area
- visible_colors: list EVERY color visible anywhere on the vehicle, tagged with location if not uniform — even minor colored parts count
- cargo: describe WHAT is being carried as specifically as possible (e.g. "another car", "motorcycle", "furniture", "boxes", "hay bales") — not just "cargo" or "items"
- cargo_location: where the cargo sits relative to the vehicle — this matters a lot for matching queries like "carrying on roof" vs "towing"
- roof_items / front_view_details / rear_view_details / side_view_details: only describe what THIS crop actually shows. If an angle isn't visible in the image, return an empty list for it — do not guess or assume
- advertisement_or_text: transcribe EVERY piece of visible text exactly as seen — brand names, shop names, ad text, license/registration plate numbers, route numbers, painted numbers, stickers, decals. Read character-by-character carefully, including partially visible or small text. This is critical for search matching (e.g. plate number searches).
- special_vehicle: true only if ambulance/police/fire/emergency vehicle
- description: describe the VEHICLE's actual real-world features/condition — body type, damage, modifications, rust, dents, decorations, unusual load, anything a person would notice
- Use as many description tags as genuinely apply (2 tags if plain/ordinary, up to 6-8 if there's a lot to notice)
- NEVER include words describing the PHOTO quality (blurry, unclear, low resolution, pixelated, dark, grainy, out of focus)
- If the vehicle itself is too obscured/occluded to see clearly, return empty lists rather than commenting on visibility
- Return raw JSON only, nothing else"""


def encode_image(path):
    """Reads an image, upscales it if it's small (helps the VLM actually see detail), returns base64 jpeg."""
    img = cv2.imread(path)
    if img is not None:
        h, w = img.shape[:2]
        m = min(h, w)
        if 0 < m < MIN_DIM_FOR_VLM:
            scale = MIN_DIM_FOR_VLM / m
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if ok:
            return base64.b64encode(buf.tobytes()).decode("utf-8")

    # fallback: just read raw bytes if decoding failed for some reason
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _call_vlm(b64_image):
    response = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",  # vision-capable, check console.groq.com/docs/models for current name
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}
                    }
                ]
            }
        ]
    )
    text = response.choices[0].message.content.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    data = json.loads(text.strip())
    if data.get("vehicle_type") not in VALID_VEHICLE_TYPES:
        data["vehicle_type"] = None
    return data


def analyze_crop(image_path, retries=3):
    """
    Runs full VLM analysis on a crop. Retries once on transient failure
    (network hiccup, rate limit, malformed response) instead of silently
    giving up — this was previously a common cause of vehicles staying
    stuck with wrong/incomplete classification.
    """
    try:
        b64_image = encode_image(image_path)
    except Exception as e:
        print(f"  Error encoding {image_path}: {e}")
        return None

    last_error = None
    for attempt in range(retries + 1):
        try:
            _rate_limiter.acquire()
            return _call_vlm(b64_image)
        except Exception as e:
            last_error = e
            if "429" in str(e) or "rate_limit" in str(e):
                _rotate_key()
                continue
            if attempt < retries:
                time.sleep(2.0)
                continue
    print(f"  VLM analysis failed for {image_path} after {retries + 1} attempt(s): {last_error}")
    return None


def main():
    crop_files = glob.glob(f"{CROPS_DIR}/*.jpg")

    results = {}
    if os.path.exists(OUTPUT_JSON):
        with open(OUTPUT_JSON, "r") as f:
            results = json.load(f)

    todo = [p for p in crop_files if os.path.basename(p) not in results
            or results[os.path.basename(p)].get("error")]

    print(f"{len(crop_files)} total, {len(todo)} need analysis (rest cached)")

    for i, path in enumerate(todo):
        filename = os.path.basename(path)
        print(f"[{i+1}/{len(todo)}] Analyzing {filename}...")
        data = analyze_crop(path)
        if data:
            results[filename] = data
            print(f"  -> {data.get('vehicle_type')}, cargo: {data.get('cargo')}")
        else:
            results[filename] = {"error": "failed to analyze"}
        with open(OUTPUT_JSON, "w") as f:
            json.dump(results, f, indent=2)

    print(f"\nDone. {len(results)} vehicles analyzed total.")


if __name__ == "__main__":
    main()