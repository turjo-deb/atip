# src/phase4_vlm.py

import os
import glob
import json
import time
import base64
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

CROPS_DIR = "outputs/crops"
OUTPUT_JSON = "outputs/vlm_descriptions.json"

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

PROMPT = """Analyze this vehicle image. Return ONLY a valid JSON object, no markdown, no explanation, no code fences. Format exactly:

{
  "vehicle_type": "car|bus|truck|motorcycle|van|other",
  "dominant_color": "main color covering most of the vehicle body",
  "visible_colors": ["list", "every", "distinct", "color", "visible", "on", "any", "part", "of", "the", "vehicle", "e.g.", "yellow back", "white front", "red stripe"],
  "cargo": ["list", "of", "visible", "cargo", "items", "or", "empty", "list"],
  "special_vehicle": false,
  "special_vehicle_type": "ambulance|police|fire|none",
  "company_logo": false,
  "description": ["tag", "words", "describing", "notable", "features"]
}

Rules:
- dominant_color: the single color covering the largest visible area
- visible_colors: list EVERY color you can see anywhere on the vehicle, tagged with location if not uniform (e.g. "yellow rear", "white front", "blue door", "black roof") — even minor colored parts count, this is important for search matching
- special_vehicle: true only if ambulance/police/fire/emergency vehicle
- cargo: only list what's clearly visible, empty list if none
- description: describe the VEHICLE's actual real-world features/condition — body type, damage, modifications, roof rack, taxi markings, brand/logo visible, rust, dents, decorations, unusual load, anything a person would notice
- Use as many description tags as genuinely apply (2 tags if plain/ordinary, up to 6-8 if there's a lot to notice)
- NEVER include words describing the PHOTO quality (blurry, unclear, low resolution, pixelated, dark, grainy, out of focus)
- If the vehicle itself is too obscured/occluded to see clearly, return an empty description list rather than commenting on visibility
- Return raw JSON only, nothing else"""


def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def analyze_crop(image_path):
    b64_image = encode_image(image_path)

    try:
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
        return json.loads(text.strip())
    except Exception as e:
        print(f"  Error: {e}")
        return None


def main():
    crop_files = glob.glob(f"{CROPS_DIR}/*.jpg")
    results = {}

    for i, path in enumerate(crop_files):
        filename = os.path.basename(path)
        print(f"[{i+1}/{len(crop_files)}] Analyzing {filename}...")

        data = analyze_crop(path)
        if data:
            results[filename] = data
            print(f"  -> {data.get('vehicle_type')}, cargo: {data.get('cargo')}")
        else:
            results[filename] = {"error": "failed to analyze"}

        time.sleep(2.5)  # ~24 req/min, safe under 30 RPM cap

    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nDone. {len(results)} vehicles analyzed.")


if __name__ == "__main__":
    main()