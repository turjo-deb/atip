# src/phase6_search.py

import sqlite3
import json
import os
import argparse
from dotenv import load_dotenv
from groq import Groq
from vlm_helper import analyze_crop

load_dotenv()

DB_PATH = "outputs/atip.db"

PARSE_PROMPT = """Convert this vehicle search query into a JSON filter object. Return ONLY raw JSON, no markdown, no explanation.

Format exactly:
{
  "vehicle_types": ["car"|"bus"|"truck"|"motorcycle", ...] or [],
  "colors": ["red"|"white"|"blue"|"black"|"yellow"|"green"|"silver/gray"|"purple/pink"|"orange", ...] or [],
  "time_after_seconds": number or null,
  "time_before_seconds": number or null,
  "direction": "up|down|null",
  "free_text": "remaining descriptive keywords not covered above as a short space-separated string, or empty string"
}

Rules:
- vehicle_types: list ALL vehicle types mentioned (e.g. "car or truck" -> ["car","truck"]). Empty list if none mentioned / "any vehicle"
- colors: list ALL overall/dominant colors mentioned. Empty list if none mentioned or query says "any color"
- If query describes a SPECIFIC PART's color (e.g. "yellow back"), do NOT add to colors — keep phrase in free_text instead
- direction: null if not mentioned
- time_after_seconds/time_before_seconds: convert phrases like "after 30 seconds" into numbers, null if not mentioned
- free_text: descriptive keywords not used as strict filters above (cargo, condition, features, part-specific colors). NEVER repeat vehicle type or color words, never include filler like "vehicle", "vehicles", "colored" — only genuinely distinguishing details

Query: """

_groq_client = None


def get_groq_client():
    global _groq_client
    if _groq_client is None:
        _groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    return _groq_client


def parse_query_simple(query):
    query_lower = query.lower()
    filters = {"vehicle_types": [], "colors": [], "time_after_seconds": None,
               "time_before_seconds": None, "direction": None, "free_text": ""}

    for vtype in ["car", "bus", "truck", "motorcycle"]:
        if vtype in query_lower:
            filters["vehicle_types"].append(vtype)

    for color in ["red", "white", "blue", "black", "yellow", "green", "silver", "purple", "orange"]:
        if color in query_lower:
            filters["colors"].append(color)

    remainder = query_lower
    for kw in filters["vehicle_types"] + filters["colors"]:
        remainder = remainder.replace(kw, "")
    filters["free_text"] = remainder.strip()

    return filters


def parse_query_llm(query):
    try:
        client = get_groq_client()
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": PARSE_PROMPT + query}],
            temperature=0
        )
        text = response.choices[0].message.content.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        return json.loads(text.strip())
    except Exception as e:
        print(f"  LLM parse failed ({e}), falling back to keyword parser")
        return parse_query_simple(query)


def sql_filter(conn, filters, video_id=None):
    cur = conn.cursor()
    query = "SELECT track_id, video_id, vehicle_type, color, crop_path, timestamp_seconds, direction, vlm_analyzed, vlm_data FROM vehicles WHERE 1=1"
    params = []

    if video_id:
        query += " AND video_id = ?"
        params.append(video_id)

    if filters.get("vehicle_types"):
        placeholders = ",".join("?" * len(filters["vehicle_types"]))
        query += f" AND vehicle_type IN ({placeholders})"
        params.extend(filters["vehicle_types"])

    if filters.get("colors"):
        placeholders = ",".join("?" * len(filters["colors"]))
        query += f" AND color IN ({placeholders})"
        params.extend(filters["colors"])

    if filters.get("direction"):
        query += " AND direction = ?"
        params.append(filters["direction"])

    if filters.get("time_after_seconds") is not None:
        query += " AND timestamp_seconds >= ?"
        params.append(filters["time_after_seconds"])

    if filters.get("time_before_seconds") is not None:
        query += " AND timestamp_seconds <= ?"
        params.append(filters["time_before_seconds"])

    cur.execute(query, params)
    return cur.fetchall()


def get_or_run_vlm(conn, track_id, video_id, crop_path, vlm_analyzed, vlm_data, on_vlm_call=None):
    if vlm_analyzed:
        return json.loads(vlm_data) if vlm_data else {}

    if on_vlm_call:
        on_vlm_call(track_id, crop_path)

    result = analyze_crop(crop_path)

    cur = conn.cursor()
    cur.execute("""
        UPDATE vehicles SET vlm_analyzed = 1, vlm_data = ?
        WHERE track_id = ? AND video_id = ?
    """, (json.dumps(result) if result else None, track_id, video_id))
    conn.commit()

    return result or {}


GENERIC_NOISE_WORDS = {
    "car", "cars", "bus", "buses", "truck", "trucks", "motorcycle", "motorcycles",
    "vehicle", "vehicles", "colored", "color", "colour", "show", "find", "all",
    "with", "carrying", "a", "an", "the", "any"
}


def matches_free_text(vlm_data, free_text):
    if not free_text or not free_text.strip():
        return True

    keywords = [kw for kw in free_text.lower().split()
                if len(kw) > 2 and kw not in GENERIC_NOISE_WORDS]

    if not keywords:
        return True  # nothing meaningful left to check, don't reject the match

    if not vlm_data:
        return False

    haystack = " ".join([
        " ".join(vlm_data.get("cargo", [])),
        " ".join(vlm_data.get("description", [])),
        " ".join(vlm_data.get("visible_colors", [])),
        str(vlm_data.get("dominant_color", "")),
        str(vlm_data.get("special_vehicle_type", "")),
    ]).lower()

    return any(kw in haystack for kw in keywords)


def search(query, video_id=None, on_vlm_call=None):
    """
    Main search entry point, importable by Streamlit.
    on_vlm_call(track_id, crop_path) is called right before each fresh (non-cached) VLM call — useful for UI feedback.
    Returns (filters_used, list_of_matches).
    """
    conn = sqlite3.connect(DB_PATH)
    filters = parse_query_llm(query)
    candidates = sql_filter(conn, filters, video_id=video_id)

    final_matches = []
    for tid, vid, vtype, color, crop_path, ts, direction, vlm_analyzed, vlm_data in candidates:
        vlm_result = get_or_run_vlm(conn, tid, vid, crop_path, vlm_analyzed, vlm_data, on_vlm_call=on_vlm_call)

        if matches_free_text(vlm_result, filters.get("free_text")):
            final_matches.append({
                "track_id": tid,
                "video_id": vid,
                "vehicle_type": vtype,
                "color": color,
                "crop_path": crop_path,
                "timestamp_seconds": ts,
                "direction": direction,
                "vlm_data": vlm_result
            })

    conn.close()
    return filters, final_matches


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("query", help="natural language search query")
    parser.add_argument("--video", default=None, help="optional: scope search to one video_id")
    args = parser.parse_args()

    def cli_vlm_notice(track_id, crop_path):
        print(f"  Running VLM on track_id={track_id} (not cached)...")

    filters, matches = search(args.query, video_id=args.video, on_vlm_call=cli_vlm_notice)
    print(f"Parsed filters: {filters}")
    print(f"Final matches: {len(matches)}")
    for m in matches:
        print(json.dumps(m, indent=2))