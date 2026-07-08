# src/phase6_search.py

import sqlite3
import json
import os
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
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
- vehicle_types: list ALL vehicle types mentioned that describe the MAIN vehicle being searched for (e.g. "car or truck" -> ["car","truck"]). Empty list if none mentioned / "any vehicle"
- colors: list ALL overall/dominant colors mentioned for the MAIN vehicle. Empty list if none mentioned or query says "any color"
- If query describes a SPECIFIC PART's color (e.g. "yellow back"), do NOT add to colors — keep phrase in free_text instead
- direction: null if not mentioned
- time_after_seconds/time_before_seconds: convert phrases like "after 30 seconds" into numbers, null if not mentioned
- free_text: descriptive keywords not used as strict filters above (cargo, condition, features, part-specific colors). NEVER repeat the vehicle type or color of the MAIN vehicle being searched for, never include filler like "vehicle", "vehicles", "colored" — only genuinely distinguishing details
- IMPORTANT: if a vehicle-type word describes CARGO rather than the main vehicle (e.g. "truck carrying a car" -> main vehicle is truck, cargo is car), KEEP that word in free_text. Only drop vehicle-type words that describe the main vehicle itself.

Query: """

RERANK_PROMPT_HEADER = """You are matching vehicle search results to a user's natural-language query. The query may describe cargo, roof items, damage, ads/text, or anything else about the vehicle, phrased however the user likes.

User query: "{query}"

Here are candidate vehicles observed by a camera system, described as JSON (one object per candidate, "id" is just an index):
{candidates_json}

Return ONLY a JSON array of the integer "id" values of candidates that plausibly match what the user described. Use your judgement about synonyms, paraphrasing, and partial matches (e.g. "carrying a car" should match cargo containing "another car" or "sedan"). Exclude candidates that clearly don't match. If nothing matches, return [].
Return ONLY the raw JSON array, nothing else."""

from vlm_helper import client as _shared_client, _rotate_key as _vlm_rotate_key
import vlm_helper


def get_groq_client():
    return vlm_helper.client


def _rotate_key():
    _vlm_rotate_key()


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
    for attempt in range(4):
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
            if "429" in str(e) or "rate_limit" in str(e):
                _rotate_key()
                continue
            print(f"  LLM parse failed ({e}), falling back to keyword parser")
            return parse_query_simple(query)
    print("  LLM parse failed after key rotation, falling back to keyword parser")
    return parse_query_simple(query)


def sql_filter(conn, filters, video_id=None):
    """Cheap hard filters only: vehicle type / direction / time.
    Color and free-text are deliberately NOT filtered here — they're handled
    downstream against the richer VLM data, since the coarse HSV bucket color
    stored in this table is only a hint, not ground truth."""
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


GENERIC_NOISE_WORDS = {
    "vehicle", "vehicles", "colored", "color", "colour", "show", "find", "all",
    "with", "a", "an", "the", "any"
}

# Colors visually close enough that a query for one should also surface the other
COLOR_CLOSE_GROUPS = [
    {"yellow", "orange"},
    {"red", "orange"},
    {"blue", "purple/pink"},
    {"white", "silver/gray"},
    {"black", "silver/gray"},
]


def expand_close_colors(colors):
    if not colors:
        return colors
    expanded = {c.lower() for c in colors}
    for group in COLOR_CLOSE_GROUPS:
        if expanded & group:
            expanded |= group
    return list(expanded)


def matches_color(hsv_color, vlm_data, requested_colors):
    """Only checks the vehicle's DOMINANT/overall color — never visible_colors,
    which can contain incidental part colors (reflections, trim, etc.) that
    caused false positives (e.g. a gray car matching a 'blue' search because
    the VLM noted a bluish window reflection in visible_colors)."""
    if not requested_colors:
        return True
    requested = {c.lower() for c in requested_colors}
    if hsv_color and hsv_color.lower() in requested:
        return True
    if vlm_data:
        dom = str(vlm_data.get("dominant_color", "")).lower()
        if any(rc in dom for rc in requested):
            return True
    return False


def matches_free_text(vlm_data, free_text, vehicle_types=None, colors=None):
    """Keyword fallback — used only if the semantic reranker call fails."""
    if not free_text or not free_text.strip():
        return True

    exclude = set(GENERIC_NOISE_WORDS)
    exclude.update(v.lower() for v in (vehicle_types or []))
    exclude.update(c.lower() for c in (colors or []))

    keywords = [kw for kw in free_text.lower().split()
                if len(kw) > 2 and kw not in exclude]

    if not keywords:
        return True

    if not vlm_data:
        return False

    haystack = " ".join([
        " ".join(vlm_data.get("cargo", [])),
        " ".join(vlm_data.get("roof_items", [])),
        " ".join(vlm_data.get("front_view_details", [])),
        " ".join(vlm_data.get("rear_view_details", [])),
        " ".join(vlm_data.get("side_view_details", [])),
        " ".join(vlm_data.get("advertisement_or_text", [])),
        str(vlm_data.get("cargo_location", "")),
        " ".join(vlm_data.get("description", [])),
        " ".join(vlm_data.get("visible_colors", [])),
        str(vlm_data.get("dominant_color", "")),
        str(vlm_data.get("special_vehicle_type", "")),
    ]).lower()

    return any(kw in haystack for kw in keywords)


def semantic_rerank(query, candidates_with_vlm):
    """
    candidates_with_vlm: list of (track_id, video_id, vlm_data dict)
    Sends the raw query + structured VLM data for each candidate to the LLM
    in ONE batched call, and asks it which candidates genuinely match.
    This is what makes flexible/arbitrary phrasing work (e.g. "car carrying
    a truck", "van with a dent near the door") instead of relying on brittle
    keyword substring matching.
    Returns a set of (track_id, video_id) tuples that matched, or None if the
    call failed (caller should fall back to keyword matching in that case).
    """
    if not candidates_with_vlm:
        return set()

    items = []
    for idx, (tid, vid, vlm) in enumerate(candidates_with_vlm):
        vlm = vlm or {}
        items.append({
            "id": idx,
            "vehicle_type": vlm.get("vehicle_type"),
            "dominant_color": vlm.get("dominant_color"),
            "cargo": vlm.get("cargo", []),
            "cargo_location": vlm.get("cargo_location"),
            "roof_items": vlm.get("roof_items", []),
            "front_view_details": vlm.get("front_view_details", []),
            "rear_view_details": vlm.get("rear_view_details", []),
            "side_view_details": vlm.get("side_view_details", []),
            "advertisement_or_text": vlm.get("advertisement_or_text", []),
            "description": vlm.get("description", []),
        })

    prompt = RERANK_PROMPT_HEADER.format(query=query, candidates_json=json.dumps(items))

    for attempt in range(4):
        try:
            client = get_groq_client()
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            text = response.choices[0].message.content.strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            matched_ids = json.loads(text.strip())

            matched_set = set()
            for mid in matched_ids:
                if isinstance(mid, int) and 0 <= mid < len(candidates_with_vlm):
                    tid, vid, _ = candidates_with_vlm[mid]
                    matched_set.add((tid, vid))
            return matched_set
        except Exception as e:
            if "429" in str(e) or "rate_limit" in str(e):
                _rotate_key()
                continue
            print(f"  Semantic rerank failed ({e}), falling back to keyword match")
            return None
    print("  Semantic rerank failed after key rotation, falling back to keyword match")
    return None


def run_eager_vlm(video_id, progress_callback=None, max_workers=6):
    """
    Runs full VLM analysis on every not-yet-analyzed vehicle for a video,
    right after indexing (phase5), instead of lazily on first search.
    Also trusts the VLM's vehicle_type to correct the stored type for ANY
    class (not just truck/bus) — this replaces the old phase_verify step,
    which only checked truck/bus and silently did nothing if its single
    VLM call failed.

    Calls run concurrently (network-bound, not CPU-bound) while a shared
    rate limiter inside vlm_helper keeps the collective request rate under
    Groq's cap — this is what makes indexing scale sub-linearly with vehicle
    count instead of taking N * ~2.5s serially.

    All sqlite writes happen on the main thread — sqlite3 connections aren't
    safe to share across threads, so worker threads only do the (rate-limited)
    network call and hand results back via as_completed().

    progress_callback(done, total) is called after each vehicle finishes.
    """
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "SELECT track_id, crop_path FROM vehicles WHERE video_id=? AND vlm_analyzed=0",
        (video_id,)
    )
    rows = cur.fetchall()
    total = len(rows)
    analyzed = 0
    corrected = 0
    done = 0

    def analyze_one(track_id, crop_path):
        if crop_path and os.path.exists(crop_path):
            return track_id, analyze_crop(crop_path)
        return track_id, None

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(analyze_one, tid, cp) for tid, cp in rows]
        for future in as_completed(futures):
            track_id, result = future.result()
            done += 1

            if result:
                analyzed += 1
                vlm_type = result.get("vehicle_type")
                if vlm_type:
                    cur.execute(
                        "SELECT vehicle_type FROM vehicles WHERE track_id=? AND video_id=?",
                        (track_id, video_id)
                    )
                    row = cur.fetchone()
                    if row and row[0] != vlm_type:
                        corrected += 1
                    cur.execute(
                        "UPDATE vehicles SET vlm_analyzed=1, vlm_data=?, vehicle_type=? WHERE track_id=? AND video_id=?",
                        (json.dumps(result), vlm_type, track_id, video_id)
                    )
                else:
                    cur.execute(
                        "UPDATE vehicles SET vlm_analyzed=1, vlm_data=? WHERE track_id=? AND video_id=?",
                        (json.dumps(result), track_id, video_id)
                    )
                conn.commit()
            # else: leave vlm_analyzed=0 so a later manual re-analyze can retry it,
            # rather than silently marking a failed call as "done"

            if progress_callback:
                progress_callback(done, total)

    conn.close()
    return {"total": total, "analyzed": analyzed, "corrected": corrected}


def search(query, video_id=None, on_vlm_call=None):
    """
    Single-video search entry point (kept for CLI / single-video use).
    For multi-video scope, use search_multi() instead — it parses the query
    and reranks only ONCE across all videos, rather than once per video.
    """
    filters, matches = search_multi(query, video_ids=[video_id] if video_id else None, on_vlm_call=on_vlm_call)
    return filters, matches


def search_multi(query, video_ids=None, on_vlm_call=None):
    """
    Searches across one or more videos with the query parsed and reranked
    ONLY ONCE total — this is the main entry point app.py should use when
    the person has multiple videos selected in scope, since looping the old
    single-video search() per video re-ran both LLM calls (parse + rerank)
    once per video, which is most of the perceived search latency.

    video_ids: list of video_ids to search, or None/empty for all videos.
    Returns (filters_used, list_of_matches).
    """
    conn = sqlite3.connect(DB_PATH)
    filters = parse_query_llm(query)
    filters["colors"] = expand_close_colors(filters.get("colors"))

    scope = video_ids if video_ids else [None]
    color_passed = []
    for vid_scope in scope:
        candidates = sql_filter(conn, filters, video_id=vid_scope)
        for tid, vid, vtype, color, crop_path, ts, direction, vlm_analyzed, vlm_data_raw in candidates:
            vlm_data = json.loads(vlm_data_raw) if vlm_data_raw else {}

            if not vlm_analyzed:
                # fallback safety net — should rarely trigger if eager indexing ran
                if on_vlm_call:
                    on_vlm_call(tid, crop_path)
                result = analyze_crop(crop_path) if crop_path and os.path.exists(crop_path) else None
                vlm_data = result or {}
                cur = conn.cursor()
                cur.execute(
                    "UPDATE vehicles SET vlm_analyzed=1, vlm_data=? WHERE track_id=? AND video_id=?",
                    (json.dumps(vlm_data) if vlm_data else None, tid, vid)
                )
                conn.commit()

            if matches_color(color, vlm_data, filters.get("colors")):
                color_passed.append({
                    "track_id": tid,
                    "video_id": vid,
                    "vehicle_type": vtype,
                    "color": color,
                    "crop_path": crop_path,
                    "timestamp_seconds": ts,
                    "direction": direction,
                    "vlm_data": vlm_data
                })

    free_text = (filters.get("free_text") or "").strip()
    exclude = set(GENERIC_NOISE_WORDS)
    exclude.update(v.lower() for v in filters.get("vehicle_types", []))
    exclude.update(c.lower() for c in filters.get("colors", []))
    meaningful_keywords = [kw for kw in free_text.lower().split()
                           if len(kw) > 2 and kw not in exclude]

    if meaningful_keywords:
        candidates_with_vlm = [(m["track_id"], m["video_id"], m["vlm_data"]) for m in color_passed]
        matched_ids = semantic_rerank(free_text, candidates_with_vlm)
        if matched_ids is not None:
            final_matches = [m for m in color_passed if (m["track_id"], m["video_id"]) in matched_ids]
        else:
            final_matches = [m for m in color_passed
                              if matches_free_text(m["vlm_data"], free_text,
                                                    filters.get("vehicle_types"), filters.get("colors"))]
    else:
        final_matches = color_passed

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