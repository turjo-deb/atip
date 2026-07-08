# src/app.py

import streamlit as st
import sqlite3
import os
import sys
import json
import shutil
import pandas as pd
import plotly.express as px


sys.path.insert(0, os.path.dirname(__file__))

from phase2_track import run_phase2
from phase3_color import run_phase3
from phase5_db import run_phase5, init_db, DB_PATH
from phase6_search import search_multi, run_eager_vlm

st.set_page_config(page_title="TrafficLens", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .main { background-color: #0b0f19; }
    section[data-testid="stSidebar"] { background-color: #10141f; border-right: 1px solid #232838; }
    .tl-sidebar-name {
        font-size: 1.3rem; font-weight: 800; color: #2dd4bf;
        text-align: center; margin: 8px 0 2px 0;
    }
    .tl-tag { color: #8b93a7; font-size: 0.78rem; text-align: center; margin: 4px 0 14px 0; }

    /* Nav buttons */
    section[data-testid="stSidebar"] .stButton button {
        text-align: left; font-size: 1rem; font-weight: 600;
        padding: 12px 16px; margin-bottom: 6px; border-radius: 8px; width: 100%;
    }
    section[data-testid="stSidebar"] .stButton button[kind="primary"] {
        background-color: #2dd4bf; color: #0b0f19; border: none;
    }
    section[data-testid="stSidebar"] .stButton button[kind="secondary"] {
        background-color: #1a1f30; color: #e5e9f0; border: 1px solid #2a3145;
    }

    .nav-divider { border-top: 1px solid #232a3d; margin: 14px 0; }

    /* Page title */
    .page-title { font-size: 1.7rem; font-weight: 800; color: #e5e9f0; margin-bottom: 2px; }
    .page-line { border-top: 1px solid #232a3d; margin: 8px 0 20px 0; }

    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #151a28 0%, #1c2333 100%);
        border: 1px solid #2a3145; padding: 18px; border-radius: 14px;
    }
    div[data-testid="stMetricValue"] { font-size: 1.8rem; font-weight: 700; color: #2dd4bf; }
    div[data-testid="stMetricLabel"] { font-weight: 600; color: #8b93a7; }

    div[data-testid="column"] {
        background: #131826; border-radius: 14px; padding: 14px;
        border: 1px solid #232a3d; margin-bottom: 10px;
    }

    .main .stButton button {
        border-radius: 8px; font-weight: 600;
        background-color: #2dd4bf; color: #0b0f19; border: none; padding: 8px 20px;
    }
    .main .stButton button:hover { background-color: #14b8a6; color: white; }

    .pill {
        display: inline-block; background: #1c2333; color: #2dd4bf;
        padding: 2px 10px; border-radius: 20px; font-size: 0.75rem; margin: 2px;
        border: 1px solid #2a3145;
    }
    img { border-radius: 10px; }

    .search-box input { font-size: 1.1rem !important; padding: 14px !important; }

    /* Sticky footer */
    .tl-footer {
        position: fixed; left: 0; bottom: 0; width: 100%;
        background: #10141f; border-top: 1px solid #232a3d;
        text-align: center; color: #8b93a7; font-size: 0.8rem;
        padding: 10px 0; line-height: 1.5; z-index: 999;
    }
    .tl-footer b { color: #2dd4bf; }
    .main .block-container { padding-bottom: 90px; padding-top: 1.5rem; }
</style>
""", unsafe_allow_html=True)

VIDEOS_DIR = "videos"
os.makedirs(VIDEOS_DIR, exist_ok=True)
init_db()

if "is_processing" not in st.session_state:
    st.session_state.is_processing = False


# ============================================================
# DB helpers
# ============================================================
def get_processed_videos():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT DISTINCT video_id FROM vehicles")
    result = [row[0] for row in cur.fetchall()]
    conn.close()
    return result


def get_video_stats(video_id):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT vehicle_type, COUNT(*) FROM vehicles WHERE video_id=? GROUP BY vehicle_type", (video_id,))
    stats = dict(cur.fetchall())
    conn.close()
    return stats


def toggle_saved(video_id, track_id, value):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("UPDATE vehicles SET saved=? WHERE video_id=? AND track_id=?", (value, video_id, track_id))
    conn.commit()
    conn.close()


def get_saved_items():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        SELECT track_id, video_id, vehicle_type, color, crop_path, timestamp_seconds, vlm_data
        FROM vehicles WHERE saved=1
    """)
    rows = cur.fetchall()
    conn.close()
    return rows

def reset_vlm_cache(video_id):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("UPDATE vehicles SET vlm_analyzed=0, vlm_data=NULL WHERE video_id=?", (video_id,))
    conn.commit()
    conn.close()


def delete_video(video_id):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("DELETE FROM vehicles WHERE video_id = ?", (video_id,))
    conn.commit()
    conn.close()

    output_dir = f"outputs/{video_id}"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    for ext in ["mp4", "avi", "mov"]:
        video_file = f"videos/{video_id}.{ext}"
        if os.path.exists(video_file):
            os.remove(video_file)


def get_video_rows(video_id):
    """All vehicle rows for a video, flattened with key VLM fields — used by export & report."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        SELECT track_id, vehicle_type, color, timestamp_seconds, direction, crop_path, vlm_data
        FROM vehicles WHERE video_id=? ORDER BY timestamp_seconds
    """, (video_id,))
    rows = cur.fetchall()
    conn.close()

    records = []
    for track_id, vtype, color, ts, direction, crop_path, vlm_raw in rows:
        vd = json.loads(vlm_raw) if vlm_raw else {}
        records.append({
            "track_id": track_id,
            "vehicle_type": vtype,
            "color": color,
            "timestamp_seconds": round(ts or 0, 2),
            "direction": direction or "",
            "dominant_color_ai": vd.get("dominant_color", ""),
            "cargo": "; ".join(vd.get("cargo", [])),
            "cargo_location": vd.get("cargo_location", ""),
            "roof_items": "; ".join(vd.get("roof_items", [])),
            "text_or_ads": "; ".join(vd.get("advertisement_or_text", [])),
            "special_vehicle": vd.get("special_vehicle_type", ""),
            "description": "; ".join(vd.get("description", [])),
            "crop_path": crop_path or "",
        })
    return records


def build_html_report(video_id, records):
    """Self-contained HTML report (charts as inline plotly PNGs skipped — pure HTML/CSS bars, prints fine)."""
    import base64 as _b64
    from datetime import datetime

    total = len(records)
    by_type, by_dir, by_color = {}, {"up": 0, "down": 0}, {}
    for r in records:
        by_type[r["vehicle_type"]] = by_type.get(r["vehicle_type"], 0) + 1
        if r["direction"] in by_dir:
            by_dir[r["direction"]] += 1
        by_color[r["color"]] = by_color.get(r["color"], 0) + 1

    def bar_rows(d):
        mx = max(d.values()) if d else 1
        return "".join(
            f"<tr><td>{k}</td><td><div class='bar' style='width:{int(280*v/mx)}px'></div> {v}</td></tr>"
            for k, v in sorted(d.items(), key=lambda x: -x[1])
        )

    # timeline buckets (10s)
    buckets = {}
    for r in records:
        b = int(r["timestamp_seconds"] // 10) * 10
        buckets[b] = buckets.get(b, 0) + 1
    timeline_rows = "".join(
        f"<tr><td>{b}–{b+10}s</td><td><div class='bar' style='width:{int(280*v/max(buckets.values()))}px'></div> {v}</td></tr>"
        for b, v in sorted(buckets.items())
    ) if buckets else ""

    # notable vehicles: has cargo, text, or special — top 12 with embedded crops
    notable = [r for r in records if r["cargo"] or r["text_or_ads"] or (r["special_vehicle"] and r["special_vehicle"] != "none")][:12]
    cards = ""
    for r in notable:
        img_tag = ""
        if r["crop_path"] and os.path.exists(r["crop_path"]):
            try:
                with open(r["crop_path"], "rb") as f:
                    img_tag = f"<img src='data:image/jpeg;base64,{_b64.b64encode(f.read()).decode()}'/>"
            except Exception:
                pass
        details = " · ".join(x for x in [
            r["cargo"] and f"Carrying: {r['cargo']}",
            r["text_or_ads"] and f"Text: {r['text_or_ads']}",
            r["special_vehicle"] not in ("", "none") and f"⚠ {r['special_vehicle']}",
        ] if x)
        cards += f"""<div class='card'>{img_tag}<div><b>{r['vehicle_type'].capitalize()}</b> · {r['color']} · {r['timestamp_seconds']}s<br><small>{details}</small></div></div>"""

    html = f"""<!DOCTYPE html><html><head><meta charset='utf-8'><title>ATIP Report — {video_id}</title>
<style>
body{{font-family:Arial,sans-serif;max-width:900px;margin:30px auto;color:#1a1a2e;padding:0 16px}}
h1{{color:#0d9488}} h2{{border-bottom:2px solid #0d9488;padding-bottom:4px;margin-top:32px}}
table{{border-collapse:collapse}} td{{padding:4px 12px 4px 0;vertical-align:middle}}
.bar{{display:inline-block;height:14px;background:#0d9488;border-radius:3px;vertical-align:middle}}
.metrics{{display:flex;gap:24px;flex-wrap:wrap;margin:16px 0}}
.metric{{background:#f0fdfa;border:1px solid #99f6e4;border-radius:10px;padding:14px 22px;text-align:center}}
.metric b{{font-size:1.6rem;color:#0d9488;display:block}}
.card{{display:flex;gap:12px;align-items:center;border:1px solid #ddd;border-radius:10px;padding:10px;margin:8px 0}}
.card img{{width:110px;border-radius:8px}}
footer{{margin-top:40px;color:#888;font-size:0.85rem;text-align:center}}
</style></head><body>
<h1>🚦 TrafficLens Report</h1>
<p><b>Video:</b> {video_id} &nbsp;·&nbsp; <b>Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
<div class='metrics'>
  <div class='metric'><b>{total}</b>Total vehicles</div>
  <div class='metric'><b>{by_dir['up']}</b>⬆ Up</div>
  <div class='metric'><b>{by_dir['down']}</b>⬇ Down</div>
</div>
<h2>Vehicles by type</h2><table>{bar_rows(by_type)}</table>
<h2>Vehicles by color</h2><table>{bar_rows(by_color)}</table>
<h2>Traffic over time</h2><table>{timeline_rows}</table>
<h2>Notable vehicles</h2>{cards if cards else "<p>None flagged.</p>"}
<footer>Generated by TrafficLens · Team KUET_Technomancers · SciiBlitz 2.0</footer>
</body></html>"""
    return html



# ============================================================
# Watchlist helpers
# ============================================================
def add_watchlist_entry(description):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("INSERT INTO watchlist (description) VALUES (?)", (description,))
    conn.commit()
    conn.close()


def get_watchlist():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT id, description, created_at FROM watchlist WHERE active=1 ORDER BY id DESC")
    rows = cur.fetchall()
    conn.close()
    return rows


def remove_watchlist_entry(entry_id):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("DELETE FROM watchlist WHERE id=?", (entry_id,))
    cur.execute("DELETE FROM watchlist_hits WHERE entry_id=?", (entry_id,))
    conn.commit()
    conn.close()


def record_watchlist_hits(entry_id, matches):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    for m in matches:
        cur.execute(
            "INSERT OR IGNORE INTO watchlist_hits (entry_id, video_id, track_id) VALUES (?, ?, ?)",
            (entry_id, m["video_id"], str(m["track_id"]))
        )
    conn.commit()
    conn.close()


def get_watchlist_hits(entry_id):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        SELECT v.track_id, v.video_id, v.vehicle_type, v.color, v.crop_path, v.timestamp_seconds
        FROM watchlist_hits h
        JOIN vehicles v ON v.video_id = h.video_id AND v.track_id = h.track_id
        WHERE h.entry_id = ?
        ORDER BY v.video_id, v.timestamp_seconds
    """, (entry_id,))
    rows = cur.fetchall()
    conn.close()
    return rows


def scan_watchlist(video_ids=None):
    """Runs every active watchlist entry through search. Returns list of (description, n_matches)."""
    results = []
    for entry_id, description, _created in get_watchlist():
        try:
            _f, matches = search_multi(description, video_ids=video_ids)
            if matches:
                record_watchlist_hits(entry_id, matches)
            results.append((description, len(matches)))
        except Exception as e:
            results.append((description, f"error: {e}"))
    return results


def guarded(clicked):
    """Wrap any button click — if processing, show a toast and ignore the click."""
    if clicked and st.session_state.is_processing:
        st.toast("⏳ Video is processing — please wait until it finishes.")
        return False
    return clicked
            
def page_header(title, caption=None):
    st.markdown(f'<div class="page-title">{title}</div>', unsafe_allow_html=True)
    if caption:
        st.caption(caption)
    st.markdown('<div class="page-line"></div>', unsafe_allow_html=True)


def render_footer():
    st.markdown("""
    <div class="tl-footer">
        <b>TrafficLens</b> v1.1 · See every vehicle, find any vehicle. ·
        Team <b>KUET_Technomancers</b> · Built for <b>SciiBlitz 2.0</b> ·
        Powered by YOLO11 · BoT-SORT · Groq · Streamlit
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# Sidebar — logo, nav, upload
# ============================================================
st.sidebar.markdown('<div class="tl-sidebar-name">TrafficLens</div>', unsafe_allow_html=True)
st.sidebar.markdown('<div class="tl-tag">AI Traffic Intelligence Platform</div>', unsafe_allow_html=True)

if "page" not in st.session_state:
    st.session_state.page = "📊 Dashboard"

for label in ["📊 Dashboard", "🔍 Analyze", "🚔 Watchlist", "💾 Library"]:
    is_active = st.session_state.page == label
    if guarded(st.sidebar.button(label, key=f"nav_{label}", use_container_width=True,
                              type="primary" if is_active else "secondary")):
        st.session_state.page = label

page = st.session_state.page

st.sidebar.markdown('<div class="nav-divider"></div>', unsafe_allow_html=True)
st.sidebar.subheader("📤 Upload video")
uploaded_file = st.sidebar.file_uploader("Drop a video file", type=["mp4", "avi", "mov"], label_visibility="collapsed")

st.sidebar.markdown("**📏 Counting line position**")
line_pos = st.sidebar.slider(
    "Where vehicles are counted",
    min_value=0.3, max_value=0.9, value=0.5, step=0.05,
    help="Vehicles are counted when they cross this horizontal line. 0.5 = middle of the frame. Set it BEFORE clicking Process."
)

if uploaded_file is not None:
    save_path = os.path.join(VIDEOS_DIR, uploaded_file.name)
    if st.sidebar.button("🚀 Process video", use_container_width=True, disabled=st.session_state.is_processing):
        st.session_state.is_processing = True
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        progress_bar = st.sidebar.progress(0, text="Detecting & tracking...")

        def progress_callback(frame_count, total_frames, confirmed):
            pct = min(frame_count / max(total_frames, 1), 1.0)
            progress_bar.progress(pct, text=f"Frame {frame_count}/{total_frames} · {confirmed} found")

        try:
            summary2 = run_phase2(save_path, progress_callback=progress_callback, line_position=line_pos)
            video_id = os.path.splitext(os.path.basename(save_path))[0]
            with st.sidebar:
                with st.spinner("Analyzing colors..."):
                    run_phase3(save_path)
                with st.spinner("Indexing..."):
                    run_phase5(save_path)

                vlm_progress = st.progress(0, text="Analyzing vehicles with AI...")

                def vlm_progress_cb(done, total):
                    pct = min(done / max(total, 1), 1.0)
                    vlm_progress.progress(pct, text=f"Analyzing vehicles with AI... {done}/{total}")

                vlm_summary = run_eager_vlm(video_id, progress_callback=vlm_progress_cb)
                if vlm_summary["corrected"] > 0:
                    st.info(f"AI corrected {vlm_summary['corrected']} vehicle classifications")

                if get_watchlist():
                    with st.spinner("Checking watchlist..."):
                        wl_results = scan_watchlist(video_ids=[video_id])
                    for desc, n in wl_results:
                        if isinstance(n, int) and n > 0:
                            st.warning(f"🚨 Watchlist hit: '{desc}' — {n} match(es) in {video_id}")

            st.session_state.is_processing = False
            st.sidebar.success(f"✅ {summary2['confirmed']} vehicles indexed")
            st.rerun()
        except Exception as e:
            st.session_state.is_processing = False
            st.sidebar.error(f"❌ Processing failed: {e}")
            st.exception(e)

st.sidebar.markdown('<div class="nav-divider"></div>', unsafe_allow_html=True)
st.sidebar.subheader("🗑️ Manage videos")

_processed_for_delete = get_processed_videos()
if _processed_for_delete:
    for vid in _processed_for_delete:
        col1, col2, col3 = st.sidebar.columns([3, 1, 1])
        col1.write(vid)
        if guarded(col2.button("🔄", key=f"reanalyze_{vid}", help="Re-analyze with latest AI prompt")):
            reset_vlm_cache(vid)
            reanalyze_progress = st.sidebar.progress(0, text="Re-analyzing...")

            def reanalyze_cb(done, total):
                reanalyze_progress.progress(min(done / max(total, 1), 1.0), text=f"Re-analyzing... {done}/{total}")

            reanalyze_summary = run_eager_vlm(vid, progress_callback=reanalyze_cb)
            st.sidebar.success(
                f"Re-analyzed {reanalyze_summary['analyzed']}/{reanalyze_summary['total']} "
                f"({reanalyze_summary['corrected']} corrected)"
            )
            st.rerun()
        if guarded(col3.button("🗑️", key=f"delvid_{vid}")):
            delete_video(vid)
            st.sidebar.success(f"Deleted {vid}")
            st.rerun()
else:
    st.sidebar.caption("No videos to manage yet.")

processed = get_processed_videos()


# ============================================================
# PAGE: Dashboard
# ============================================================
if page == "📊 Dashboard":
    page_header("📊 Dashboard", "Overview of a processed video.")

    if not processed:
        st.info("No processed videos yet. Upload one from the sidebar.")
    else:
        selected_video = st.selectbox("Choose video", processed)
        stats = get_video_stats(selected_video)
        total = sum(stats.values())

        cols = st.columns(len(stats) + 1)
        cols[0].metric("Total", total)
        for i, (vtype, count) in enumerate(stats.items()):
            cols[i + 1].metric(vtype.capitalize(), count)

        df = pd.DataFrame(list(stats.items()), columns=["Vehicle Type", "Count"])
        fig = px.bar(df, x="Vehicle Type", y="Count", color="Vehicle Type",
                     color_discrete_sequence=["#2dd4bf", "#818cf8", "#f472b6", "#fbbf24"])
        fig.update_layout(showlegend=False, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                          font_color="#e5e9f0", height=300)
        st.plotly_chart(fig, use_container_width=True)

        records = get_video_rows(selected_video)

        # Direction split
        ups = sum(1 for r in records if r["direction"] == "up")
        downs = sum(1 for r in records if r["direction"] == "down")
        dcol1, dcol2 = st.columns(2)
        dcol1.metric("⬆️ Going up", ups)
        dcol2.metric("⬇️ Going down", downs)

        # Emergency vehicles panel
        emergencies = [r for r in records if r["special_vehicle"] not in ("", "none", None)]
        if emergencies:
            st.error(f"🚨 {len(emergencies)} emergency vehicle(s) detected in this video")
            ecols = st.columns(min(len(emergencies), 4))
            bucket_counts = {}
            for r in records:
                b = int(r["timestamp_seconds"] // 10) * 10
                bucket_counts[b] = bucket_counts.get(b, 0) + 1
            for ei, r in enumerate(emergencies):
                with ecols[ei % len(ecols)]:
                    if r["crop_path"] and os.path.exists(r["crop_path"]):
                        st.image(r["crop_path"], use_container_width=True)
                    b = int(r["timestamp_seconds"] // 10) * 10
                    n_around = bucket_counts.get(b, 1)
                    density = "heavy" if n_around >= 8 else ("moderate" if n_around >= 4 else "light")
                    st.markdown(f"**🚨 {r['special_vehicle'].capitalize()}** · ⏱️ {r['timestamp_seconds']}s")
                    st.caption(f"Passed during {density} traffic ({n_around} vehicles in that 10s window)")
        else:
            st.caption("✅ No emergency vehicles detected in this video.")

        # Traffic timeline (10s buckets)
        if records:
            tdf = pd.DataFrame(records)
            tdf["time_bucket"] = (tdf["timestamp_seconds"] // 10 * 10).astype(int)
            timeline = tdf.groupby(["time_bucket", "vehicle_type"]).size().reset_index(name="count")
            timeline["time"] = timeline["time_bucket"].astype(str) + "s"
            fig2 = px.bar(timeline, x="time", y="count", color="vehicle_type",
                          title="Traffic over time (per 10s)",
                          color_discrete_sequence=["#2dd4bf", "#818cf8", "#f472b6", "#fbbf24"])
            fig2.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                               font_color="#e5e9f0", height=300, legend_title_text="")
            st.plotly_chart(fig2, use_container_width=True)

        # Exports + report
        ecol1, ecol2, ecol3 = st.columns(3)
        if records:
            edf = pd.DataFrame(records).drop(columns=["crop_path"])
            ecol1.download_button("⬇️ Export CSV", edf.to_csv(index=False).encode(),
                                  file_name=f"atip_{selected_video}.csv", mime="text/csv",
                                  use_container_width=True)
            ecol2.download_button("⬇️ Export JSON", json.dumps(records, indent=2).encode(),
                                  file_name=f"atip_{selected_video}.json", mime="application/json",
                                  use_container_width=True)
            report_html = build_html_report(selected_video, records)
            ecol3.download_button("📄 Download report", report_html.encode(),
                                  file_name=f"atip_report_{selected_video}.html", mime="text/html",
                                  use_container_width=True)

        video_path = f"outputs/{selected_video}/phase2_test.mp4"
        if os.path.exists(video_path):
            st.video(video_path)

    render_footer()


# ============================================================
# PAGE: Analyze (search)
# ============================================================
elif page == "🔍 Analyze":
    page_header("🔍 Search Vehicles", "Describe what you're looking for in plain language.")

    if not processed:
        st.info("No processed videos yet. Upload one from the sidebar to get started.")
    else:
        st.write("**Search in:**")
        scope_cols = st.columns(min(len(processed), 6))
        search_scope = []
        for i, vid in enumerate(processed):
            with scope_cols[i % len(scope_cols)]:
                if st.checkbox(vid, value=True, key=f"scope_{vid}"):
                    search_scope.append(vid)

        if "recent_searches" not in st.session_state:
            st.session_state.recent_searches = []
        if "search_prefill" not in st.session_state:
            st.session_state.search_prefill = ""

        st.markdown('<div class="search-box">', unsafe_allow_html=True)
        query = st.text_input(
            "Search",
            value=st.session_state.search_prefill,
            placeholder="e.g. red car, bus with yellow back, truck carrying ladder",
            label_visibility="collapsed"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        st.session_state.search_prefill = ""

        if st.session_state.recent_searches:
            st.caption("🕘 Recent searches:")
            _rs = st.session_state.recent_searches
            for row_start in range(0, len(_rs), 5):
                row = _rs[row_start:row_start + 5]
                rcols = st.columns(5)
                for ri, rq in enumerate(row):
                    if rcols[ri].button(rq, key=f"recent_{row_start + ri}", use_container_width=True):
                        st.session_state.search_prefill = rq
                        st.rerun()

        FUN_FACTS = [
            "🤖 AI is peeking at the pixels...",
            "🔍 Cross-referencing vehicle DNA...",
            "🧠 Appreciating fine automotive detail...",
            "👀 Zooming in like a traffic cop...",
        ]

        if query and search_scope:
            vlm_call_count = {"n": 0}
            status_placeholder = st.empty()

            def on_vlm_call(track_id, crop_path):
                vlm_call_count["n"] += 1
                fact = FUN_FACTS[vlm_call_count["n"] % len(FUN_FACTS)]
                status_placeholder.info(f"{fact} (vehicle #{track_id})")

            q_clean = query.strip()
            if q_clean and (not st.session_state.recent_searches or st.session_state.recent_searches[0] != q_clean):
                if q_clean in st.session_state.recent_searches:
                    st.session_state.recent_searches.remove(q_clean)
                st.session_state.recent_searches.insert(0, q_clean)
                st.session_state.recent_searches = st.session_state.recent_searches[:10]

            all_matches = []
            with st.spinner("Searching..."):
                filters, all_matches = search_multi(query, video_ids=search_scope, on_vlm_call=on_vlm_call)
            status_placeholder.empty()

            st.write(f"**{len(all_matches)} result(s)** across {len(search_scope)} video(s)")

            if all_matches:
                cols = st.columns(3)
                for i, m in enumerate(all_matches):
                    with cols[i % 3]:
                        if m["crop_path"] and os.path.exists(m["crop_path"]):
                            st.image(m["crop_path"], use_container_width=True)
                        st.markdown(f"**{m['vehicle_type'].capitalize()}** · {m['color']} · ⏱️ {m['timestamp_seconds']:.1f}s")
                        st.caption(f"📹 {m['video_id']}")

                        vd = m["vlm_data"] or {}
                        tags = vd.get("description", [])
                        cargo = vd.get("cargo", [])
                        roof = vd.get("roof_items", [])
                        ads = vd.get("advertisement_or_text", [])
                        pills = "".join(f'<span class="pill">{t}</span>' for t in (tags + cargo + roof + ads))
                        if pills:
                            st.markdown(pills, unsafe_allow_html=True)
                        if cargo:
                            loc = vd.get("cargo_location")
                            loc_txt = f" ({loc.replace('_', ' ')})" if loc and loc != "none" else ""
                            st.caption(f"🚛 Carrying: {', '.join(cargo)}{loc_txt}")
                        if ads:
                            st.caption(f"📝 Text/logo: {', '.join(ads)}")

                        bcol1, bcol2 = st.columns(2)
                        if guarded(bcol1.button("💾 Save", key=f"save_{m['video_id']}_{m['track_id']}", use_container_width=True)):
                            toggle_saved(m["video_id"], m["track_id"], 1)
                            st.toast("Saved to Library!")
                        if m["crop_path"] and os.path.exists(m["crop_path"]):
                            with open(m["crop_path"], "rb") as _f:
                                bcol2.download_button("⬇️", _f.read(),
                                    file_name=os.path.basename(m["crop_path"]), mime="image/jpeg",
                                    key=f"dl_{m['video_id']}_{m['track_id']}", use_container_width=True)
            else:
                st.warning("No matches found. Try a different query.")
        elif query and not search_scope:
            st.warning("Select at least one video to search in.")

    render_footer()


# ============================================================
# PAGE: Watchlist
# ============================================================
elif page == "🚔 Watchlist":
    page_header("🚔 Suspect Watchlist", "Standing BOLO descriptions — new videos are checked automatically.")

    if "wl_prefill" not in st.session_state:
        st.session_state.wl_prefill = ""

    st.caption("💡 Examples (click to use):")
    EXAMPLES = ["white pickup with dented door", "red car carrying furniture",
                "truck with company logo", "motorcycle with two riders"]
    excols = st.columns(len(EXAMPLES))
    for xi, ex in enumerate(EXAMPLES):
        if excols[xi].button(ex, key=f"wlex_{xi}", use_container_width=True):
            st.session_state.wl_prefill = ex
            st.rerun()

    wcol1, wcol2 = st.columns([4, 1])
    new_desc = wcol1.text_input("Describe the vehicle to watch for",
                                value=st.session_state.wl_prefill,
                                placeholder="e.g. white pickup truck with a dented door and ladder rack",
                                label_visibility="collapsed")
    st.session_state.wl_prefill = ""
    if guarded(wcol2.button("➕ Add", use_container_width=True)) and new_desc.strip():
        add_watchlist_entry(new_desc.strip())
        st.toast("Added to watchlist")
        st.rerun()

    entries = get_watchlist()

    if entries and processed:
        if guarded(st.button("🔍 Scan all videos now")):
            with st.spinner("Scanning all videos against watchlist..."):
                wl_results = scan_watchlist(video_ids=processed)
            for desc, n in wl_results:
                if isinstance(n, int):
                    st.write(f"• '{desc}' → {n} match(es)")
                else:
                    st.write(f"• '{desc}' → {n}")
            st.rerun()

    st.markdown('<div class="page-line"></div>', unsafe_allow_html=True)

    if not entries:
        st.info("No watchlist entries yet. Add a description above — every newly processed video will be checked against it automatically.")
    else:
        for entry_id, description, created_at in entries:
            hcol1, hcol2 = st.columns([5, 1])
            hcol1.markdown(f"**🎯 {description}**")
            if guarded(hcol2.button("🗑️", key=f"wldel_{entry_id}")):
                remove_watchlist_entry(entry_id)
                st.rerun()

            hits = get_watchlist_hits(entry_id)
            if hits:
                st.markdown(f"🚨 **{len(hits)} match(es):**")
                hcols = st.columns(min(len(hits), 4))
                for hi, (track_id, video_id, vtype, color, crop_path, ts) in enumerate(hits):
                    with hcols[hi % len(hcols)]:
                        if crop_path and os.path.exists(crop_path):
                            st.image(crop_path, use_container_width=True)
                        st.markdown(f"**{(vtype or 'vehicle').capitalize()}** · {color} · ⏱️ {ts:.1f}s")
                        st.caption(f"📹 {video_id}")
            else:
                st.caption("No matches yet.")
            st.markdown('<div class="page-line"></div>', unsafe_allow_html=True)

    render_footer()


# ============================================================
# PAGE: Library
# ============================================================
else:
    page_header("💾 Library", "Saved vehicles from your searches.")

    saved = get_saved_items()

    if not saved:
        st.info("Nothing saved yet. Go to Analyze → search → hit 💾 Save on results you want to keep.")
    else:
        cols = st.columns(3)
        for i, (track_id, video_id, vtype, color, crop_path, ts, vlm_data_raw) in enumerate(saved):
            with cols[i % 3]:
                if crop_path and os.path.exists(crop_path):
                    st.image(crop_path, use_container_width=True)
                st.markdown(f"**{vtype.capitalize()}** · {color} · ⏱️ {ts:.1f}s")
                st.caption(f"Source: {video_id}")

                if vlm_data_raw:
                    vlm_data = json.loads(vlm_data_raw)
                    cargo = vlm_data.get("cargo", [])
                    roof = vlm_data.get("roof_items", [])
                    ads = vlm_data.get("advertisement_or_text", [])
                    tags = vlm_data.get("description", []) + cargo + roof + ads
                    pills = "".join(f'<span class="pill">{t}</span>' for t in tags)
                    if pills:
                        st.markdown(pills, unsafe_allow_html=True)
                    if cargo:
                        loc = vlm_data.get("cargo_location")
                        loc_txt = f" ({loc.replace('_', ' ')})" if loc and loc != "none" else ""
                        st.caption(f"🚛 Carrying: {', '.join(cargo)}{loc_txt}")

                lcol1, lcol2 = st.columns(2)
                if guarded(lcol1.button("🗑️ Remove", key=f"del_{video_id}_{track_id}", use_container_width=True)):
                    toggle_saved(video_id, track_id, 0)
                    st.rerun()
                if crop_path and os.path.exists(crop_path):
                    with open(crop_path, "rb") as _f:
                        lcol2.download_button("⬇️", _f.read(),
                            file_name=os.path.basename(crop_path), mime="image/jpeg",
                            key=f"libdl_{video_id}_{track_id}", use_container_width=True)

    render_footer()