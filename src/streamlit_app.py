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
_k = os.getenv("GROQ_API_KEYS")
st.info(f"DEBUG: GROQ_API_KEYS is {'SET (' + str(len(_k.split(','))) + ' keys)' if _k else 'MISSING'}")

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

for label in ["📊 Dashboard", "🔍 Analyze", "💾 Library"]:
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

        st.markdown('<div class="search-box">', unsafe_allow_html=True)
        query = st.text_input(
            "Search",
            placeholder="e.g. red car, bus with yellow back, truck carrying ladder",
            label_visibility="collapsed"
        )
        st.markdown('</div>', unsafe_allow_html=True)

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

                        if guarded(st.button("💾 Save", key=f"save_{m['video_id']}_{m['track_id']}")):
                            toggle_saved(m["video_id"], m["track_id"], 1)
                            st.toast("Saved to Library!")
            else:
                st.warning("No matches found. Try a different query.")
        elif query and not search_scope:
            st.warning("Select at least one video to search in.")

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

                if guarded(st.button("🗑️ Remove", key=f"del_{video_id}_{track_id}")):
                    toggle_saved(video_id, track_id, 0)
                    st.rerun()

    render_footer()