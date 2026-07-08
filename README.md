# 🚦 TrafficLens (ATIP — Automated Traffic Intelligence Platform)

**Team KUET_Technomancers · SciBlitz AI Challenge 2026 · IEEE Student Branch, CUET**

🔴 **Live demo:** https://huggingface.co/spaces/turjodeb/atip

Turn raw traffic video into a **searchable vehicle database**. Upload footage → vehicles are detected, tracked, counted and described by AI → then search them in plain language: *"red car carrying furniture"*, *"bus with yellow back"*, *"truck with DHAKA written on it"*. Includes a police-style **suspect watchlist (BOLO)** that automatically scans every new video, an **emergency-vehicle panel**, analytics, exports and one-click evidence reports.

## ✨ Features
- **Detect · track · count** — YOLO11 + BoT-SORT, user-adjustable counting line, per-vehicle direction & timestamp
- **AI vehicle descriptions** — vision-language model extracts type, part-wise colors, cargo & location, roof items, damage, readable text/logos/plate fragments, emergency status
- **Natural-language search** — LLM query parsing + semantic re-ranking across one or many videos
- **🚔 Suspect Watchlist** — standing descriptions auto-checked against every newly processed video; alert cards with evidence crops
- **🚨 Emergency panel** — ambulance/police/fire detections with traffic-density context
- **Analytics & export** — traffic-over-time chart, direction split, CSV/JSON export, self-contained HTML report, crop downloads, saved library

## 🧠 Tech stack
| Layer | Tech |
|---|---|
| Detection | YOLO11n (Ultralytics, pretrained on COCO) |
| Tracking | BoT-SORT (Ultralytics) + class voting + IoU track recovery |
| Descriptions | Llama-4-Scout-17B (vision) via Groq API |
| Search LLM | Llama-3.3-70B via Groq API (parse + batched semantic re-rank) |
| Color hint | OpenCV HSV masking |
| Storage | SQLite |
| Frontend | Streamlit |
| Deploy | Docker Space on Hugging Face (CPU, free tier) |

## 🗂️ Pipeline
```
video ─► Phase 2: detect+track+count (crops saved)
      ─► Phase 3: HSV color hint
      ─► Phase 5: index into SQLite
      ─► Eager VLM: JSON description per vehicle (rate-limited, multi-key)
      ─► Phase 6: NL search (LLM parse → SQL prefilter → color match → LLM re-rank)
      ─► Watchlist auto-scan on every new video
```

## 🚀 Run locally
```bash
git clone https://github.com/turjo-deb/atip.git
cd atip
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# ffmpeg must be installed and on PATH (https://ffmpeg.org)

# API keys — create .env in the repo root:
echo 'GROQ_API_KEYS=gsk_key1,gsk_key2' > .env       # one key also works

streamlit run src/streamlit_app.py
```
Open http://localhost:8501, upload a traffic video (mp4/avi/mov), wait for processing, then search.

### Docker
```bash
docker build -t atip .
docker run -p 8501:8501 -e GROQ_API_KEYS=gsk_yourkey atip
```

## ⚙️ Configuration
- `GROQ_API_KEYS` — comma-separated Groq API keys (free at https://console.groq.com); the app rotates keys on rate limits
- Counting line position — slider in the sidebar (set before processing)

## 📁 Repo layout
```
src/
  streamlit_app.py   # UI: dashboard, search, watchlist, library
  phase1_detect.py   # (experiment) raw detection pass
  phase2_track.py    # tracking, counting line, crop saving
  phase3_color.py    # HSV dominant color
  phase5_db.py       # SQLite schema & indexing
  phase6_search.py   # NL search + eager VLM indexing + watchlist scan backend
  vlm_helper.py      # Groq vision calls, rate limiter, key rotation
Dockerfile
requirements.txt
```

## 🙏 Attribution (third-party models, data & libraries)
- **YOLO11 / BoT-SORT** — [Ultralytics](https://github.com/ultralytics/ultralytics), AGPL-3.0. Weights pretrained on **COCO** (annotations CC BY 4.0, https://cocodataset.org). No training/fine-tuning performed by us.
- **Llama-4-Scout-17B-16E-Instruct** & **Llama-3.3-70B-Versatile** — Meta, under their respective Llama Community Licenses, accessed via the [Groq API](https://groq.com).
- **Libraries:** PyTorch (BSD-3), OpenCV (Apache-2.0), Streamlit (Apache-2.0), Groq SDK (Apache-2.0), pandas (BSD-3), Plotly (MIT), NumPy (BSD-3), lapx (BSD-2), FFmpeg (LGPL/GPL), SQLite (public domain). Full list: `requirements.txt`.
- Test clips: publicly available traffic-camera footage, used for inference only.

See `ATIP_Model_Data_Card.pdf` for limitations and ethical considerations.

## ⚠️ Notes & limitations
- Plate/text transcription is assistive, **not evidence-grade** — always verify with the crop.
- Free-tier Space storage is ephemeral: DB and crops reset on restart.
- The tool describes **vehicles, not people** — no face detection/recognition.

---
*TrafficLens v1.2 · See every vehicle, find any vehicle.*
