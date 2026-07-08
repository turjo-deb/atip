---
title: ATIP - Automated Traffic Intelligence Platform
emoji: 🚦
colorFrom: teal
colorTo: indigo
sdk: streamlit
sdk_version: "1.38.0"
app_file: app.py
pinned: false
---

# ATIP — Automated Traffic Intelligence Platform

Team **KUET_Technomancers** — built for **SciiBlitz 2.0**

## Migrating from Streamlit Cloud to Hugging Face Spaces

1. Create a new Space at huggingface.co/new-space, SDK = **Streamlit**.
2. Push this folder's contents to the Space's git repo root (this `README.md`'s
   YAML front matter above is what HF reads for Space config — keep it at the
   very top of the file).
3. `requirements.txt` and `packages.txt` (for the `ffmpeg` apt dependency) work
   exactly the same way as on Streamlit Cloud — no changes needed there.
4. Set `GROQ_API_KEY` under the Space's **Settings → Repository secrets**
   (equivalent of Streamlit Cloud's `st.secrets` / env vars). `python-dotenv`'s
   `load_dotenv()` calls in `vlm_helper.py` / `phase6_search.py` are no-ops if
   there's no `.env` file, so the env var set via HF secrets is picked up fine.
5. Free tier CPU Spaces give **16GB RAM** vs Streamlit Cloud's 1GB — this
   matters once you add the helmet-detection and motorcycle-overload-detection
   models, since those will run alongside YOLO11n + torch + the tracker.
6. Persistent storage: Spaces' local filesystem resets on rebuild/restart just
   like Streamlit Cloud's does — `outputs/`, `videos/`, and `atip.db` are not
   durable across redeploys on the free tier either way. If you want the DB and
   crops to survive restarts, that needs a Space with persistent storage
   (paid) or an external store (e.g. a small S3-compatible bucket / a hosted
   SQLite replacement) — flag if you want this wired in before your next demo.
