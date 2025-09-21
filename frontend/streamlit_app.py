# frontend/streamlit_app.py
import os, io, sys, zipfile, base64
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# --- make sure we can import backend.omr_core no matter where we run from ---
ROOT = Path(__file__).resolve().parents[1]   # project root (folder that contains "backend")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.omr_core import grade_from_bytes, SUBJECTS  # uses your OpenCV pipeline

st.set_page_config(page_title="OMR Grader", layout="wide")
st.markdown("""
<style>
/* subtle polish */
.block-container {padding-top: 1rem; padding-bottom: 2rem;}
.stMetric {background: #0e1117; border-radius: 12px; padding: 10px;}
</style>
""", unsafe_allow_html=True)

# --------------------------- Sidebar ---------------------------
with st.sidebar:
    st.title("⚙️ Settings")
    st.caption("Upload answer key once, then grade any number of sheets.")
    key_file = st.file_uploader("Answer Key (.xlsx / .csv)", type=["xlsx", "xls", "csv"])
    out_dir = st.text_input("Save overlays to folder", value="overlays")
    if st.button("Clear Session"):
        st.experimental_rerun()

# --------------------------- Header ----------------------------
st.title("📝 OMR Grader (OpenCV)")
st.caption("Upload one or more OMR answer sheets. View overlays, scores, and download CSVs.")

# ------------------------ Upload Images ------------------------
uploads = st.file_uploader("OMR Images (JPG/PNG)", type=["jpg","jpeg","png"], accept_multiple_files=True)

colA, colB, colC = st.columns([1,1,2])
start_btn = colA.button("🚀 Grade", type="primary", use_container_width=True)
colB.write("")  # spacing

# ---------------------- Results Containers ---------------------
summary_placeholder = st.empty()
details_placeholder = st.empty()
gallery = st.container()

def np_to_png_bytes(img_bgr: np.ndarray) -> bytes:
    if img_bgr.ndim == 2:
        img = img_bgr
    else:
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return buf.getvalue()

def df_download_button(df: pd.DataFrame, filename: str, label: str):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(label, data=csv, file_name=filename, mime="text/csv")

def zip_download_button(files: dict, filename: str, label: str):
    # files: { "name.png": bytes, ... }
    b = io.BytesIO()
    with zipfile.ZipFile(b, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in files.items():
            z.writestr(name, data)
    st.download_button(label, data=b.getvalue(), file_name=filename, mime="application/zip")

if start_btn:
    if not key_file:
        st.error("Please upload an **Answer Key** in the sidebar.")
    elif not uploads:
        st.error("Please upload at least one **OMR image**.")
    else:
        os.makedirs(out_dir, exist_ok=True)
        summaries = []
        all_details = []
        overlay_zip = {}

        pbar = st.progress(0, text="Grading…")
        for i, img in enumerate(uploads, start=1):
            try:
                img_bytes = img.getvalue()
                key_bytes = key_file.getvalue()

                summary, details, overlay = grade_from_bytes(img_bytes, key_bytes)
                # summary: {'total_correct': int, '<Subject>_score': int, ...}
                # details: [{Subject, QNo, Pred}, ...]

                # build display row
                row = {
                    "image": img.name,
                    "total_correct": summary.get("total_correct", 0)
                }
                for s in SUBJECTS:
                    row[f"{s}_score"] = summary.get(f"{s}_score", 0)
                summaries.append(row)

                # enrich details with image name
                for d in details:
                    all_details.append({"image": img.name, **d})

                # overlay preview + save
                png_bytes = np_to_png_bytes(overlay)
                ts = datetime.now().strftime("%Y%m%d-%H%M%S")
                oname = f"{Path(img.name).stem}_{ts}.png"
                # save to disk for audit
                Path(out_dir, oname).write_bytes(png_bytes)
                overlay_zip[oname] = png_bytes

                # show in gallery
                with gallery:
                    st.subheader(f"📄 {img.name}")
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Total Correct", row["total_correct"])
                    for j, s in enumerate(SUBJECTS[:2], start=1):
                        (m2 if j==1 else m3).metric(s, row[f"{s}_score"])
                    n1, n2, n3 = st.columns(3)
                    n1.metric(SUBJECTS[2], row[f"{SUBJECTS[2]}_score"])
                    n2.metric(SUBJECTS[3], row[f"{SUBJECTS[3]}_score"])
                    n3.metric(SUBJECTS[4], row[f"{SUBJECTS[4]}_score"])
                    st.image(png_bytes, caption="Graded overlay", use_container_width=True)
                    st.divider()

            except Exception as e:
                st.error(f"❌ {img.name}: {e}")

            pbar.progress(i/len(uploads), text=f"Graded {i}/{len(uploads)}")

        # tables + downloads
        if summaries:
            df_sum = pd.DataFrame(summaries)
            df_det = pd.DataFrame(all_details)

            with summary_placeholder.container():
                st.subheader("📊 Summary")
                st.dataframe(df_sum, use_container_width=True, height=250)
                c1, c2, c3 = st.columns(3)
                with c1: df_download_button(df_sum, "Summary.csv", "⬇️ Download Summary.csv")
                with c2: df_download_button(df_det, "Details.csv", "⬇️ Download Details.csv")
                with c3: zip_download_button(overlay_zip, "Overlays.zip", "⬇️ Download Overlays.zip")

            with details_placeholder.container():
                st.subheader("🔎 Details")
                st.dataframe(df_det, use_container_width=True, height=350)

else:
    st.info("Upload key + images, then click **Grade**.")

st.caption("Built with Streamlit + OpenCV. No cloud calls.")
