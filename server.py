import os, io, uuid, zipfile
from typing import List

import cv2
import numpy as np
import pandas as pd
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates

from backend.omr_core import grade_from_bytes, SUBJECTS
from backend.models import init_db, SessionLocal, Submission, Detail

APP_TITLE = "OMR Grader"
BASE_DIR = os.path.dirname(__file__)
STATIC_DIR = os.path.join(BASE_DIR, "static")
OVERLAY_DIR = os.path.join(STATIC_DIR, "overlays")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

os.makedirs(OVERLAY_DIR, exist_ok=True)

app = FastAPI(title=APP_TITLE)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Fail fast if templates dir is wrong
if not os.path.isdir(TEMPLATES_DIR):
    raise RuntimeError(f"Templates folder not found at: {TEMPLATES_DIR}")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

init_db()

# ---------- helpers ----------
def save_overlay_png(overlay_nd: np.ndarray) -> str:
    name = f"{uuid.uuid4().hex}.png"
    out_path = os.path.join(OVERLAY_DIR, name)
    if overlay_nd.ndim == 3 and overlay_nd.shape[2] >= 3:
        overlay_nd = overlay_nd[:, :, :3]
    overlay_nd = overlay_nd.astype("uint8")
    ok = cv2.imwrite(out_path, overlay_nd)
    if not ok:
        raise RuntimeError("Failed to write overlay image.")
    return f"/static/overlays/{name}"

def df_to_csv_stream(df: pd.DataFrame, name="data.csv") -> StreamingResponse:
    data = df.to_csv(index=False).encode("utf-8")
    return StreamingResponse(io.BytesIO(data), media_type="text/csv",
                             headers={"Content-Disposition": f'attachment; filename="{name}"'})

# ---------- sanity routes ----------
@app.get("/hello", response_class=PlainTextResponse)
def hello():
    return "Hello from OMR Grader"

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    try:
        db = SessionLocal()
        rows = db.query(Submission).order_by(Submission.id.desc()).limit(20).all()
        return templates.TemplateResponse("index.html",
            {"request": request, "rows": rows, "title": APP_TITLE})
    except Exception as e:
        # Show a readable error instead of a blank page
        return HTMLResponse(f"<pre>Home error:\n{e}</pre>", status_code=500)
    finally:
        db.close()

# ---------- main actions ----------
@app.post("/upload", response_class=HTMLResponse)
async def upload(request: Request,
                 key: UploadFile = File(...),
                 images: List[UploadFile] = File(...)):
    key_bytes = await key.read()
    db = SessionLocal()
    try:
        for img in images:
            img_bytes = await img.read()
            summary, details, overlay = grade_from_bytes(img_bytes, key_bytes)
            overlay_url = save_overlay_png(overlay)

            sub = Submission(
                image_name=img.filename,
                overlay_path=overlay_url,
                total_correct=summary.get("total_correct", 0),
                python_score=summary.get("Python_score", 0),
                data_analysis_score=summary.get("Data Analysis_score", 0),
                mysql_score=summary.get("MySQL_score", 0),
                power_bi_score=summary.get("Power BI_score", 0),
                adv_stats_score=summary.get("Adv Stats_score", 0),
            )
            db.add(sub); db.flush()
            for d in details:
                db.add(Detail(submission_id=sub.id, subject=d["Subject"], qno=d["QNo"], pred=d["Pred"]))
        db.commit()
    except Exception as e:
        db.rollback()
        return HTMLResponse(f"<pre>Upload error:\n{e}</pre>", status_code=400)
    finally:
        db.close()
    return RedirectResponse(url="/submissions", status_code=303)

@app.get("/submissions", response_class=HTMLResponse)
def submissions(request: Request):
    try:
        db = SessionLocal()
        rows = db.query(Submission).order_by(Submission.id.desc()).all()
        return templates.TemplateResponse("submissions.html",
            {"request": request, "rows": rows, "title": APP_TITLE})
    except Exception as e:
        return HTMLResponse(f"<pre>Submissions error:\n{e}</pre>", status_code=500)
    finally:
        db.close()

@app.get("/submissions/{sid}", response_class=HTMLResponse)
def submission_details(request: Request, sid: int):
    try:
        db = SessionLocal()
        sub = db.query(Submission).filter(Submission.id==sid).first()
        det = db.query(Detail).filter(Detail.submission_id==sid).all()
        if not sub:
            return HTMLResponse("<pre>Not found</pre>", status_code=404)

        df = pd.DataFrame([{"Subject":d.subject,"QNo":d.qno,"Pred":d.pred} for d in det])
        piv_rows, piv_cols = [], ["QNo"] + SUBJECTS
        if not df.empty:
            piv = (df.pivot(index="QNo", columns="Subject", values="Pred")
                     .reindex(index=range(1,101), columns=SUBJECTS).fillna(""))
            piv.reset_index(inplace=True)
            piv_cols = list(piv.columns)
            piv_rows = piv.to_dict("records")

        return templates.TemplateResponse("details.html",
            {"request": request, "sub": sub, "piv_rows": piv_rows,
             "piv_cols": piv_cols, "title": APP_TITLE})
    except Exception as e:
        return HTMLResponse(f"<pre>Details error:\n{e}</pre>", status_code=500)
    finally:
        db.close()

# ---------- downloads ----------
@app.get("/download/summary.csv")
def download_summary():
    db = SessionLocal()
    try:
        rows = db.query(Submission).order_by(Submission.id).all()
        df = pd.DataFrame([{
            "id": r.id, "image": r.image_name, "total_correct": r.total_correct,
            "Python": r.python_score, "Data Analysis": r.data_analysis_score,
            "MySQL": r.mysql_score, "Power BI": r.power_bi_score, "Adv Stats": r.adv_stats_score
        } for r in rows])
    finally:
        db.close()
    return df_to_csv_stream(df, "Summary.csv")

@app.get("/download/details.csv")
def download_details():
    db = SessionLocal()
    try:
        det = db.query(Detail).order_by(Detail.submission_id, Detail.qno).all()
        df = pd.DataFrame([{"submission_id":d.submission_id,"subject":d.subject,"QNo":d.qno,"Pred":d.pred} for d in det])
    finally:
        db.close()
    return df_to_csv_stream(df, "Details.csv")

@app.get("/download/overlays.zip")
def download_overlays():
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, "w", zipfile.ZIP_DEFLATED) as z:
        for name in os.listdir(OVERLAY_DIR):
            z.write(os.path.join(OVERLAY_DIR, name), arcname=name)
    mem.seek(0)
    return StreamingResponse(mem, media_type="application/zip",
                             headers={"Content-Disposition": 'attachment; filename="Overlays.zip"'})
