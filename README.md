# **OMR Grader** — Fast, Accurate, Open-Source

A lightweight web app that grades OMR (Optical Mark Recognition) answer sheets using **OpenCV**, with a clean **FastAPI + Jinja** UI.  
Upload an **answer key** (Excel/CSV) and one or more **OMR images** → get per-subject scores, per-question predictions, and downloadable overlays + CSV reports. No cloud calls.

---

## **Table of Contents**
- [Why This Exists](#why-this-exists)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Quickstart](#quickstart)
- [Using the App](#using-the-app)
- [Outputs](#outputs)
- [Answer Key Examples](#answer-key-examples)
- [How It Works (Short Version)](#how-it-works-short-version)
- [Docker (Optional)](#docker-optional)
- [Troubleshooting](#troubleshooting)
- [Roadmap](#roadmap)
- [Security & Privacy](#security--privacy)
- [License](#license)

---

## **Why This Exists**
Manual OMR checking is slow and error-prone. This app turns the process into a one-click workflow:

- Drag-and-drop key + images  
- Automatic deskew + bubble detection  
- Visual overlays for audit  
- Summary & detailed CSVs for instructors

---

## **Features**
- **Multi-subject grading**: Python, Data Analysis (EDA), MySQL/SQL, Power BI, Adv Stats (configurable).
- **Robust detection**: Deskews sheets, tolerates lighting/shadow variance, distinguishes filled vs. empty bubbles.
- **Batch processing**: Upload many sheets at once.
- **Auditable outputs**:
  - Color **overlays** for every sheet  
  - **Summary.csv** (per sheet + per subject)  
  - **Details.csv** (per sheet + per question)
- **Simple web app**: FastAPI + Jinja + Bootstrap (dark UI).
- **Local & private**: Runs on your machine (or your server).
- **SQLite storage**: Keeps submission history.

---

## **Tech Stack**
- **Backend:** FastAPI (Python), SQLAlchemy, SQLite  
- **Vision:** OpenCV, NumPy  
- **Parsing:** pandas, openpyxl  
- **Frontend:** Jinja templates + Bootstrap  
- **Dev:** Uvicorn (hot-reload), optional Docker

---

## **Project Structure**
```text
omr-grader/
├─ backend/
│  ├─ __init__.py
│  ├─ omr_core.py        # OpenCV pipeline (returns NumPy overlay image)
│  └─ models.py          # SQLAlchemy models + init_db()
├─ static/
│  └─ overlays/          # generated overlays (git-ignored)
├─ templates/
│  ├─ base.html
│  ├─ index.html         # upload page + recent cards
│  ├─ submissions.html   # table of all runs
│  └─ details.html       # per-sheet overlay + pivot table
├─ server.py             # FastAPI app (UI routes + downloads)
├─ requirements.txt
└─ README.md
```

---

## **Quickstart**

### **1) Install**
**Windows (PowerShell)**
```powershell
python -m venv .venv
& .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### **2) Run the web app**
```bash
uvicorn server:app --reload --host 0.0.0.0 --port 8000
```
Open **http://localhost:8000**  
Health check: **http://localhost:8000/hello** should return *Hello from OMR Grader*.

---

## **Using the App**
1. **Upload Answer Key** (`.xlsx` or `.csv`).  
   Supported column names (case/spacing flexible):
   - `Python`
   - `Data Analysis` or `EDA`
   - `MySQL` or `SQL`
   - `Power BI`
   - `Adv Stats` or `Statistics` (even “Satistics” typo is handled)
2. **Answer cells**: `a`, `b`, `c`, `d` (case-insensitive). Multiple correct answers allowed: `a,b`.
3. Upload one or more **OMR images** (`.jpg` / `.jpeg` / `.png`).
4. Click **Run Grading**.
5. Review overlays and scores on the **Submissions** page.
6. Use the navbar to download **Summary.csv**, **Details.csv**, and **Overlays.zip**.

---

## **Outputs**
- **Overlays** → saved to `static/overlays/` with green/red markings on each bubble.
- **Summary.csv** → columns:
  ```text
  id, image, total_correct, Python, Data Analysis, MySQL, Power BI, Adv Stats
  ```
- **Details.csv** → columns:
  ```text
  submission_id, subject, QNo, Pred   # Pred ∈ {A,B,C,D,""}
  ```

---

## **Answer Key Examples**
**Excel (conceptual)**
```text
| Python | Data Analysis | SQL | Power BI | Statistics |
| a      | b             | c   | b        | a          |
| c      | d             | c   | a        | b          |
| b      | b             | a   | d        | c          |
...
```

**CSV**
```csv
Python,EDA,SQL,Power BI,Statistics
a,b,c,b,a
c,d,c,a,b
b,b,a,d,c
```

---

## **How It Works (Short Version)**
- **Preprocess** → grayscale, adaptive threshold; estimate skew via Hough lines; rotate.
- **Detect** → Hough circle transform; group into 4-bubble sets (A–D).
- **Align** → cluster groups into 5 subject columns; map to row indices using median vertical pitch.
- **Decide** → measure ink inside vs. local background; pick best candidate with margin.
- **Score** → compare predictions with key; compute per-subject + total.
- **Render** → draw colored circles, save overlay.

---

## **Docker (Optional)**
```bash
docker build -t omr-grader .
docker run --rm -p 8000:8000 \
  -v "$(pwd)/static/overlays:/app/static/overlays" \
  omr-grader
```
Open **http://localhost:8000**.

---

## **Troubleshooting**
- **Blank page** → ensure `templates/` is at the repo root and `server.py` points to it. Visit `/hello` to verify.
- **“attempted relative import with no known parent package”** → run Uvicorn from the **project root** and ensure `backend/__init__.py` exists.
- **Excel read error** → `pip install openpyxl`.
- **Overlays not saved** → check write permissions for `static/overlays/`.
- **Predictions look off** → scan at a higher resolution; tune thresholds in `backend/omr_core.py` (radii, fill ratios, spacing tolerances).

---

## **Roadmap**
- Admin settings page (tune thresholds, subject names, question counts)  
- Bulk ZIP uploads  
- Per-question ✔ / ✖ overlays  
- Authentication / roles (instructors vs. TAs)  
- Export to styled XLSX

---

## **Security & Privacy**
Runs locally by default and **does not** send images to external services.  
For server deployments, add access control and HTTPS.

---

## **License**
MIT — see `LICENSE`.

---

*Built with FastAPI, OpenCV, NumPy, pandas, SQLAlchemy, and Bootstrap.*
