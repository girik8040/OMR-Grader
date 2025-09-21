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
