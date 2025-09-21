# models.py — SQLite using SQLAlchemy (no migrations needed)

from sqlalchemy import create_engine, Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from datetime import datetime
import os

DB_URL = os.getenv("DATABASE_URL", "sqlite:///omr.db")

engine = create_engine(DB_URL, connect_args={"check_same_thread": False} if DB_URL.startswith("sqlite") else {})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()

class Submission(Base):
    __tablename__ = "submissions"
    id = Column(Integer, primary_key=True)
    image_name = Column(String)
    overlay_path = Column(String)
    total_correct = Column(Integer, default=0)
    python_score = Column(Integer, default=0)
    data_analysis_score = Column(Integer, default=0)
    mysql_score = Column(Integer, default=0)
    power_bi_score = Column(Integer, default=0)
    adv_stats_score = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    details = relationship("Detail", back_populates="submission")

class Detail(Base):
    __tablename__ = "details"
    id = Column(Integer, primary_key=True)
    submission_id = Column(Integer, ForeignKey("submissions.id"))
    subject = Column(String)
    qno = Column(Integer)
    pred = Column(String)
    
    submission = relationship("Submission", back_populates="details")

def init_db():
    Base.metadata.create_all(bind=engine)