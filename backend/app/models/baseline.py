from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, JSON, Float
from sqlalchemy.orm import relationship
from datetime import datetime

from .base import Base

class Baseline(Base):
    __tablename__ = "baseline"

    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey("project.id", ondelete="CASCADE"), nullable=False, index=True)
    name = Column(String(255), nullable=False)
    description = Column(String(1024), nullable=True)
    config = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    project = relationship("Project", back_populates="baselines")
    deviations = relationship("BaselineDeviation", back_populates="baseline", cascade="all, delete-orphan")

class BaselineDeviation(Base):
    __tablename__ = "baseline_deviation"

    id = Column(Integer, primary_key=True, index=True)
    baseline_id = Column(Integer, ForeignKey("baseline.id", ondelete="CASCADE"), nullable=False, index=True)
    metric_name = Column(String(255), nullable=False)
    deviation_value = Column(Float, nullable=False)
    severity = Column(String(50), nullable=False, default="medium")
    detected_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    baseline = relationship("Baseline", back_populates="deviations")