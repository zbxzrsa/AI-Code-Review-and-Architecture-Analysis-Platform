from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, JSON, Boolean
from sqlalchemy.orm import relationship
from datetime import datetime

from .base import Base

class AlertRule(Base):
    __tablename__ = "alert_rule"

    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey("project.id", ondelete="CASCADE"), nullable=False, index=True)
    rule_name = Column(String(255), nullable=False)
    condition = Column(JSON, nullable=False)  # 定义触发条件（阈值、表达式等）
    severity = Column(String(50), nullable=False, default="medium")
    is_active = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    events = relationship("AlertEvent", back_populates="rule", cascade="all, delete-orphan")

class AlertEvent(Base):
    __tablename__ = "alert_event"

    id = Column(Integer, primary_key=True, index=True)
    rule_id = Column(Integer, ForeignKey("alert_rule.id", ondelete="CASCADE"), nullable=False, index=True)
    session_id = Column(Integer, ForeignKey("analysis_session.id", ondelete="SET NULL"), nullable=True, index=True)
    triggered_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    payload = Column(JSON, nullable=True)
    status = Column(String(50), nullable=False, default="open")  # open/ack/closed

    rule = relationship("AlertRule", back_populates="events")