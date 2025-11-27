from sqlalchemy import Column, Integer, String, Float, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()


class AiEvalRun(Base):
    __tablename__ = "ai_eval_runs"
    
    id = Column(Integer, primary_key=True, index=True)
    channel = Column(String(50), nullable=False, index=True)
    dataset = Column(String(255), nullable=False)
    score = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    
    # Additional metadata
    total_items = Column(Integer, default=0)
    successful_items = Column(Integer, default=0)
    avg_latency_ms = Column(Float, default=0.0)
    avg_tokens_out = Column(Float, default=0.0)
    offline_mode = Column(String(10), default="false")
    
    def __repr__(self):
        return f"<AiEvalRun(channel={self.channel}, score={self.score:.3f}, created_at={self.created_at})>"