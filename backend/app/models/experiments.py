from sqlalchemy import Column, Integer, String, DateTime, Text, Float, Boolean
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()


class AiPromptVersion(Base):
    __tablename__ = "ai_prompt_versions"
    
    id = Column(Integer, primary_key=True, index=True)
    channel = Column(String(50), nullable=False, index=True)
    version = Column(String(20), nullable=False)
    path = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    
    def __repr__(self):
        return f"<AiPromptVersion(channel={self.channel}, version={self.version})>"


class AiExperiment(Base):
    __tablename__ = "ai_experiments"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    channel = Column(String(50), nullable=False, index=True)
    config_json = Column(Text)  # JSON string of experiment config
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    
    def __repr__(self):
        return f"<AiExperiment(name={self.name}, channel={self.channel})>"


class AiRun(Base):
    __tablename__ = "ai_runs"
    
    id = Column(Integer, primary_key=True, index=True)
    channel = Column(String(50), nullable=False, index=True)
    prompt_hash = Column(String(64), nullable=False, index=True)
    model = Column(String(100), nullable=False)
    prompt_version = Column(String(20), nullable=False)
    latency_ms = Column(Float, nullable=False)
    tokens_out = Column(Integer, default=0)
    cache_hit = Column(Boolean, default=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    
    # Additional metadata
    client_ip = Column(String(45))
    user_agent = Column(Text)
    request_id = Column(String(36), index=True)
    
    def __repr__(self):
        return f"<AiRun(channel={self.channel}, model={self.model}, latency_ms={self.latency_ms})>"