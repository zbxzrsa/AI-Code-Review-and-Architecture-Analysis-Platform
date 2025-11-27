from sqlalchemy import Column, Integer, String, DateTime, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()


class AiAuditEvent(Base):
    __tablename__ = "ai_audit_events"
    
    id = Column(Integer, primary_key=True, index=True)
    channel = Column(String(50), nullable=False, index=True)
    prompt_hash = Column(String(64), nullable=False, index=True)
    redactions_applied = Column(Text)  # JSON string of redaction counts
    deny_hits = Column(Text)  # JSON string of deny violations
    risky_violations = Column(Text)  # JSON string of risky patterns
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    
    # Additional metadata
    client_ip = Column(String(45))  # IPv6 compatible
    user_agent = Column(Text)
    request_id = Column(String(36), index=True)  # UUID for request tracking
    
    def __repr__(self):
        return f"<AiAuditEvent(channel={self.channel}, prompt_hash={self.prompt_hash[:8]}..., created_at={self.created_at})>"