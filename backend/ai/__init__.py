import os
from .engines import stable, next as nxt, legacy

CHANNEL = os.getenv("AI_CHANNEL", "stable")

def pick(channel: str = None):
    ch = (channel or CHANNEL).lower()
    return {"stable": stable, "next": nxt, "legacy": legacy}.get(ch, stable)