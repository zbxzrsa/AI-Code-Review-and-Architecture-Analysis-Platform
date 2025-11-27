"""
Legacy AI engine with fast baseline and structured JSON output.
"""

import os
import json
import requests
from typing import Dict, Any, Optional, List
import logging

from .shared import parse_and_validate, load_prompt
from ..rag.citations import format_context, extract_citations_from_text, merge_citations

logger = logging.getLogger(__name__)


def review(
    text: str, 
    context: Optional[List[Dict[str, Any]]] = None, 
    meta: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Perform fast baseline code review using legacy channel.
    
    Args:
        text: Code text to review
        context: Optional RAG context snippets (usually ignored for speed)
        meta: Optional metadata (use_rag flag, etc.)
        
    Returns:
        Structured review output matching legacy schema
        
    Raises:
        ValueError: If model call fails or validation fails
    """
    # Load structured prompt
    try:
        prompt_body, front_matter = load_prompt("legacy")
    except Exception as e:
        raise ValueError(f"Failed to load legacy prompt: {e}")
    
    # For legacy, ignore context by default for speed (unless explicitly requested)
    user_message = text
    citations = []
    
    if context and meta and meta.get("use_context", False):
        context_text, context_citations = format_context(context)
        user_message = f"{context_text}\n\nCode to review:\n{text}"
        citations.extend(context_citations)
    elif meta and meta.get("use_rag", False):
        # Legacy typically skips RAG for speed
        logger.debug("RAG requested but legacy channel prioritizes speed")
    
    # Build messages for Ollama chat API
    messages = [
        {"role": "system", "content": prompt_body},
        {"role": "user", "content": user_message}
    ]
    
    # Load model configuration (fastest model)
    model = os.getenv("LEGACY_MODEL", "qwen2:0.5b-instruct")
    ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")
    
    # Legacy uses minimal parameters for speed
    payload = {
        "model": model,
        "messages": messages,
        "options": {
            "temperature": 0.1,  # Low temperature for consistency
            "top_p": 0.8,
            "num_ctx": 2048,   # Smaller context for speed
            "max_tokens": 400   # Smaller output for speed
        },
        "format": "json"  # Enforce JSON output
    }
    
    try:
        response = requests.post(ollama_url, json=payload, timeout=60)  # Shorter timeout
        response.raise_for_status()
        response_data = response.json()
        
        if "message" not in response_data or "content" not in response_data["message"]:
            raise ValueError("Invalid response format from Ollama")
            
        raw_output = response_data["message"]["content"]
        
    except requests.RequestException as e:
        raise ValueError(f"Failed to call Ollama: {e}")
    except Exception as e:
        raise ValueError(f"Failed to parse Ollama response: {e}")
    
    # Parse and validate structured output
    try:
        parsed_output = parse_and_validate("legacy", raw_output)
    except ValueError as e:
        # Retry once with repair instruction
        logger.warning(f"First parse failed, retrying with repair: {e}")
        
        retry_messages = messages + [
            {"role": "assistant", "content": raw_output},
            {"role": "user", "content": "Return only VALID JSON for the schema; no prose."}
        ]
        
        payload["messages"] = retry_messages
        
        try:
            response = requests.post(ollama_url, json=payload, timeout=60)
            response.raise_for_status()
            response_data = response.json()
            raw_output = response_data["message"]["content"]
            parsed_output = parse_and_validate("legacy", raw_output)
        except Exception as retry_e:
            raise ValueError(f"JSON validation failed even after retry: {retry_e}")
    
    # Extract additional citations from the response (minimal for legacy)
    response_citations = extract_citations_from_text(raw_output)
    all_citations = merge_citations(citations, response_citations)
    
    # Enhance metadata
    if "meta" not in parsed_output:
        parsed_output["meta"] = {}
    
    parsed_output["meta"].update({
        "model": model,
        "prompt_version": front_matter.get("version", "unknown"),
        "channel": "legacy"
    })
    
    # Add citations if any (usually minimal for legacy)
    if all_citations:
        parsed_output["citations"] = all_citations
    
    return parsed_output


def get_model_info() -> Dict[str, Any]:
    """Get information about legacy model configuration."""
    return {
        "model": os.getenv("LEGACY_MODEL", "qwen2:0.5b-instruct"),
        "channel": "legacy",
        "temperature": 0.1,
        "top_p": 0.8,
        "format": "json",
        "supports_context": False,  # Usually disabled for speed
        "supports_citations": True,
        "priority": "speed"
    }