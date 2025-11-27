"""
Stable AI engine with structured JSON output and RAG support.
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
    Perform structured code review using the stable channel.
    
    Args:
        text: Code text to review
        context: Optional RAG context snippets
        meta: Optional metadata (use_rag flag, etc.)
        
    Returns:
        Structured review output matching stable schema
        
    Raises:
        ValueError: If model call fails or validation fails
    """
    # Load structured prompt
    try:
        prompt_body, front_matter = load_prompt("stable")
    except Exception as e:
        raise ValueError(f"Failed to load stable prompt: {e}")
    
    # Prepare user message with optional context
    user_message = text
    citations = []
    
    if context:
        context_text, context_citations = format_context(context)
        user_message = f"{context_text}\n\nCode to review:\n{text}"
        citations.extend(context_citations)
    elif meta and meta.get("use_rag", True):
        # TODO: Implement actual RAG search when available
        logger.debug("RAG requested but not implemented, proceeding without context")
    
    # Build messages for Ollama chat API
    messages = [
        {"role": "system", "content": prompt_body},
        {"role": "user", "content": user_message}
    ]
    
    # Load model configuration
    model = os.getenv("STABLE_MODEL", "mistral:7b-instruct")
    ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")
    
    # Call Ollama with JSON format
    payload = {
        "model": model,
        "messages": messages,
        "options": {
            "temperature": 0.2,
            "top_p": 0.9,
            "num_ctx": 4096
        },
        "format": "json"  # Enforce JSON output
    }
    
    try:
        response = requests.post(ollama_url, json=payload, timeout=120)
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
        parsed_output = parse_and_validate("stable", raw_output)
    except ValueError as e:
        # Retry once with repair instruction
        logger.warning(f"First parse failed, retrying with repair: {e}")
        
        retry_messages = messages + [
            {"role": "assistant", "content": raw_output},
            {"role": "user", "content": "Return only VALID JSON for the schema; no prose."}
        ]
        
        payload["messages"] = retry_messages
        
        try:
            response = requests.post(ollama_url, json=payload, timeout=120)
            response.raise_for_status()
            response_data = response.json()
            raw_output = response_data["message"]["content"]
            parsed_output = parse_and_validate("stable", raw_output)
        except Exception as retry_e:
            raise ValueError(f"JSON validation failed even after retry: {retry_e}")
    
    # Extract additional citations from the response
    response_citations = extract_citations_from_text(raw_output)
    all_citations = merge_citations(citations, response_citations)
    
    # Enhance metadata
    if "meta" not in parsed_output:
        parsed_output["meta"] = {}
    
    parsed_output["meta"].update({
        "model": model,
        "prompt_version": front_matter.get("version", "unknown"),
        "channel": "stable"
    })
    
    # Add citations if any
    if all_citations:
        parsed_output["citations"] = all_citations
    
    return parsed_output


def get_model_info() -> Dict[str, Any]:
    """Get information about the stable model configuration."""
    return {
        "model": os.getenv("STABLE_MODEL", "mistral:7b-instruct"),
        "channel": "stable",
        "temperature": 0.2,
        "top_p": 0.9,
        "format": "json",
        "supports_context": True,
        "supports_citations": True
    }