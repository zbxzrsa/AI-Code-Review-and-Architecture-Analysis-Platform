"""
Next AI engine with experimental features and structured JSON output.
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
    Perform experimental code review using the next channel.
    
    Args:
        text: Code text to review
        context: Optional RAG context snippets
        meta: Optional metadata (use_rag flag, experiments, etc.)
        
    Returns:
        Structured review output matching next schema
        
    Raises:
        ValueError: If model call fails or validation fails
    """
    # Load structured prompt
    try:
        prompt_body, front_matter = load_prompt("next")
    except Exception as e:
        raise ValueError(f"Failed to load next prompt: {e}")
    
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
    
    # Add experimental context if provided
    if meta and meta.get("experiments"):
        exp_context = "\n".join([f"EXP {k}: {v}" for k, v in meta["experiments"].items()])
        user_message = f"{exp_context}\n\n{user_message}"
    
    # Build messages for Ollama chat API
    messages = [
        {"role": "system", "content": prompt_body},
        {"role": "user", "content": user_message}
    ]
    
    # Load model configuration with experimental parameters
    model = os.getenv("NEXT_MODEL", "qwen2:1.5b-instruct")
    ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")
    
    # Experimental parameters from meta or defaults
    temperature = meta.get("temperature", 0.4) if meta else 0.4
    top_p = meta.get("top_p", 0.8) if meta else 0.8
    
    # Call Ollama with JSON format
    payload = {
        "model": model,
        "messages": messages,
        "options": {
            "temperature": temperature,
            "top_p": top_p,
            "num_ctx": 4096,
            "max_tokens": 800
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
        parsed_output = parse_and_validate("next", raw_output)
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
            parsed_output = parse_and_validate("next", raw_output)
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
        "channel": "next",
        "params_hint": {
            "temperature": f"<={temperature}",
            "top_p": f"{top_p}",
            "max_output_tokens": "<=800"
        }
    })
    
    # Add citations if any
    if all_citations:
        parsed_output["citations"] = all_citations
    
    return parsed_output


def get_model_info() -> Dict[str, Any]:
    """Get information about the next model configuration."""
    return {
        "model": os.getenv("NEXT_MODEL", "qwen2:1.5b-instruct"),
        "channel": "next",
        "temperature": 0.4,
        "top_p": 0.8,
        "format": "json",
        "supports_context": True,
        "supports_citations": True,
        "supports_experiments": True
    }