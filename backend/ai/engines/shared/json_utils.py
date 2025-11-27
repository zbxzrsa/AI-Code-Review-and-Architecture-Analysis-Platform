"""
JSON utilities for parsing and repairing model outputs.
"""

import json
import re
from typing import Any, Dict, Optional


def safe_json_loads(text: str) -> Dict[str, Any]:
    """
    Safely load JSON with common error fixes.
    
    Args:
        text: Raw text that should contain JSON
        
    Returns:
        Parsed JSON dictionary
        
    Raises:
        ValueError: If JSON cannot be parsed after repairs
    """
    # Common fixes
    fixed_text = text.strip()
    
    # Remove markdown code blocks
    fixed_text = re.sub(r'^```json\s*', '', fixed_text)
    fixed_text = re.sub(r'```\s*$', '', fixed_text)
    
    # Fix trailing commas
    fixed_text = re.sub(r',(\s*[}\]])', r'\1', fixed_text)
    
    # Fix missing quotes around keys (common in model outputs)
    fixed_text = re.sub(r'(\w+)\s*:', r'"\1":', fixed_text)
    
    # Fix single quotes to double quotes
    fixed_text = re.sub(r"'([^']*)'", r'"\1"', fixed_text)
    
    try:
        return json.loads(fixed_text)
    except json.JSONDecodeError as e:
        # Try more aggressive repairs
        try:
            return _aggressive_json_repair(fixed_text)
        except Exception:
            raise ValueError(f"JSON parsing failed: {e}")


def ensure_json_only(text: str) -> str:
    """
    Extract only the JSON object from mixed text.
    
    Args:
        text: Text that may contain JSON mixed with other content
        
    Returns:
        JSON string only
    """
    # Look for JSON object boundaries
    brace_count = 0
    start_idx = None
    end_idx = None
    
    for i, char in enumerate(text):
        if char == '{':
            if brace_count == 0:
                start_idx = i
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0 and start_idx is not None:
                end_idx = i + 1
                break
    
    if start_idx is not None and end_idx is not None:
        return text[start_idx:end_idx]
    
    # Fallback: try to find any JSON-like structure
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if json_match:
        return json_match.group(0)
    
    return text


def add_schema_hint(schema_str: str) -> str:
    """
    Add schema hint to system message for better JSON compliance.
    
    Args:
        schema_str: JSON schema string
        
    Returns:
        Enhanced system message with schema hint
    """
    return f"""
IMPORTANT: You must respond with valid JSON only. No additional text, no explanations, no markdown formatting.

Required JSON schema structure:
{schema_str}

Your entire response must be a single valid JSON object matching this schema exactly.
"""


def _aggressive_json_repair(text: str) -> Dict[str, Any]:
    """
    More aggressive JSON repair for badly formatted model outputs.
    
    Args:
        text: Badly formatted JSON text
        
    Returns:
        Reparsed JSON dictionary
        
    Raises:
        ValueError: If repair fails
    """
    # Remove all non-JSON content
    json_only = ensure_json_only(text)
    
    # Fix common issues
    repairs = [
        # Fix unescaped quotes in strings
        (r':\s*"([^"]*)"([^",\}\]]*)"', r': "\1\\"\\2\\""'),
        # Fix newlines in strings
        (r'\n', r'\\n'),
        # Fix tabs in strings  
        (r'\t', r'\\t'),
        # Fix unescaped backslashes
        (r'\\\\', r'\\\\\\\\'),
    ]
    
    for pattern, replacement in repairs:
        json_only = re.sub(pattern, replacement, json_only)
    
    try:
        return json.loads(json_only)
    except json.JSONDecodeError:
        # Last resort: try to extract key-value pairs manually
        return _manual_json_parse(json_only)


def _manual_json_parse(text: str) -> Dict[str, Any]:
    """
    Manual JSON parsing as last resort for severely malformed output.
    
    Args:
        text: Severely malformed JSON text
        
    Returns:
        Partially parsed dictionary
        
    Raises:
        ValueError: If manual parsing also fails
    """
    result = {}
    
    # Simple regex-based key-value extraction
    pattern = r'"([^"]+)"\s*:\s*"([^"]*)"'
    matches = re.findall(pattern, text)
    
    for key, value in matches:
        result[key] = value
    
    if not result:
        raise ValueError("Failed to parse JSON even with manual extraction")
    
    return result


def validate_json_structure(data: Dict[str, Any], required_keys: list) -> bool:
    """
    Validate that JSON contains required top-level keys.
    
    Args:
        data: Parsed JSON dictionary
        required_keys: List of required keys
        
    Returns:
        True if valid, False otherwise
    """
    return all(key in data for key in required_keys)