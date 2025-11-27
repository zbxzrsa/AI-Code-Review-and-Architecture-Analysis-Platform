"""
Prompt loading utilities for structured AI prompts.
"""

import os
import re
from typing import Dict, Tuple, Any
from pathlib import Path


def load_prompt(channel: str) -> Tuple[str, Dict[str, Any]]:
    """
    Load structured prompt with front matter parsing.
    
    Args:
        channel: The AI channel (stable, next, legacy)
        
    Returns:
        Tuple of (prompt_body, front_matter_dict)
        
    Raises:
        FileNotFoundError: If prompt file doesn't exist
        ValueError: If front matter is malformed
    """
    prompt_dir = Path(__file__).parent.parent.parent / "prompts" / channel
    prompt_file = prompt_dir / "review.md"
    
    if not prompt_file.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
    
    with open(prompt_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Parse YAML front matter
    front_matter = {}
    body = content
    
    if content.startswith('---\n'):
        parts = content.split('---\n', 2)
        if len(parts) >= 3:
            try:
                import yaml
                front_matter = yaml.safe_load(parts[1]) or {}
                body = parts[2].strip()
            except ImportError:
                # Fallback: simple key-value parsing
                front_matter_text = parts[1]
                for line in front_matter_text.split('\n'):
                    if ':' in line:
                        key, value = line.split(':', 1)
                        front_matter[key.strip()] = value.strip().strip('"')
                body = parts[2].strip()
            except Exception as e:
                raise ValueError(f"Failed to parse front matter: {e}")
    
    return body, front_matter


def get_prompt_version(channel: str) -> str:
    """
    Get the version of a prompt channel.
    
    Args:
        channel: The AI channel
        
    Returns:
        Version string or "unknown" if not found
    """
    try:
        _, front_matter = load_prompt(channel)
        return front_matter.get('version', 'unknown')
    except Exception:
        return 'unknown'


def validate_prompt_structure(channel: str) -> bool:
    """
    Validate that a prompt has the required structure.
    
    Args:
        channel: The AI channel to validate
        
    Returns:
        True if valid, False otherwise
    """
    try:
        body, front_matter = load_prompt(channel)
        
        # Check required front matter fields
        required_fields = ['version', 'owner']
        for field in required_fields:
            if field not in front_matter:
                return False
        
        # Check that body contains JSON schema hint
        if 'JSON' not in body.upper():
            return False
            
        return True
    except Exception:
        return False