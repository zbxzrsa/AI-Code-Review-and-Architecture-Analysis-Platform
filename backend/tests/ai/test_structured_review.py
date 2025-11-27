"""
Comprehensive tests for structured AI review functionality.
"""

import pytest
import json
import os
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

# Set environment for testing
os.environ["AI_STRUCTURED_MODE"] = "true"
os.environ["STABLE_MODEL"] = "mistral:7b-instruct"
os.environ["NEXT_MODEL"] = "qwen2:1.5b-instruct" 
os.environ["LEGACY_MODEL"] = "qwen2:0.5b-instruct"
os.environ["OLLAMA_URL"] = "http://localhost:11434/api/chat"

# Import after setting environment
from backend.ai.engines.shared.schema import parse_and_validate, StableOutput, NextOutput, LegacyOutput
from backend.ai.engines.shared.prompt_loader import load_prompt, validate_prompt_structure
from backend.ai.engines.shared.json_utils import safe_json_loads, ensure_json_only
from backend.ai.rag.citations import format_context, extract_citations_from_text, merge_citations


class TestStructuredPrompts:
    """Test structured prompt loading and validation."""
    
    def test_load_stable_prompt(self):
        """Test loading stable channel prompt."""
        body, front_matter = load_prompt("stable")
        
        assert "JSON" in body.upper()
        assert front_matter["version"] == "v1.2.3"
        assert front_matter["owner"] == "team-ml"
        assert "conservative" in body.lower()
    
    def test_load_next_prompt(self):
        """Test loading next channel prompt."""
        body, front_matter = load_prompt("next")
        
        assert "JSON" in body.upper()
        assert front_matter["version"] == "v1.2.3"
        assert "experimental" in body.lower()
    
    def test_load_legacy_prompt(self):
        """Test loading legacy channel prompt."""
        body, front_matter = load_prompt("legacy")
        
        assert "JSON" in body.upper()
        assert front_matter["version"] == "v1.2.3"
        assert "baseline" in body.lower()
    
    def test_validate_prompt_structure(self):
        """Test prompt structure validation."""
        assert validate_prompt_structure("stable") is True
        assert validate_prompt_structure("next") is True
        assert validate_prompt_structure("legacy") is True
        assert validate_prompt_structure("nonexistent") is False


class TestJSONUtilities:
    """Test JSON parsing and repair utilities."""
    
    def test_safe_json_loads_valid(self):
        """Test parsing valid JSON."""
        valid_json = '{"key": "value", "number": 42}'
        result = safe_json_loads(valid_json)
        
        assert result["key"] == "value"
        assert result["number"] == 42
    
    def test_safe_json_loads_with_fixes(self):
        """Test JSON parsing with common fixes."""
        # Test trailing comma
        json_with_trailing = '{"key": "value",}'
        result = safe_json_loads(json_with_trailing)
        assert result["key"] == "value"
        
        # Test single quotes
        json_single_quotes = "{'key': 'value'}"
        result = safe_json_loads(json_single_quotes)
        assert result["key"] == "value"
        
        # Test markdown wrapper
        json_markdown = "```json\n{\"key\": \"value\"}\n```"
        result = safe_json_loads(json_markdown)
        assert result["key"] == "value"
    
    def test_ensure_json_only(self):
        """Test extracting JSON from mixed text."""
        mixed_text = "Some preamble text here\n{\"key\": \"value\"}\nSome postamble text"
        json_only = ensure_json_only(mixed_text)
        
        parsed = json.loads(json_only)
        assert parsed["key"] == "value"
    
    def test_ensure_json_only_nested(self):
        """Test extracting nested JSON objects."""
        nested_text = "Text before {\"outer\": {\"inner\": \"value\"}} text after"
        json_only = ensure_json_only(nested_text)
        
        parsed = json.loads(json_only)
        assert parsed["outer"]["inner"] == "value"


class TestSchemaValidation:
    """Test schema validation for all channels."""
    
    def test_parse_and_validate_stable(self):
        """Test stable channel schema validation."""
        valid_stable = {
            "summary": "Code review completed",
            "risk_level": "low",
            "key_findings": [
                {
                    "area": "security",
                    "title": "Potential SQL injection",
                    "severity": "major",
                    "evidence": "Direct string concatenation in query",
                    "source": "app.py:45-50"
                }
            ],
            "recommendations": [
                {
                    "title": "Use parameterized queries",
                    "rationale": "Prevents SQL injection attacks",
                    "est_impact": "safety",
                    "effort": "M"
                }
            ],
            "safe_patch_sketch": [],
            "tests": [],
            "observability": [],
            "citations": [],
            "meta": {
                "channel": "stable",
                "style": "conservative",
                "constraints": ["offline", "no-paid-apis", "deterministic-json"]
            }
        }
        
        result = parse_and_validate("stable", json.dumps(valid_stable))
        assert result["summary"] == "Code review completed"
        assert result["risk_level"] == "low"
        assert len(result["key_findings"]) == 1
    
    def test_parse_and_validate_next(self):
        """Test next channel schema validation."""
        valid_next = {
            "summary": "Experimental review completed",
            "risk_level": "medium",
            "experiments_considered": [
                {
                    "name": "async_optimization",
                    "toggle": "on",
                    "reason": "Potential performance improvement"
                }
            ],
            "key_findings": [
                {
                    "area": "perf",
                    "title": "Inefficient loop detected",
                    "severity": "minor",
                    "evidence": "O(n^2) complexity in data processing"
                }
            ],
            "recommendations": [
                {
                    "title": "Use list comprehension",
                    "rationale": "More Pythonic and faster",
                    "expected_gain": "p95 -20%",
                    "validate": "benchmark with timeit",
                    "effort": "S"
                }
            ],
            "safe_patch_sketch": [],
            "tests": [],
            "guardrails": [],
            "citations": [],
            "meta": {
                "channel": "next",
                "style": "experimental",
                "constraints": ["offline", "no-paid-apis", "deterministic-json"]
            }
        }
        
        result = parse_and_validate("next", json.dumps(valid_next))
        assert result["summary"] == "Experimental review completed"
        assert len(result["experiments_considered"]) == 1
    
    def test_parse_and_validate_legacy(self):
        """Test legacy channel schema validation."""
        valid_legacy = {
            "summary": "Quick review completed",
            "risk_level": "low",
            "key_findings": [
                {
                    "rule": "missing_error_handling",
                    "evidence": "Function doesn't handle exceptions",
                    "severity": "minor",
                    "source": "utils.py:23"
                }
            ],
            "recommendations": [
                {
                    "title": "Add try-catch block",
                    "effort": "S",
                    "hint": "Wrap in try: except Exception:"
                }
            ],
            "meta": {
                "channel": "legacy",
                "style": "baseline-fast",
                "constraints": ["offline", "no-paid-apis", "deterministic-json"]
            }
        }
        
        result = parse_and_validate("legacy", json.dumps(valid_legacy))
        assert result["summary"] == "Quick review completed"
        assert len(result["key_findings"]) == 1
    
    def test_parse_and_validate_invalid_json(self):
        """Test handling of invalid JSON."""
        with pytest.raises(ValueError, match="Invalid JSON"):
            parse_and_validate("stable", "not valid json at all")
    
    def test_parse_and_validate_schema_violation(self):
        """Test handling of schema violations."""
        invalid_schema = {
            "summary": "test",
            # Missing required risk_level
            "key_findings": [],
            "recommendations": [],
            "meta": {"channel": "stable", "style": "conservative", "constraints": []}
        }
        
        with pytest.raises(ValueError, match="Schema validation failed"):
            parse_and_validate("stable", json.dumps(invalid_schema))


class TestRAGCitations:
    """Test RAG context formatting and citation extraction."""
    
    def test_format_context_empty(self):
        """Test formatting empty context."""
        context_text, citations = format_context([])
        
        assert context_text == ""
        assert citations == []
    
    def test_format_context_with_snippets(self):
        """Test formatting context with snippets."""
        snippets = [
            {
                "file": "app.py",
                "start": 10,
                "end": 15,
                "text": "def process_data(data):\n    return data.strip()"
            },
            {
                "file": "utils.py", 
                "start": 5,
                "end": 8,
                "text": "def helper():\n    pass"
            }
        ]
        
        context_text, citations = format_context(snippets)
        
        assert "app.py:10-15" in context_text
        assert "utils.py:5-8" in context_text
        assert "def process_data" in context_text
        assert len(citations) == 2
        assert citations[0]["source"] == "app.py:10-15"
        assert citations[0]["reason"] == "retrieved via RAG"
    
    def test_extract_citations_from_text(self):
        """Test extracting citations from text."""
        text_with_refs = "The issue is in app.py:45 where we have utils.py:123-125 calling main.js:10"
        citations = extract_citations_from_text(text_with_refs)
        
        assert len(citations) == 3
        assert any(c["source"] == "app.py:45" for c in citations)
        assert any(c["source"] == "utils.py:123-125" for c in citations)
        assert any(c["source"] == "main.js:10" for c in citations)
    
    def test_merge_citations(self):
        """Test merging citation lists."""
        citations1 = [
            {"source": "app.py:10", "reason": "test1"},
            {"source": "utils.py:20", "reason": "test2"}
        ]
        citations2 = [
            {"source": "app.py:10", "reason": "duplicate"},
            {"source": "main.py:30", "reason": "test3"}
        ]
        
        merged = merge_citations(citations1, citations2)
        
        assert len(merged) == 3  # Duplicate removed
        sources = [c["source"] for c in merged]
        assert "app.py:10" in sources
        assert "utils.py:20" in sources
        assert "main.py:30" in sources


class TestEngineIntegration:
    """Test engine integration with structured mode."""
    
    @patch('backend.ai.engines.stable.requests.post')
    def test_stable_engine_structured_mode(self, mock_post):
        """Test stable engine in structured mode."""
        # Mock Ollama response
        mock_response = Mock()
        mock_response.json.return_value = {
            "message": {
                "content": json.dumps({
                    "summary": "Review completed",
                    "risk_level": "low",
                    "key_findings": [],
                    "recommendations": [],
                    "meta": {
                        "channel": "stable",
                        "style": "conservative",
                        "constraints": ["offline", "no-paid-apis", "deterministic-json"]
                    }
                })
            }
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response
        
        # Import and test engine
        from backend.ai.engines.stable import review
        
        result = review(
            text="def test(): pass",
            context=[],
            meta={"use_rag": False}
        )
        
        assert result["summary"] == "Review completed"
        assert result["risk_level"] == "low"
        assert result["meta"]["channel"] == "stable"
        assert "model" in result["meta"]
        assert "prompt_version" in result["meta"]
    
    @patch('backend.ai.engines.next.requests.post')
    def test_next_engine_structured_mode(self, mock_post):
        """Test next engine in structured mode."""
        # Mock Ollama response
        mock_response = Mock()
        mock_response.json.return_value = {
            "message": {
                "content": json.dumps({
                    "summary": "Experimental review",
                    "risk_level": "medium",
                    "experiments_considered": [],
                    "key_findings": [],
                    "recommendations": [],
                    "guardrails": [],
                    "meta": {
                        "channel": "next",
                        "style": "experimental",
                        "constraints": ["offline", "no-paid-apis", "deterministic-json"]
                    }
                })
            }
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response
        
        from backend.ai.engines.next import review
        
        result = review(
            text="def test(): pass",
            context=[],
            meta={"use_rag": False, "temperature": 0.3}
        )
        
        assert result["summary"] == "Experimental review"
        assert result["meta"]["channel"] == "next"
    
    @patch('backend.ai.engines.legacy.requests.post')
    def test_legacy_engine_structured_mode(self, mock_post):
        """Test legacy engine in structured mode."""
        # Mock Ollama response
        mock_response = Mock()
        mock_response.json.return_value = {
            "message": {
                "content": json.dumps({
                    "summary": "Quick review",
                    "risk_level": "low",
                    "key_findings": [],
                    "recommendations": [],
                    "meta": {
                        "channel": "legacy",
                        "style": "baseline-fast",
                        "constraints": ["offline", "no-paid-apis", "deterministic-json"]
                    }
                })
            }
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response
        
        from backend.ai.engines.legacy import review
        
        result = review(
            text="def test(): pass",
            context=[],
            meta={"use_rag": False}
        )
        
        assert result["summary"] == "Quick review"
        assert result["meta"]["channel"] == "legacy"


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    @patch('backend.ai.engines.stable.requests.post')
    def test_json_repair_mechanism(self, mock_post):
        """Test JSON repair mechanism on malformed output."""
        # First call returns malformed JSON, second call returns valid JSON
        malformed_response = Mock()
        malformed_response.json.return_value = {
            "message": {
                "content": '{"summary": "test", "risk_level": "low",}'  # Trailing comma
            }
        }
        malformed_response.raise_for_status.return_value = None
        
        valid_response = Mock()
        valid_response.json.return_value = {
            "message": {
                "content": json.dumps({
                    "summary": "test",
                    "risk_level": "low",
                    "key_findings": [],
                    "recommendations": [],
                    "meta": {
                        "channel": "stable",
                        "style": "conservative",
                        "constraints": ["offline", "no-paid-apis", "deterministic-json"]
                    }
                })
            }
        }
        valid_response.raise_for_status.return_value = None
        
        mock_post.side_effect = [malformed_response, valid_response]
        
        from backend.ai.engines.stable import review
        
        # Should succeed after retry
        result = review(
            text="def test(): pass",
            context=[],
            meta={"use_rag": False}
        )
        
        assert result["summary"] == "test"
        assert mock_post.call_count == 2  # Initial call + retry
    
    @patch('backend.ai.engines.stable.requests.post')
    def test_engine_request_failure(self, mock_post):
        """Test handling of request failures."""
        mock_post.side_effect = Exception("Network error")
        
        from backend.ai.engines.stable import review
        
        with pytest.raises(ValueError, match="Failed to call Ollama"):
            review(
                text="def test(): pass",
                context=[],
                meta={"use_rag": False}
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])