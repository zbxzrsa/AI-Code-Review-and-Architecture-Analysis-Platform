#!/usr/bin/env python3
"""
AI Reviewer with Dynamic Model Loading
Supports v1/v2/v3 models with offline capabilities and chunking
"""

import logging
import time
import hashlib
from typing import Dict, Any, List, Optional, Union
import json
import asyncio
from pathlib import Path

# AI/ML imports
try:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("Transformers not available, using mock responses")

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    logging.warning("Ollama not available, local LLM support disabled")

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logging.warning("OpenAI not available, cloud LLM support disabled")

logger = logging.getLogger(__name__)

class AIReviewer:
    """AI code reviewer with dynamic model loading for v1/v2/v3"""
    
    def __init__(self, version: str = "v1", config: Optional[Dict[str, Any]] = None):
        self.version = version
        self.config = config or {}
        self.model = None
        self.tokenizer = None
        self.client = None
        self.model_name = None
        self.max_tokens = self.config.get('max_tokens', 512)
        self.temperature = self.config.get('temperature', 0.1)
        self.cache_enabled = self.config.get('cache_enabled', True)
        
        # Response cache
        self.response_cache = {}
        
        # Initialize model
        self._load_model()
        
    def _load_model(self):
        """Load model based on version"""
        try:
            if self.version == "v1":
                self._load_v1_model()
            elif self.version == "v2":
                self._load_v2_model()
            elif self.version == "v3":
                self._load_v3_model()
            else:
                raise ValueError(f"Invalid version: {self.version}")
                
            logger.info(f"Loaded {self.version} model: {self.model_name}")
            
        except Exception as e:
            logger.error(f"Failed to load {self.version} model: {e}")
            # Fallback to mock model
            self._load_mock_model()
    
    def _load_v1_model(self):
        """Load v1 stable model (CodeBERT-based)"""
        self.model_name = "microsoft/codebert-base"
        
        if OLLAMA_AVAILABLE and self.config.get('prefer_local', False):
            # Try local Ollama first
            try:
                self.client = ollama.Client()
                # Pull model if not available
                ollama.pull('codebert')
                self.model_name = "codebert"
                return
            except Exception as e:
                logger.warning(f"Ollama CodeBERT not available: {e}")
        
        if TRANSFORMERS_AVAILABLE:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
                # Set padding token
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
            except Exception as e:
                logger.warning(f"Failed to load CodeBERT: {e}")
                raise
        else:
            raise ImportError("Transformers not available for v1 model")
    
    def _load_v2_model(self):
        """Load v2 experimental model (StarCoder-based)"""
        self.model_name = "bigcode/starcoder"
        
        if OLLAMA_AVAILABLE:
            # Prefer local Ollama for v2
            try:
                self.client = ollama.Client()
                # Pull model if not available
                ollama.pull('starcoder')
                self.model_name = "starcoder"
                return
            except Exception as e:
                logger.warning(f"Ollama StarCoder not available: {e}")
        
        if TRANSFORMERS_AVAILABLE:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
                # Set padding token
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
            except Exception as e:
                logger.warning(f"Failed to load StarCoder: {e}")
                raise
        else:
            raise ImportError("Transformers not available for v2 model")
    
    def _load_v3_model(self):
        """Load v3 deprecated model (GPT-2 fallback)"""
        self.model_name = "gpt2"
        
        if TRANSFORMERS_AVAILABLE:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
                # Set padding token
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
            except Exception as e:
                logger.warning(f"Failed to load GPT-2: {e}")
                raise
        else:
            raise ImportError("Transformers not available for v3 model")
    
    def _load_mock_model(self):
        """Load mock model for fallback"""
        self.model_name = "mock"
        self.client = None
        self.model = None
        self.tokenizer = None
        logger.warning("Using mock model for AI review")
    
    async def review(self, code: str, language: str = "python", focus_areas: Optional[List[str]] = None) -> Dict[str, Any]:
        """Generate AI code review using the selected model"""
        start_time = time.time()
        
        # Generate cache key
        cache_key = self._generate_cache_key(code, language, focus_areas)
        
        # Check cache
        if self.cache_enabled and cache_key in self.response_cache:
            logger.debug(f"Using cached review for {language} code")
            cached_result = self.response_cache[cache_key]
            cached_result['cached'] = True
            return cached_result
        
        try:
            # Chunk code if too long
            chunks = self._chunk_code(code)
            reviews = []
            
            for chunk in chunks:
                if self.client and OLLAMA_AVAILABLE:
                    # Use Ollama (local)
                    chunk_review = await self._review_with_ollama(chunk, language, focus_areas)
                elif self.model and TRANSFORMERS_AVAILABLE:
                    # Use Hugging Face transformers
                    chunk_review = await self._review_with_transformers(chunk, language, focus_areas)
                else:
                    # Use mock
                    chunk_review = await self._review_with_mock(chunk, language, focus_areas)
                
                reviews.append(chunk_review)
            
            # Combine reviews
            combined_review = self._combine_reviews(reviews, code, language)
            
            # Add metadata
            combined_review.update({
                'version': self.version,
                'model_used': self.model_name,
                'processing_time': time.time() - start_time,
                'chunks_processed': len(chunks),
                'cached': False,
                'focus_areas': focus_areas or []
            })
            
            # Cache result
            if self.cache_enabled:
                self.response_cache[cache_key] = combined_review
            
            return combined_review
            
        except Exception as e:
            logger.error(f"AI review failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'version': self.version,
                'model_used': self.model_name,
                'processing_time': time.time() - start_time,
                'cached': False
            }
    
    def _chunk_code(self, code: str, max_length: int = 400) -> List[str]:
        """Split code into chunks to handle token limits"""
        lines = code.splitlines()
        chunks = []
        current_chunk = []
        current_length = 0
        
        for line in lines:
            # Rough estimation of tokens (1 token ≈ 4 characters)
            line_tokens = len(line) // 4 + 1
            
            if current_length + line_tokens > max_length and current_chunk:
                # Start new chunk
                chunks.append('\n'.join(current_chunk))
                current_chunk = [line]
                current_length = line_tokens
            else:
                current_chunk.append(line)
                current_length += line_tokens
        
        # Add final chunk
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
        
        return chunks
    
    async def _review_with_ollama(self, code: str, language: str, focus_areas: Optional[List[str]]) -> Dict[str, Any]:
        """Review code using Ollama local LLM"""
        try:
            prompt = self._build_prompt(code, language, focus_areas)
            
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, 
                lambda: self.client.generate(
                    model=self.model_name,
                    prompt=prompt,
                    options={
                        'temperature': self.temperature,
                        'num_predict': self.max_tokens
                    }
                )
            )
            
            review_text = response.get('response', '')
            return self._parse_review_response(review_text, code)
            
        except Exception as e:
            logger.error(f"Ollama review failed: {e}")
            raise
    
    async def _review_with_transformers(self, code: str, language: str, focus_areas: Optional[List[str]]) -> Dict[str, Any]:
        """Review code using Hugging Face transformers"""
        try:
            prompt = self._build_prompt(code, language, focus_areas)
            
            # Tokenize input
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=self.max_tokens
            )
            
            # Generate response
            loop = asyncio.get_event_loop()
            outputs = await loop.run_in_executor(
                None,
                lambda: self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=self.temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            )
            
            # Decode response
            review_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Remove the prompt part
            if prompt in review_text:
                review_text = review_text.replace(prompt, '').strip()
            
            return self._parse_review_response(review_text, code)
            
        except Exception as e:
            logger.error(f"Transformers review failed: {e}")
            raise
    
    async def _review_with_mock(self, code: str, language: str, focus_areas: Optional[List[str]]) -> Dict[str, Any]:
        """Mock review for testing/fallback"""
        await asyncio.sleep(0.1)  # Simulate processing time
        
        lines = code.splitlines()
        issues = []
        
        # Simple mock analysis
        for i, line in enumerate(lines, 1):
            if 'password' in line.lower() and '=' in line:
                issues.append({
                    'type': 'security',
                    'severity': 'high',
                    'line': i,
                    'message': 'Potential hardcoded password detected',
                    'suggestion': 'Use environment variables or secure storage'
                })
            elif len(line) > 120:
                issues.append({
                    'type': 'style',
                    'severity': 'low',
                    'line': i,
                    'message': 'Line too long',
                    'suggestion': 'Break line into multiple lines'
                })
        
        return {
            'success': True,
            'issues': issues,
            'summary': f"Found {len(issues)} issues in {len(lines)} lines of {language} code",
            'score': max(0, 100 - len(issues) * 5),
            'suggestions': ['Consider adding more documentation', 'Review security practices']
        }
    
    def _build_prompt(self, code: str, language: str, focus_areas: Optional[List[str]]) -> str:
        """Build prompt for AI model"""
        focus_text = ""
        if focus_areas:
            focus_text = f"Focus on: {', '.join(focus_areas)}. "
        
        prompt = f"""Please review this {language} code. {focus_text}Provide specific feedback on:

1. Security vulnerabilities
2. Code quality issues
3. Performance concerns
4. Best practices violations
5. Potential bugs

Code to review:
```{language}
{code}
```

Provide your review in JSON format:
{{
    "issues": [
        {{
            "type": "security|quality|performance|style",
            "severity": "low|medium|high|critical",
            "line": line_number,
            "message": "description of issue",
            "suggestion": "how to fix"
        }}
    ],
    "summary": "overall summary",
    "score": 0-100,
    "suggestions": ["general improvement suggestions"]
}}
"""
        return prompt
    
    def _parse_review_response(self, review_text: str, original_code: str) -> Dict[str, Any]:
        """Parse AI review response"""
        try:
            # Try to extract JSON from response
            start_idx = review_text.find('{')
            end_idx = review_text.rfind('}') + 1
            
            if start_idx != -1 and end_idx != -1:
                json_str = review_text[start_idx:end_idx]
                parsed = json.loads(json_str)
                
                # Ensure required fields
                parsed.setdefault('success', True)
                parsed.setdefault('issues', [])
                parsed.setdefault('summary', 'Review completed')
                parsed.setdefault('score', 75)
                parsed.setdefault('suggestions', [])
                
                return parsed
            else:
                # Fallback: treat as text response
                return {
                    'success': True,
                    'issues': [],
                    'summary': review_text[:200] + '...' if len(review_text) > 200 else review_text,
                    'score': 75,
                    'suggestions': [],
                    'raw_response': review_text
                }
                
        except json.JSONDecodeError:
            # Fallback for non-JSON responses
            return {
                'success': True,
                'issues': [],
                'summary': review_text[:200] + '...' if len(review_text) > 200 else review_text,
                'score': 75,
                'suggestions': [],
                'raw_response': review_text
            }
    
    def _combine_reviews(self, reviews: List[Dict[str, Any]], original_code: str, language: str) -> Dict[str, Any]:
        """Combine multiple chunk reviews into one"""
        if not reviews:
            return {
                'success': False,
                'error': 'No reviews to combine'
            }
        
        # Merge all issues
        all_issues = []
        total_score = 0
        summaries = []
        all_suggestions = set()
        
        for review in reviews:
            if review.get('success', False):
                issues = review.get('issues', [])
                # Adjust line numbers for chunks
                chunk_start = self._get_chunk_line_offset(original_code, review)
                for issue in issues:
                    if 'line' in issue:
                        issue['line'] += chunk_start
                all_issues.extend(issues)
                
                total_score += review.get('score', 75)
                summaries.append(review.get('summary', ''))
                suggestions = review.get('suggestions', [])
                all_suggestions.update(suggestions)
        
        # Calculate average score
        avg_score = total_score // len(reviews) if reviews else 75
        
        # Remove duplicate issues
        unique_issues = self._deduplicate_issues(all_issues)
        
        # Combine summaries
        combined_summary = ' | '.join(filter(None, summaries))
        
        return {
            'success': True,
            'issues': unique_issues,
            'summary': combined_summary or f"Review of {language} code completed",
            'score': avg_score,
            'suggestions': list(all_suggestions)
        }
    
    def _get_chunk_line_offset(self, original_code: str, review: Dict[str, Any]) -> int:
        """Calculate line offset for a chunk (simplified)"""
        # This is a simplified implementation
        # In practice, you'd need to track chunk boundaries
        return 0
    
    def _deduplicate_issues(self, issues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate issues"""
        seen = set()
        unique_issues = []
        
        for issue in issues:
            # Create a key for deduplication
            key = (
                issue.get('line', 0),
                issue.get('type', ''),
                issue.get('message', '')[:50]  # First 50 chars
            )
            
            if key not in seen:
                seen.add(key)
                unique_issues.append(issue)
        
        return unique_issues
    
    def _generate_cache_key(self, code: str, language: str, focus_areas: Optional[List[str]]) -> str:
        """Generate cache key for review results"""
        content = f"{self.version}:{language}:{code}:{sorted(focus_areas or [])}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def clear_cache(self):
        """Clear response cache"""
        self.response_cache.clear()
        logger.info("AI reviewer cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            'cache_size': len(self.response_cache),
            'cache_enabled': self.cache_enabled,
            'model_name': self.model_name,
            'version': self.version
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Check health of AI reviewer"""
        status = {
            'version': self.version,
            'model_name': self.model_name,
            'status': 'healthy',
            'capabilities': []
        }
        
        if OLLAMA_AVAILABLE and self.client:
            status['capabilities'].append('local_llm')
        
        if TRANSFORMERS_AVAILABLE and self.model:
            status['capabilities'].append('transformers')
        
        if not OLLAMA_AVAILABLE and not TRANSFORMERS_AVAILABLE:
            status['status'] = 'degraded'
            status['message'] = 'Using mock responses only'
        
        return status

class ReviewService:
    """Unified service combining AI reviewer and static analysis"""
    
    def __init__(self, version: str = "v1", config: Optional[Dict[str, Any]] = None):
        self.version = version
        self.config = config or {}
        self.ai_reviewer = AIReviewer(version, config)
        
        # Import static analyzer
        try:
            from .static_analyzer import StaticAnalyzer
            self.static_analyzer = StaticAnalyzer(config.get('static_analysis', {}))
            self.static_available = True
        except ImportError:
            self.static_analyzer = None
            self.static_available = False
            logger.warning("Static analyzer not available")
    
    async def review(self, code: str, language: str = "python", 
                   focus_areas: Optional[List[str]] = None,
                   include_static: bool = True) -> Dict[str, Any]:
        """Generate comprehensive review with AI and static analysis"""
        start_time = time.time()
        
        # AI review
        ai_review = await self.ai_reviewer.review(code, language, focus_areas)
        
        # Static analysis
        static_issues = []
        if include_static and self.static_available and self.static_analyzer:
            try:
                # For static analysis, we need a file path
                # Create a temporary file path
                import tempfile
                with tempfile.NamedTemporaryFile(mode='w', suffix=f'.{language}', delete=False) as f:
                    f.write(code)
                    temp_path = f.name
                
                static_issues = self.static_analyzer.analyze_file(temp_path)
                
                # Clean up
                import os
                os.unlink(temp_path)
                
            except Exception as e:
                logger.warning(f"Static analysis failed: {e}")
        
        # Combine results
        combined_result = {
            'success': ai_review.get('success', True),
            'ai_review': ai_review,
            'static_issues': static_issues,
            'total_issues': len(ai_review.get('issues', [])) + len(static_issues),
            'version': self.version,
            'processing_time': time.time() - start_time,
            'language': language,
            'focus_areas': focus_areas or []
        }
        
        # Calculate combined score
        ai_score = ai_review.get('score', 75)
        static_penalty = min(20, len(static_issues) * 2)
        combined_result['combined_score'] = max(0, ai_score - static_penalty)
        
        return combined_result
    
    def health_check(self) -> Dict[str, Any]:
        """Health check for the review service"""
        ai_health = self.ai_reviewer.health_check()
        
        status = {
            'service': 'review_service',
            'version': self.version,
            'status': ai_health['status'],
            'ai_reviewer': ai_health,
            'static_analyzer': {
                'available': self.static_available,
                'status': 'healthy' if self.static_available else 'unavailable'
            }
        }
        
        if not self.static_available:
            status['status'] = 'degraded'
            status['message'] = 'Static analyzer unavailable'
        
        return status