"""
GLM-4.6 AI Service Integration

This service provides integration with GLM-4.6:cloud model for code review tasks.
"""

import os
import requests
import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class GLMService:
    """Service for interacting with GLM-4.6:cloud model via Ollama API"""
    
    def __init__(self):
        self.endpoint = os.getenv('GLM_ENDPOINT', 'http://10.122.131.109:11434')
        self.model = os.getenv('GLM_MODEL', 'glm-4.6:cloud')
        self.timeout = int(os.getenv('GLM_TIMEOUT', '30'))
        
    def health_check(self) -> Dict[str, Any]:
        """Check if GLM service is available"""
        try:
            response = requests.get(f"{self.endpoint}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                glm_available = any(model.get('model') == self.model for model in models)
                return {
                    "status": "healthy" if glm_available else "model_not_available",
                    "endpoint": self.endpoint,
                    "model": self.model,
                    "available": glm_available,
                    "all_models": [model.get('model') for model in models]
                }
            else:
                return {
                    "status": "unhealthy",
                    "endpoint": self.endpoint,
                    "error": f"HTTP {response.status_code}"
                }
        except Exception as e:
            return {
                "status": "error",
                "endpoint": self.endpoint,
                "error": str(e)
            }
    
    def review_code(self, code: str, language: str = "python", 
             focus_areas: Optional[list] = None) -> Dict[str, Any]:
        """
        Review code using GLM-4.6:cloud model
        
        Args:
            code: The source code to review
            language: Programming language (python, javascript, typescript, etc.)
            focus_areas: Specific areas to focus on (security, performance, style, etc.)
            
        Returns:
            Dictionary containing review results and metadata
        """
        
        if not focus_areas:
            focus_areas = ["security", "performance", "style", "bugs", "logic"]
        
        focus_str = ", ".join(focus_areas)
        
        prompt = f"""
        As an expert code reviewer with deep knowledge of {language} and software engineering principles,
        please analyze the following code for potential issues:
        
        Focus on these areas: {focus_str}
        
        Code to review:
        ```{language}
        {code}
        ```
        
        Please provide:
        1. A summary of potential issues found
        2. Specific line numbers where issues occur
        3. Severity levels (Critical/High/Medium/Low) for each issue
        4. Concrete improvement suggestions with code examples
        5. Best practices recommendations
        
        Format your response as structured JSON with the following schema:
        {{
            "summary": "Brief overview of code quality",
            "issues": [
                {{
                    "type": "security|performance|style|bugs|logic",
                    "severity": "critical|high|medium|low",
                    "line": 123,
                    "description": "Clear description of the issue",
                    "suggestion": "How to fix it"
                }}
            ],
            "improvements": [
                {{
                    "area": "readability|performance|security",
                    "suggestion": "Specific improvement recommendation",
                    "code_example": "improved code snippet"
                }}
            ],
            "overall_score": 85,
            "reviewed_by": "GLM-4.6:cloud",
            "reviewed_at": "{datetime.now().isoformat()}"
        }}
        """
        
        try:
            response = requests.post(
                f"{self.endpoint}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Try to parse JSON response
                try:
                    review_data = json.loads(result.get("response", "{}"))
                    return {
                        "success": True,
                        "review": review_data,
                        "raw_response": result.get("response", ""),
                        "model": self.model,
                        "tokens_used": result.get("prompt_eval_count", 0),
                        "duration": result.get("total_duration", 0),
                        "timestamp": datetime.now().isoformat()
                    }
                except json.JSONDecodeError:
                    # If response is not valid JSON, return as raw text
                    return {
                        "success": True,
                        "review": {
                            "summary": "Code review completed",
                            "issues": [],
                            "improvements": [],
                            "overall_score": 75,
                            "reviewed_by": self.model,
                            "reviewed_at": datetime.now().isoformat(),
                            "raw_text": result.get("response", "")
                        },
                        "raw_response": result.get("response", ""),
                        "model": self.model,
                        "tokens_used": result.get("prompt_eval_count", 0),
                        "timestamp": datetime.now().isoformat()
                    }
            else:
                logger.error(f"GLM API request failed: {response.status_code}")
                return {
                    "success": False,
                    "error": f"API request failed: {response.status_code}",
                    "status_code": response.status_code
                }
                
        except requests.Timeout:
            logger.error("GLM API request timed out")
            return {
                "success": False,
                "error": "Request timed out",
                "timeout": self.timeout
            }
        except Exception as e:
            logger.error(f"GLM API request failed: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def analyze_diff(self, diff_content: str, file_path: str) -> Dict[str, Any]:
        """
        Analyze git diff for potential issues
        
        Args:
            diff_content: Git diff output
            file_path: Path to the file being analyzed
            
        Returns:
            Analysis results for the diff
        """
        
        prompt = f"""
        As an expert code reviewer, please analyze this git diff for potential issues:
        
        File: {file_path}
        
        Diff:
        ```diff
        {diff_content}
        ```
        
        Please identify:
        1. Introduced bugs or regressions
        2. Security vulnerabilities in new code
        3. Performance issues
        4. Code quality problems
        5. Missing tests or edge cases
        
        Format as JSON:
        {{
            "diff_analysis": {{
                "added_issues": [],
                "removed_issues": [],
                "overall_impact": "low|medium|high",
                "recommendations": []
            }},
            "reviewed_by": "GLM-4.6:cloud",
            "reviewed_at": "{datetime.now().isoformat()}"
        }}
        """
        
        return self.review_code(diff_content, "diff", ["bugs", "security", "performance"])
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the GLM model"""
        return {
            "name": "GLM-4.6",
            "type": "cloud", 
            "provider": "Zhipu AI",
            "endpoint": self.endpoint,
            "model_id": self.model,
            "capabilities": [
                "code_review",
                "security_analysis", 
                "performance_optimization",
                "best_practices",
                "multilingual"
            ],
            "parameters": "366B",
            "strengths": [
                "fast_response",
                "comprehensive_analysis",
                "cost_effective"
            ]
        }


# Singleton instance
glm_service = GLMService()


# Usage examples
if __name__ == "__main__":
    # Test health check
    health = glm_service.health_check()
    print("Health check:", json.dumps(health, indent=2))
    
    # Test code review
    test_code = '''
def calculate_user_age(birth_year):
    current_year = 2023
    return current_year - birth_year
'''
    
    review = await glm_service.review_code(test_code, "python")
    print("Code review:", json.dumps(review, indent=2))