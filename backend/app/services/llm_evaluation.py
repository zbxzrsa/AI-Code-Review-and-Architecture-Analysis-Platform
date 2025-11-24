"""
LLM Evaluation Harness for AI Code Review Platform.
Provides prompt templates, evaluation metrics, and guardrails.
"""

import os
import json
import time
import hashlib
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import jinja2
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM


class PromptType(str, Enum):
    """Types of prompts for LLM evaluation."""
    
    CODE_ANALYSIS = "code_analysis"
    SECURITY_REVIEW = "security_review"
    PERFORMANCE_ANALYSIS = "performance_analysis"
    CODE_REFACTORING = "code_refactoring"
    DOCUMENTATION_GENERATION = "documentation_generation"
    BUG_DETECTION = "bug_detection"
    CODE_COMPLETION = "code_completion"
    EXPLANATION = "explanation"


class EvaluationMetric(str, Enum):
    """Evaluation metrics for LLM responses."""
    
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    BLEU_SCORE = "bleu_score"
    ROUGE_SCORE = "rouge_score"
    RESPONSE_TIME = "response_time"
    TOKEN_COUNT = "token_count"
    COHERENCE = "coherence"
    RELEVANCE = "relevance"
    SAFETY = "safety"


@dataclass
class PromptTemplate:
    """Prompt template configuration."""
    
    name: str
    type: PromptType
    template: str
    variables: List[str]
    description: str
    version: str = "1.0"
    examples: Optional[List[Dict[str, Any]]] = None
    guardrails: Optional[List[str]] = None


@dataclass
class EvaluationResult:
    """Result of LLM evaluation."""
    
    model_name: str
    prompt_type: PromptType
    input_data: Dict[str, Any]
    output: str
    metrics: Dict[str, float]
    guardrails_violations: List[str]
    evaluation_time: float
    timestamp: str


@dataclass
class GoldenSet:
    """Golden dataset for evaluation."""
    
    name: str
    description: str
    examples: List[Dict[str, Any]]
    metrics_weights: Dict[str, float]


class PromptManager:
    """Manage prompt templates with versioning."""
    
    def __init__(self, templates_dir: str = "prompts"):
        self.templates_dir = Path(templates_dir)
        self.templates_dir.mkdir(exist_ok=True)
        self.env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(self.templates_dir),
            autoescape=jinja2.select_autoescape(['html', 'xml'])
        )
        self._templates: Dict[str, PromptTemplate] = {}
        self._load_templates()
    
    def _load_templates(self):
        """Load all prompt templates from files."""
        for template_file in self.templates_dir.glob("*.j2"):
            try:
                with open(template_file, 'r', encoding='utf-8') as f:
                    template_data = json.load(f)
                
                template = PromptTemplate(**template_data)
                self._templates[template.name] = template
                
            except Exception as e:
                print(f"Error loading template {template_file}: {e}")
    
    def get_template(self, name: str) -> Optional[PromptTemplate]:
        """Get a prompt template by name."""
        return self._templates.get(name)
    
    def render_prompt(
        self,
        template_name: str,
        variables: Dict[str, Any]
    ) -> Optional[str]:
        """Render a prompt template with variables."""
        template = self.get_template(template_name)
        if not template:
            return None
        
        try:
            jinja_template = self.env.get_template(f"{template_name}.j2")
            return jinja_template.render(**variables)
        except Exception as e:
            print(f"Error rendering template {template_name}: {e}")
            return None
    
    def list_templates(self) -> List[PromptTemplate]:
        """List all available templates."""
        return list(self._templates.values())
    
    def save_template(self, template: PromptTemplate):
        """Save a prompt template to file."""
        template_file = self.templates_dir / f"{template.name}.j2"
        
        # Save JSON metadata
        metadata_file = self.templates_dir / f"{template.name}.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(asdict(template), f, indent=2)
        
        # Save Jinja2 template
        with open(template_file, 'w', encoding='utf-8') as f:
            f.write(template.template)
        
        self._templates[template.name] = template


class GuardrailsManager:
    """Manage LLM response guardrails."""
    
    def __init__(self):
        self.rules = {
            "no_secrets": self._check_secrets,
            "no_malicious_code": self._check_malicious_code,
            "no_pii": self._check_pii,
            "code_quality": self._check_code_quality,
            "length_limit": self._check_length_limit,
            "language_filter": self._check_language_filter,
        }
    
    def check_guardrails(
        self,
        response: str,
        active_rules: List[str] = None
    ) -> Tuple[bool, List[str]]:
        """Check response against guardrails."""
        if active_rules is None:
            active_rules = list(self.rules.keys())
        
        violations = []
        
        for rule in active_rules:
            if rule in self.rules:
                is_violated, message = self.rules[rule](response)
                if is_violated:
                    violations.append(message)
        
        return len(violations) == 0, violations
    
    def _check_secrets(self, text: str) -> Tuple[bool, str]:
        """Check for potential secrets in response."""
        secret_patterns = [
            r'password\s*=\s*["\'][^"\']+["\']',
            r'api_key\s*=\s*["\'][^"\']+["\']',
            r'secret\s*=\s*["\'][^"\']+["\']',
            r'token\s*=\s*["\'][^"\']+["\']',
        ]
        
        import re
        for pattern in secret_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True, "Potential secret detected in response"
        
        return False, ""
    
    def _check_malicious_code(self, text: str) -> Tuple[bool, str]:
        """Check for potentially malicious code patterns."""
        malicious_patterns = [
            r'eval\s*\(',
            r'exec\s*\(',
            r'system\s*\(',
            r'subprocess\.call\s*\(',
            r'os\.system\s*\(',
        ]
        
        import re
        for pattern in malicious_patterns:
            if re.search(pattern, text):
                return True, "Potentially malicious code pattern detected"
        
        return False, ""
    
    def _check_pii(self, text: str) -> Tuple[bool, str]:
        """Check for personally identifiable information."""
        pii_patterns = [
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
            r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',  # Credit card
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
        ]
        
        import re
        for pattern in pii_patterns:
            if re.search(pattern, text):
                return True, "Potential PII detected in response"
        
        return False, ""
    
    def _check_code_quality(self, text: str) -> Tuple[bool, str]:
        """Check for basic code quality issues."""
        # This is a simplified check - in practice, you'd use more sophisticated analysis
        if text.count("TODO") > 3:
            return True, "Excessive TODO comments in generated code"
        
        return False, ""
    
    def _check_length_limit(self, text: str) -> Tuple[bool, str]:
        """Check if response exceeds length limits."""
        max_length = 10000  # Configurable
        if len(text) > max_length:
            return True, f"Response exceeds maximum length of {max_length} characters"
        
        return False, ""
    
    def _check_language_filter(self, text: str) -> Tuple[bool, str]:
        """Check for inappropriate language."""
        # Simplified profanity check - in practice, use a comprehensive list
        profanity_list = ["profanity1", "profanity2"]  # Add actual words
        
        words = text.lower().split()
        for word in words:
            if word in profanity_list:
                return True, "Inappropriate language detected"
        
        return False, ""


class EvaluationMetrics:
    """Calculate evaluation metrics for LLM responses."""
    
    @staticmethod
    def calculate_bleu_score(reference: str, candidate: str) -> float:
        """Calculate BLEU score for text similarity."""
        try:
            from nltk.translate.bleu_score import sentence_bleu
            reference_tokens = reference.split()
            candidate_tokens = candidate.split()
            return sentence_bleu([reference_tokens], candidate_tokens)
        except ImportError:
            # Fallback to simple word overlap
            ref_words = set(reference.split())
            cand_words = set(candidate.split())
            overlap = len(ref_words & cand_words)
            return overlap / max(len(ref_words), len(cand_words))
    
    @staticmethod
    def calculate_rouge_score(reference: str, candidate: str) -> Dict[str, float]:
        """Calculate ROUGE scores for text similarity."""
        try:
            from rouge_score import rouge_scorer
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            scores = scorer.score(reference, candidate)
            return {
                'rouge1': scores['rouge1'].fmeasure,
                'rouge2': scores['rouge2'].fmeasure,
                'rougeL': scores['rougeL'].fmeasure,
            }
        except ImportError:
            # Fallback to simple character n-gram overlap
            return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
    
    @staticmethod
    def calculate_coherence(text: str) -> float:
        """Calculate text coherence score."""
        # Simplified coherence based on sentence structure
        sentences = text.split('.')
        if len(sentences) <= 1:
            return 1.0
        
        # Check for consistent sentence length variation
        sentence_lengths = [len(s.strip()) for s in sentences if s.strip()]
        if not sentence_lengths:
            return 0.0
        
        avg_length = sum(sentence_lengths) / len(sentence_lengths)
        variance = sum((l - avg_length) ** 2 for l in sentence_lengths) / len(sentence_lengths)
        
        # Lower variance = higher coherence (simplified)
        coherence = 1.0 / (1.0 + variance / 1000)
        return min(coherence, 1.0)
    
    @staticmethod
    def calculate_relevance(query: str, response: str) -> float:
        """Calculate relevance score between query and response."""
        # Simple keyword overlap - in practice, use semantic similarity
        query_words = set(query.lower().split())
        response_words = set(response.lower().split())
        
        if not query_words:
            return 0.0
        
        overlap = len(query_words & response_words)
        relevance = overlap / len(query_words)
        return relevance


class LLMEvaluator:
    """Main LLM evaluation harness."""
    
    def __init__(
        self,
        model_name: str,
        prompt_manager: PromptManager,
        guardrails_manager: GuardrailsManager
    ):
        self.model_name = model_name
        self.prompt_manager = prompt_manager
        self.guardrails_manager = guardrails_manager
        self.tokenizer = None
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the LLM model and tokenizer."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        except Exception as e:
            print(f"Error loading model {self.model_name}: {e}")
    
    def evaluate_on_golden_set(
        self,
        golden_set: GoldenSet,
        active_guardrails: List[str] = None
    ) -> List[EvaluationResult]:
        """Evaluate model on a golden dataset."""
        results = []
        
        for example in golden_set.examples:
            result = self.evaluate_single_example(
                example,
                golden_set.metrics_weights,
                active_guardrails
            )
            results.append(result)
        
        return results
    
    def evaluate_single_example(
        self,
        example: Dict[str, Any],
        metrics_weights: Dict[str, float],
        active_guardrails: List[str] = None
    ) -> EvaluationResult:
        """Evaluate model on a single example."""
        start_time = time.time()
        
        # Get prompt template and render
        template_name = example.get('template_name')
        variables = example.get('variables', {})
        prompt = self.prompt_manager.render_prompt(template_name, variables)
        
        if not prompt:
            raise ValueError(f"Template {template_name} not found")
        
        # Generate response
        output = self._generate_response(prompt)
        
        # Calculate metrics
        metrics = self._calculate_metrics(
            example.get('expected_output', ''),
            output,
            metrics_weights
        )
        
        # Check guardrails
        is_safe, violations = self.guardrails_manager.check_guardrails(
            output, active_guardrails
        )
        
        evaluation_time = time.time() - start_time
        
        return EvaluationResult(
            model_name=self.model_name,
            prompt_type=PromptType(example.get('prompt_type', 'code_analysis')),
            input_data=example,
            output=output,
            metrics=metrics,
            guardrails_violations=violations,
            evaluation_time=evaluation_time,
            timestamp=time.strftime('%Y-%m-%d %H:%M:%S')
        )
    
    def _generate_response(self, prompt: str) -> str:
        """Generate response from the model."""
        if not self.model or not self.tokenizer:
            return "Model not loaded"
        
        try:
            inputs = self.tokenizer.encode(prompt, return_tensors='pt')
            outputs = self.model.generate(
                inputs,
                max_length=500,
                num_return_sequences=1,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response[len(prompt):].strip()  # Remove prompt from response
            
        except Exception as e:
            return f"Error generating response: {e}"
    
    def _calculate_metrics(
        self,
        reference: str,
        candidate: str,
        weights: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate all evaluation metrics."""
        metrics = {}
        
        # Text similarity metrics
        if 'bleu_score' in weights:
            metrics['bleu_score'] = EvaluationMetrics.calculate_bleu_score(reference, candidate)
        
        if 'rouge_score' in weights:
            rouge_scores = EvaluationMetrics.calculate_rouge_score(reference, candidate)
            metrics.update(rouge_scores)
        
        # Quality metrics
        if 'coherence' in weights:
            metrics['coherence'] = EvaluationMetrics.calculate_coherence(candidate)
        
        if 'relevance' in weights:
            metrics['relevance'] = EvaluationMetrics.calculate_relevance(reference, candidate)
        
        # Basic metrics
        metrics['response_length'] = len(candidate)
        metrics['word_count'] = len(candidate.split())
        
        # Apply weights
        weighted_metrics = {}
        for metric, value in metrics.items():
            weight = weights.get(metric, 1.0)
            weighted_metrics[metric] = value * weight
        
        return weighted_metrics
    
    def generate_evaluation_report(
        self,
        results: List[EvaluationResult]
    ) -> Dict[str, Any]:
        """Generate comprehensive evaluation report."""
        if not results:
            return {"error": "No results to analyze"}
        
        # Aggregate metrics
        all_metrics = {}
        for result in results:
            for metric, value in result.metrics.items():
                if metric not in all_metrics:
                    all_metrics[metric] = []
                all_metrics[metric].append(value)
        
        # Calculate statistics
        report = {
            "model_name": self.model_name,
            "total_examples": len(results),
            "evaluation_timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
            "metrics_summary": {},
            "guardrails_summary": {},
            "performance_summary": {}
        }
        
        # Metrics statistics
        for metric, values in all_metrics.items():
            report["metrics_summary"][metric] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "min": np.min(values),
                "max": np.max(values),
                "median": np.median(values)
            }
        
        # Guardrails statistics
        total_violations = sum(len(r.guardrails_violations) for r in results)
        report["guardrails_summary"] = {
            "total_violations": total_violations,
            "violation_rate": total_violations / len(results),
            "common_violations": self._get_common_violations(results)
        }
        
        # Performance statistics
        evaluation_times = [r.evaluation_time for r in results]
        report["performance_summary"] = {
            "avg_evaluation_time": np.mean(evaluation_times),
            "total_evaluation_time": sum(evaluation_times),
            "examples_per_second": len(results) / sum(evaluation_times)
        }
        
        return report
    
    def _get_common_violations(
        self,
        results: List[EvaluationResult]
    ) -> Dict[str, int]:
        """Get most common guardrails violations."""
        violation_counts = {}
        
        for result in results:
            for violation in result.guardrails_violations:
                violation_counts[violation] = violation_counts.get(violation, 0) + 1
        
        return violation_counts


# Sample prompt templates
SAMPLE_TEMPLATES = [
    PromptTemplate(
        name="code_analysis",
        type=PromptType.CODE_ANALYSIS,
        template="""Analyze the following code for quality, security, and performance issues:

Language: {{ language }}
Code:
```{{ language }}
{{ code }}
```

Please provide:
1. Overall quality score (1-10)
2. Security vulnerabilities
3. Performance issues
4. Code style suggestions
5. Best practices recommendations

Focus on actionable feedback that would improve the code.""",
        variables=["language", "code"],
        description="Analyze code for quality, security, and performance",
        guardrails=["no_secrets", "no_malicious_code", "language_filter"]
    ),
    
    PromptTemplate(
        name="security_review",
        type=PromptType.SECURITY_REVIEW,
        template="""Perform a comprehensive security review of the following code:

Language: {{ language }}
Code:
```{{ language }}
{{ code }}
```

Identify and categorize:
1. Critical security vulnerabilities
2. Medium risk issues
3. Low risk issues
4. Security best practices violations

For each issue, provide:
- Severity level (Critical/High/Medium/Low)
- Line number (if applicable)
- Description
- Remediation steps
- Code example (if applicable)

Focus on OWASP Top 10 and common security anti-patterns.""",
        variables=["language", "code"],
        description="Comprehensive security code review",
        guardrails=["no_secrets", "no_malicious_code"]
    ),
    
    PromptTemplate(
        name="code_refactoring",
        type=PromptType.CODE_REFACTORING,
        template="""Refactor the following code to improve quality, readability, and performance:

Language: {{ language }}
Original Code:
```{{ language }}
{{ code }}
```

Requirements:
{{ requirements }}

Please provide:
1. Refactored code with improvements
2. Explanation of changes made
3. Benefits of the refactoring
4. Any trade-offs or considerations

Ensure the refactored code maintains the same functionality.""",
        variables=["language", "code", "requirements"],
        description="Refactor code for better quality and performance",
        guardrails=["no_secrets", "no_malicious_code", "code_quality"]
    )
]


def create_sample_golden_set() -> GoldenSet:
    """Create a sample golden dataset for evaluation."""
    return GoldenSet(
        name="code_review_basics",
        description="Basic code review scenarios",
        metrics_weights={
            "bleu_score": 0.3,
            "rouge1": 0.2,
            "coherence": 0.2,
            "relevance": 0.3
        },
        examples=[
            {
                "template_name": "code_analysis",
                "prompt_type": "code_analysis",
                "variables": {
                    "language": "python",
                    "code": "def add(a,b): return a+b"
                },
                "expected_output": "The function should include type hints and docstring. Consider adding input validation."
            },
            {
                "template_name": "security_review",
                "prompt_type": "security_review",
                "variables": {
                    "language": "python",
                    "code": "eval(user_input)"
                },
                "expected_output": "Critical: Use of eval() function allows arbitrary code execution. Replace with safer alternatives."
            }
        ]
    )


# Usage example
def main():
    """Example usage of the LLM evaluation harness."""
    
    # Initialize components
    prompt_manager = PromptManager()
    guardrails_manager = GuardrailsManager()
    
    # Add sample templates
    for template in SAMPLE_TEMPLATES:
        prompt_manager.save_template(template)
    
    # Create evaluator
    evaluator = LLMEvaluator(
        model_name="microsoft/DialoGPT-medium",  # Example model
        prompt_manager=prompt_manager,
        guardrails_manager=guardrails_manager
    )
    
    # Create golden set
    golden_set = create_sample_golden_set()
    
    # Run evaluation
    results = evaluator.evaluate_on_golden_set(
        golden_set,
        active_guardrails=["no_secrets", "no_malicious_code", "language_filter"]
    )
    
    # Generate report
    report = evaluator.generate_evaluation_report(results)
    
    # Save results
    with open("evaluation_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"Evaluation completed. Results saved to evaluation_report.json")
    print(f"Total examples evaluated: {len(results)}")
    print(f"Average evaluation time: {report['performance_summary']['avg_evaluation_time']:.2f}s")


if __name__ == "__main__":
    main()