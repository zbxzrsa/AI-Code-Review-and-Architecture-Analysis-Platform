import re
import hashlib
import time
from typing import Dict, List, Any, Tuple
from datetime import datetime

class SecurityFilter:
    def __init__(self):
        # PII detection patterns
        self.email_pattern = re.compile(
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            re.IGNORECASE
        )
        
        self.phone_pattern = re.compile(
            r'(?:(?:\+?\d{1,3})?[-. (]?\d{3}[-. )]\d{3}[-. ]?\d{4})',
            re.IGNORECASE
        )
        
        self.api_key_pattern = re.compile(
            r'(?i)(api[-_]?key|secret|token|password|private[_-]?key)\s*[:=]\s*[\'"]?[A-Za-z0-9\-_+/]{12,}[\'"]?',
            re.IGNORECASE
        )
        
        self.ip_pattern = re.compile(
            r'\b(?:\d{1,3}\.){3}\d{1,3}\b|\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b'
        )
        
        self.credit_card_pattern = re.compile(
            r'\b(?:\d{4}[-\s]?){3}\d{4}\b',
            re.IGNORECASE
        )
        
        self.ssn_pattern = re.compile(
            r'\b\d{3}[-]?\d{2}[-]?\d{4}\b'
        )
        
        # Output guard patterns (phrases that should trigger alerts)
        self.deny_list_phrases = [
            'send to external',
            'upload code to',
            'exfiltrate data',
            'bypass security',
            'ignore authentication',
            'disable logging',
            'hide evidence',
            'delete logs',
            'transmit sensitive',
            'leak information',
            'circumvent controls'
        ]
        
        # Risky code patterns
        self.risky_patterns = [
            (r'eval\s*\(', 'Dynamic code execution'),
            (r'exec\s*\(', 'Code execution'),
            (r'system\s*\(', 'System command execution'),
            (r'shell_exec\s*\(', 'Shell execution'),
            (r'passthru\s*\(', 'Command passthrough'),
            (r'file_get_contents\s*\(', 'File read operation'),
            (r'file_put_contents\s*\(', 'File write operation'),
            (r'curl_exec\s*\(', 'URL execution'),
            (r'base64_decode\s*\(', 'Obfuscation attempt'),
        ]
        
        for pattern, desc in self.risky_patterns:
            self.risky_patterns.append((re.compile(pattern, re.IGNORECASE), desc))
    
    def redact_pii(self, text: str) -> Tuple[str, Dict[str, int]]:
        """Redact PII from text and return redaction report"""
        redacted_text = text
        redaction_counts = {}
        
        # Redact emails
        matches = self.email_pattern.findall(redacted_text)
        if matches:
            redaction_counts['emails'] = len(matches)
            redacted_text = self.email_pattern.sub('[EMAIL_REDACTED]', redacted_text)
        
        # Redact phone numbers
        matches = self.phone_pattern.findall(redacted_text)
        if matches:
            redaction_counts['phone_numbers'] = len(matches)
            redacted_text = self.phone_pattern.sub('[PHONE_REDACTED]', redacted_text)
        
        # Redact API keys and secrets
        matches = self.api_key_pattern.findall(redacted_text)
        if matches:
            redaction_counts['api_keys'] = len(matches)
            redacted_text = self.api_key_pattern.sub('[SECRET_REDACTED]', redacted_text)
        
        # Redact IP addresses
        matches = self.ip_pattern.findall(redacted_text)
        if matches:
            redaction_counts['ip_addresses'] = len(matches)
            redacted_text = self.ip_pattern.sub('[IP_REDACTED]', redacted_text)
        
        # Redact credit card numbers
        matches = self.credit_card_pattern.findall(redacted_text)
        if matches:
            redaction_counts['credit_cards'] = len(matches)
            redacted_text = self.credit_card_pattern.sub('[CARD_REDACTED]', redacted_text)
        
        # Redact SSNs
        matches = self.ssn_pattern.findall(redacted_text)
        if matches:
            redaction_counts['ssns'] = len(matches)
            redacted_text = self.ssn_pattern.sub('[SSN_REDACTED]', redacted_text)
        
        return redacted_text, redaction_counts
    
    def check_output_guards(self, text: str) -> List[Dict[str, str]]:
        """Check for denied phrases in output"""
        violations = []
        text_lower = text.lower()
        
        for phrase in self.deny_list_phrases:
            if phrase in text_lower:
                violations.append({
                    'type': 'deny_phrase',
                    'phrase': phrase,
                    'severity': 'high',
                    'description': f'Potentially dangerous phrase detected: {phrase}'
                })
        
        return violations
    
    def check_risky_patterns(self, text: str) -> List[Dict[str, str]]:
        """Check for risky code patterns"""
        violations = []
        
        for pattern, description in self.risky_patterns:
            matches = pattern.findall(text)
            if matches:
                violations.append({
                    'type': 'risky_pattern',
                    'pattern': pattern.pattern,
                    'description': description,
                    'matches': len(matches),
                    'severity': 'medium'
                })
        
        return violations
    
    def filter_input(self, text: str) -> Dict[str, Any]:
        """Apply all input filters"""
        # Redact PII
        redacted_text, redaction_counts = self.redact_pii(text)
        
        # Check for risky patterns
        risky_violations = self.check_risky_patterns(text)
        
        # Generate content hash
        content_hash = hashlib.sha256(text.encode()).hexdigest()
        
        return {
            'original_text': text,
            'filtered_text': redacted_text,
            'redactions_applied': redaction_counts,
            'risky_violations': risky_violations,
            'content_hash': content_hash,
            'filtered_at': datetime.utcnow().isoformat()
        }
    
    def filter_output(self, text: str) -> Dict[str, Any]:
        """Apply output filters"""
        # Check for deny phrases
        deny_violations = self.check_output_guards(text)
        
        # Add safety warnings if risky content detected
        if deny_violations:
            text = "\n⚠️  SECURITY WARNING ⚠️\n" + \
                   "The following potentially dangerous content was detected:\n"
            for violation in deny_violations:
                text += f"- {violation['description']}\n"
            text += "\nPlease review and ensure this is safe before proceeding.\n\n" + \
                   text
        
        return {
            'original_output': text,
            'deny_violations': deny_violations,
            'safety_warnings': len(deny_violations) > 0,
            'filtered_at': datetime.utcnow().isoformat()
        }

# Global security filter instance
security_filter = SecurityFilter()