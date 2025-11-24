"""
安全优化模块
提供认证、授权、加密、输入验证和安全监控功能
"""
import asyncio
import hashlib
import hmac
import secrets
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Union, Callable, Set
from enum import Enum
import re
import json
import base64
from functools import wraps
import logging

# 尝试导入安全相关依赖
try:
    import bcrypt
    import jwt
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    import rsa
    SECURITY_LIBS_AVAILABLE = True
except ImportError:
    SECURITY_LIBS_AVAILABLE = False

try:
    import bleach
    HTML_SANITIZER_AVAILABLE = True
except ImportError:
    HTML_SANITIZER_AVAILABLE = False

logger = logging.getLogger(__name__)

class SecurityLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ThreatType(Enum):
    SQL_INJECTION = "sql_injection"
    XSS = "xss"
    CSRF = "csrf"
    BRUTE_FORCE = "brute_force"
    RATE_LIMIT = "rate_limit"
    MALICIOUS_INPUT = "malicious_input"

@dataclass
class SecurityEvent:
    timestamp: datetime
    threat_type: ThreatType
    source_ip: str
    user_id: Optional[str]
    description: str
    severity: SecurityLevel
    blocked: bool = True

@dataclass
class SecurityConfig:
    max_login_attempts: int = 5
    lockout_duration_minutes: int = 15
    session_timeout_minutes: int = 30
    password_min_length: int = 8
    password_require_special_chars: bool = True
    enable_rate_limiting: bool = True
    rate_limit_requests_per_minute: int = 60
    enable_input_sanitization: bool = True
    enable_encryption: bool = True
    jwt_secret_key: Optional[str] = None
    jwt_algorithm: str = "HS256"
    jwt_expiration_hours: int = 24

class PasswordValidator:
    """密码验证器"""
    
    @staticmethod
    def validate_password(password: str, config: SecurityConfig) -> List[str]:
        """验证密码强度"""
        errors = []
        
        if len(password) < config.password_min_length:
            errors.append(f"Password must be at least {config.password_min_length} characters long")
        
        if config.password_require_special_chars:
            if not re.search(r'[A-Z]', password):
                errors.append("Password must contain at least one uppercase letter")
            if not re.search(r'[a-z]', password):
                errors.append("Password must contain at least one lowercase letter")
            if not re.search(r'\d', password):
                errors.append("Password must contain at least one digit")
            if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
                errors.append("Password must contain at least one special character")
        
        # 检查常见弱密码
        weak_passwords = [
            'password', '123456', 'qwerty', 'admin', 'letmein',
            'welcome', 'monkey', 'dragon', 'master', 'hello'
        ]
        if password.lower() in weak_passwords:
            errors.append("Password is too common")
        
        return errors
    
    @staticmethod
    def hash_password(password: str) -> str:
        """哈希密码"""
        if SECURITY_LIBS_AVAILABLE:
            salt = bcrypt.gensalt()
            return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')
        else:
            # 回退到简单哈希
            return hashlib.sha256(password.encode()).hexdigest()
    
    @staticmethod
    def verify_password(password: str, hashed: str) -> bool:
        """验证密码"""
        if SECURITY_LIBS_AVAILABLE:
            return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))
        else:
            return hashlib.sha256(password.encode()).hexdigest() == hashed

class InputSanitizer:
    """输入清理器"""
    
    @staticmethod
    def sanitize_html(html: str) -> str:
        """清理HTML输入"""
        if HTML_SANITIZER_AVAILABLE:
            allowed_tags = ['p', 'br', 'strong', 'em', 'ul', 'ol', 'li']
            allowed_attributes = {'*': ['class']}
            return bleach.clean(
                html, 
                tags=allowed_tags, 
                attributes=allowed_attributes,
                strip=True
            )
        else:
            # 简单的HTML标签移除
            return re.sub(r'<[^>]+>', '', html)
    
    @staticmethod
    def sanitize_sql(input_str: str) -> str:
        """清理SQL输入"""
        # 移除危险字符
        dangerous_chars = ["'", '"', ';', '--', '/*', '*/', 'xp_', 'sp_']
        sanitized = input_str
        for char in dangerous_chars:
            sanitized = sanitized.replace(char, '')
        return sanitized
    
    @staticmethod
    def validate_email(email: str) -> bool:
        """验证邮箱格式"""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None
    
    @staticmethod
    def validate_url(url: str) -> bool:
        """验证URL格式"""
        pattern = r'^https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)$'
        return re.match(pattern, url) is not None

class RateLimiter:
    """速率限制器"""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.requests: Dict[str, List[float]] = {}
        self.lock = asyncio.Lock()
    
    async def is_allowed(self, identifier: str) -> bool:
        """检查是否允许请求"""
        if not self.config.enable_rate_limiting:
            return True
        
        async with self.lock:
            now = time.time()
            minute_ago = now - 60
            
            # 清理过期记录
            if identifier in self.requests:
                self.requests[identifier] = [
                    req_time for req_time in self.requests[identifier]
                    if req_time > minute_ago
                ]
            else:
                self.requests[identifier] = []
            
            # 检查是否超过限制
            if len(self.requests[identifier]) >= self.config.rate_limit_requests_per_minute:
                return False
            
            # 记录新请求
            self.requests[identifier].append(now)
            return True
    
    async def get_remaining_requests(self, identifier: str) -> int:
        """获取剩余请求数"""
        async with self.lock:
            if identifier not in self.requests:
                return self.config.rate_limit_requests_per_minute
            
            now = time.time()
            minute_ago = now - 60
            
            valid_requests = [
                req_time for req_time in self.requests[identifier]
                if req_time > minute_ago
            ]
            
            return max(0, self.config.rate_limit_requests_per_minute - len(valid_requests))

class JWTManager:
    """JWT令牌管理器"""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        if not config.jwt_secret_key:
            self.secret_key = secrets.token_urlsafe(32)
        else:
            self.secret_key = config.jwt_secret_key
    
    def generate_token(self, user_id: str, additional_claims: Dict[str, Any] = None) -> str:
        """生成JWT令牌"""
        if not SECURITY_LIBS_AVAILABLE:
            raise ImportError("PyJWT is required for JWT functionality")
        
        payload = {
            'user_id': user_id,
            'exp': datetime.now(timezone.utc) + timedelta(hours=self.config.jwt_expiration_hours),
            'iat': datetime.now(timezone.utc),
            'iss': 'ai-code-review-platform'
        }
        
        if additional_claims:
            payload.update(additional_claims)
        
        return jwt.encode(payload, self.secret_key, algorithm=self.config.jwt_algorithm)
    
    def verify_token(self, token: str) -> Dict[str, Any]:
        """验证JWT令牌"""
        if not SECURITY_LIBS_AVAILABLE:
            raise ImportError("PyJWT is required for JWT functionality")
        
        try:
            payload = jwt.decode(
                token, 
                self.secret_key, 
                algorithms=[self.config.jwt_algorithm]
            )
            return payload
        except jwt.ExpiredSignatureError:
            raise ValueError("Token has expired")
        except jwt.InvalidTokenError:
            raise ValueError("Invalid token")
    
    def refresh_token(self, token: str) -> str:
        """刷新JWT令牌"""
        payload = self.verify_token(token)
        user_id = payload.pop('user_id')
        return self.generate_token(user_id, payload)

class EncryptionManager:
    """加密管理器"""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self._cipher: Optional[Fernet] = None
        
        if config.enable_encryption and SECURITY_LIBS_AVAILABLE:
            self._init_cipher()
    
    def _init_cipher(self) -> None:
        """初始化加密器"""
        # 生成密钥
        password = secrets.token_bytes(32)
        salt = secrets.token_bytes(16)
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password))
        self._cipher = Fernet(key)
    
    def encrypt(self, data: str) -> str:
        """加密数据"""
        if not self._cipher:
            return data  # 如果加密未启用，返回原数据
        
        encrypted_data = self._cipher.encrypt(data.encode())
        return base64.urlsafe_b64encode(encrypted_data).decode()
    
    def decrypt(self, encrypted_data: str) -> str:
        """解密数据"""
        if not self._cipher:
            return encrypted_data  # 如果加密未启用，返回原数据
        
        try:
            decoded_data = base64.urlsafe_b64decode(encrypted_data.encode())
            decrypted_data = self._cipher.decrypt(decoded_data)
            return decrypted_data.decode()
        except Exception:
            raise ValueError("Failed to decrypt data")

class SecurityMonitor:
    """安全监控器"""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.events: List[SecurityEvent] = []
        self.blocked_ips: Set[str] = set()
        self.suspicious_patterns: Dict[str, re.Pattern] = {
            'sql_injection': re.compile(r'(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER)\b)', re.IGNORECASE),
            'xss': re.compile(r'<script[^>]*>.*?</script>', re.IGNORECASE | re.DOTALL),
            'path_traversal': re.compile(r'\.\.[\\/]', re.IGNORECASE),
        }
        self.lock = asyncio.Lock()
    
    async def log_event(self, event: SecurityEvent) -> None:
        """记录安全事件"""
        async with self.lock:
            self.events.append(event)
            
            # 如果事件过多，清理旧事件
            if len(self.events) > 10000:
                self.events = self.events[-5000:]
        
        # 如果是高危事件，阻止IP
        if event.severity in [SecurityLevel.HIGH, SecurityLevel.CRITICAL] and event.blocked:
            await self.block_ip(event.source_ip)
        
        logger.warning(f"Security event: {event.threat_type.value} from {event.source_ip}")
    
    async def block_ip(self, ip: str, duration_minutes: int = 60) -> None:
        """阻止IP地址"""
        self.blocked_ips.add(ip)
        
        # 设置定时解除阻止
        asyncio.create_task(self._unblock_ip_after(ip, duration_minutes))
    
    async def _unblock_ip_after(self, ip: str, duration_minutes: int) -> None:
        """定时解除IP阻止"""
        await asyncio.sleep(duration_minutes * 60)
        self.blocked_ips.discard(ip)
    
    def is_ip_blocked(self, ip: str) -> bool:
        """检查IP是否被阻止"""
        return ip in self.blocked_ips
    
    def analyze_input(self, input_str: str, source_ip: str) -> List[ThreatType]:
        """分析输入威胁"""
        threats = []
        
        for threat_name, pattern in self.suspicious_patterns.items():
            if pattern.search(input_str):
                threats.append(ThreatType(threat_name))
        
        return threats
    
    async def get_security_summary(self) -> Dict[str, Any]:
        """获取安全摘要"""
        async with self.lock:
            recent_events = [
                event for event in self.events
                if event.timestamp > datetime.now(timezone.utc) - timedelta(hours=24)
            ]
            
            threat_counts = {}
            for event in recent_events:
                threat_type = event.threat_type.value
                threat_counts[threat_type] = threat_counts.get(threat_type, 0) + 1
            
            return {
                'total_events_24h': len(recent_events),
                'blocked_ips_count': len(self.blocked_ips),
                'threat_distribution': threat_counts,
                'high_severity_events': len([
                    e for e in recent_events 
                    if e.severity in [SecurityLevel.HIGH, SecurityLevel.CRITICAL]
                ])
            }

class SecurityManager:
    """综合安全管理器"""
    
    def __init__(self, config: SecurityConfig = None):
        self.config = config or SecurityConfig()
        self.password_validator = PasswordValidator()
        self.input_sanitizer = InputSanitizer()
        self.rate_limiter = RateLimiter(self.config)
        self.jwt_manager = JWTManager(self.config)
        self.encryption_manager = EncryptionManager(self.config)
        self.security_monitor = SecurityMonitor(self.config)
    
    async def authenticate_user(self, username: str, password: str, ip: str) -> Dict[str, Any]:
        """用户认证"""
        # 检查IP是否被阻止
        if self.security_monitor.is_ip_blocked(ip):
            return {'success': False, 'reason': 'IP blocked'}
        
        # 检查速率限制
        if not await self.rate_limiter.is_allowed(f"auth:{ip}"):
            return {'success': False, 'reason': 'Rate limit exceeded'}
        
        # 这里应该查询数据库验证用户
        # 示例实现
        if username == "admin" and password == "password":
            # 生成JWT令牌
            token = self.jwt_manager.generate_token(username)
            return {
                'success': True,
                'token': token,
                'user_id': username,
                'expires_in': self.config.jwt_expiration_hours * 3600
            }
        else:
            # 记录失败尝试
            await self.security_monitor.log_event(SecurityEvent(
                timestamp=datetime.now(timezone.utc),
                threat_type=ThreatType.BRUTE_FORCE,
                source_ip=ip,
                user_id=username,
                description="Failed login attempt",
                severity=SecurityLevel.MEDIUM
            ))
            return {'success': False, 'reason': 'Invalid credentials'}
    
    def validate_and_sanitize_input(self, input_data: Any, input_type: str = "text") -> Any:
        """验证和清理输入"""
        if isinstance(input_data, str):
            # 检测威胁
            threats = self.security_monitor.analyze_input(input_data, "unknown")
            if threats:
                logger.warning(f"Potential threats detected: {[t.value for t in threats]}")
            
            # 根据类型清理
            if input_type == "html":
                return self.input_sanitizer.sanitize_html(input_data)
            elif input_type == "sql":
                return self.input_sanitizer.sanitize_sql(input_data)
            else:
                # 基本清理
                return input_data.strip()[:1000]  # 限制长度
        
        elif isinstance(input_data, dict):
            return {
                key: self.validate_and_sanitize_input(value, input_type)
                for key, value in input_data.items()
            }
        
        return input_data
    
    async def check_rate_limit(self, identifier: str) -> Dict[str, Any]:
        """检查速率限制"""
        allowed = await self.rate_limiter.is_allowed(identifier)
        remaining = await self.rate_limiter.get_remaining_requests(identifier)
        
        return {
            'allowed': allowed,
            'remaining': remaining,
            'limit': self.config.rate_limit_requests_per_minute
        }
    
    def encrypt_sensitive_data(self, data: str) -> str:
        """加密敏感数据"""
        return self.encryption_manager.encrypt(data)
    
    def decrypt_sensitive_data(self, encrypted_data: str) -> str:
        """解密敏感数据"""
        return self.encryption_manager.decrypt(encrypted_data)
    
    async def get_security_status(self) -> Dict[str, Any]:
        """获取安全状态"""
        return await self.security_monitor.get_security_summary()

# 装饰器
def require_auth(auth_manager: SecurityManager):
    """认证装饰器"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # 这里应该从请求中提取token
            # 示例实现
            token = kwargs.get('token') or kwargs.get('authorization', '').replace('Bearer ', '')
            
            if not token:
                raise ValueError("Authentication required")
            
            try:
                payload = auth_manager.jwt_manager.verify_token(token)
                kwargs['user_id'] = payload['user_id']
                return await func(*args, **kwargs)
            except Exception as e:
                raise ValueError(f"Invalid authentication: {str(e)}")
        
        return wrapper
    return decorator

def rate_limit_check(auth_manager: SecurityManager, identifier_extractor: Callable = None):
    """速率限制装饰器"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # 提取标识符
            if identifier_extractor:
                identifier = identifier_extractor(*args, **kwargs)
            else:
                identifier = kwargs.get('ip', 'unknown')
            
            # 检查速率限制
            rate_check = await auth_manager.check_rate_limit(identifier)
            if not rate_check['allowed']:
                raise ValueError(f"Rate limit exceeded. Try again later.")
            
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator

def sanitize_inputs(auth_manager: SecurityManager):
    """输入清理装饰器"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # 清理所有字符串输入
            sanitized_kwargs = {}
            for key, value in kwargs.items():
                if isinstance(value, str):
                    sanitized_kwargs[key] = auth_manager.validate_and_sanitize_input(value)
                else:
                    sanitized_kwargs[key] = value
            
            return await func(*args, **sanitized_kwargs)
        
        return wrapper
    return decorator

# 全局安全管理器实例
security_manager = SecurityManager()

def get_security_manager() -> SecurityManager:
    """获取全局安全管理器"""
    return security_manager