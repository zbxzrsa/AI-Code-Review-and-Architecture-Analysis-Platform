"""
安全模块 - 提供后端安全功能
包括认证、授权、数据加密和安全配置
"""
import os
import secrets
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Union

import jwt
from passlib.context import CryptContext
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.user import User

# 密码哈希上下文
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT相关配置
ALGORITHM = "HS256"
access_token_jwt_subject = "access"


def create_access_token(
    subject: Union[str, Any], expires_delta: Optional[timedelta] = None
) -> str:
    """
    创建JWT访问令牌
    
    Args:
        subject: 令牌主题（通常是用户ID）
        expires_delta: 过期时间增量
        
    Returns:
        编码后的JWT令牌
    """
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(
            minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES
        )
    
    to_encode = {"exp": expire, "sub": str(subject), "type": access_token_jwt_subject}
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    验证密码
    
    Args:
        plain_password: 明文密码
        hashed_password: 哈希后的密码
        
    Returns:
        密码是否匹配
    """
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """
    获取密码哈希
    
    Args:
        password: 明文密码
        
    Returns:
        哈希后的密码
    """
    return pwd_context.hash(password)


def authenticate_user(db: Session, email: str, password: str) -> Optional[User]:
    """
    认证用户
    
    Args:
        db: 数据库会话
        email: 用户邮箱
        password: 用户密码
        
    Returns:
        认证成功返回用户对象，失败返回None
    """
    user = db.query(User).filter(User.email == email).first()
    if not user:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user


def generate_secure_random_string(length: int = 32) -> str:
    """
    生成安全的随机字符串
    
    Args:
        length: 字符串长度
        
    Returns:
        随机字符串
    """
    return secrets.token_urlsafe(length)


def encrypt_sensitive_data(data: str) -> str:
    """
    加密敏感数据
    
    Args:
        data: 需要加密的数据
        
    Returns:
        加密后的数据
    """
    # 实际项目中应使用更强的加密方法，如AES
    # 这里使用简单的方法作为示例
    key = settings.SECRET_KEY[:32].encode()
    return data  # 占位，实际应实现加密


def decrypt_sensitive_data(encrypted_data: str) -> str:
    """
    解密敏感数据
    
    Args:
        encrypted_data: 加密后的数据
        
    Returns:
        解密后的数据
    """
    # 实际项目中应使用对应的解密方法
    # 这里使用简单的方法作为示例
    key = settings.SECRET_KEY[:32].encode()
    return encrypted_data  # 占位，实际应实现解密


def sanitize_input(input_data: str) -> str:
    """
    清理用户输入，防止注入攻击
    
    Args:
        input_data: 用户输入
        
    Returns:
        清理后的输入
    """
    if not input_data:
        return ""
    
    # 移除可能导致SQL注入的字符
    sanitized = input_data.replace("'", "''")
    
    # 移除可能导致命令注入的字符
    sanitized = sanitized.replace(";", "")
    sanitized = sanitized.replace("|", "")
    sanitized = sanitized.replace("&", "")
    
    return sanitized


def generate_csrf_token() -> str:
    """
    生成CSRF令牌
    
    Returns:
        CSRF令牌
    """
    return secrets.token_hex(32)


def validate_csrf_token(token: str, stored_token: str) -> bool:
    """
    验证CSRF令牌
    
    Args:
        token: 提交的令牌
        stored_token: 存储的令牌
        
    Returns:
        令牌是否有效
    """
    if not token or not stored_token:
        return False
    return secrets.compare_digest(token, stored_token)


def check_password_strength(password: str) -> Dict[str, Any]:
    """
    检查密码强度
    
    Args:
        password: 密码
        
    Returns:
        包含密码强度信息的字典
    """
    result = {
        "score": 0,
        "is_strong": False,
        "suggestions": [],
    }
    
    if len(password) < 8:
        result["suggestions"].append("密码长度应至少为8个字符")
    else:
        result["score"] += 1
    
    if any(c.isupper() for c in password):
        result["score"] += 1
    else:
        result["suggestions"].append("密码应包含大写字母")
    
    if any(c.islower() for c in password):
        result["score"] += 1
    else:
        result["suggestions"].append("密码应包含小写字母")
    
    if any(c.isdigit() for c in password):
        result["score"] += 1
    else:
        result["suggestions"].append("密码应包含数字")
    
    if any(not c.isalnum() for c in password):
        result["score"] += 1
    else:
        result["suggestions"].append("密码应包含特殊字符")
    
    result["is_strong"] = result["score"] >= 4
    
    return result


def rate_limit_check(user_id: str, action: str, max_attempts: int, time_window: int) -> bool:
    """
    检查是否超过速率限制
    
    Args:
        user_id: 用户ID
        action: 操作类型
        max_attempts: 最大尝试次数
        time_window: 时间窗口（秒）
        
    Returns:
        是否允许操作
    """
    # 实际项目中应使用Redis等存储尝试记录
    # 这里使用简单的方法作为示例
    return True  # 占位，实际应实现速率限制检查