from typing import Any, Dict, List, Optional, Union
from fastapi import HTTPException, status
from pydantic import BaseModel

class ErrorDetail(BaseModel):
    """错误详情模型"""
    loc: Optional[List[str]] = None
    msg: str
    type: str
    ctx: Optional[Dict[str, Any]] = None


class AppException(HTTPException):
    """应用自定义异常基类"""
    def __init__(
        self,
        status_code: int,
        detail: str,
        error_code: str = "INTERNAL_ERROR",
        headers: Optional[Dict[str, str]] = None,
        error_details: Optional[List[ErrorDetail]] = None
    ):
        super().__init__(status_code=status_code, detail=detail, headers=headers)
        self.error_code = error_code
        self.error_details = error_details or []


class BadRequestException(AppException):
    """400 错误请求异常"""
    def __init__(
        self,
        detail: str = "请求参数无效",
        error_code: str = "BAD_REQUEST",
        headers: Optional[Dict[str, str]] = None,
        error_details: Optional[List[ErrorDetail]] = None
    ):
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=detail,
            error_code=error_code,
            headers=headers,
            error_details=error_details
        )


class UnauthorizedException(AppException):
    """401 未授权异常"""
    def __init__(
        self,
        detail: str = "未授权访问",
        error_code: str = "UNAUTHORIZED",
        headers: Optional[Dict[str, str]] = None,
        error_details: Optional[List[ErrorDetail]] = None
    ):
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=detail,
            error_code=error_code,
            headers=headers,
            error_details=error_details
        )


class ForbiddenException(AppException):
    """403 禁止访问异常"""
    def __init__(
        self,
        detail: str = "禁止访问",
        error_code: str = "FORBIDDEN",
        headers: Optional[Dict[str, str]] = None,
        error_details: Optional[List[ErrorDetail]] = None
    ):
        super().__init__(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=detail,
            error_code=error_code,
            headers=headers,
            error_details=error_details
        )


class NotFoundException(AppException):
    """404 资源不存在异常"""
    def __init__(
        self,
        detail: str = "资源不存在",
        error_code: str = "NOT_FOUND",
        headers: Optional[Dict[str, str]] = None,
        error_details: Optional[List[ErrorDetail]] = None
    ):
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=detail,
            error_code=error_code,
            headers=headers,
            error_details=error_details
        )


class ConflictException(AppException):
    """409 资源冲突异常"""
    def __init__(
        self,
        detail: str = "资源冲突",
        error_code: str = "CONFLICT",
        headers: Optional[Dict[str, str]] = None,
        error_details: Optional[List[ErrorDetail]] = None
    ):
        super().__init__(
            status_code=status.HTTP_409_CONFLICT,
            detail=detail,
            error_code=error_code,
            headers=headers,
            error_details=error_details
        )


class UnprocessableEntityException(AppException):
    """422 无法处理的实体异常"""
    def __init__(
        self,
        detail: str = "无法处理的实体",
        error_code: str = "UNPROCESSABLE_ENTITY",
        headers: Optional[Dict[str, str]] = None,
        error_details: Optional[List[ErrorDetail]] = None
    ):
        super().__init__(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=detail,
            error_code=error_code,
            headers=headers,
            error_details=error_details
        )


class InternalServerErrorException(AppException):
    """500 服务器内部错误异常"""
    def __init__(
        self,
        detail: str = "服务器内部错误",
        error_code: str = "INTERNAL_SERVER_ERROR",
        headers: Optional[Dict[str, str]] = None,
        error_details: Optional[List[ErrorDetail]] = None
    ):
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=detail,
            error_code=error_code,
            headers=headers,
            error_details=error_details
        )


class ServiceUnavailableException(AppException):
    """503 服务不可用异常"""
    def __init__(
        self,
        detail: str = "服务暂时不可用",
        error_code: str = "SERVICE_UNAVAILABLE",
        headers: Optional[Dict[str, str]] = None,
        error_details: Optional[List[ErrorDetail]] = None
    ):
        super().__init__(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=detail,
            error_code=error_code,
            headers=headers,
            error_details=error_details
        )


class DependencyException(AppException):
    """依赖服务异常"""
    def __init__(
        self,
        detail: str = "依赖服务异常",
        error_code: str = "DEPENDENCY_ERROR",
        headers: Optional[Dict[str, str]] = None,
        error_details: Optional[List[ErrorDetail]] = None
    ):
        super().__init__(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=detail,
            error_code=error_code,
            headers=headers,
            error_details=error_details
        )


class ValidationException(AppException):
    """数据验证异常"""
    def __init__(
        self,
        detail: str = "数据验证失败",
        error_code: str = "VALIDATION_ERROR",
        headers: Optional[Dict[str, str]] = None,
        error_details: Optional[List[ErrorDetail]] = None
    ):
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=detail,
            error_code=error_code,
            headers=headers,
            error_details=error_details
        )


class DatabaseException(AppException):
    """数据库操作异常"""
    def __init__(
        self,
        detail: str = "数据库操作失败",
        error_code: str = "DATABASE_ERROR",
        headers: Optional[Dict[str, str]] = None,
        error_details: Optional[List[ErrorDetail]] = None
    ):
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=detail,
            error_code=error_code,
            headers=headers,
            error_details=error_details
        )


class BusinessLogicException(AppException):
    """业务逻辑异常"""
    def __init__(
        self,
        detail: str = "业务逻辑错误",
        error_code: str = "BUSINESS_LOGIC_ERROR",
        headers: Optional[Dict[str, str]] = None,
        error_details: Optional[List[ErrorDetail]] = None
    ):
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=detail,
            error_code=error_code,
            headers=headers,
            error_details=error_details
        )