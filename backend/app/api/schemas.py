"""
Standardized API schemas and error responses for AI Code Review Platform.
"""

from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from pydantic import BaseModel, Field, validator
from enum import Enum


class APIResponse(BaseModel):
    """Standard API response wrapper."""
    
    success: bool = Field(description="Whether the request was successful")
    data: Optional[Any] = Field(default=None, description="Response data")
    message: Optional[str] = Field(default=None, description="Response message")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")
    request_id: Optional[str] = Field(default=None, description="Request correlation ID")


class PaginatedResponse(BaseModel):
    """Paginated response wrapper."""
    
    items: List[Any] = Field(description="List of items")
    total: int = Field(description="Total number of items")
    page: int = Field(description="Current page number")
    size: int = Field(description="Page size")
    pages: int = Field(description="Total number of pages")
    has_next: bool = Field(description="Whether there is a next page")
    has_prev: bool = Field(description="Whether there is a previous page")


class ErrorDetail(BaseModel):
    """Detailed error information."""
    
    code: str = Field(description="Error code")
    message: str = Field(description="Error message")
    field: Optional[str] = Field(default=None, description="Field name if validation error")
    value: Optional[Any] = Field(default=None, description="Invalid value if validation error")


class APIError(BaseModel):
    """Standard API error response (RFC 7807 Problem Details)."""
    
    type: str = Field(description="Error type identifier")
    title: str = Field(description="Error title")
    status: int = Field(description="HTTP status code")
    detail: str = Field(description="Error description")
    instance: Optional[str] = Field(default=None, description="Error instance identifier")
    errors: Optional[List[ErrorDetail]] = Field(default=None, description="Detailed error list")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")
    request_id: Optional[str] = Field(default=None, description="Request correlation ID")


class ErrorCode(str, Enum):
    """Standard error codes."""
    
    # Validation errors
    VALIDATION_ERROR = "VALIDATION_ERROR"
    INVALID_INPUT = "INVALID_INPUT"
    MISSING_FIELD = "MISSING_FIELD"
    INVALID_FORMAT = "INVALID_FORMAT"
    
    # Authentication/Authorization errors
    UNAUTHORIZED = "UNAUTHORIZED"
    FORBIDDEN = "FORBIDDEN"
    INVALID_TOKEN = "INVALID_TOKEN"
    TOKEN_EXPIRED = "TOKEN_EXPIRED"
    INSUFFICIENT_PERMISSIONS = "INSUFFICIENT_PERMISSIONS"
    
    # Resource errors
    NOT_FOUND = "NOT_FOUND"
    ALREADY_EXISTS = "ALREADY_EXISTS"
    RESOURCE_CONFLICT = "RESOURCE_CONFLICT"
    RESOURCE_LOCKED = "RESOURCE_LOCKED"
    
    # Rate limiting
    RATE_LIMIT_EXCEEDED = "RATE_LIMIT_EXCEEDED"
    QUOTA_EXCEEDED = "QUOTA_EXCEEDED"
    
    # Service errors
    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"
    INTERNAL_ERROR = "INTERNAL_ERROR"
    DATABASE_ERROR = "DATABASE_ERROR"
    EXTERNAL_SERVICE_ERROR = "EXTERNAL_SERVICE_ERROR"
    
    # Business logic errors
    INVALID_OPERATION = "INVALID_OPERATION"
    OPERATION_NOT_ALLOWED = "OPERATION_NOT_ALLOWED"
    DEPENDENCY_NOT_MET = "DEPENDENCY_NOT_MET"
    
    # File/Upload errors
    FILE_TOO_LARGE = "FILE_TOO_LARGE"
    INVALID_FILE_TYPE = "INVALID_FILE_TYPE"
    UPLOAD_FAILED = "UPLOAD_FAILED"
    
    # AI/Analysis errors
    ANALYSIS_FAILED = "ANALYSIS_FAILED"
    MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"
    ANALYSIS_TIMEOUT = "ANALYSIS_TIMEOUT"
    INVALID_CODE_FORMAT = "INVALID_CODE_FORMAT"


class HealthStatus(str, Enum):
    """Health check status."""
    
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class HealthCheck(BaseModel):
    """Health check response."""
    
    status: HealthStatus = Field(description="Overall health status")
    version: str = Field(description="API version")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Check timestamp")
    uptime: float = Field(description="Service uptime in seconds")
    checks: Dict[str, Any] = Field(description="Individual health checks")


class SortOrder(str, Enum):
    """Sort order options."""
    
    ASC = "asc"
    DESC = "desc"


class FilterOperator(str, Enum):
    """Filter operators."""
    
    EQ = "eq"
    NE = "ne"
    GT = "gt"
    GTE = "gte"
    LT = "lt"
    LTE = "lte"
    IN = "in"
    NIN = "nin"
    LIKE = "like"
    ILIKE = "ilike"
    CONTAINS = "contains"
    STARTS_WITH = "starts_with"
    ENDS_WITH = "ends_with"


class Filter(BaseModel):
    """Generic filter model."""
    
    field: str = Field(description="Field to filter on")
    operator: FilterOperator = Field(description="Filter operator")
    value: Any = Field(description="Filter value")


class Sort(BaseModel):
    """Generic sort model."""
    
    field: str = Field(description="Field to sort by")
    order: SortOrder = Field(default=SortOrder.ASC, description="Sort order")


class PaginationParams(BaseModel):
    """Pagination parameters."""
    
    page: int = Field(default=1, ge=1, description="Page number")
    size: int = Field(default=20, ge=1, le=100, description="Page size")
    
    @property
    def offset(self) -> int:
        """Calculate offset from page and size."""
        return (self.page - 1) * self.size


class SearchParams(BaseModel):
    """Search parameters."""
    
    query: Optional[str] = Field(default=None, description="Search query")
    filters: Optional[List[Filter]] = Field(default=None, description="Filter conditions")
    sort: Optional[List[Sort]] = Field(default=None, description="Sort conditions")
    pagination: Optional[PaginationParams] = Field(default=None, description="Pagination parameters")


# Request/Response schemas for common operations
class CreateRequest(BaseModel):
    """Base create request schema."""
    
    name: str = Field(description="Resource name")
    description: Optional[str] = Field(default=None, description="Resource description")


class UpdateRequest(BaseModel):
    """Base update request schema."""
    
    name: Optional[str] = Field(default=None, description="Resource name")
    description: Optional[str] = Field(default=None, description="Resource description")


class DeleteResponse(BaseModel):
    """Delete operation response."""
    
    deleted: bool = Field(description="Whether deletion was successful")
    message: str = Field(description="Deletion message")


class BulkOperation(BaseModel):
    """Bulk operation request."""
    
    operation: str = Field(description="Operation type")
    items: List[Any] = Field(description="Items to operate on")
    options: Optional[Dict[str, Any]] = Field(default=None, description="Operation options")


class BulkOperationResponse(BaseModel):
    """Bulk operation response."""
    
    total: int = Field(description="Total items processed")
    successful: int = Field(description="Successful operations")
    failed: int = Field(description="Failed operations")
    errors: Optional[List[ErrorDetail]] = Field(default=None, description="Operation errors")


# API versioning
class APIVersion(BaseModel):
    """API version information."""
    
    version: str = Field(description="API version")
    deprecated: bool = Field(default=False, description="Whether version is deprecated")
    sunset_date: Optional[datetime] = Field(default=None, description="Version sunset date")
    supported_until: Optional[datetime] = Field(default=None, description="Version support end date")


# Metadata schemas
class Metadata(BaseModel):
    """Resource metadata."""
    
    created_at: datetime = Field(description="Creation timestamp")
    updated_at: datetime = Field(description="Last update timestamp")
    created_by: Optional[str] = Field(default=None, description="Creator identifier")
    updated_by: Optional[str] = Field(default=None, description="Last updater identifier")
    version: int = Field(default=1, description="Resource version")


class AuditLog(BaseModel):
    """Audit log entry."""
    
    id: str = Field(description="Log entry ID")
    action: str = Field(description="Action performed")
    resource_type: str = Field(description="Resource type")
    resource_id: str = Field(description="Resource identifier")
    user_id: Optional[str] = Field(default=None, description="User identifier")
    timestamp: datetime = Field(description="Action timestamp")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Action details")
    ip_address: Optional[str] = Field(default=None, description="Client IP address")
    user_agent: Optional[str] = Field(default=None, description="Client user agent")


# Utility functions for creating responses
def create_success_response(
    data: Any = None,
    message: str = "Operation successful",
    request_id: Optional[str] = None
) -> APIResponse:
    """Create a success response."""
    return APIResponse(
        success=True,
        data=data,
        message=message,
        request_id=request_id
    )


def create_error_response(
    error_code: ErrorCode,
    detail: str,
    status_code: int = 400,
    errors: Optional[List[ErrorDetail]] = None,
    request_id: Optional[str] = None
) -> APIError:
    """Create an error response."""
    return APIError(
        type=f"https://api.example.com/errors/{error_code.value}",
        title=error_code.value.replace("_", " ").title(),
        status=status_code,
        detail=detail,
        errors=errors,
        request_id=request_id
    )


def create_paginated_response(
    items: List[Any],
    total: int,
    page: int,
    size: int
) -> PaginatedResponse:
    """Create a paginated response."""
    pages = (total + size - 1) // size
    return PaginatedResponse(
        items=items,
        total=total,
        page=page,
        size=size,
        pages=pages,
        has_next=page < pages,
        has_prev=page > 1
    )