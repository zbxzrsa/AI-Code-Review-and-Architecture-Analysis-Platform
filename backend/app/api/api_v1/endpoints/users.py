from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db

router = APIRouter()

@router.get("/me")
async def read_users_me():
    """
    获取当前用户信息
    """
    return {"id": 1, "username": "admin", "email": "admin@example.com"}