from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db

router = APIRouter()

@router.get("/")
async def list_projects(
    db: AsyncSession = Depends(get_db)
):
    """
    获取项目列表
    """
    # 这里将实现获取项目列表的逻辑
    return [
        {"id": 1, "name": "Project 1", "description": "Description 1"},
        {"id": 2, "name": "Project 2", "description": "Description 2"}
    ]

@router.post("/")
async def create_project(
    db: AsyncSession = Depends(get_db)
):
    """
    创建新项目
    """
    # 这里将实现创建项目的逻辑
    return {"id": 3, "name": "New Project", "description": "New Description"}