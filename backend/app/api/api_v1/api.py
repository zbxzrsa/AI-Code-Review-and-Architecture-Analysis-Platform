from fastapi import APIRouter

from app.api.api_v1.endpoints import code_analysis, users, projects

api_router = APIRouter()
api_router.include_router(users.router, prefix="/users", tags=["users"])
api_router.include_router(projects.router, prefix="/projects", tags=["projects"])
api_router.include_router(code_analysis.router, prefix="/analysis", tags=["code-analysis"])