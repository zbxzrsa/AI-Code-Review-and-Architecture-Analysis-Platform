from fastapi import APIRouter, HTTPException, Header
from pydantic import BaseModel

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])

class LoginRequest(BaseModel):
    username: str
    password: str

class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    username: str

MOCK_USERS = {
    "admin": "admin123",
    "user": "password"
}

@router.post("/login", response_model=LoginResponse)
async def login(payload: LoginRequest):
    # 简易验证：存在即通过；实际应查询数据库并校验密码散列
    if payload.username in MOCK_USERS:
        # 演示固定token；生产应签发JWT
        return LoginResponse(access_token="demo-token", username=payload.username)
    raise HTTPException(status_code=401, detail="用户名或密码错误")

@router.get("/me")
async def me(authorization: str | None = Header(default=None)):
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="未认证")
    token = authorization.split(" ", 1)[1]
    if token != "demo-token":
        raise HTTPException(status_code=401, detail="令牌无效")
    return {"id": 1, "username": "admin", "roles": ["admin"], "email": "admin@example.com"}