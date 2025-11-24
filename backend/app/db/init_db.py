from app.db.session import AsyncSessionLocal

async def init_db() -> None:
    """
    初始化数据库，创建表和初始数据
    """
    async with AsyncSessionLocal() as session:
        # 在这里添加初始化逻辑
        # 例如创建初始用户、角色等
        pass