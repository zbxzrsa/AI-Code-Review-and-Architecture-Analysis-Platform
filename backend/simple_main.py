from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="AI Code Review Platform API",
    description="Simple API for development",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "ai-analysis-api"}

@app.get("/api/v1/projects")
async def get_projects():
    return [
        {
            "id": 1,
            "name": "Sample Project",
            "description": "A sample project for testing",
            "status": "active"
        }
    ]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)