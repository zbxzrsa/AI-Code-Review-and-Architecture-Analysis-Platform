"""
Search API endpoints
"""
from datetime import datetime
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Query
from pydantic import BaseModel

router = APIRouter(prefix="/api/v1/search", tags=["search"])


class WebSearchResult(BaseModel):
    title: str
    url: str
    snippet: str


class SearchResult(BaseModel):
    type: str  # "project", "session", "file", "issue"
    id: int
    title: str
    description: str
    score: float
    metadata: Dict[str, Any]
    created_at: datetime


class SearchQuery(BaseModel):
    query: str
    filters: Optional[Dict[str, Any]] = None
    limit: int = 20


# Mock search data
MOCK_SEARCH_DATA = [
    SearchResult(
        type="project",
        id=1,
        title="E-commerce Platform",
        description="Main e-commerce application with user management and payment processing",
        score=0.95,
        metadata={"language": "Python", "framework": "Django", "lines_of_code": 15000},
        created_at=datetime(2024, 1, 10, 9, 0)
    ),
    SearchResult(
        type="session",
        id=1,
        title="Initial Analysis - E-commerce Platform",
        description="Comprehensive security and quality analysis session",
        score=0.88,
        metadata={"status": "completed", "issues_found": 12, "duration_minutes": 30},
        created_at=datetime(2024, 1, 15, 10, 0)
    ),
    SearchResult(
        type="file",
        id=1,
        title="src/main.py",
        description="Main application entry point with FastAPI setup",
        score=0.82,
        metadata={"size_bytes": 2048, "language": "Python", "complexity": "medium"},
        created_at=datetime(2024, 1, 15, 10, 0)
    ),
    SearchResult(
        type="issue",
        id=1,
        title="SQL Injection Vulnerability",
        description="Potential SQL injection in user authentication module",
        score=0.91,
        metadata={"severity": "high", "category": "security", "file": "auth/models.py", "line": 45},
        created_at=datetime(2024, 1, 15, 10, 15)
    ),
    SearchResult(
        type="project",
        id=2,
        title="Data Analytics Dashboard",
        description="Business intelligence dashboard with real-time analytics",
        score=0.78,
        metadata={"language": "JavaScript", "framework": "React", "lines_of_code": 8500},
        created_at=datetime(2024, 1, 12, 14, 0)
    ),
    SearchResult(
        type="session",
        id=2,
        title="Security Scan - E-commerce Platform",
        description="Focused security vulnerability assessment",
        score=0.85,
        metadata={"status": "running", "issues_found": 0, "duration_minutes": 0},
        created_at=datetime(2024, 1, 20, 14, 0)
    ),
    SearchResult(
        type="file",
        id=2,
        title="src/utils.py",
        description="Utility functions for data processing and validation",
        score=0.75,
        metadata={"size_bytes": 1536, "language": "Python", "complexity": "low"},
        created_at=datetime(2024, 1, 15, 10, 5)
    ),
    SearchResult(
        type="issue",
        id=2,
        title="Code Duplication",
        description="Duplicate code blocks found in multiple modules",
        score=0.68,
        metadata={"severity": "medium", "category": "quality", "files": ["utils.py", "helpers.py"], "similarity": 0.85},
        created_at=datetime(2024, 1, 15, 10, 20)
    )
]


@router.get("/", response_model=List[SearchResult])
async def search(
    q: str = Query(..., description="Search query"),
    type_filter: Optional[str] = Query(None, description="Filter by result type"),
    limit: int = Query(20, ge=1, le=100, description="Number of results to return")
):
    """Search across projects, sessions, files, and issues"""
    query_lower = q.lower()
    results = []
    
    for item in MOCK_SEARCH_DATA:
        # Simple text matching in title and description
        if (query_lower in item.title.lower() or 
            query_lower in item.description.lower() or
            any(query_lower in str(v).lower() for v in item.metadata.values())):
            
            if type_filter is None or item.type == type_filter:
                results.append(item)
    
    # Sort by score descending
    results.sort(key=lambda x: x.score, reverse=True)
    
    return results[:limit]


@router.post("/", response_model=List[SearchResult])
async def advanced_search(search_query: SearchQuery):
    """Advanced search with filters and complex queries"""
    query_lower = search_query.query.lower()
    results = []
    
    for item in MOCK_SEARCH_DATA:
        # Text matching
        text_match = (query_lower in item.title.lower() or 
                     query_lower in item.description.lower() or
                     any(query_lower in str(v).lower() for v in item.metadata.values()))
        
        if not text_match:
            continue
        
        # Apply filters if provided
        if search_query.filters:
            filter_match = True
            for filter_key, filter_value in search_query.filters.items():
                if filter_key == "type" and item.type != filter_value:
                    filter_match = False
                    break
                elif filter_key == "created_after":
                    filter_date = datetime.fromisoformat(filter_value.replace('Z', '+00:00'))
                    if item.created_at < filter_date:
                        filter_match = False
                        break
                elif filter_key == "created_before":
                    filter_date = datetime.fromisoformat(filter_value.replace('Z', '+00:00'))
                    if item.created_at > filter_date:
                        filter_match = False
                        break
                elif filter_key in item.metadata and item.metadata[filter_key] != filter_value:
                    filter_match = False
                    break
            
            if not filter_match:
                continue
        
        results.append(item)
    
    # Sort by score descending
    results.sort(key=lambda x: x.score, reverse=True)
    
    return results[:search_query.limit]


@router.get("/suggestions")
async def get_search_suggestions(
    q: str = Query(..., description="Partial search query"),
    limit: int = Query(10, ge=1, le=20, description="Number of suggestions")
):
    """Get search suggestions based on partial query"""
    query_lower = q.lower()
    suggestions = set()
    
    for item in MOCK_SEARCH_DATA:
        # Extract words from title and description
        words = (item.title + " " + item.description).lower().split()
        for word in words:
            if word.startswith(query_lower) and len(word) > len(query_lower):
                suggestions.add(word)
        
        # Add metadata values as suggestions
        for value in item.metadata.values():
            if isinstance(value, str) and value.lower().startswith(query_lower):
                suggestions.add(value.lower())
    
    return sorted(list(suggestions))[:limit]


@router.get("/filters")
async def get_available_filters():
    """Get available filter options for advanced search"""
    types = set(item.type for item in MOCK_SEARCH_DATA)
    
    # Extract unique metadata keys
    metadata_keys = set()
    for item in MOCK_SEARCH_DATA:
        metadata_keys.update(item.metadata.keys())
    
    return {
        "types": sorted(list(types)),
        "metadata_fields": sorted(list(metadata_keys)),
        "date_fields": ["created_after", "created_before"]
    }


@router.get("/web", response_model=List[WebSearchResult])
async def web_search(
    q: str = Query(..., description="Search query for the web"),
    limit: int = Query(10, ge=1, le=50, description="Number of web results")
):
    """Perform web search via DuckDuckGo Instant Answer API"""
    import httpx

    params = {
        "q": q,
        "format": "json",
        "no_html": 1,
        "no_redirect": 1,
    }

    results: List[WebSearchResult] = []
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get("https://api.duckduckgo.com/", params=params)
            resp.raise_for_status()
            data = resp.json()

        def extract(item: Dict[str, Any]) -> Optional[WebSearchResult]:
            title = item.get("Text")
            url = item.get("FirstURL")
            if title and url:
                return WebSearchResult(title=title, url=url, snippet=title)
            return None

        related = data.get("RelatedTopics") or []
        for topic in related:
            if isinstance(topic, dict) and "Topics" in topic:
                for sub in topic["Topics"]:
                    r = extract(sub)
                    if r:
                        results.append(r)
            else:
                r = extract(topic)
                if r:
                    results.append(r)

        # Fallback to abstract if no related topics
        if not results and data.get("AbstractURL"):
            heading = data.get("Heading") or q
            abstract_url = data.get("AbstractURL")
            abstract_text = data.get("AbstractText") or heading
            results.append(WebSearchResult(title=heading, url=abstract_url, snippet=abstract_text))

    except Exception:
        # Gracefully return empty list on errors
        results = []

    return results[:limit]