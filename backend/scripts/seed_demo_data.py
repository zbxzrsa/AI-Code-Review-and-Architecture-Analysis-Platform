"""
Seed script for demo data
"""
import asyncio
import uuid
from datetime import datetime, timedelta
from sqlalchemy.orm import Session

from app.db.session import engine, SessionLocal
from app.models import Project, Repository, AnalysisSession, Finding, Baseline


def create_demo_user():
    """Create demo user"""
    # This would normally create a user, for now we'll use existing
    pass


def create_demo_project(db: Session):
    """Create demo project"""
    project = Project(
        name="AI Code Review Demo",
        slug="ai-code-review-demo",
        description="Demo project for AI Code Review Platform",
        visibility="public",
        status="active",
        settings={
            "analysis_rules": ["security", "quality", "performance"],
            "notification_settings": {
                "email": True,
                "slack": False
            }
        }
    )
    
    db.add(project)
    db.commit()
    db.refresh(project)
    return project


def create_demo_repository(db: Session, project_id: uuid.UUID):
    """Create demo repository"""
    repo = Repository(
        project_id=project_id,
        name="demo-repository",
        full_name="demo-user/demo-repository",
        provider="github",
        external_id=123456789,
        external_url="https://github.com/demo-user/demo-repository",
        clone_url="https://github.com/demo-user/demo-repository.git",
        default_branch="main",
        is_active=True,
        sync_status="success",
        last_synced_at=datetime.utcnow(),
        settings={
            "analysis_enabled": True,
            "auto_scan": True
        }
    )
    
    db.add(repo)
    db.commit()
    db.refresh(repo)
    return repo


def create_demo_analysis_session(db: Session, project_id: uuid.UUID, repo_id: uuid.UUID):
    """Create demo analysis session"""
    session = AnalysisSession(
        project_id=project_id,
        repository_id=repo_id,
        analysis_type="full",
        config={
            "rules": ["security", "quality", "performance"],
            "depth": "full"
        },
        rules_enabled=["security", "quality", "performance"],
        status="completed",
        progress=100.0,
        started_at=datetime.utcnow() - timedelta(hours=2),
        completed_at=datetime.utcnow() - timedelta(hours=1),
        duration_seconds=3600,
        total_findings=15,
        critical_findings=2,
        high_findings=5,
        medium_findings=6,
        low_findings=2,
        commit_sha="abc123def456",
        branch="main"
    )
    
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


def create_demo_findings(db: Session, session_id: uuid.UUID):
    """Create demo findings"""
    findings = [
        Finding(
            session_id=session_id,
            rule_id="SEC001",
            rule_name="Hardcoded Secret Detected",
            category="security",
            severity="critical",
            confidence="high",
            file_path="src/config/database.py",
            line_number=15,
            end_line_number=15,
            title="Potential hardcoded secret in configuration",
            description="A hardcoded secret was detected in the configuration file. This could lead to security vulnerabilities.",
            recommendation="Move secrets to environment variables or secure secret management system.",
            code_snippet="DATABASE_PASSWORD = 'hardcoded_password'",
            cwe_id="798",
            owasp_category="A05:2021 - Security Misconfiguration",
            ai_analysis={
                "risk_score": 9.5,
                "exploitability": "High",
                "impact": "Critical"
            },
            ai_suggestion="Replace hardcoded password with environment variable: os.getenv('DB_PASSWORD')",
            ai_confidence=0.95
        ),
        Finding(
            session_id=session_id,
            rule_id="QUAL001",
            rule_name="Complex Function",
            category="quality",
            severity="medium",
            confidence="medium",
            file_path="src/services/analysis.py",
            line_number=45,
            end_line_number=78,
            title="Function complexity too high",
            description="The function has high cyclomatic complexity, making it difficult to maintain and test.",
            recommendation="Refactor the function into smaller, more focused functions.",
            code_snippet="def analyze_code(data, options, config, rules, filters):\n    # Complex logic here",
            ai_analysis={
                "complexity_score": 15,
                "maintainability": "Low"
            },
            ai_suggestion="Extract separate functions for data processing, option handling, and rule application",
            ai_confidence=0.87
        ),
        Finding(
            session_id=session_id,
            rule_id="PERF001",
            rule_name="Inefficient Database Query",
            category="performance",
            severity="high",
            confidence="high",
            file_path="src/models/user.py",
            line_number=23,
            end_line_number=25,
            title="N+1 query detected",
            description="A potential N+1 query issue was detected in the database access pattern.",
            recommendation="Use eager loading or batch queries to avoid N+1 problems.",
            code_snippet="for user in users:\n    user.posts  # This creates N+1 queries",
            ai_analysis={
                "performance_impact": "High",
                "estimated_slowdown": "10x"
            },
            ai_suggestion="Use joinedload or selectinload to fetch related data in a single query",
            ai_confidence=0.92
        ),
        Finding(
            session_id=session_id,
            rule_id="ARCH001",
            rule_name="Circular Dependency",
            category="architecture",
            severity="medium",
            confidence="medium",
            file_path="src/modules/auth.py",
            line_number=12,
            end_line_number=12,
            title="Circular import detected",
            description="A circular dependency was detected between modules, which can lead to runtime errors.",
            recommendation="Refactor the module structure to eliminate circular dependencies.",
            code_snippet="from .services import auth_service",
            ai_analysis={
                "dependency_type": "circular",
                "impact": "Runtime error risk"
            },
            ai_suggestion="Create a separate module for shared functionality or use dependency injection",
            ai_confidence=0.78
        )
    ]
    
    for finding in findings:
        db.add(finding)
    
    db.commit()
    return findings


def create_demo_baseline(db: Session, project_id: uuid.UUID):
    """Create demo baseline"""
    baseline = Baseline(
        project_id=project_id,
        name="Production Quality Baseline",
        description="Quality standards for production code",
        thresholds={
            "max_critical_issues": 0,
            "max_high_issues": 5,
            "max_medium_issues": 20,
            "min_coverage_percent": 80,
            "max_complexity": 10
        },
        rules=["security", "quality", "performance"],
        exceptions=[
            {
                "rule_id": "QUAL001",
                "file_pattern": "test_*.py",
                "reason": "Test files may have higher complexity"
            }
        ],
        is_active=True,
        version=1,
        applied_at=datetime.utcnow()
    )
    
    db.add(baseline)
    db.commit()
    db.refresh(baseline)
    return baseline


async def seed_demo_data():
    """Seed demo data for testing"""
    db = SessionLocal()
    try:
        print("Creating demo data...")
        
        # Create demo project
        project = create_demo_project(db)
        print(f"Created project: {project.name}")
        
        # Create demo repository
        repo = create_demo_repository(db, project.id)
        print(f"Created repository: {repo.name}")
        
        # Create demo analysis session
        session = create_demo_analysis_session(db, project.id, repo.id)
        print(f"Created analysis session: {session.id}")
        
        # Create demo findings
        findings = create_demo_findings(db, session.id)
        print(f"Created {len(findings)} findings")
        
        # Create demo baseline
        baseline = create_demo_baseline(db, project.id)
        print(f"Created baseline: {baseline.name}")
        
        print("Demo data created successfully!")
        print(f"Project ID: {project.id}")
        print(f"Repository ID: {repo.id}")
        print(f"Session ID: {session.id}")
        print(f"Total Findings: {len(findings)}")
        
    except Exception as e:
        print(f"Error creating demo data: {e}")
        db.rollback()
    finally:
        db.close()


if __name__ == "__main__":
    asyncio.run(seed_demo_data())