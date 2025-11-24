# Contributing to AI Code Review and Architecture Analysis Platform

Thank you for your interest in contributing to our platform! This document provides guidelines and information for contributors.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Code Standards](#code-standards)
- [Testing](#testing)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)
- [Community Guidelines](#community-guidelines)

## Getting Started

### Prerequisites

- Python 3.11+
- Node.js 18+
- Docker and Docker Compose
- Git

### Setup Development Environment

1. **Fork and Clone**
   ```bash
   git clone https://github.com/your-username/ai-code-review-platform.git
   cd ai-code-review-platform
   ```

2. **Install Dependencies**
   ```bash
   make setup-dev
   ```

3. **Start Development Services**
   ```bash
   make start-dev
   ```

4. **Verify Installation**
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000
   - API Documentation: http://localhost:8000/docs

## Development Workflow

### 1. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
```

### 2. Make Your Changes

- Follow the code standards outlined below
- Write tests for new functionality
- Update documentation as needed

### 3. Run Quality Checks

```bash
make pre-commit
make test
make lint
make type-check
```

### 4. Commit Your Changes

We use [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `test`: Test additions or changes
- `chore`: Maintenance tasks
- `ci`: CI/CD changes
- `build`: Build system changes
- `security`: Security-related changes
- `deps`: Dependency updates

Examples:
```bash
git commit -m "feat(api): add code analysis endpoint"
git commit -m "fix(auth): resolve JWT token validation issue"
git commit -m "docs: update API documentation"
```

### 5. Push and Create Pull Request

```bash
git push origin feature/your-feature-name
```

Create a pull request following the template in the GitHub interface.

## Code Standards

### Python (Backend)

#### Style Guide
- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/)
- Use [Black](https://black.readthedocs.io/) for formatting
- Use [isort](https://isort.readthedocs.io/) for import sorting
- Maximum line length: 88 characters

#### Type Hints
- All functions must have type hints
- Use `Optional[T]` for nullable types
- Use `Union[T, U]` for multiple types
- Use `Literal` for string constants

#### Documentation
- All public functions and classes must have docstrings
- Use Google-style docstrings
- Include parameter types, return types, and examples

```python
def analyze_code(
    code: str,
    language: str,
    options: Optional[Dict[str, Any]] = None
) -> AnalysisResult:
    """Analyze code for quality and security issues.
    
    Args:
        code: Source code to analyze
        language: Programming language identifier
        options: Optional analysis configuration
        
    Returns:
        AnalysisResult containing metrics and suggestions
        
    Raises:
        ValueError: If code is empty or language is unsupported
    """
    pass
```

#### Error Handling
- Use specific exception types
- Include meaningful error messages
- Log errors with context
- Use structured logging

### TypeScript (Frontend)

#### Style Guide
- Follow [ESLint](https://eslint.org/) configuration
- Use [Prettier](https://prettier.io/) for formatting
- Maximum line length: 100 characters

#### Component Structure
```typescript
interface ComponentProps {
  // Define props here
}

const Component: React.FC<ComponentProps> = ({ prop1, prop2 }) => {
  // Component logic
  
  return (
    <div>
      {/* JSX */}
    </div>
  );
};

export default Component;
```

#### State Management
- Use React hooks for local state
- Use context for global state
- Prefer functional components over class components

## Testing

### Backend Testing

#### Unit Tests
- Test all public functions and methods
- Use pytest fixtures for setup
- Mock external dependencies
- Aim for >80% code coverage

```python
import pytest
from app.services.analysis import analyze_code

def test_analyze_code_success():
    """Test successful code analysis."""
    code = "def hello(): return 'world'"
    result = analyze_code(code, "python")
    
    assert result.quality_score > 0
    assert len(result.suggestions) >= 0
```

#### Integration Tests
- Test API endpoints
- Test database operations
- Test external service integrations

#### E2E Tests
- Test complete user workflows
- Use real services when possible

### Frontend Testing

#### Unit Tests
- Test component rendering
- Test user interactions
- Test state changes

```typescript
import { render, screen } from '@testing-library/react';
import Component from './Component';

test('renders component correctly', () => {
  render(<Component />);
  expect(screen.getByText('Expected Text')).toBeInTheDocument();
});
```

#### Integration Tests
- Test component interactions
- Test API integrations

## Documentation

### Code Documentation
- Keep docstrings up to date
- Document complex algorithms
- Add inline comments for non-obvious code

### API Documentation
- Use OpenAPI specifications
- Include request/response examples
- Document error responses

### README Updates
- Update feature lists
- Update setup instructions
- Update configuration examples

## Pull Request Process

### Before Submitting

1. **Run All Checks**
   ```bash
   make ci-test
   ```

2. **Update Documentation**
   - README if needed
   - API documentation
   - Changelog

3. **Review Your Changes**
   - Check for TODO comments
   - Remove debug code
   - Verify no secrets are committed

### Pull Request Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Tests added/updated
- [ ] No breaking changes (or documented)
```

### Review Process

1. **Automated Checks**
   - CI/CD pipeline runs
   - Code quality checks
   - Security scans

2. **Code Review**
   - At least one reviewer approval required
   - Address all review comments
   - Update based on feedback

3. **Merge**
   - Squash and merge commits
   - Delete feature branch
   - Update changelog

## Community Guidelines

### Code of Conduct

We are committed to providing a welcoming and inclusive environment. Please read our [Code of Conduct](CODE_OF_CONDUCT.md) for detailed guidelines.

### Communication

- Use GitHub issues for bug reports and feature requests
- Use discussions for questions and ideas
- Be respectful and constructive in all communications

### Getting Help

- Check existing issues and documentation
- Ask questions in GitHub discussions
- Join our community Slack/Discord

## Development Tools

### IDE Configuration

#### VS Code
Install these extensions:
- Python
- TypeScript and JavaScript Language Features
- ESLint
- Prettier
- Docker
- GitLens

#### PyCharm
- Configure code style to match project standards
- Set up type checking
- Configure test runner

### Useful Commands

```bash
# Development
make start-dev          # Start all services
make stop-dev           # Stop all services
make docker-logs        # View service logs

# Quality
make lint               # Run linting
make format             # Format code
make type-check         # Run type checking
make pre-commit         # Run all pre-commit hooks

# Testing
make test               # Run unit tests
make test-coverage      # Run tests with coverage
make test-integration   # Run integration tests

# Security
make security-scan      # Run security scans
make dependency-check   # Check for vulnerable dependencies

# Documentation
make docs-serve         # Serve documentation locally
make docs-build         # Build documentation
```

## Troubleshooting

### Common Issues

#### Docker Issues
```bash
# Clean up Docker resources
docker system prune -f
docker-compose down -v
docker-compose up -d --build
```

#### Python Environment Issues
```bash
# Recreate virtual environment
rm -rf .venv
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate     # Windows
pip install -e ".[dev]"
```

#### Frontend Build Issues
```bash
# Clear node modules and reinstall
rm -rf node_modules package-lock.json
npm install
```

### Getting Help

1. Check the [troubleshooting guide](docs/troubleshooting.md)
2. Search existing [GitHub issues](https://github.com/example/ai-code-review/issues)
3. Create a new issue with detailed information
4. Join our community discussions

Thank you for contributing to the AI Code Review and Architecture Analysis Platform!