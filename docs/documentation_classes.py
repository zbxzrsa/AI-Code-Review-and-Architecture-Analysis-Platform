"""
Documentation Classes for AI Code Review and Architecture Analysis Platform
This module contains structured classes for organizing and managing all project documentation.
"""

import os
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
from enum import Enum
from pathlib import Path

class DocumentType(Enum):
    """Document type enumeration"""
    ARCHITECTURE = "architecture"
    API = "api"
    USER_GUIDE = "user_guide"
    IMPLEMENTATION = "implementation"
    OPTIMIZATION = "optimization"
    DELIVERY = "delivery"
    STANDARDS = "standards"
    ROADMAP = "roadmap"
    CHECKLIST = "checklist"
    REPORT = "report"
    CLEANUP = "cleanup"
    USABILITY = "usability"
    PROGRESS = "progress"

class DocumentStatus(Enum):
    """Document status enumeration"""
    DRAFT = "draft"
    REVIEW = "review"
    APPROVED = "approved"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"

@dataclass
class DocumentMetadata:
    """Document metadata information"""
    title: str
    description: str
    author: str = "AI Platform Team"
    version: str = "1.0.0"
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    status: DocumentStatus = DocumentStatus.DRAFT
    tags: List[str] = field(default_factory=list)
    category: str = ""
    language: str = "zh-CN"
    word_count: int = 0
    reading_time_minutes: int = 0

@dataclass
class DocumentSection:
    """Document section structure"""
    title: str
    content: str
    level: int = 1  # Header level (1-6)
    subsections: List['DocumentSection'] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CodeExample:
    """Code example within documentation"""
    language: str
    code: str
    description: str
    file_path: Optional[str] = None
    line_numbers: bool = True
    theme: str = "dark"

@dataclass
class DiagramReference:
    """Diagram reference in documentation"""
    title: str
    file_path: str
    description: str
    diagram_type: str = "svg"  # svg, png, jpg, etc.
    width: Optional[int] = None
    height: Optional[int] = None

class BaseDocumentation:
    """Base class for all documentation types"""
    
    def __init__(self, metadata: DocumentMetadata):
        self.metadata = metadata
        self.sections: List[DocumentSection] = []
        self.code_examples: List[CodeExample] = []
        self.diagrams: List[DiagramReference] = []
        self.references: List[str] = []
        self.appendices: List[DocumentSection] = []
    
    def add_section(self, title: str, content: str, level: int = 1) -> DocumentSection:
        """Add a new section to the document"""
        section = DocumentSection(title=title, content=content, level=level)
        self.sections.append(section)
        return section
    
    def add_subsection(self, parent_section: DocumentSection, title: str, content: str) -> DocumentSection:
        """Add a subsection to an existing section"""
        subsection = DocumentSection(title=title, content=content, level=parent_section.level + 1)
        parent_section.subsections.append(subsection)
        return subsection
    
    def add_code_example(self, language: str, code: str, description: str, file_path: str = None) -> CodeExample:
        """Add a code example to the document"""
        example = CodeExample(
            language=language,
            code=code,
            description=description,
            file_path=file_path
        )
        self.code_examples.append(example)
        return example
    
    def add_diagram(self, title: str, file_path: str, description: str, diagram_type: str = "svg") -> DiagramReference:
        """Add a diagram reference to the document"""
        diagram = DiagramReference(
            title=title,
            file_path=file_path,
            description=description,
            diagram_type=diagram_type
        )
        self.diagrams.append(diagram)
        return diagram
    
    def add_reference(self, reference: str) -> None:
        """Add a reference to the document"""
        if reference not in self.references:
            self.references.append(reference)
    
    def add_appendix(self, title: str, content: str) -> DocumentSection:
        """Add an appendix to the document"""
        appendix = DocumentSection(title=title, content=content, level=1)
        self.appendices.append(appendix)
        return appendix
    
    def get_table_of_contents(self) -> List[Dict[str, Any]]:
        """Generate table of contents"""
        toc = []
        for section in self.sections:
            toc.append({
                'title': section.title,
                'level': section.level,
                'subsections': self._get_subsection_toc(section.subsections)
            })
        return toc
    
    def _get_subsection_toc(self, subsections: List[DocumentSection]) -> List[Dict[str, Any]]:
        """Recursively get subsection TOC"""
        toc = []
        for subsection in subsections:
            toc.append({
                'title': subsection.title,
                'level': subsection.level,
                'subsections': self._get_subsection_toc(subsection.subsections)
            })
        return toc
    
    def update_metadata(self, **kwargs) -> None:
        """Update document metadata"""
        for key, value in kwargs.items():
            if hasattr(self.metadata, key):
                setattr(self.metadata, key, value)
        self.metadata.updated_at = datetime.now()
    
    def calculate_statistics(self) -> Dict[str, Any]:
        """Calculate document statistics"""
        total_content = " ".join([section.content for section in self.sections])
        word_count = len(total_content.split())
        reading_time = max(1, word_count // 200)  # Average reading speed
        
        return {
            'word_count': word_count,
            'reading_time_minutes': reading_time,
            'section_count': len(self.sections),
            'code_example_count': len(self.code_examples),
            'diagram_count': len(self.diagrams),
            'reference_count': len(self.references)
        }

class ArchitectureDocumentation(BaseDocumentation):
    """Architecture documentation class"""
    
    def __init__(self, metadata: DocumentMetadata = None):
        if metadata is None:
            metadata = DocumentMetadata(
                title="System Architecture Documentation",
                description="Comprehensive system architecture overview and design patterns",
                category="technical",
                tags=["architecture", "design", "patterns", "system"]
            )
        super().__init__(metadata)
        self.document_type = DocumentType.ARCHITECTURE
        self.principles: List[str] = []
        self.components: Dict[str, Any] = {}
        self.design_patterns: List[str] = []
        self.technology_stack: Dict[str, List[str]] = {}
    
    def add_architecture_principle(self, principle: str, description: str) -> None:
        """Add an architecture principle"""
        self.principles.append(principle)
        # Add as section for detailed description
        self.add_section(f"Principle: {principle}", description)
    
    def add_component(self, name: str, description: str, responsibilities: List[str]) -> None:
        """Add a system component"""
        self.components[name] = {
            'description': description,
            'responsibilities': responsibilities
        }
        
        content = f"**Description**: {description}\n\n**Responsibilities**:\n"
        for resp in responsibilities:
            content += f"- {resp}\n"
        
        self.add_section(f"Component: {name}", content)
    
    def add_design_pattern(self, pattern: str, description: str, usage: str) -> None:
        """Add a design pattern"""
        self.design_patterns.append(pattern)
        
        content = f"**Description**: {description}\n\n**Usage**: {usage}"
        self.add_section(f"Pattern: {pattern}", content)
    
    def add_technology_stack(self, category: str, technologies: List[str]) -> None:
        """Add technology stack information"""
        self.technology_stack[category] = technologies
        
        content = "**Technologies**:\n"
        for tech in technologies:
            content += f"- {tech}\n"
        
        self.add_section(f"Technology Stack: {category}", content)

class APIDocumentation(BaseDocumentation):
    """API documentation class"""
    
    def __init__(self, metadata: DocumentMetadata = None):
        if metadata is None:
            metadata = DocumentMetadata(
                title="API Documentation",
                description="Complete API reference and usage examples",
                category="technical",
                tags=["api", "endpoints", "reference", "integration"]
            )
        super().__init__(metadata)
        self.document_type = DocumentType.API
        self.endpoints: List[Dict[str, Any]] = []
        self.schemas: Dict[str, Any] = {}
        self.authentication: Dict[str, Any] = {}
    
    def add_endpoint(self, method: str, path: str, description: str, 
                   parameters: List[Dict] = None, responses: Dict[str, Any] = None) -> None:
        """Add an API endpoint"""
        endpoint = {
            'method': method.upper(),
            'path': path,
            'description': description,
            'parameters': parameters or [],
            'responses': responses or {}
        }
        self.endpoints.append(endpoint)
        
        content = f"**{method.upper()} {path}**\n\n{description}\n\n"
        if parameters:
            content += "**Parameters**:\n"
            for param in parameters:
                content += f"- `{param['name']}` ({param['type']}): {param['description']}\n"
            content += "\n"
        
        if responses:
            content += "**Responses**:\n"
            for status, response in responses.items():
                content += f"- `{status}`: {response['description']}\n"
        
        self.add_section(f"Endpoint: {method.upper()} {path}", content)
    
    def add_schema(self, name: str, schema: Dict[str, Any]) -> None:
        """Add a data schema"""
        self.schemas[name] = schema
        
        content = f"**Schema**: {name}\n\n```json\n{json.dumps(schema, indent=2)}\n```"
        self.add_section(f"Schema: {name}", content)
    
    def add_authentication_method(self, method: str, description: str, example: str = None) -> None:
        """Add authentication method"""
        self.authentication[method] = {
            'description': description,
            'example': example
        }
        
        content = f"**{method}**\n\n{description}"
        if example:
            content += f"\n\n**Example**:\n```{example}```"
        
        self.add_section(f"Authentication: {method}", content)

class UserGuideDocumentation(BaseDocumentation):
    """User guide documentation class"""
    
    def __init__(self, metadata: DocumentMetadata = None):
        if metadata is None:
            metadata = DocumentMetadata(
                title="User Guide",
                description="Comprehensive user guide for the AI Code Review Platform",
                category="user",
                tags=["user-guide", "tutorial", "getting-started", "how-to"]
            )
        super().__init__(metadata)
        self.document_type = DocumentType.USER_GUIDE
        self.tutorials: List[Dict[str, Any]] = []
        self.faq: List[Dict[str, str]] = []
        self.troubleshooting: List[Dict[str, str]] = []
    
    def add_tutorial(self, title: str, description: str, steps: List[str], 
                   difficulty: str = "beginner") -> None:
        """Add a tutorial"""
        tutorial = {
            'title': title,
            'description': description,
            'steps': steps,
            'difficulty': difficulty
        }
        self.tutorials.append(tutorial)
        
        content = f"**Difficulty**: {difficulty}\n\n{description}\n\n**Steps**:\n"
        for i, step in enumerate(steps, 1):
            content += f"{i}. {step}\n"
        
        self.add_section(f"Tutorial: {title}", content)
    
    def add_faq_item(self, question: str, answer: str) -> None:
        """Add an FAQ item"""
        self.faq.append({'question': question, 'answer': answer})
        
        content = f"**Q**: {question}\n\n**A**: {answer}"
        self.add_section(f"FAQ: {question[:50]}...", content)
    
    def add_troubleshooting_item(self, problem: str, solution: str) -> None:
        """Add a troubleshooting item"""
        self.troubleshooting.append({'problem': problem, 'solution': solution})
        
        content = f"**Problem**: {problem}\n\n**Solution**: {solution}"
        self.add_section(f"Troubleshooting: {problem[:50]}...", content)

class ImplementationDocumentation(BaseDocumentation):
    """Implementation documentation class"""
    
    def __init__(self, metadata: DocumentMetadata = None):
        if metadata is None:
            metadata = DocumentMetadata(
                title="Implementation Guide",
                description="Detailed implementation guidelines and best practices",
                category="development",
                tags=["implementation", "development", "guidelines", "best-practices"]
            )
        super().__init__(metadata)
        self.document_type = DocumentType.IMPLEMENTATION
        self.guidelines: List[Dict[str, Any]] = []
        self.code_standards: Dict[str, Any] = {}
        self.testing_guidelines: List[str] = []
    
    def add_guideline(self, title: str, description: str, examples: List[str] = None) -> None:
        """Add an implementation guideline"""
        guideline = {
            'title': title,
            'description': description,
            'examples': examples or []
        }
        self.guidelines.append(guideline)
        
        content = description
        if examples:
            content += "\n\n**Examples**:\n"
            for example in examples:
                content += f"- {example}\n"
        
        self.add_section(f"Guideline: {title}", content)
    
    def add_code_standard(self, category: str, rules: List[str]) -> None:
        """Add code standards"""
        self.code_standards[category] = rules
        
        content = f"**Category**: {category}\n\n**Rules**:\n"
        for rule in rules:
            content += f"- {rule}\n"
        
        self.add_section(f"Code Standard: {category}", content)

class DocumentationManager:
    """Manager for all documentation classes"""
    
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.documents: Dict[str, BaseDocumentation] = {}
        self.index: Dict[str, Any] = {}
    
    def create_architecture_doc(self, doc_id: str = "architecture") -> ArchitectureDocumentation:
        """Create architecture documentation"""
        doc = ArchitectureDocumentation()
        self.documents[doc_id] = doc
        return doc
    
    def create_api_doc(self, doc_id: str = "api") -> APIDocumentation:
        """Create API documentation"""
        doc = APIDocumentation()
        self.documents[doc_id] = doc
        return doc
    
    def create_user_guide(self, doc_id: str = "user_guide") -> UserGuideDocumentation:
        """Create user guide documentation"""
        doc = UserGuideDocumentation()
        self.documents[doc_id] = doc
        return doc
    
    def create_implementation_doc(self, doc_id: str = "implementation") -> ImplementationDocumentation:
        """Create implementation documentation"""
        doc = ImplementationDocumentation()
        self.documents[doc_id] = doc
        return doc
    
    def get_document(self, doc_id: str) -> Optional[BaseDocumentation]:
        """Get a document by ID"""
        return self.documents.get(doc_id)
    
    def list_documents(self) -> List[Dict[str, Any]]:
        """List all documents"""
        return [
            {
                'id': doc_id,
                'title': doc.metadata.title,
                'type': doc.document_type.value if hasattr(doc, 'document_type') else 'unknown',
                'status': doc.metadata.status.value,
                'updated_at': doc.metadata.updated_at.isoformat(),
                'category': doc.metadata.category,
                'tags': doc.metadata.tags
            }
            for doc_id, doc in self.documents.items()
        ]
    
    def generate_index(self) -> None:
        """Generate documentation index"""
        self.index = {
            'generated_at': datetime.now().isoformat(),
            'total_documents': len(self.documents),
            'categories': {},
            'documents': self.list_documents()
        }
        
        # Group by category
        for doc_info in self.index['documents']:
            category = doc_info['category']
            if category not in self.index['categories']:
                self.index['categories'][category] = []
            self.index['categories'][category].append(doc_info)
    
    def save_document(self, doc_id: str, format: str = "markdown") -> str:
        """Save document to file"""
        doc = self.documents.get(doc_id)
        if not doc:
            raise ValueError(f"Document {doc_id} not found")
        
        # Update statistics
        stats = doc.calculate_statistics()
        doc.metadata.word_count = stats['word_count']
        doc.metadata.reading_time_minutes = stats['reading_time_minutes']
        
        # Generate content
        content = self._generate_document_content(doc, format)
        
        # Save to file
        file_path = self.base_path / f"{doc_id}.{format}"
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return str(file_path)
    
    def _generate_document_content(self, doc: BaseDocumentation, format: str) -> str:
        """Generate document content in specified format"""
        if format.lower() == "markdown":
            return self._generate_markdown(doc)
        elif format.lower() == "html":
            return self._generate_html(doc)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _generate_markdown(self, doc: BaseDocumentation) -> str:
        """Generate Markdown content"""
        content = []
        
        # Header
        content.append(f"# {doc.metadata.title}")
        content.append("")
        
        # Metadata
        content.append("## Document Information")
        content.append("")
        content.append(f"- **Author**: {doc.metadata.author}")
        content.append(f"- **Version**: {doc.metadata.version}")
        content.append(f"- **Status**: {doc.metadata.status.value}")
        content.append(f"- **Created**: {doc.metadata.created_at.strftime('%Y-%m-%d')}")
        content.append(f"- **Updated**: {doc.metadata.updated_at.strftime('%Y-%m-%d')}")
        content.append(f"- **Reading Time**: {doc.metadata.reading_time_minutes} minutes")
        content.append("")
        
        # Tags
        if doc.metadata.tags:
            content.append("**Tags**: " + ", ".join(doc.metadata.tags))
            content.append("")
        
        # Table of Contents
        if doc.sections:
            content.append("## Table of Contents")
            content.append("")
            toc = doc.get_table_of_contents()
            for item in toc:
                content.append(f"{'  ' * (item['level'] - 1)}- [{item['title']}](#{self._slugify(item['title'])})")
            content.append("")
        
        # Sections
        for section in doc.sections:
            content.extend(self._generate_section_markdown(section))
        
        # Code Examples
        if doc.code_examples:
            content.append("## Code Examples")
            content.append("")
            for example in doc.code_examples:
                content.append(f"### {example.description}")
                content.append("")
                content.append(f"```{example.language}")
                content.append(example.code)
                content.append("```")
                content.append("")
        
        # Diagrams
        if doc.diagrams:
            content.append("## Diagrams")
            content.append("")
            for diagram in doc.diagrams:
                content.append(f"### {diagram.title}")
                content.append("")
                content.append(f"{diagram.description}")
                content.append("")
                content.append(f"![{diagram.title}]({diagram.file_path})")
                content.append("")
        
        # References
        if doc.references:
            content.append("## References")
            content.append("")
            for i, ref in enumerate(doc.references, 1):
                content.append(f"{i}. {ref}")
            content.append("")
        
        return "\n".join(content)
    
    def _generate_section_markdown(self, section: DocumentSection, level: int = None) -> List[str]:
        """Generate Markdown for a section"""
        if level is None:
            level = section.level
        
        content = []
        header = "#" * level
        content.append(f"{header} {section.title}")
        content.append("")
        content.append(section.content)
        content.append("")
        
        # Subsections
        for subsection in section.subsections:
            content.extend(self._generate_section_markdown(subsection, level + 1))
        
        return content
    
    def _generate_html(self, doc: BaseDocumentation) -> str:
        """Generate HTML content"""
        # Basic HTML generation (can be enhanced with proper HTML library)
        content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{doc.metadata.title}</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }}
        h1, h2, h3, h4, h5, h6 {{ color: #333; }}
        code {{ background-color: #f4f4f4; padding: 2px 4px; border-radius: 3px; }}
        pre {{ background-color: #f4f4f4; padding: 10px; border-radius: 5px; overflow-x: auto; }}
        .metadata {{ background-color: #f9f9f9; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
    </style>
</head>
<body>
    <div class="metadata">
        <h2>Document Information</h2>
        <p><strong>Author:</strong> {doc.metadata.author}</p>
        <p><strong>Version:</strong> {doc.metadata.version}</p>
        <p><strong>Status:</strong> {doc.metadata.status.value}</p>
        <p><strong>Created:</strong> {doc.metadata.created_at.strftime('%Y-%m-%d')}</p>
        <p><strong>Updated:</strong> {doc.metadata.updated_at.strftime('%Y-%m-%d')}</p>
        <p><strong>Reading Time:</strong> {doc.metadata.reading_time_minutes} minutes</p>
    </div>
    
    <h1>{doc.metadata.title}</h1>
    
    {self._generate_sections_html(doc.sections)}
</body>
</html>
        """
        return content
    
    def _generate_sections_html(self, sections: List[DocumentSection]) -> str:
        """Generate HTML for sections"""
        content = []
        for section in sections:
            header_tag = f"h{section.level}"
            content.append(f"<{header_tag}>{section.title}</{header_tag}>")
            content.append(f"<div>{section.content}</div>")
            
            if section.subsections:
                content.append(self._generate_sections_html(section.subsections))
        
        return "".join(content)
    
    def _slugify(self, text: str) -> str:
        """Convert text to URL-friendly slug"""
        # Convert to lowercase and replace spaces with hyphens
        slug = text.lower()
        slug = re.sub(r'[^\w\s-]', '', slug)
        slug = re.sub(r'[-\s]+', '-', slug)
        return slug.strip('-')
    
    def save_all_documents(self, format: str = "markdown") -> List[str]:
        """Save all documents"""
        saved_files = []
        for doc_id in self.documents:
            file_path = self.save_document(doc_id, format)
            saved_files.append(file_path)
        
        # Save index
        self.generate_index()
        index_path = self.base_path / "index.json"
        with open(index_path, 'w', encoding='utf-8') as f:
            json.dump(self.index, f, indent=2, ensure_ascii=False)
        saved_files.append(str(index_path))
        
        return saved_files
    
    def load_existing_documentation(self) -> None:
        """Load existing documentation from files"""
        # This would parse existing markdown files and convert to class structure
        # Implementation depends on existing file formats
        pass

# Factory function
def create_documentation_manager(base_path: str = "D:\\Desktop\\AI Code Review and Architecture Analysis Platform\\docs") -> DocumentationManager:
    """Create and return a documentation manager"""
    # Ensure the directory exists
    Path(base_path).mkdir(parents=True, exist_ok=True)
    
    return DocumentationManager(base_path)

# Example usage
if __name__ == "__main__":
    # Create documentation manager
    doc_manager = create_documentation_manager()
    
    # Create architecture documentation
    arch_doc = doc_manager.create_architecture_doc()
    
    # Add architecture principles
    arch_doc.add_architecture_principle(
        "Separation of Concerns",
        "Each module should focus on a single responsibility and be independent of other modules."
    )
    
    # Add components
    arch_doc.add_component(
        "Frontend",
        "React-based user interface",
        ["Render UI components", "Handle user interactions", "Manage client state"]
    )
    
    # Add technology stack
    arch_doc.add_technology_stack("Frontend", ["React", "TypeScript", "Ant Design", "Redux"])
    arch_doc.add_technology_stack("Backend", ["FastAPI", "Python", "PostgreSQL", "Redis"])
    
    # Create API documentation
    api_doc = doc_manager.create_api_doc()
    
    # Add endpoints
    api_doc.add_endpoint(
        "POST",
        "/api/v1/analyze",
        "Analyze code for defects and quality issues",
        parameters=[
            {"name": "code", "type": "string", "description": "Code to analyze"},
            {"name": "language", "type": "string", "description": "Programming language"}
        ],
        responses={
            "200": {"description": "Analysis results"},
            "400": {"description": "Invalid input"}
        }
    )
    
    # Create user guide
    user_guide = doc_manager.create_user_guide()
    
    # Add tutorials
    user_guide.add_tutorial(
        "Getting Started",
        "Learn how to use the AI Code Review Platform",
        [
            "Sign up for an account",
            "Create a new project",
            "Upload your code",
            "Review analysis results"
        ]
    )
    
    # Save all documentation
    saved_files = doc_manager.save_all_documents("markdown")
    print(f"Documentation saved to: {saved_files}")