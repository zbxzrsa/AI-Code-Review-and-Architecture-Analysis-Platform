"""
Documentation Processor for AI Code Review and Architecture Analysis Platform
Converts existing documentation to class-based structure
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Any, Optional
from documentation_classes import (
    DocumentationManager, DocumentType, DocumentStatus, DocumentMetadata,
    ArchitectureDocumentation, APIDocumentation, UserGuideDocumentation,
    ImplementationDocumentation
)

class DocumentationProcessor:
    """Processes existing documentation files and converts them to class-based structure"""
    
    def __init__(self, docs_path: str):
        self.docs_path = Path(docs_path)
        self.doc_manager = DocumentationManager(docs_path)
        self.processed_files = {}
    
    def process_all_documentation(self) -> None:
        """Process all documentation files in the docs directory"""
        print("Processing all documentation files...")
        
        # Process architecture documentation
        self._process_architecture_docs()
        
        # Process API documentation
        self._process_api_docs()
        
        # Process user guides
        self._process_user_guides()
        
        # Process implementation guides
        self._process_implementation_docs()
        
        # Process optimization reports
        self._process_optimization_docs()
        
        # Process delivery summaries
        self._process_delivery_docs()
        
        # Process checklists
        self._process_checklist_docs()
        
        # Save all processed documentation
        self._save_all_processed_docs()
        
        print(f"Documentation processing complete. Files saved to: {self.docs_path}")
    
    def _process_architecture_docs(self) -> None:
        """Process architecture documentation"""
        arch_file = self.docs_path / "architecture.md"
        if arch_file.exists():
            print(f"Processing architecture documentation: {arch_file}")
            
            with open(arch_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Create architecture documentation
            arch_doc = self.doc_manager.create_architecture_doc("system_architecture")
            
            # Parse and add architecture principles
            principles = self._extract_section(content, "架构原则")
            if principles:
                principle_list = self._extract_list_items(principles)
                for principle in principle_list:
                    arch_doc.add_architecture_principle(
                        principle.split('：')[0].strip(),
                        principle.split('：')[1].strip() if '：' in principle else principle
                    )
            
            # Parse and add frontend architecture
            frontend_section = self._extract_section(content, "前端架构")
            if frontend_section:
                tech_stack = self._extract_tech_stack(frontend_section)
                for category, technologies in tech_stack.items():
                    arch_doc.add_technology_stack(f"Frontend - {category}", technologies)
                
                # Add frontend components
                arch_doc.add_component(
                    "Frontend Application",
                    "React-based user interface with TypeScript",
                    ["UI rendering", "User interaction handling", "State management", "API communication"]
                )
            
            # Parse and add backend architecture
            backend_section = self._extract_section(content, "后端架构")
            if backend_section:
                tech_stack = self._extract_tech_stack(backend_section)
                for category, technologies in tech_stack.items():
                    arch_doc.add_technology_stack(f"Backend - {category}", technologies)
                
                # Add backend components
                arch_doc.add_component(
                    "Backend API",
                    "FastAPI-based REST API with Python",
                    ["HTTP request handling", "Business logic implementation", "Data access", "Authentication"]
                )
            
            # Add design patterns
            patterns_section = self._extract_section(content, "设计模式应用")
            if patterns_section:
                patterns = self._extract_list_items(patterns_section)
                for pattern in patterns:
                    arch_doc.add_design_pattern(
                        pattern.split('：')[0].strip(),
                        pattern.split('：')[1].strip() if '：' in pattern else pattern,
                        "Applied throughout the system architecture"
                    )
            
            # Add architecture diagram reference
            arch_doc.add_diagram(
                "System Architecture Overview",
                "architecture.svg",
                "High-level system architecture showing all major components and their interactions"
            )
            
            # Update metadata
            arch_doc.update_metadata(
                title="System Architecture Documentation",
                description="Comprehensive system architecture overview including principles, components, and design patterns",
                status=DocumentStatus.APPROVED,
                tags=["architecture", "design", "patterns", "components", "technology-stack"]
            )
            
            self.processed_files['architecture'] = arch_doc
    
    def _process_api_docs(self) -> None:
        """Process API documentation"""
        api_files = [
            "ai_analysis_api.md",
            "api_documentation.md"
        ]
        
        for api_file in api_files:
            file_path = self.docs_path / api_file
            if file_path.exists():
                print(f"Processing API documentation: {file_path}")
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Create API documentation
                api_doc = self.doc_manager.create_api_doc(api_file.replace('.md', ''))
                
                # Extract endpoints (simplified parsing)
                endpoints = self._extract_api_endpoints(content)
                for endpoint in endpoints:
                    api_doc.add_endpoint(
                        endpoint.get('method', 'GET'),
                        endpoint.get('path', '/'),
                        endpoint.get('description', ''),
                        endpoint.get('parameters', []),
                        endpoint.get('responses', {})
                    )
                
                # Add authentication
                api_doc.add_authentication_method(
                    "JWT Bearer Token",
                    "Authentication using JSON Web Tokens",
                    "Authorization: Bearer <token>"
                )
                
                # Update metadata
                api_doc.update_metadata(
                    title=f"API Documentation - {api_file.replace('.md', '').replace('_', ' ').title()}",
                    description=f"API reference for {api_file.replace('.md', '')}",
                    status=DocumentStatus.APPROVED,
                    tags=["api", "endpoints", "authentication", "reference"]
                )
                
                self.processed_files[api_file.replace('.md', '')] = api_doc
    
    def _process_user_guides(self) -> None:
        """Process user guide documentation"""
        user_guide_files = [
            "user_guide.md",
            "QUICK_START.md",
            "shortcut-creation-guide.md"
        ]
        
        for guide_file in user_guide_files:
            file_path = self.docs_path / guide_file
            if file_path.exists():
                print(f"Processing user guide: {file_path}")
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Create user guide documentation
                user_doc = self.doc_manager.create_user_guide(guide_file.replace('.md', ''))
                
                # Extract tutorials or steps
                title = self._extract_title(content)
                if title:
                    # Add as tutorial
                    steps = self._extract_steps_from_content(content)
                    if steps:
                        user_doc.add_tutorial(
                            title,
                            self._extract_description(content),
                            steps,
                            "beginner"
                        )
                
                # Add FAQ if present
                faq_section = self._extract_section(content, "FAQ") or self._extract_section(content, "常见问题")
                if faq_section:
                    faq_items = self._extract_faq_items(faq_section)
                    for question, answer in faq_items:
                        user_doc.add_faq_item(question, answer)
                
                # Update metadata
                user_doc.update_metadata(
                    title=title or guide_file.replace('.md', '').replace('_', ' ').title(),
                    description=self._extract_description(content),
                    status=DocumentStatus.APPROVED,
                    tags=["user-guide", "tutorial", "getting-started"]
                )
                
                self.processed_files[guide_file.replace('.md', '')] = user_doc
    
    def _process_implementation_docs(self) -> None:
        """Process implementation documentation"""
        impl_files = [
            "code-standards.md",
            "IMPLEMENTATION_ROADMAP.md",
            "cleanup_plan.md"
        ]
        
        for impl_file in impl_files:
            file_path = self.docs_path / impl_file
            if file_path.exists():
                print(f"Processing implementation documentation: {file_path}")
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Create implementation documentation
                impl_doc = self.doc_manager.create_implementation_doc(impl_file.replace('.md', ''))
                
                # Extract guidelines
                guidelines = self._extract_guidelines(content)
                for guideline in guidelines:
                    impl_doc.add_guideline(
                        guideline.get('title', ''),
                        guideline.get('description', ''),
                        guideline.get('examples', [])
                    )
                
                # Extract code standards
                if 'standards' in impl_file.lower():
                    standards = self._extract_code_standards(content)
                    for category, rules in standards.items():
                        impl_doc.add_code_standard(category, rules)
                
                # Update metadata
                impl_doc.update_metadata(
                    title=self._extract_title(content) or impl_file.replace('.md', '').replace('_', ' ').title(),
                    description=self._extract_description(content),
                    status=DocumentStatus.APPROVED,
                    tags=["implementation", "guidelines", "standards", "development"]
                )
                
                self.processed_files[impl_file.replace('.md', '')] = impl_doc
    
    def _process_optimization_docs(self) -> None:
        """Process optimization documentation"""
        opt_files = [
            "OPTIMIZATION_REPORT.md",
            "FINAL_OPTIMIZATION_REPORT.md"
        ]
        
        for opt_file in opt_files:
            file_path = self.docs_path / opt_file
            if file_path.exists():
                print(f"Processing optimization documentation: {file_path}")
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Create implementation documentation for optimization
                opt_doc = self.doc_manager.create_implementation_doc(f"optimization_{opt_file.replace('.md', '')}")
                
                # Extract optimization guidelines
                optimizations = self._extract_optimization_items(content)
                for opt in optimizations:
                    opt_doc.add_guideline(
                        opt.get('title', ''),
                        opt.get('description', ''),
                        opt.get('examples', [])
                    )
                
                # Update metadata
                opt_doc.update_metadata(
                    title=f"Optimization Report - {opt_file.replace('.md', '').replace('_', ' ').title()}",
                    description=self._extract_description(content),
                    status=DocumentStatus.APPROVED,
                    tags=["optimization", "performance", "improvements"]
                )
                
                self.processed_files[f"optimization_{opt_file.replace('.md', '')}"] = opt_doc
    
    def _process_delivery_docs(self) -> None:
        """Process delivery documentation"""
        delivery_files = [
            "DELIVERY_SUMMARY.md",
            "PHASE_1_DELIVERY_SUMMARY.md",
            "PHASE_2_DELIVERY_SUMMARY.md"
        ]
        
        for delivery_file in delivery_files:
            file_path = self.docs_path / delivery_file
            if file_path.exists():
                print(f"Processing delivery documentation: {file_path}")
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Create implementation documentation for delivery
                delivery_doc = self.doc_manager.create_implementation_doc(f"delivery_{delivery_file.replace('.md', '')}")
                
                # Extract delivery information
                delivery_info = self._extract_delivery_info(content)
                for info in delivery_info:
                    delivery_doc.add_guideline(
                        info.get('title', ''),
                        info.get('description', ''),
                        info.get('details', [])
                    )
                
                # Update metadata
                delivery_doc.update_metadata(
                    title=f"Delivery Summary - {delivery_file.replace('.md', '').replace('_', ' ').title()}",
                    description=self._extract_description(content),
                    status=DocumentStatus.APPROVED,
                    tags=["delivery", "summary", "milestone"]
                )
                
                self.processed_files[f"delivery_{delivery_file.replace('.md', '')}"] = delivery_doc
    
    def _process_checklist_docs(self) -> None:
        """Process checklist documentation"""
        checklist_files = [
            "PHASE_1_ACCEPTANCE_CHECKLIST.md",
            "PHASE_2_ACCEPTANCE_CHECKLIST.md",
            "usability-checklist.md"
        ]
        
        for checklist_file in checklist_files:
            file_path = self.docs_path / checklist_file
            if file_path.exists():
                print(f"Processing checklist documentation: {file_path}")
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Create implementation documentation for checklists
                checklist_doc = self.doc_manager.create_implementation_doc(f"checklist_{checklist_file.replace('.md', '')}")
                
                # Extract checklist items
                checklist_items = self._extract_checklist_items(content)
                for item in checklist_items:
                    checklist_doc.add_guideline(
                        item.get('title', ''),
                        item.get('description', ''),
                        item.get('criteria', [])
                    )
                
                # Update metadata
                checklist_doc.update_metadata(
                    title=f"Checklist - {checklist_file.replace('.md', '').replace('_', ' ').title()}",
                    description=self._extract_description(content),
                    status=DocumentStatus.APPROVED,
                    tags=["checklist", "acceptance", "criteria", "quality"]
                )
                
                self.processed_files[f"checklist_{checklist_file.replace('.md', '')}"] = checklist_doc
    
    def _extract_section(self, content: str, section_name: str) -> Optional[str]:
        """Extract a specific section from content"""
        pattern = rf'## {re.escape(section_name)}\s*\n(.*?)(?=\n## |\Z)'
        match = re.search(pattern, content, re.DOTALL)
        return match.group(1).strip() if match else None
    
    def _extract_title(self, content: str) -> Optional[str]:
        """Extract the main title from content"""
        match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
        return match.group(1).strip() if match else None
    
    def _extract_description(self, content: str) -> str:
        """Extract description from content"""
        # Look for the first paragraph after the title
        lines = content.split('\n')
        for line in lines[1:]:  # Skip title
            line = line.strip()
            if line and not line.startswith('#'):
                return line
        return "No description available"
    
    def _extract_list_items(self, content: str) -> List[str]:
        """Extract list items from content"""
        items = []
        for match in re.finditer(r'^\d+\.\s*\*\*(.+?)\*\*:\s*(.+)$', content, re.MULTILINE):
            items.append(f"{match.group(1).strip()}: {match.group(2).strip()}")
        return items
    
    def _extract_tech_stack(self, content: str) -> Dict[str, List[str]]:
        """Extract technology stack from content"""
        tech_stack = {}
        
        # Look for technology lists
        tech_section = self._extract_section(content, "技术栈")
        if tech_section:
            lines = tech_section.split('\n')
            current_category = None
            
            for line in lines:
                line = line.strip()
                if line.startswith('-'):
                    # Technology item
                    if current_category:
                        tech = line.replace('-', '').strip()
                        if current_category not in tech_stack:
                            tech_stack[current_category] = []
                        tech_stack[current_category].append(tech)
                elif line and not line.startswith('#'):
                    # Category (heuristic)
                    current_category = line.replace(':', '').strip()
        
        return tech_stack
    
    def _extract_api_endpoints(self, content: str) -> List[Dict[str, Any]]:
        """Extract API endpoints from content"""
        endpoints = []
        
        # Look for endpoint patterns
        endpoint_pattern = r'(GET|POST|PUT|DELETE|PATCH)\s+(/[^\s]+)'
        for match in re.finditer(endpoint_pattern, content):
            endpoints.append({
                'method': match.group(1),
                'path': match.group(2),
                'description': f"{match.group(1)} {match.group(2)} endpoint",
                'parameters': [],
                'responses': {'200': {'description': 'Success'}}
            })
        
        return endpoints
    
    def _extract_steps_from_content(self, content: str) -> List[str]:
        """Extract steps from content"""
        steps = []
        
        # Look for numbered lists
        for match in re.finditer(r'^\d+\.\s+(.+)$', content, re.MULTILINE):
            steps.append(match.group(1).strip())
        
        return steps
    
    def _extract_faq_items(self, content: str) -> List[tuple]:
        """Extract FAQ items from content"""
        faq_items = []
        
        # Look for Q&A patterns
        qa_pattern = r'([Qq]\d*[:\.]\s*.+?)\s*([Aa]\d*[:\.]\s*.+?)(?=\n[Qq]|\n[Aa]|\Z)'
        for match in re.finditer(qa_pattern, content, re.DOTALL):
            question = re.sub(r'^[Qq]\d*[:\.]\s*', '', match.group(1).strip())
            answer = re.sub(r'^[Aa]\d*[:\.]\s*', '', match.group(2).strip())
            faq_items.append((question, answer))
        
        return faq_items
    
    def _extract_guidelines(self, content: str) -> List[Dict[str, Any]]:
        """Extract guidelines from content"""
        guidelines = []
        
        # Look for guideline patterns
        sections = re.split(r'\n#{1,3}\s+', content)
        for section in sections[1:]:  # Skip title
            lines = section.strip().split('\n')
            if lines:
                title = lines[0].strip()
                description = '\n'.join(lines[1:]).strip()
                
                if title and description:
                    guidelines.append({
                        'title': title,
                        'description': description,
                        'examples': []
                    })
        
        return guidelines
    
    def _extract_code_standards(self, content: str) -> Dict[str, List[str]]:
        """Extract code standards from content"""
        standards = {}
        
        # Look for standard patterns
        sections = re.split(r'\n#{1,3}\s+', content)
        current_category = None
        
        for section in sections[1:]:  # Skip title
            lines = section.strip().split('\n')
            if lines:
                title = lines[0].strip()
                
                # Check if this looks like a category
                if any(keyword in title.lower() for keyword in ['style', 'format', 'naming', 'structure']):
                    current_category = title
                    standards[current_category] = []
                elif current_category and lines[0].startswith('-'):
                    # This is a rule
                    rule = lines[0].replace('-', '').strip()
                    standards[current_category].append(rule)
        
        return standards
    
    def _extract_optimization_items(self, content: str) -> List[Dict[str, Any]]:
        """Extract optimization items from content"""
        optimizations = []
        
        # Look for optimization sections
        sections = re.split(r'\n#{1,3}\s+', content)
        for section in sections[1:]:  # Skip title
            lines = section.strip().split('\n')
            if lines:
                title = lines[0].strip()
                description = '\n'.join(lines[1:]).strip()
                
                if title and description:
                    optimizations.append({
                        'title': title,
                        'description': description,
                        'examples': []
                    })
        
        return optimizations
    
    def _extract_delivery_info(self, content: str) -> List[Dict[str, Any]]:
        """Extract delivery information from content"""
        delivery_info = []
        
        # Look for delivery sections
        sections = re.split(r'\n#{1,3}\s+', content)
        for section in sections[1:]:  # Skip title
            lines = section.strip().split('\n')
            if lines:
                title = lines[0].strip()
                description = '\n'.join(lines[1:]).strip()
                
                if title and description:
                    delivery_info.append({
                        'title': title,
                        'description': description,
                        'details': []
                    })
        
        return delivery_info
    
    def _extract_checklist_items(self, content: str) -> List[Dict[str, Any]]:
        """Extract checklist items from content"""
        checklist_items = []
        
        # Look for checkbox patterns
        sections = re.split(r'\n#{1,3}\s+', content)
        for section in sections[1:]:  # Skip title
            lines = section.strip().split('\n')
            if lines:
                title = lines[0].strip()
                description = '\n'.join(lines[1:]).strip()
                
                if title and description:
                    checklist_items.append({
                        'title': title,
                        'description': description,
                        'criteria': []
                    })
        
        return checklist_items
    
    def _save_all_processed_docs(self) -> None:
        """Save all processed documentation"""
        print("Saving all processed documentation...")
        
        # Save each document
        for doc_id, doc in self.processed_files.items():
            try:
                file_path = self.doc_manager.save_document(doc_id, "markdown")
                print(f"Saved: {file_path}")
            except Exception as e:
                print(f"Error saving {doc_id}: {e}")
        
        # Generate and save index
        try:
            self.doc_manager.generate_index()
            index_path = self.docs_path / "index.json"
            with open(index_path, 'w', encoding='utf-8') as f:
                import json
                json.dump(self.doc_manager.index, f, indent=2, ensure_ascii=False)
            print(f"Saved index: {index_path}")
        except Exception as e:
            print(f"Error saving index: {e}")

def main():
    """Main function to process all documentation"""
    docs_path = "/mnt/d/Desktop/AI Code Review and Architecture Analysis Platform/docs"
    
    # Create processor and process all documentation
    processor = DocumentationProcessor(docs_path)
    processor.process_all_documentation()
    
    print("\nDocumentation processing complete!")
    print(f"Processed {len(processor.processed_files)} documents")
    print("All files have been class-structured and saved to the docs directory.")

if __name__ == "__main__":
    main()