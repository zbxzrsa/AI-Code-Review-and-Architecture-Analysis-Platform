#!/usr/bin/env python3
"""
Simple Test for Core AI Platform Functionality
Tests the basic functionality without complex imports
"""

import sys
import os
import asyncio
import json
from pathlib import Path

def test_version_router():
    """Test the version router directly"""
    print("=== Testing Version Router ===")
    
    try:
        # Add ai_versions to path
        ai_versions_path = Path(__file__).parent / "ai_versions"
        sys.path.insert(0, str(ai_versions_path))
        
        from version_router import VersionRouter
        
        router = VersionRouter()
        
        # Test getting active config
        config = router.get_active_config()
        print(f"✅ Version router successful")
        print(f"   Current version: {config.get('version', 'unknown')}")
        print(f"   Model name: {config.get('model', {}).get('name', 'unknown')}")
        
        # Test version comparison
        comparison = router.compare_versions('v1', 'v2')
        print(f"✅ Version comparison successful")
        print(f"   Latency difference: {comparison['differences']['latency']:.2f}s")
        print(f"   Accuracy difference: {comparison['differences']['accuracy']:.2f}")
        print(f"   Recommendation: {comparison['recommendation']}")
        
        # Test status check
        status = router.get_all_versions_status()
        print(f"✅ Status check successful")
        print(f"   Available versions: {list(status['versions'].keys())}")
        
        return True
        
    except Exception as e:
        print(f"❌ Version router test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_config_files():
    """Test that all config files are valid YAML"""
    print("\n=== Testing Config Files ===")
    
    try:
        import yaml
        
        base_path = Path(__file__).parent / "ai_versions"
        versions = ['v1_stable', 'v2_experimental', 'v3_deprecated']
        
        for version in versions:
            config_file = base_path / version / "config.yaml"
            
            if not config_file.exists():
                print(f"❌ Config file missing: {config_file}")
                return False
            
            try:
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)
                
                print(f"✅ {version} config valid")
                print(f"   Version: {config.get('version', 'unknown')}")
                print(f"   Model: {config.get('model', {}).get('name', 'unknown')}")
                print(f"   Status: {config.get('status', 'unknown')}")
                
            except yaml.YAMLError as e:
                print(f"❌ {version} config invalid: {e}")
                return False
        
        return True
        
    except ImportError:
        print("❌ PyYAML not available for config testing")
        return False
    except Exception as e:
        print(f"❌ Config file test failed: {e}")
        return False

def test_example_files():
    """Test that example files exist and are readable"""
    print("\n=== Testing Example Files ===")
    
    try:
        base_path = Path(__file__).parent / "ai_versions"
        
        examples = [
            base_path / "v1_stable" / "examples" / "factorial.py",
            base_path / "v2_experimental" / "examples" / "javascript_issues.js",
            base_path / "v3_deprecated" / "examples" / "expensive_api_example.py"
        ]
        
        for example_file in examples:
            if example_file.exists():
                print(f"✅ Example file exists: {example_file.name}")
                
                # Try to read it
                try:
                    with open(example_file, 'r') as f:
                        content = f.read()
                    
                    # Basic checks
                    lines = len(content.splitlines())
                    chars = len(content)
                    
                    print(f"   Lines: {lines}, Characters: {chars}")
                    
                except Exception as e:
                    print(f"❌ Could not read {example_file.name}: {e}")
                    return False
            else:
                print(f"❌ Example file missing: {example_file}")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Example files test failed: {e}")
        return False

def test_basic_python_analysis():
    """Test basic Python AST analysis"""
    print("\n=== Testing Basic Python Analysis ===")
    
    try:
        import ast
        
        test_code = '''
def calculate_factorial(n):
    """Calculate factorial"""
    if n < 0:
        return None
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result

class Calculator:
    def __init__(self):
        self.history = []
    
    def add(self, a, b):
        result = a + b
        self.history.append(f"{a} + {b} = {result}")
        return result
'''
        
        # Parse the code
        tree = ast.parse(test_code)
        
        # Count functions and classes
        functions = len([node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)])
        classes = len([node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)])
        
        print(f"✅ Python AST analysis successful")
        print(f"   Functions found: {functions}")
        print(f"   Classes found: {classes}")
        
        # Test complexity calculation
        complexity = 1  # Base complexity
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For)):
                complexity += 1
        
        print(f"   Cyclomatic complexity: {complexity}")
        
        return True
        
    except Exception as e:
        print(f"❌ Python analysis test failed: {e}")
        return False

def test_directory_structure():
    """Test that the expected directory structure exists"""
    print("\n=== Testing Directory Structure ===")
    
    try:
        base_path = Path(__file__).parent
        
        expected_dirs = [
            "ai_versions",
            "ai_versions/v1_stable",
            "ai_versions/v2_experimental", 
            "ai_versions/v3_deprecated",
            "backend",
            "backend/app",
            "backend/app/core",
            "backend/app/services",
            "docs",
            "scripts"
        ]
        
        for dir_path in expected_dirs:
            full_path = base_path / dir_path
            if full_path.exists() and full_path.is_dir():
                print(f"✅ Directory exists: {dir_path}")
            else:
                print(f"❌ Directory missing: {dir_path}")
                return False
        
        # Check for key files
        expected_files = [
            "ai_versions/version_router.py",
            "backend/app/core/code_analyzer.py",
            "backend/app/services/static_analyzer.py",
            "backend/app/services/ai_reviewer.py",
            "README.md"
        ]
        
        for file_path in expected_files:
            full_path = base_path / file_path
            if full_path.exists() and full_path.is_file():
                print(f"✅ File exists: {file_path}")
            else:
                print(f"❌ File missing: {file_path}")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Directory structure test failed: {e}")
        return False

async def main():
    """Run all tests"""
    print("🚀 Testing AI Code Review Platform - Core Functionality")
    print("=" * 65)
    
    results = []
    
    # Run all tests
    results.append(test_directory_structure())
    results.append(test_config_files())
    results.append(test_example_files())
    results.append(test_version_router())
    results.append(test_basic_python_analysis())
    
    # Summary
    print("\n" + "=" * 65)
    print("📊 TEST SUMMARY")
    print("=" * 65)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! Core functionality is working.")
        print("\n📋 NEXT STEPS:")
        print("1. Install required dependencies:")
        print("   - pip install tree-sitter")
        print("   - pip install transformers")
        print("   - pip install ollama")
        print("   - pip install pylint bandit radon")
        print("2. Set up AI models:")
        print("   - ollama pull starcoder")
        print("   - ollama pull codebert")
        print("3. Run the platform:")
        print("   - cd backend && python -m app.main")
        print("   - Or use: npm run build")
        return 0
    else:
        print("❌ Some tests failed. Check the logs above.")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)