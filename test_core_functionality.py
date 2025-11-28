#!/usr/bin/env python3
"""
Test Core AI Platform Functionality
Tests the enhanced code analyzer, AI reviewer, and version router
"""

import sys
import os
import asyncio
import json
from pathlib import Path

# Add the backend to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

async def test_code_analyzer():
    """Test the enhanced code analyzer"""
    print("=== Testing Code Analyzer ===")
    
    try:
        from app.core.code_analyzer import CodeAnalyzer
        
        # Test with Python code
        python_code = '''
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
        
        analyzer = CodeAnalyzer()
        result = analyzer.analyze_code(python_code, 'python', 'test.py')
        
        print(f"✅ Python analysis successful")
        print(f"   Functions: {result['metrics']['functions']}")
        print(f"   Classes: {result['metrics']['classes']}")
        print(f"   Complexity: {result['metrics']['complexity']}")
        print(f"   Issues found: {len(result['issues'])}")
        
        # Test with JavaScript code
        js_code = '''
function calculateTotal(items) {
    let total = 0;
    for(let i = 0; i < items.length; i++) {
        total += items[i].price;
    }
    return total;
}
'''
        
        js_result = analyzer.analyze_code(js_code, 'javascript', 'test.js')
        print(f"✅ JavaScript analysis successful")
        print(f"   Functions: {js_result['metrics']['functions']}")
        print(f"   Issues found: {len(js_result['issues'])}")
        
        return True
        
    except Exception as e:
        print(f"❌ Code analyzer test failed: {e}")
        return False

async def test_ai_reviewer():
    """Test the AI reviewer with different versions"""
    print("\n=== Testing AI Reviewer ===")
    
    try:
        from app.services.ai_reviewer import AIReviewer
        
        test_code = '''
def process_user_input(user_data):
    # Process user input without validation
    password = user_data.get('password', 'default123')
    return password
'''
        
        # Test v1 (stable)
        print("Testing v1 (stable)...")
        reviewer_v1 = AIReviewer('v1', {'cache_enabled': False})
        result_v1 = await reviewer_v1.review(test_code, 'python', ['security'])
        
        print(f"✅ v1 review successful")
        print(f"   Model: {result_v1.get('model_used', 'unknown')}")
        print(f"   Issues: {len(result_v1.get('issues', []))}")
        print(f"   Score: {result_v1.get('score', 0)}")
        
        # Test v2 (experimental)
        print("Testing v2 (experimental)...")
        reviewer_v2 = AIReviewer('v2', {'cache_enabled': False})
        result_v2 = await reviewer_v2.review(test_code, 'python', ['security'])
        
        print(f"✅ v2 review successful")
        print(f"   Model: {result_v2.get('model_used', 'unknown')}")
        print(f"   Issues: {len(result_v2.get('issues', []))}")
        print(f"   Score: {result_v2.get('score', 0)}")
        
        # Test v3 (deprecated)
        print("Testing v3 (deprecated)...")
        reviewer_v3 = AIReviewer('v3', {'cache_enabled': False})
        result_v3 = await reviewer_v3.review(test_code, 'python', ['security'])
        
        print(f"✅ v3 review successful")
        print(f"   Model: {result_v3.get('model_used', 'unknown')}")
        print(f"   Issues: {len(result_v3.get('issues', []))}")
        print(f"   Score: {result_v3.get('score', 0)}")
        
        return True
        
    except Exception as e:
        print(f"❌ AI reviewer test failed: {e}")
        return False

def test_static_analyzer():
    """Test the enhanced static analyzer"""
    print("\n=== Testing Static Analyzer ===")
    
    try:
        from app.services.static_analyzer import StaticAnalyzer
        
        # Create a temporary Python file with issues
        test_file = "/tmp/test_analysis.py"
        test_code = '''
import os
import subprocess

def insecure_function():
    password = "hardcoded_password_123"
    eval(user_input)
    os.system("rm -rf /")
    return password

def very_long_function_that_should_be_refactored():
    # This function is too long and complex
    x = 1
    if x > 0:
        if x > 1:
            if x > 2:
                if x > 3:
                    if x > 4:
                        return x * 2
    return x
'''
        
        with open(test_file, 'w') as f:
            f.write(test_code)
        
        # Test static analysis
        config = {
            'rules': ['security', 'quality', 'complexity'],
            'language': 'python'
        }
        
        analyzer = StaticAnalyzer(config)
        findings = analyzer.analyze_file(test_file)
        
        print(f"✅ Static analysis successful")
        print(f"   Total findings: {len(findings)}")
        
        # Count by type
        security_issues = len([f for f in findings if f.get('type') == 'security'])
        quality_issues = len([f for f in findings if f.get('type') == 'quality'])
        
        print(f"   Security issues: {security_issues}")
        print(f"   Quality issues: {quality_issues}")
        
        # Clean up
        os.unlink(test_file)
        
        return True
        
    except Exception as e:
        print(f"❌ Static analyzer test failed: {e}")
        return False

def test_version_router():
    """Test the version router"""
    print("\n=== Testing Version Router ===")
    
    try:
        # Add ai_versions to path
        sys.path.insert(0, str(Path(__file__).parent))
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
        
        # Test status check
        status = router.get_all_versions_status()
        print(f"✅ Status check successful")
        print(f"   Available versions: {list(status['versions'].keys())}")
        
        return True
        
    except Exception as e:
        print(f"❌ Version router test failed: {e}")
        return False

async def test_review_service():
    """Test the unified review service"""
    print("\n=== Testing Review Service ===")
    
    try:
        from app.services.ai_reviewer import ReviewService
        
        test_code = '''
def authenticate_user(username, password):
    # Simple authentication with issues
    if username == "admin" and password == "admin123":
        return True
    return False
'''
        
        # Test unified service
        service = ReviewService('v1', {'cache_enabled': False})
        result = await service.review(test_code, 'python', ['security'], include_static=True)
        
        print(f"✅ Review service successful")
        print(f"   Total issues: {result.get('total_issues', 0)}")
        print(f"   AI issues: {len(result.get('ai_review', {}).get('issues', []))}")
        print(f"   Static issues: {len(result.get('static_issues', []))}")
        print(f"   Combined score: {result.get('combined_score', 0)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Review service test failed: {e}")
        return False

async def main():
    """Run all tests"""
    print("🚀 Testing AI Code Review Platform Core Functionality")
    print("=" * 60)
    
    results = []
    
    # Run all tests
    results.append(await test_code_analyzer())
    results.append(await test_ai_reviewer())
    results.append(test_static_analyzer())
    results.append(test_version_router())
    results.append(await test_review_service())
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! Core functionality is working.")
        return 0
    else:
        print("❌ Some tests failed. Check the logs above.")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)