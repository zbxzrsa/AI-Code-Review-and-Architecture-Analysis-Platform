"""
第三方框架转换的集成测试：Django→Express、Spring→ASP.NET Core
"""
import unittest
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from backend.app.services.framework_converter.frameworks import (
    convert_frameworks,
    convert_framework_dependencies,
    generate_framework_migration_guide,
)


class TestFrameworkConversion(unittest.TestCase):
    def test_django_to_express(self):
        src = """
from django.http import JsonResponse

def api_view(request):
    data = {"message": "Hello"}
    return JsonResponse(data)
"""
        out = convert_frameworks(src, "python", "javascript")
        self.assertIn("app.get('/api'", out)
        self.assertIn("res.json({ message: 'Hello' })", out)

        guide = generate_framework_migration_guide(src, "python", "javascript")
        self.assertIn("Django → Express", guide)

    def test_spring_to_aspnet(self):
        src = """
@RestController
public class UserController {
    @GetMapping("/users/{id}")
    public User getUser(@PathVariable Long id) {
        return userService.findById(id);
    }
}
"""
        out = convert_frameworks(src, "java", "csharp")
        self.assertIn("[ApiController]", out)
        self.assertIn("[HttpGet(\"{id}\")]", out)

    def test_dependency_conversion_python_to_js(self):
        src = """
# requirements.txt
Django==3.2
requests>=2.28
"""
        dep_out = convert_framework_dependencies(src, "python", "javascript")
        self.assertIn("dependencies", dep_out)
        self.assertIn("express", dep_out)
        self.assertIn("axios", dep_out)

    def test_dependency_conversion_gradle_to_csproj(self):
        src = """
dependencies {
    implementation "org.springframework.boot:spring-boot-starter-web:2.7.0"
}
"""
        dep_out = convert_framework_dependencies(src, "java", "csharp")
        self.assertIn("PackageReference", dep_out)
        self.assertIn("Microsoft.AspNetCore.App", dep_out)


if __name__ == '__main__':
    unittest.main()