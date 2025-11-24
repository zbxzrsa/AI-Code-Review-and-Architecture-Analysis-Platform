"""
标准库 API 映射与迁移指南的集成测试
"""
import unittest
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from backend.app.services.api_mapping.standard_lib import convert_standard_api, generate_api_migration_guide


class TestStandardApiMapping(unittest.TestCase):
    def test_python_to_javascript_stdlib(self):
        src = """
import os, json, datetime

path = os.path.join('dir', 'file.txt')
data = json.loads('{"key": "value"}')
now = datetime.datetime.now()
"""
        out = convert_standard_api(src, "python", "javascript")
        self.assertIn("const path = require('path');", out)
        self.assertIn("path.join('dir', 'file.txt')", out)
        self.assertIn("JSON.parse('{\"key\": \"value\"}')", out)
        self.assertIn("new Date()", out)

        guide = generate_api_migration_guide(src, "python", "javascript")
        self.assertIn("从 python 迁移到 javascript", guide)
        self.assertIn("os.path.join → path.join", guide)

    def test_java_to_csharp_collections(self):
        src = """
// Java集合使用
List<String> list = new ArrayList<>();
list.add("item");
Map<String, Integer> map = new HashMap<>();
map.put("key", 1);
"""
        out = convert_standard_api(src, "java", "csharp")
        self.assertIn("var list = new List<string>();", out)
        self.assertIn("list.Add(\"item\");", out)
        self.assertIn("var map = new Dictionary<string, int>();", out)
        self.assertIn("map[\"key\"] = 1;", out)


if __name__ == '__main__':
    unittest.main()