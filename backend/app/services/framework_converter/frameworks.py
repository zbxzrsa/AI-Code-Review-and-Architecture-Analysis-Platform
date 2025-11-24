"""
第三方库与框架转换支持

转换策略：
- 直接映射转换：功能等价库、API 兼容层、配置文件转换
- 架构模式转换：MVC 框架、ORM 映射、测试框架适配（本文件聚焦 MVC 路由/控制器）
- 依赖管理转换：包管理器、版本兼容、构建脚本迁移（简版）

场景示例：
- Python Django → JavaScript Express
- Java Spring → C# ASP.NET Core
"""
from typing import Dict, Any, List
import re


def _convert_django_to_express(code: str) -> str:
    """将简单 Django 视图转换为 Express 路由（示例级别）"""
    converted = code
    # 若存在 data = {...}，捕获字面量以便内联到 res.json
    data_assign = re.search(r"\b(\w+)\s*=\s*\{([\s\S]*?)\}\s*$", code, re.MULTILINE)
    inline_obj = None
    if data_assign:
        var_name = data_assign.group(1)
        obj_body = data_assign.group(2)
        # 将 Python 字典风格转换为 JS 对象：键去引号，值使用单引号
        js_obj = obj_body
        js_obj = re.sub(r'"([A-Za-z0-9_]+)"\s*:', r"\1: ", js_obj)
        js_obj = js_obj.replace('"', "'")
        inline_obj = f"{{ {js_obj} }}"
        inline_obj = re.sub(r":\s+", ": ", inline_obj)

    # 识别 Django JsonResponse 模式
    def _json_replace(m: re.Match) -> str:
        arg = m.group(1).strip()
        if inline_obj and data_assign and arg == data_assign.group(1):
            return f"res.json({inline_obj})"
        return f"res.json({arg})"
    converted = re.sub(r"return\s+JsonResponse\(([^)]+)\)", _json_replace, converted)

    # 识别函数视图定义并生成 Express 路由
    func_view = re.compile(r"def\s+(\w+)\(request\):")
    if func_view.search(code):
        # 插入 Express 代码骨架
        header = [
            "const express = require('express');",
            "const app = express();",
            "",
        ]
        # 映射到 GET 路由（简单情况）
        converted = "\n".join(header) + re.sub(
            r"def\s+(\w+)\(request\):\n\s*([\s\S]*)",
            r"app.get('/api', (req, res) => {\n\2\n\n});",
            converted,
        )
        # 追加启动说明注释
        converted += "\n// 启动：app.listen(3000)"
    return converted


def _convert_spring_to_aspnet(code: str) -> str:
    """将简单 Spring @RestController 转换为 ASP.NET Core 控制器（示例级）"""
    # 识别控制器名
    class_match = re.search(r"@RestController[\s\S]*class\s+(\w+)", code)
    controller_name = class_match.group(1) if class_match else "Controller"

    # 识别 Get 路由与方法签名
    route_match = re.search(r"@GetMapping\(\"([^\"]+)\"\)", code)
    route = route_match.group(1) if route_match else "/"

    method_sig = re.search(r"public\s+([A-Za-z0-9_<>]+)\s+(\w+)\([^)]*\)", code)
    return_type = method_sig.group(1) if method_sig else "ActionResult"
    method_name = method_sig.group(2) if method_sig else "Get"

    # 识别返回调用
    svc = re.search(r"return\s+([A-Za-z0-9_\.]+)\([^)]*\);", code)
    service_call = svc.group(1) if svc else None

    converted_lines: List[str] = [
        "using Microsoft.AspNetCore.Mvc;",
        "",
        "[ApiController]",
        "[Route(\"[controller]\")]",
        f"public class {controller_name} : ControllerBase",
        "{",
        ("    [HttpGet(\"{id}\")]" if "{id}" in route else f"    [HttpGet(\"{route}\")]"),
        f"    public ActionResult<{return_type}> {method_name}(long id)",
        "    {",
        (f"        return {service_call}(id);" if service_call else "        return Ok();"),
        "    }",
        "}",
    ]
    return "\n".join(converted_lines)


def convert_frameworks(source_code: str, source_language: str, target_language: str) -> str:
    """框架转换入口：根据源/目标语言路由到具体转换"""
    src = source_language.lower()
    dst = target_language.lower()
    if src == "python" and dst in ("javascript", "typescript"):
        # 目前以 Django→Express 简化实现
        return _convert_django_to_express(source_code)
    if src == "java" and dst == "csharp":
        return _convert_spring_to_aspnet(source_code)
    return source_code


def convert_framework_dependencies(source_code: str, source_language: str, target_language: str) -> str:
    """依赖管理转换：pip/requirements, Gradle/Maven → npm/NuGet（简版映射）"""
    src = source_language.lower()
    dst = target_language.lower()
    output_lines: List[str] = []

    # Python (pip/requirements.txt) → JavaScript (npm package.json)
    if src == "python" and dst in ("javascript", "typescript"):
        # 简单解析 requirements 或文本中的包名
        pkgs = []
        for line in source_code.splitlines():
            m = re.match(r"\s*([A-Za-z0-9_\-]+)([<>=!~].*)?", line)
            if m:
                name = m.group(1).lower()
                pkgs.append(name)
        # 基础映射
        mapping = {
            "django": "express",
            "requests": "axios",
            "flask": "express",
            "uvicorn": "nodemon",
            "gunicorn": "nodemon",
            "pydantic": "zod",
        }
        npm_dependencies = {}
        dev_dependencies = {}
        for p in pkgs:
            if p in ("uvicorn", "gunicorn"):
                dev_dependencies["nodemon"] = "^3"
            elif p in mapping:
                npm_dependencies[mapping[p]] = "latest"

        # 始终添加基础 Web 依赖
        if "express" not in npm_dependencies:
            npm_dependencies["express"] = "latest"
        npm_dependencies.setdefault("cors", "^2")
        npm_dependencies.setdefault("body-parser", "^1")

        output_lines += [
            "目标 package.json 依赖（示例）:",
            "{",
            "  \"dependencies\": {",
        ]
        for k, v in npm_dependencies.items():
            output_lines.append(f"    \"{k}\": \"{v}\",")
        if output_lines[-1].endswith(","):
            output_lines[-1] = output_lines[-1].rstrip(",")
        output_lines += [
            "  },",
            "  \"devDependencies\": {",
        ]
        if dev_dependencies:
            for k, v in dev_dependencies.items():
                output_lines.append(f"    \"{k}\": \"{v}\",")
            if output_lines[-1].endswith(","):
                output_lines[-1] = output_lines[-1].rstrip(",")
        output_lines += [
            "  }",
            "}",
            "安装命令示例：",
            "npm install " + " ".join(npm_dependencies.keys()),
            *( ["npm install -D " + " ".join(dev_dependencies.keys())] if dev_dependencies else [] ),
        ]
        return "\n".join(output_lines)

    # Java (Gradle/Maven) → C# (.csproj NuGet)
    if src == "java" and dst == "csharp":
        nuget_map = {
            "spring-boot-starter-web": "Microsoft.AspNetCore.App",
            "spring-boot-starter": "Microsoft.AspNetCore.App",
            "spring-web": "Microsoft.AspNetCore.App",
            "spring-data-jpa": "Microsoft.EntityFrameworkCore",
            "hibernate": "NHibernate",
            "junit": "xunit",
        }
        refs: List[str] = []
        for line in source_code.splitlines():
            # Gradle: implementation "group:name:version"
            mg = re.search(r"implementation\s+\"([A-Za-z0-9_.\-]+):([A-Za-z0-9_.\-]+):([A-Za-z0-9_.\-]+)\"", line)
            if mg:
                artifact = mg.group(2)
                mapped = nuget_map.get(artifact)
                if mapped:
                    refs.append(f"<PackageReference Include=\"{mapped}\" Version=\"latest\" />")
            # Maven: <artifactId>name</artifactId>
            mm = re.search(r"<artifactId>([A-Za-z0-9_.\-]+)</artifactId>", line)
            if mm:
                artifact = mm.group(1)
                mapped = nuget_map.get(artifact)
                if mapped:
                    refs.append(f"<PackageReference Include=\"{mapped}\" Version=\"latest\" />")
        if not refs:
            refs = [
                "<PackageReference Include=\"Microsoft.AspNetCore.App\" Version=\"latest\" />",
                "<PackageReference Include=\"Microsoft.EntityFrameworkCore\" Version=\"latest\" />",
            ]
        return "\n".join([
            "目标 .csproj 依赖（示例）：",
            "<ItemGroup>",
            *refs,
            "</ItemGroup>",
            "安装命令示例：",
            "dotnet restore",
        ])

    return source_code


def generate_framework_migration_guide(source_code: str, source_language: str, target_language: str) -> str:
    """生成框架迁移指南（简版）：列出识别到的模式与目标骨架"""
    lines: List[str] = [
        f"从 {source_language} 迁移到 {target_language} 的框架转换指南",
        "",
        "识别的架构模式：",
    ]
    if re.search(r"@RestController", source_code):
        lines.append("- Spring REST 控制器 → ASP.NET Core ApiController")
    if re.search(r"JsonResponse\(", source_code):
        lines.append("- Django → Express | JsonResponse → res.json")
    lines += [
        "",
        "迁移步骤建议：",
        "- 路由与控制器：先建立目标语言的路由骨架，再逐步迁移业务逻辑",
        "- 依赖与配置：根据目标生态更新包管理与启动脚本",
        "- 测试与验证：执行端到端测试确保行为与错误处理一致",
        "",
        "依赖管理转换建议：",
        "- Python→JS：将 requirements.txt 映射到 package.json，requests→axios，Django/Flask→Express",
        "- Java→C#：Gradle/Maven 依赖映射到 .csproj PackageReference，Spring Web→AspNetCore",
    ]
    return "\n".join(lines)