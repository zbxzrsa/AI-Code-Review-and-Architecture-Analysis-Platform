"""
类型映射系统模块

提供基础类型映射表、泛型类型转换和类型别名处理功能
"""
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass, field

from .inference import TypeInfo, TypeCategory, TypeFactory


@dataclass
class TypeMapping:
    """类型映射"""
    source_type: TypeInfo
    target_type: TypeInfo
    conversion_rules: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class TypeMappingRegistry:
    """类型映射注册表"""
    
    def __init__(self):
        self.mappings: Dict[Tuple[str, str], Dict[str, TypeMapping]] = {}
    
    def register_mapping(self, source_lang: str, target_lang: str, source_type_name: str, 
                         target_type_name: str, mapping: TypeMapping) -> None:
        """注册类型映射"""
        lang_pair = (source_lang, target_lang)
        if lang_pair not in self.mappings:
            self.mappings[lang_pair] = {}
        
        self.mappings[lang_pair][source_type_name] = mapping
    
    def get_mapping(self, source_lang: str, target_lang: str, source_type_name: str) -> Optional[TypeMapping]:
        """获取类型映射"""
        lang_pair = (source_lang, target_lang)
        if lang_pair not in self.mappings:
            return None
        
        return self.mappings[lang_pair].get(source_type_name)
    
    def get_all_mappings(self, source_lang: str, target_lang: str) -> Dict[str, TypeMapping]:
        """获取所有类型映射"""
        lang_pair = (source_lang, target_lang)
        return self.mappings.get(lang_pair, {})


class TypeMappingSystem:
    """类型映射系统"""
    
    def __init__(self):
        self.registry = TypeMappingRegistry()
        self._init_basic_mappings()
    
    def _init_basic_mappings(self) -> None:
        """初始化基本类型映射"""
        # Python -> TypeScript
        self._init_python_to_typescript_mappings()
        
        # TypeScript -> Python
        self._init_typescript_to_python_mappings()
        
        # Python -> JavaScript
        self._init_python_to_javascript_mappings()
        
        # JavaScript -> Python
        self._init_javascript_to_python_mappings()
        
        # JavaScript -> TypeScript
        self._init_javascript_to_typescript_mappings()
        
        # TypeScript -> JavaScript
        self._init_typescript_to_javascript_mappings()
        
        # Java -> C#
        self._init_java_to_csharp_mappings()
        
        # C# -> Java
        self._init_csharp_to_java_mappings()
    
    def _init_python_to_typescript_mappings(self) -> None:
        """初始化Python到TypeScript的类型映射"""
        factory = TypeFactory
        
        # 原始类型映射
        self._register_primitive_mapping("python", "typescript", "int", "number")
        self._register_primitive_mapping("python", "typescript", "float", "number")
        self._register_primitive_mapping("python", "typescript", "str", "string")
        self._register_primitive_mapping("python", "typescript", "bool", "boolean")
        self._register_primitive_mapping("python", "typescript", "None", "null")
        self._register_primitive_mapping("python", "typescript", "bytes", "Uint8Array")
        
        # 容器类型映射
        list_mapping = TypeMapping(
            source_type=factory.create_container_type("List", "python"),
            target_type=factory.create_container_type("Array", "typescript")
        )
        self.registry.register_mapping("python", "typescript", "List", "Array", list_mapping)
        
        dict_mapping = TypeMapping(
            source_type=factory.create_container_type("Dict", "python"),
            target_type=factory.create_container_type("Record", "typescript")
        )
        self.registry.register_mapping("python", "typescript", "Dict", "Record", dict_mapping)
        
        set_mapping = TypeMapping(
            source_type=factory.create_container_type("Set", "python"),
            target_type=factory.create_container_type("Set", "typescript")
        )
        self.registry.register_mapping("python", "typescript", "Set", "Set", set_mapping)
        
        tuple_mapping = TypeMapping(
            source_type=factory.create_container_type("Tuple", "python"),
            target_type=factory.create_container_type("Array", "typescript")
        )
        self.registry.register_mapping("python", "typescript", "Tuple", "Array", tuple_mapping)
        
        # 特殊类型映射
        any_mapping = TypeMapping(
            source_type=factory.create_any_type("python"),
            target_type=factory.create_any_type("typescript")
        )
        self.registry.register_mapping("python", "typescript", "Any", "any", any_mapping)
        
        # 类类型映射
        class_mapping = TypeMapping(
            source_type=factory.create_class_type("object", "python"),
            target_type=factory.create_class_type("Object", "typescript")
        )
        self.registry.register_mapping("python", "typescript", "object", "Object", class_mapping)
    
    def _init_typescript_to_python_mappings(self) -> None:
        """初始化TypeScript到Python的类型映射"""
        factory = TypeFactory
        
        # 原始类型映射
        self._register_primitive_mapping("typescript", "python", "number", "float")
        self._register_primitive_mapping("typescript", "python", "string", "str")
        self._register_primitive_mapping("typescript", "python", "boolean", "bool")
        self._register_primitive_mapping("typescript", "python", "null", "None")
        self._register_primitive_mapping("typescript", "python", "undefined", "None")
        self._register_primitive_mapping("typescript", "python", "Uint8Array", "bytes")
        
        # 容器类型映射
        array_mapping = TypeMapping(
            source_type=factory.create_container_type("Array", "typescript"),
            target_type=factory.create_container_type("List", "python")
        )
        self.registry.register_mapping("typescript", "python", "Array", "List", array_mapping)
        
        record_mapping = TypeMapping(
            source_type=factory.create_container_type("Record", "typescript"),
            target_type=factory.create_container_type("Dict", "python")
        )
        self.registry.register_mapping("typescript", "python", "Record", "Dict", record_mapping)
        
        set_mapping = TypeMapping(
            source_type=factory.create_container_type("Set", "typescript"),
            target_type=factory.create_container_type("Set", "python")
        )
        self.registry.register_mapping("typescript", "python", "Set", "Set", set_mapping)
        
        # 特殊类型映射
        any_mapping = TypeMapping(
            source_type=factory.create_any_type("typescript"),
            target_type=factory.create_any_type("python")
        )
        self.registry.register_mapping("typescript", "python", "any", "Any", any_mapping)
        
        unknown_mapping = TypeMapping(
            source_type=factory.create_primitive_type("unknown", "typescript"),
            target_type=factory.create_any_type("python")
        )
        self.registry.register_mapping("typescript", "python", "unknown", "Any", unknown_mapping)
        
        # 类类型映射
        class_mapping = TypeMapping(
            source_type=factory.create_class_type("Object", "typescript"),
            target_type=factory.create_class_type("object", "python")
        )
        self.registry.register_mapping("typescript", "python", "Object", "object", class_mapping)
    
    def _init_python_to_javascript_mappings(self) -> None:
        """初始化Python到JavaScript的类型映射"""
        factory = TypeFactory
        
        # 原始类型映射
        self._register_primitive_mapping("python", "javascript", "int", "number")
        self._register_primitive_mapping("python", "javascript", "float", "number")
        self._register_primitive_mapping("python", "javascript", "str", "string")
        self._register_primitive_mapping("python", "javascript", "bool", "boolean")
        self._register_primitive_mapping("python", "javascript", "None", "null")
        
        # 容器类型映射
        list_mapping = TypeMapping(
            source_type=factory.create_container_type("List", "python"),
            target_type=factory.create_container_type("Array", "javascript")
        )
        self.registry.register_mapping("python", "javascript", "List", "Array", list_mapping)
        
        dict_mapping = TypeMapping(
            source_type=factory.create_container_type("Dict", "python"),
            target_type=factory.create_class_type("Object", "javascript")
        )
        self.registry.register_mapping("python", "javascript", "Dict", "Object", dict_mapping)
        
        # 特殊类型映射
        any_mapping = TypeMapping(
            source_type=factory.create_any_type("python"),
            target_type=factory.create_any_type("javascript")
        )
        self.registry.register_mapping("python", "javascript", "Any", "any", any_mapping)
    
    def _init_javascript_to_python_mappings(self) -> None:
        """初始化JavaScript到Python的类型映射"""
        factory = TypeFactory
        
        # 原始类型映射
        self._register_primitive_mapping("javascript", "python", "number", "float")
        self._register_primitive_mapping("javascript", "python", "string", "str")
        self._register_primitive_mapping("javascript", "python", "boolean", "bool")
        self._register_primitive_mapping("javascript", "python", "null", "None")
        self._register_primitive_mapping("javascript", "python", "undefined", "None")
        
        # 容器类型映射
        array_mapping = TypeMapping(
            source_type=factory.create_container_type("Array", "javascript"),
            target_type=factory.create_container_type("List", "python")
        )
        self.registry.register_mapping("javascript", "python", "Array", "List", array_mapping)
        
        # 特殊类型映射
        any_mapping = TypeMapping(
            source_type=factory.create_any_type("javascript"),
            target_type=factory.create_any_type("python")
        )
        self.registry.register_mapping("javascript", "python", "any", "Any", any_mapping)
    
    def _init_javascript_to_typescript_mappings(self) -> None:
        """初始化JavaScript到TypeScript的类型映射"""
        factory = TypeFactory
        
        # 原始类型映射
        self._register_primitive_mapping("javascript", "typescript", "number", "number")
        self._register_primitive_mapping("javascript", "typescript", "string", "string")
        self._register_primitive_mapping("javascript", "typescript", "boolean", "boolean")
        self._register_primitive_mapping("javascript", "typescript", "null", "null")
        self._register_primitive_mapping("javascript", "typescript", "undefined", "undefined")
        
        # 容器类型映射
        array_mapping = TypeMapping(
            source_type=factory.create_container_type("Array", "javascript"),
            target_type=factory.create_container_type("Array", "typescript")
        )
        self.registry.register_mapping("javascript", "typescript", "Array", "Array", array_mapping)
        
        # 特殊类型映射
        any_mapping = TypeMapping(
            source_type=factory.create_any_type("javascript"),
            target_type=factory.create_any_type("typescript")
        )
        self.registry.register_mapping("javascript", "typescript", "any", "any", any_mapping)
    
    def _init_typescript_to_javascript_mappings(self) -> None:
        """初始化TypeScript到JavaScript的类型映射"""
        factory = TypeFactory
        
        # 原始类型映射
        self._register_primitive_mapping("typescript", "javascript", "number", "number")
        self._register_primitive_mapping("typescript", "javascript", "string", "string")
        self._register_primitive_mapping("typescript", "javascript", "boolean", "boolean")
        self._register_primitive_mapping("typescript", "javascript", "null", "null")
        self._register_primitive_mapping("typescript", "javascript", "undefined", "undefined")
        
        # 容器类型映射
        array_mapping = TypeMapping(
            source_type=factory.create_container_type("Array", "typescript"),
            target_type=factory.create_container_type("Array", "javascript")
        )
        self.registry.register_mapping("typescript", "javascript", "Array", "Array", array_mapping)
        
        # 特殊类型映射
        any_mapping = TypeMapping(
            source_type=factory.create_any_type("typescript"),
            target_type=factory.create_any_type("javascript")
        )
        self.registry.register_mapping("typescript", "javascript", "any", "any", any_mapping)
        
        unknown_mapping = TypeMapping(
            source_type=factory.create_primitive_type("unknown", "typescript"),
            target_type=factory.create_any_type("javascript")
        )
        self.registry.register_mapping("typescript", "javascript", "unknown", "any", unknown_mapping)
    
    def _init_java_to_csharp_mappings(self) -> None:
        """初始化Java到C#的类型映射"""
        factory = TypeFactory
        
        # 原始类型映射
        self._register_primitive_mapping("java", "csharp", "int", "int")
        self._register_primitive_mapping("java", "csharp", "long", "long")
        self._register_primitive_mapping("java", "csharp", "float", "float")
        self._register_primitive_mapping("java", "csharp", "double", "double")
        self._register_primitive_mapping("java", "csharp", "boolean", "bool")
        self._register_primitive_mapping("java", "csharp", "char", "char")
        self._register_primitive_mapping("java", "csharp", "byte", "byte")
        self._register_primitive_mapping("java", "csharp", "short", "short")
        
        # 引用类型映射
        self._register_primitive_mapping("java", "csharp", "String", "string")
        self._register_primitive_mapping("java", "csharp", "Object", "object")
        
        # 容器类型映射
        list_mapping = TypeMapping(
            source_type=factory.create_container_type("List", "java"),
            target_type=factory.create_container_type("List", "csharp")
        )
        self.registry.register_mapping("java", "csharp", "List", "List", list_mapping)
        
        map_mapping = TypeMapping(
            source_type=factory.create_container_type("Map", "java"),
            target_type=factory.create_container_type("Dictionary", "csharp")
        )
        self.registry.register_mapping("java", "csharp", "Map", "Dictionary", map_mapping)
        
        set_mapping = TypeMapping(
            source_type=factory.create_container_type("Set", "java"),
            target_type=factory.create_container_type("HashSet", "csharp")
        )
        self.registry.register_mapping("java", "csharp", "Set", "HashSet", set_mapping)
    
    def _init_csharp_to_java_mappings(self) -> None:
        """初始化C#到Java的类型映射"""
        factory = TypeFactory
        
        # 原始类型映射
        self._register_primitive_mapping("csharp", "java", "int", "int")
        self._register_primitive_mapping("csharp", "java", "long", "long")
        self._register_primitive_mapping("csharp", "java", "float", "float")
        self._register_primitive_mapping("csharp", "java", "double", "double")
        self._register_primitive_mapping("csharp", "java", "bool", "boolean")
        self._register_primitive_mapping("csharp", "java", "char", "char")
        self._register_primitive_mapping("csharp", "java", "byte", "byte")
        self._register_primitive_mapping("csharp", "java", "short", "short")
        
        # 引用类型映射
        self._register_primitive_mapping("csharp", "java", "string", "String")
        self._register_primitive_mapping("csharp", "java", "object", "Object")
        
        # 容器类型映射
        list_mapping = TypeMapping(
            source_type=factory.create_container_type("List", "csharp"),
            target_type=factory.create_container_type("List", "java")
        )
        self.registry.register_mapping("csharp", "java", "List", "List", list_mapping)
        
        dictionary_mapping = TypeMapping(
            source_type=factory.create_container_type("Dictionary", "csharp"),
            target_type=factory.create_container_type("Map", "java")
        )
        self.registry.register_mapping("csharp", "java", "Dictionary", "Map", dictionary_mapping)
        
        hashset_mapping = TypeMapping(
            source_type=factory.create_container_type("HashSet", "csharp"),
            target_type=factory.create_container_type("Set", "java")
        )
        self.registry.register_mapping("csharp", "java", "HashSet", "Set", hashset_mapping)
    
    def _register_primitive_mapping(self, source_lang: str, target_lang: str, 
                                   source_type_name: str, target_type_name: str) -> None:
        """注册原始类型映射"""
        factory = TypeFactory
        mapping = TypeMapping(
            source_type=factory.create_primitive_type(source_type_name, source_lang),
            target_type=factory.create_primitive_type(target_type_name, target_lang)
        )
        self.registry.register_mapping(source_lang, target_lang, source_type_name, target_type_name, mapping)
    
    def map_type(self, type_info: TypeInfo, target_lang: str) -> TypeInfo:
        """映射类型"""
        source_lang = type_info.source_lang
        
        # 如果源语言和目标语言相同，直接返回
        if source_lang == target_lang:
            return type_info
        
        # 根据类型分类处理
        if type_info.category == TypeCategory.PRIMITIVE:
            return self._map_primitive_type(type_info, target_lang)
        
        elif type_info.category == TypeCategory.CONTAINER:
            return self._map_container_type(type_info, target_lang)
        
        elif type_info.category == TypeCategory.UNION:
            return self._map_union_type(type_info, target_lang)
        
        elif type_info.category == TypeCategory.OPTIONAL:
            return self._map_optional_type(type_info, target_lang)
        
        elif type_info.category == TypeCategory.CLASS:
            return self._map_class_type(type_info, target_lang)
        
        elif type_info.category == TypeCategory.FUNCTION:
            return self._map_function_type(type_info, target_lang)
        
        elif type_info.category == TypeCategory.ANY:
            return self._map_any_type(type_info, target_lang)
        
        # 默认返回未知类型
        return TypeFactory.create_unknown_type(target_lang)
    
    def _map_primitive_type(self, type_info: TypeInfo, target_lang: str) -> TypeInfo:
        """映射原始类型"""
        source_lang = type_info.source_lang
        source_type_name = type_info.name
        
        # 查找映射
        mapping = self.registry.get_mapping(source_lang, target_lang, source_type_name)
        if mapping:
            return mapping.target_type
        
        # 如果没有找到映射，返回未知类型
        return TypeFactory.create_unknown_type(target_lang)
    
    def _map_container_type(self, type_info: TypeInfo, target_lang: str) -> TypeInfo:
        """映射容器类型"""
        source_lang = type_info.source_lang
        source_type_name = type_info.name
        
        # 查找映射
        mapping = self.registry.get_mapping(source_lang, target_lang, source_type_name)
        if not mapping:
            return TypeFactory.create_unknown_type(target_lang)
        
        # 映射类型参数
        mapped_type_args = []
        for arg in type_info.type_args:
            mapped_arg = self.map_type(arg, target_lang)
            mapped_type_args.append(mapped_arg)
        
        # 创建目标类型
        target_type = TypeFactory.create_container_type(
            mapping.target_type.name,
            target_lang,
            mapped_type_args
        )
        
        return target_type
    
    def _map_union_type(self, type_info: TypeInfo, target_lang: str) -> TypeInfo:
        """映射联合类型"""
        # 映射类型参数
        mapped_type_args = []
        for arg in type_info.type_args:
            mapped_arg = self.map_type(arg, target_lang)
            mapped_type_args.append(mapped_arg)
        
        # 创建目标类型
        return TypeFactory.create_union_type(target_lang, mapped_type_args)
    
    def _map_optional_type(self, type_info: TypeInfo, target_lang: str) -> TypeInfo:
        """映射可选类型"""
        # 映射内部类型
        inner_type = type_info.type_args[0]
        mapped_inner_type = self.map_type(inner_type, target_lang)
        
        # 创建目标类型
        return TypeFactory.create_optional_type(target_lang, mapped_inner_type)
    
    def _map_class_type(self, type_info: TypeInfo, target_lang: str) -> TypeInfo:
        """映射类类型"""
        source_lang = type_info.source_lang
        source_type_name = type_info.name
        
        # 查找映射
        mapping = self.registry.get_mapping(source_lang, target_lang, source_type_name)
        if mapping:
            return mapping.target_type
        
        # 如果没有找到映射，使用相同的类名
        return TypeFactory.create_class_type(source_type_name, target_lang)
    
    def _map_function_type(self, type_info: TypeInfo, target_lang: str) -> TypeInfo:
        """映射函数类型"""
        # 简单处理：直接创建目标语言的函数类型
        return TypeFactory.create_function_type(target_lang)
    
    def _map_any_type(self, type_info: TypeInfo, target_lang: str) -> TypeInfo:
        """映射任意类型"""
        # 简单处理：直接创建目标语言的任意类型
        return TypeFactory.create_any_type(target_lang)
    
    def register_custom_mapping(self, source_lang: str, target_lang: str, 
                               source_type_name: str, target_type_name: str, 
                               conversion_rules: Dict[str, Any] = None) -> None:
        """注册自定义类型映射"""
        factory = TypeFactory
        
        # 创建源类型
        if source_type_name in ["int", "float", "str", "bool", "None"]:
            source_type = factory.create_primitive_type(source_type_name, source_lang)
        else:
            source_type = factory.create_class_type(source_type_name, source_lang)
        
        # 创建目标类型
        if target_type_name in ["int", "float", "str", "bool", "None"]:
            target_type = factory.create_primitive_type(target_type_name, target_lang)
        else:
            target_type = factory.create_class_type(target_type_name, target_lang)
        
        # 创建映射
        mapping = TypeMapping(
            source_type=source_type,
            target_type=target_type,
            conversion_rules=conversion_rules or {}
        )
        
        # 注册映射
        self.registry.register_mapping(source_lang, target_lang, source_type_name, target_type_name, mapping)
    
    def get_all_mappings(self, source_lang: str, target_lang: str) -> Dict[str, TypeMapping]:
        """获取所有类型映射"""
        return self.registry.get_all_mappings(source_lang, target_lang)