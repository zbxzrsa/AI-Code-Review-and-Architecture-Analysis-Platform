"""
示例插件实现 - 展示如何开发各种类型的插件
"""

from typing import Dict, List, Any
from datetime import datetime
import asyncio
import re
import json
import logging

from .plugin_manager import (
    PluginMetadata, PluginType, TranslationEnginePlugin, 
    ContentProcessorPlugin, QualityCheckerPlugin
)

logger = logging.getLogger(__name__)


class MockTranslationEngine(TranslationEnginePlugin):
    """模拟翻译引擎插件"""
    
    def __init__(self):
        self.initialized = False
        self.translation_count = 0
        self.supported_languages = ['en', 'zh', 'ja', 'ko', 'fr', 'de', 'es']
        
        # 简单的翻译映射（仅用于演示）
        self.translations = {
            ('en', 'zh'): {
                'hello': '你好',
                'world': '世界',
                'thank you': '谢谢',
                'goodbye': '再见'
            },
            ('zh', 'en'): {
                '你好': 'hello',
                '世界': 'world',
                '谢谢': 'thank you',
                '再见': 'goodbye'
            }
        }
    
    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            id="mock_translation_engine",
            name="Mock Translation Engine",
            version="1.0.0",
            description="A mock translation engine for testing purposes",
            author="System",
            plugin_type=PluginType.TRANSLATION_ENGINE,
            supported_features=["translate", "detect_language", "batch_translate"],
            dependencies=[]
        )
    
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """初始化插件"""
        try:
            self.config = config
            self.initialized = True
            logger.info("Mock translation engine initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize mock translation engine: {e}")
            return False
    
    async def cleanup(self) -> bool:
        """清理资源"""
        self.initialized = False
        logger.info("Mock translation engine cleaned up")
        return True
    
    def get_health_status(self) -> Dict[str, Any]:
        """获取健康状态"""
        return {
            'status': 'healthy' if self.initialized else 'unhealthy',
            'translation_count': self.translation_count,
            'supported_languages': len(self.supported_languages),
            'last_check': datetime.now().isoformat()
        }
    
    async def translate(self, text: str, source_lang: str, target_lang: str, **kwargs) -> str:
        """翻译文本"""
        if not self.initialized:
            raise RuntimeError("Plugin not initialized")
        
        self.translation_count += 1
        
        # 模拟翻译延迟
        await asyncio.sleep(0.1)
        
        # 查找翻译映射
        lang_pair = (source_lang, target_lang)
        if lang_pair in self.translations:
            text_lower = text.lower().strip()
            if text_lower in self.translations[lang_pair]:
                return self.translations[lang_pair][text_lower]
        
        # 如果没有找到映射，返回带前缀的原文
        return f"[{source_lang}->{target_lang}] {text}"
    
    async def detect_language(self, text: str) -> str:
        """检测语言"""
        if not self.initialized:
            raise RuntimeError("Plugin not initialized")
        
        # 简单的语言检测逻辑
        if re.search(r'[\u4e00-\u9fff]', text):
            return 'zh'
        elif re.search(r'[a-zA-Z]', text):
            return 'en'
        else:
            return 'auto'
    
    def get_supported_languages(self) -> List[str]:
        """获取支持的语言列表"""
        return self.supported_languages.copy()


class HTMLContentProcessor(ContentProcessorPlugin):
    """HTML内容处理器插件"""
    
    def __init__(self):
        self.initialized = False
        self.processing_count = 0
        self.supported_types = ['html', 'xhtml']
    
    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            id="html_content_processor",
            name="HTML Content Processor",
            version="1.0.0",
            description="Processes HTML content for translation",
            author="System",
            plugin_type=PluginType.CONTENT_PROCESSOR,
            supported_features=["extract_text", "preserve_structure", "clean_html"],
            dependencies=[]
        )
    
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """初始化插件"""
        try:
            self.config = config
            self.initialized = True
            logger.info("HTML content processor initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize HTML content processor: {e}")
            return False
    
    async def cleanup(self) -> bool:
        """清理资源"""
        self.initialized = False
        logger.info("HTML content processor cleaned up")
        return True
    
    def get_health_status(self) -> Dict[str, Any]:
        """获取健康状态"""
        return {
            'status': 'healthy' if self.initialized else 'unhealthy',
            'processing_count': self.processing_count,
            'supported_types': self.supported_types,
            'last_check': datetime.now().isoformat()
        }
    
    async def process_content(self, content: str, content_type: str, **kwargs) -> str:
        """处理HTML内容"""
        if not self.initialized:
            raise RuntimeError("Plugin not initialized")
        
        if content_type not in self.supported_types:
            return content
        
        self.processing_count += 1
        
        # 模拟处理延迟
        await asyncio.sleep(0.05)
        
        processed = content
        
        # 清理HTML注释
        processed = re.sub(r'<!--.*?-->', '', processed, flags=re.DOTALL)
        
        # 标准化空白字符
        processed = re.sub(r'\s+', ' ', processed)
        
        # 移除空的标签
        processed = re.sub(r'<(\w+)[^>]*>\s*</\1>', '', processed)
        
        # 提取可翻译文本（简化实现）
        if kwargs.get('extract_text_only', False):
            # 移除所有HTML标签，只保留文本
            processed = re.sub(r'<[^>]+>', '', processed)
            processed = processed.strip()
        
        return processed
    
    def get_supported_content_types(self) -> List[str]:
        """获取支持的内容类型"""
        return self.supported_types.copy()


class BasicQualityChecker(QualityCheckerPlugin):
    """基础质量检查器插件"""
    
    def __init__(self):
        self.initialized = False
        self.check_count = 0
        self.quality_metrics = [
            'length_score',
            'readability_score',
            'completeness_score',
            'language_consistency'
        ]
    
    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            id="basic_quality_checker",
            name="Basic Quality Checker",
            version="1.0.0",
            description="Basic quality checking for translated content",
            author="System",
            plugin_type=PluginType.QUALITY_CHECKER,
            supported_features=["length_check", "readability_check", "completeness_check"],
            dependencies=[]
        )
    
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """初始化插件"""
        try:
            self.config = config
            self.initialized = True
            logger.info("Basic quality checker initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize basic quality checker: {e}")
            return False
    
    async def cleanup(self) -> bool:
        """清理资源"""
        self.initialized = False
        logger.info("Basic quality checker cleaned up")
        return True
    
    def get_health_status(self) -> Dict[str, Any]:
        """获取健康状态"""
        return {
            'status': 'healthy' if self.initialized else 'unhealthy',
            'check_count': self.check_count,
            'metrics_count': len(self.quality_metrics),
            'last_check': datetime.now().isoformat()
        }
    
    async def check_quality(self, content: str, **kwargs) -> Dict[str, Any]:
        """检查内容质量"""
        if not self.initialized:
            raise RuntimeError("Plugin not initialized")
        
        self.check_count += 1
        
        # 模拟检查延迟
        await asyncio.sleep(0.02)
        
        original_content = kwargs.get('original_content', '')
        
        quality_result = {
            'overall_score': 0.0,
            'metrics': {},
            'issues': [],
            'suggestions': []
        }
        
        # 长度评分
        length_score = self._check_length(content, original_content)
        quality_result['metrics']['length_score'] = length_score
        
        # 可读性评分
        readability_score = self._check_readability(content)
        quality_result['metrics']['readability_score'] = readability_score
        
        # 完整性评分
        completeness_score = self._check_completeness(content)
        quality_result['metrics']['completeness_score'] = completeness_score
        
        # 语言一致性评分
        consistency_score = self._check_language_consistency(content)
        quality_result['metrics']['language_consistency'] = consistency_score
        
        # 计算总体评分
        scores = [length_score, readability_score, completeness_score, consistency_score]
        quality_result['overall_score'] = sum(scores) / len(scores)
        
        # 生成问题和建议
        quality_result['issues'] = self._identify_issues(quality_result['metrics'])
        quality_result['suggestions'] = self._generate_suggestions(quality_result['metrics'])
        
        return quality_result
    
    def _check_length(self, content: str, original_content: str) -> float:
        """检查长度合理性"""
        if not original_content:
            return 0.8  # 没有原文对比，给中等分数
        
        content_len = len(content.strip())
        original_len = len(original_content.strip())
        
        if original_len == 0:
            return 0.0
        
        ratio = content_len / original_len
        
        # 理想的长度比例在0.8-1.5之间
        if 0.8 <= ratio <= 1.5:
            return 1.0
        elif 0.5 <= ratio <= 2.0:
            return 0.7
        else:
            return 0.3
    
    def _check_readability(self, content: str) -> float:
        """检查可读性"""
        if not content.strip():
            return 0.0
        
        # 简单的可读性指标
        sentences = len(re.findall(r'[.!?]+', content))
        words = len(content.split())
        
        if sentences == 0:
            return 0.5
        
        avg_words_per_sentence = words / sentences
        
        # 理想的句子长度
        if 10 <= avg_words_per_sentence <= 20:
            return 1.0
        elif 5 <= avg_words_per_sentence <= 30:
            return 0.7
        else:
            return 0.4
    
    def _check_completeness(self, content: str) -> float:
        """检查完整性"""
        content = content.strip()
        
        if not content:
            return 0.0
        
        completeness_score = 0.0
        
        # 检查是否有开头
        if len(content) > 10:
            completeness_score += 0.25
        
        # 检查是否有结尾标点
        if content.endswith(('.', '!', '?', '。', '！', '？')):
            completeness_score += 0.25
        
        # 检查长度是否合理
        if 20 <= len(content) <= 5000:
            completeness_score += 0.25
        
        # 检查是否有明显的截断
        if not content.endswith(('...', '…', '[truncated]')):
            completeness_score += 0.25
        
        return completeness_score
    
    def _check_language_consistency(self, content: str) -> float:
        """检查语言一致性"""
        if not content.strip():
            return 0.0
        
        # 简单的语言一致性检查
        has_chinese = bool(re.search(r'[\u4e00-\u9fff]', content))
        has_english = bool(re.search(r'[a-zA-Z]', content))
        has_numbers = bool(re.search(r'\d', content))
        
        # 如果同时包含中英文，可能是混合语言，降低评分
        if has_chinese and has_english:
            # 检查是否是合理的混合（如专有名词）
            english_ratio = len(re.findall(r'[a-zA-Z]', content)) / len(content)
            if english_ratio < 0.1:  # 少量英文可能是专有名词
                return 0.9
            else:
                return 0.6
        
        return 1.0
    
    def _identify_issues(self, metrics: Dict[str, float]) -> List[str]:
        """识别质量问题"""
        issues = []
        
        if metrics.get('length_score', 0) < 0.5:
            issues.append("翻译长度与原文差异过大")
        
        if metrics.get('readability_score', 0) < 0.5:
            issues.append("句子长度不合理，影响可读性")
        
        if metrics.get('completeness_score', 0) < 0.5:
            issues.append("内容可能不完整")
        
        if metrics.get('language_consistency', 0) < 0.7:
            issues.append("语言使用不一致")
        
        return issues
    
    def _generate_suggestions(self, metrics: Dict[str, float]) -> List[str]:
        """生成改进建议"""
        suggestions = []
        
        if metrics.get('length_score', 0) < 0.7:
            suggestions.append("建议检查翻译的完整性和准确性")
        
        if metrics.get('readability_score', 0) < 0.7:
            suggestions.append("建议调整句子结构，提高可读性")
        
        if metrics.get('completeness_score', 0) < 0.7:
            suggestions.append("建议检查内容的开头和结尾")
        
        if metrics.get('language_consistency', 0) < 0.8:
            suggestions.append("建议统一语言使用，避免不必要的混合")
        
        return suggestions
    
    def get_quality_metrics(self) -> List[str]:
        """获取质量指标列表"""
        return self.quality_metrics.copy()


class AdvancedTranslationEngine(TranslationEnginePlugin):
    """高级翻译引擎插件（模拟）"""
    
    def __init__(self):
        self.initialized = False
        self.translation_count = 0
        self.supported_languages = ['en', 'zh', 'ja', 'ko', 'fr', 'de', 'es', 'ru', 'ar']
        self.cache = {}
    
    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            id="advanced_translation_engine",
            name="Advanced Translation Engine",
            version="2.0.0",
            description="Advanced translation engine with caching and quality optimization",
            author="System",
            plugin_type=PluginType.TRANSLATION_ENGINE,
            supported_features=[
                "translate", "detect_language", "batch_translate", 
                "quality_scoring", "context_aware", "caching"
            ],
            dependencies=[]
        )
    
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """初始化插件"""
        try:
            self.config = config
            self.cache_size = config.get('cache_size', 1000)
            self.quality_threshold = config.get('quality_threshold', 0.8)
            self.initialized = True
            logger.info("Advanced translation engine initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize advanced translation engine: {e}")
            return False
    
    async def cleanup(self) -> bool:
        """清理资源"""
        self.cache.clear()
        self.initialized = False
        logger.info("Advanced translation engine cleaned up")
        return True
    
    def get_health_status(self) -> Dict[str, Any]:
        """获取健康状态"""
        return {
            'status': 'healthy' if self.initialized else 'unhealthy',
            'translation_count': self.translation_count,
            'cache_size': len(self.cache),
            'cache_hit_rate': self._calculate_cache_hit_rate(),
            'supported_languages': len(self.supported_languages),
            'last_check': datetime.now().isoformat()
        }
    
    async def translate(self, text: str, source_lang: str, target_lang: str, **kwargs) -> str:
        """高级翻译功能"""
        if not self.initialized:
            raise RuntimeError("Plugin not initialized")
        
        # 检查缓存
        cache_key = f"{source_lang}:{target_lang}:{hash(text)}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        self.translation_count += 1
        
        # 模拟高级翻译处理
        await asyncio.sleep(0.2)  # 模拟更长的处理时间
        
        # 上下文感知翻译（简化实现）
        context = kwargs.get('context', '')
        if context:
            translated = f"[Context-aware] {text} -> {target_lang}"
        else:
            translated = f"[Advanced] {text} -> {target_lang}"
        
        # 缓存结果
        if len(self.cache) < self.cache_size:
            self.cache[cache_key] = translated
        
        return translated
    
    async def detect_language(self, text: str) -> str:
        """高级语言检测"""
        if not self.initialized:
            raise RuntimeError("Plugin not initialized")
        
        # 模拟更准确的语言检测
        await asyncio.sleep(0.05)
        
        # 多种语言特征检测
        if re.search(r'[\u4e00-\u9fff]', text):
            return 'zh'
        elif re.search(r'[\u3040-\u309f\u30a0-\u30ff]', text):
            return 'ja'
        elif re.search(r'[\uac00-\ud7af]', text):
            return 'ko'
        elif re.search(r'[а-яё]', text, re.IGNORECASE):
            return 'ru'
        elif re.search(r'[a-zA-Z]', text):
            return 'en'
        else:
            return 'auto'
    
    def get_supported_languages(self) -> List[str]:
        """获取支持的语言列表"""
        return self.supported_languages.copy()
    
    def _calculate_cache_hit_rate(self) -> float:
        """计算缓存命中率"""
        if self.translation_count == 0:
            return 0.0
        # 这里简化计算，实际应该跟踪缓存命中次数
        return min(0.8, len(self.cache) / max(self.translation_count, 1))


# 插件注册函数
def get_available_plugins():
    """获取可用的插件列表"""
    return [
        MockTranslationEngine,
        HTMLContentProcessor,
        BasicQualityChecker,
        AdvancedTranslationEngine
    ]