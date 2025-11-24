"""
内容分析微服务 - 文本预处理、质量检查和内容优化
支持插件化的内容处理器和质量检查工具
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import re
import uuid
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class ContentType(Enum):
    """内容类型"""
    PLAIN_TEXT = "plain_text"
    HTML = "html"
    MARKDOWN = "markdown"
    CODE = "code"
    JSON = "json"
    XML = "xml"


class QualityLevel(Enum):
    """质量等级"""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"


class ProcessingStage(Enum):
    """处理阶段"""
    PREPROCESSING = "preprocessing"
    ANALYSIS = "analysis"
    OPTIMIZATION = "optimization"
    VALIDATION = "validation"


@dataclass
class ContentSegment:
    """内容片段"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    content_type: ContentType = ContentType.PLAIN_TEXT
    start_position: int = 0
    end_position: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def length(self) -> int:
        return len(self.content)


@dataclass
class QualityMetrics:
    """质量指标"""
    readability_score: float = 0.0
    complexity_score: float = 0.0
    coherence_score: float = 0.0
    completeness_score: float = 0.0
    overall_score: float = 0.0
    quality_level: QualityLevel = QualityLevel.FAIR
    issues: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)


@dataclass
class AnalysisResult:
    """分析结果"""
    original_content: str
    processed_content: str
    segments: List[ContentSegment] = field(default_factory=list)
    quality_metrics: QualityMetrics = field(default_factory=QualityMetrics)
    processing_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


class ContentProcessorInterface(ABC):
    """内容处理器接口"""
    
    @abstractmethod
    async def process(self, content: str, content_type: ContentType) -> str:
        """处理内容"""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """获取处理器名称"""
        pass
    
    @abstractmethod
    def get_supported_types(self) -> List[ContentType]:
        """获取支持的内容类型"""
        pass


class QualityCheckerInterface(ABC):
    """质量检查器接口"""
    
    @abstractmethod
    async def check_quality(self, content: str, content_type: ContentType) -> QualityMetrics:
        """检查内容质量"""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """获取检查器名称"""
        pass


class TextPreprocessor(ContentProcessorInterface):
    """文本预处理器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    async def process(self, content: str, content_type: ContentType) -> str:
        """预处理文本内容"""
        processed = content
        
        # 清理多余空白
        if self.config.get('clean_whitespace', True):
            processed = re.sub(r'\s+', ' ', processed.strip())
        
        # 标准化换行符
        if self.config.get('normalize_newlines', True):
            processed = processed.replace('\r\n', '\n').replace('\r', '\n')
        
        # 移除控制字符
        if self.config.get('remove_control_chars', True):
            processed = ''.join(char for char in processed if ord(char) >= 32 or char in '\n\t')
        
        # HTML特殊处理
        if content_type == ContentType.HTML:
            processed = await self._process_html(processed)
        
        # Markdown特殊处理
        elif content_type == ContentType.MARKDOWN:
            processed = await self._process_markdown(processed)
        
        return processed
    
    async def _process_html(self, content: str) -> str:
        """处理HTML内容"""
        # 移除HTML注释
        content = re.sub(r'<!--.*?-->', '', content, flags=re.DOTALL)
        
        # 标准化标签
        content = re.sub(r'<(\w+)([^>]*?)/?>', r'<\1\2>', content)
        
        return content
    
    async def _process_markdown(self, content: str) -> str:
        """处理Markdown内容"""
        # 标准化标题
        content = re.sub(r'^#{1,6}\s*(.+)$', lambda m: '#' * len(m.group(0).split()[0]) + ' ' + m.group(1), content, flags=re.MULTILINE)
        
        return content
    
    def get_name(self) -> str:
        return "TextPreprocessor"
    
    def get_supported_types(self) -> List[ContentType]:
        return [ContentType.PLAIN_TEXT, ContentType.HTML, ContentType.MARKDOWN]


class ContentSegmenter(ContentProcessorInterface):
    """内容分段器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.max_segment_length = config.get('max_segment_length', 1000)
        self.overlap_length = config.get('overlap_length', 100)
    
    async def process(self, content: str, content_type: ContentType) -> str:
        """分段处理内容"""
        # 这里返回原内容，实际分段在segment_content方法中
        return content
    
    async def segment_content(self, content: str, content_type: ContentType) -> List[ContentSegment]:
        """将内容分段"""
        segments = []
        
        if content_type == ContentType.HTML:
            segments = await self._segment_html(content)
        elif content_type == ContentType.MARKDOWN:
            segments = await self._segment_markdown(content)
        else:
            segments = await self._segment_text(content)
        
        return segments
    
    async def _segment_text(self, content: str) -> List[ContentSegment]:
        """分段纯文本"""
        segments = []
        
        # 按句子分段
        sentences = re.split(r'[.!?]+\s+', content)
        
        current_segment = ""
        start_pos = 0
        
        for sentence in sentences:
            if len(current_segment) + len(sentence) > self.max_segment_length:
                if current_segment:
                    segments.append(ContentSegment(
                        content=current_segment.strip(),
                        content_type=ContentType.PLAIN_TEXT,
                        start_position=start_pos,
                        end_position=start_pos + len(current_segment)
                    ))
                    start_pos += len(current_segment)
                    current_segment = sentence
                else:
                    # 单个句子太长，强制分段
                    segments.append(ContentSegment(
                        content=sentence[:self.max_segment_length],
                        content_type=ContentType.PLAIN_TEXT,
                        start_position=start_pos,
                        end_position=start_pos + self.max_segment_length
                    ))
                    start_pos += self.max_segment_length
            else:
                current_segment += " " + sentence if current_segment else sentence
        
        # 添加最后一段
        if current_segment:
            segments.append(ContentSegment(
                content=current_segment.strip(),
                content_type=ContentType.PLAIN_TEXT,
                start_position=start_pos,
                end_position=start_pos + len(current_segment)
            ))
        
        return segments
    
    async def _segment_html(self, content: str) -> List[ContentSegment]:
        """分段HTML内容"""
        # 按HTML标签分段
        tag_pattern = r'<[^>]+>'
        parts = re.split(tag_pattern, content)
        
        segments = []
        position = 0
        
        for part in parts:
            if part.strip():
                segments.append(ContentSegment(
                    content=part.strip(),
                    content_type=ContentType.HTML,
                    start_position=position,
                    end_position=position + len(part)
                ))
            position += len(part)
        
        return segments
    
    async def _segment_markdown(self, content: str) -> List[ContentSegment]:
        """分段Markdown内容"""
        # 按标题分段
        sections = re.split(r'^#+\s+', content, flags=re.MULTILINE)
        
        segments = []
        position = 0
        
        for section in sections:
            if section.strip():
                segments.append(ContentSegment(
                    content=section.strip(),
                    content_type=ContentType.MARKDOWN,
                    start_position=position,
                    end_position=position + len(section)
                ))
            position += len(section)
        
        return segments
    
    def get_name(self) -> str:
        return "ContentSegmenter"
    
    def get_supported_types(self) -> List[ContentType]:
        return [ContentType.PLAIN_TEXT, ContentType.HTML, ContentType.MARKDOWN]


class BasicQualityChecker(QualityCheckerInterface):
    """基础质量检查器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    async def check_quality(self, content: str, content_type: ContentType) -> QualityMetrics:
        """检查内容质量"""
        metrics = QualityMetrics()
        
        # 可读性评分
        metrics.readability_score = await self._calculate_readability(content)
        
        # 复杂度评分
        metrics.complexity_score = await self._calculate_complexity(content)
        
        # 连贯性评分
        metrics.coherence_score = await self._calculate_coherence(content)
        
        # 完整性评分
        metrics.completeness_score = await self._calculate_completeness(content)
        
        # 计算总体评分
        metrics.overall_score = (
            metrics.readability_score * 0.3 +
            metrics.complexity_score * 0.2 +
            metrics.coherence_score * 0.3 +
            metrics.completeness_score * 0.2
        )
        
        # 确定质量等级
        if metrics.overall_score >= 0.9:
            metrics.quality_level = QualityLevel.EXCELLENT
        elif metrics.overall_score >= 0.7:
            metrics.quality_level = QualityLevel.GOOD
        elif metrics.overall_score >= 0.5:
            metrics.quality_level = QualityLevel.FAIR
        else:
            metrics.quality_level = QualityLevel.POOR
        
        # 生成问题和建议
        metrics.issues = await self._identify_issues(content, metrics)
        metrics.suggestions = await self._generate_suggestions(content, metrics)
        
        return metrics
    
    async def _calculate_readability(self, content: str) -> float:
        """计算可读性评分"""
        if not content.strip():
            return 0.0
        
        # 简单的可读性指标
        words = len(content.split())
        sentences = len(re.findall(r'[.!?]+', content))
        
        if sentences == 0:
            return 0.5
        
        avg_words_per_sentence = words / sentences
        
        # 理想的句子长度是15-20个词
        if 15 <= avg_words_per_sentence <= 20:
            return 1.0
        elif 10 <= avg_words_per_sentence <= 25:
            return 0.8
        elif 5 <= avg_words_per_sentence <= 30:
            return 0.6
        else:
            return 0.4
    
    async def _calculate_complexity(self, content: str) -> float:
        """计算复杂度评分"""
        if not content.strip():
            return 0.0
        
        # 词汇复杂度
        words = content.split()
        unique_words = set(words)
        
        if len(words) == 0:
            return 0.0
        
        vocabulary_diversity = len(unique_words) / len(words)
        
        # 适中的词汇多样性得分更高
        if 0.4 <= vocabulary_diversity <= 0.7:
            return 1.0
        elif 0.3 <= vocabulary_diversity <= 0.8:
            return 0.8
        else:
            return 0.6
    
    async def _calculate_coherence(self, content: str) -> float:
        """计算连贯性评分"""
        if not content.strip():
            return 0.0
        
        sentences = re.split(r'[.!?]+', content)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) < 2:
            return 0.5
        
        # 简单的连贯性检查：相邻句子的词汇重叠
        coherence_scores = []
        
        for i in range(len(sentences) - 1):
            words1 = set(sentences[i].lower().split())
            words2 = set(sentences[i + 1].lower().split())
            
            if len(words1) == 0 or len(words2) == 0:
                continue
            
            overlap = len(words1.intersection(words2))
            total = len(words1.union(words2))
            
            if total > 0:
                coherence_scores.append(overlap / total)
        
        if not coherence_scores:
            return 0.5
        
        return min(1.0, sum(coherence_scores) / len(coherence_scores) * 3)
    
    async def _calculate_completeness(self, content: str) -> float:
        """计算完整性评分"""
        if not content.strip():
            return 0.0
        
        # 检查内容是否看起来完整
        completeness_indicators = 0
        total_indicators = 4
        
        # 有开头
        if len(content) > 50:
            completeness_indicators += 1
        
        # 有结尾标点
        if content.strip().endswith(('.', '!', '?')):
            completeness_indicators += 1
        
        # 长度适中
        if 100 <= len(content) <= 5000:
            completeness_indicators += 1
        
        # 没有明显的截断
        if not content.endswith('...') and not content.endswith('…'):
            completeness_indicators += 1
        
        return completeness_indicators / total_indicators
    
    async def _identify_issues(self, content: str, metrics: QualityMetrics) -> List[str]:
        """识别内容问题"""
        issues = []
        
        if metrics.readability_score < 0.5:
            issues.append("文本可读性较差，句子可能过长或过短")
        
        if metrics.complexity_score < 0.5:
            issues.append("词汇使用单调或过于复杂")
        
        if metrics.coherence_score < 0.5:
            issues.append("句子之间缺乏连贯性")
        
        if metrics.completeness_score < 0.5:
            issues.append("内容可能不完整")
        
        # 检查特殊字符
        if re.search(r'[^\w\s\.,!?;:()\-\'\"]+', content):
            issues.append("包含特殊字符，可能影响翻译质量")
        
        return issues
    
    async def _generate_suggestions(self, content: str, metrics: QualityMetrics) -> List[str]:
        """生成改进建议"""
        suggestions = []
        
        if metrics.readability_score < 0.7:
            suggestions.append("建议调整句子长度，保持在15-20个词之间")
        
        if metrics.complexity_score < 0.7:
            suggestions.append("建议增加词汇多样性或简化复杂词汇")
        
        if metrics.coherence_score < 0.7:
            suggestions.append("建议增加句子间的逻辑连接词")
        
        if metrics.completeness_score < 0.7:
            suggestions.append("建议检查内容完整性，确保有明确的开头和结尾")
        
        return suggestions
    
    def get_name(self) -> str:
        return "BasicQualityChecker"


class ContentAnalysisService:
    """内容分析服务"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.processors: Dict[str, ContentProcessorInterface] = {}
        self.quality_checkers: Dict[str, QualityCheckerInterface] = {}
        
        # 注册默认处理器
        self.register_processor(TextPreprocessor(config))
        self.register_processor(ContentSegmenter(config))
        
        # 注册默认质量检查器
        self.register_quality_checker(BasicQualityChecker(config))
    
    def register_processor(self, processor: ContentProcessorInterface):
        """注册内容处理器"""
        self.processors[processor.get_name()] = processor
        logger.info(f"Registered content processor: {processor.get_name()}")
    
    def register_quality_checker(self, checker: QualityCheckerInterface):
        """注册质量检查器"""
        self.quality_checkers[checker.get_name()] = checker
        logger.info(f"Registered quality checker: {checker.get_name()}")
    
    async def analyze_content(self, content: str, content_type: ContentType = ContentType.PLAIN_TEXT) -> AnalysisResult:
        """分析内容"""
        start_time = datetime.now()
        
        try:
            # 预处理
            processed_content = content
            for processor in self.processors.values():
                if content_type in processor.get_supported_types():
                    processed_content = await processor.process(processed_content, content_type)
            
            # 内容分段
            segments = []
            segmenter = self.processors.get('ContentSegmenter')
            if segmenter and hasattr(segmenter, 'segment_content'):
                segments = await segmenter.segment_content(processed_content, content_type)
            
            # 质量检查
            quality_metrics = QualityMetrics()
            for checker in self.quality_checkers.values():
                metrics = await checker.check_quality(processed_content, content_type)
                # 合并质量指标（这里简化为使用第一个检查器的结果）
                quality_metrics = metrics
                break
            
            # 计算处理时间
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return AnalysisResult(
                original_content=content,
                processed_content=processed_content,
                segments=segments,
                quality_metrics=quality_metrics,
                processing_time=processing_time,
                metadata={
                    'content_type': content_type.value,
                    'processors_used': list(self.processors.keys()),
                    'quality_checkers_used': list(self.quality_checkers.keys())
                }
            )
            
        except Exception as e:
            logger.error(f"Content analysis failed: {e}")
            raise
    
    async def batch_analyze(self, contents: List[Tuple[str, ContentType]]) -> List[AnalysisResult]:
        """批量分析内容"""
        tasks = [self.analyze_content(content, content_type) for content, content_type in contents]
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    def get_supported_content_types(self) -> List[ContentType]:
        """获取支持的内容类型"""
        supported_types = set()
        for processor in self.processors.values():
            supported_types.update(processor.get_supported_types())
        return list(supported_types)


# 全局内容分析服务实例
content_analysis_service: Optional[ContentAnalysisService] = None


def initialize_content_analysis_service(config: Dict[str, Any]) -> ContentAnalysisService:
    """初始化内容分析服务"""
    global content_analysis_service
    content_analysis_service = ContentAnalysisService(config)
    return content_analysis_service


def get_content_analysis_service() -> ContentAnalysisService:
    """获取内容分析服务实例"""
    if content_analysis_service is None:
        raise RuntimeError("Content analysis service not initialized")
    return content_analysis_service