"""
预测性缓存策略模块
基于机器学习和历史数据预测可能需要的分析结果，提前缓存以提升命中率
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
import hashlib

# Optional ML dependencies - will be imported only if available
try:
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    np = None
    TfidfVectorizer = None
    cosine_similarity = None

from app.core.cache.cache_manager import CacheManager
from app.models.analysis_cache import AnalysisCache

logger = logging.getLogger(__name__)


@dataclass
class CachePrediction:
    """缓存预测结果"""
    file_path: str
    predicted_score: float
    reason: str
    priority: int
    estimated_ttl: int


@dataclass
class AccessPattern:
    """文件访问模式"""
    file_path: str
    access_count: int
    last_access: datetime
    access_frequency: float  # 每天访问次数
    co_accessed_files: Set[str]  # 经常一起访问的文件
    change_frequency: float  # 文件变更频率


class PredictiveCacheManager:
    """预测性缓存管理器"""
    
    def __init__(
        self,
        cache_manager: CacheManager,
        redis_client,
        dependency_service=None,
        prediction_window_days: int = 30,
        min_access_count: int = 3
    ):
        self.cache_manager = cache_manager
        self.dependency_service = dependency_service
        self.redis_client = redis_client
        self.prediction_window_days = prediction_window_days
        self.min_access_count = min_access_count
        
        # ML模型组件
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self._is_trained = False
        
        # 缓存键前缀
        self.PATTERN_KEY_PREFIX = "predictive:pattern:"
        self.PREDICTION_KEY_PREFIX = "predictive:prediction:"
        self.ACCESS_LOG_KEY = "predictive:access_log"
        
    async def initialize(self):
        """初始化预测模型"""
        await self._load_historical_patterns()
        await self._train_similarity_model()
        self._is_trained = True
        logger.info("Predictive cache manager initialized")
    
    async def record_access(self, file_path: str, tenant_id: str, repo_id: str):
        """记录文件访问"""
        timestamp = datetime.utcnow()
        
        # 记录到访问日志
        access_record = {
            "file_path": file_path,
            "tenant_id": tenant_id,
            "repo_id": repo_id,
            "timestamp": timestamp.isoformat(),
            "day_key": timestamp.strftime("%Y-%m-%d")
        }
        
        await self.redis_client.lpush(
            self.ACCESS_LOG_KEY,
            json.dumps(access_record)
        )
        
        # 更新访问模式
        await self._update_access_pattern(file_path, tenant_id, repo_id, timestamp)
        
        # 清理旧日志（保留30天）
        await self._cleanup_old_access_logs()
    
    async def predict_cache_needs(
        self, 
        changed_files: List[str],
        tenant_id: str,
        repo_id: str,
        limit: int = 50
    ) -> List[CachePrediction]:
        """预测可能需要的缓存项"""
        if not self._is_trained:
            logger.warning("Prediction model not trained, returning empty predictions")
            return []
        
        predictions = []
        
        # 1. 基于依赖关系的预测
        dep_predictions = await self._predict_by_dependencies(
            changed_files, tenant_id, repo_id
        )
        predictions.extend(dep_predictions)
        
        # 2. 基于访问模式的预测
        pattern_predictions = await self._predict_by_access_patterns(
            changed_files, tenant_id, repo_id
        )
        predictions.extend(pattern_predictions)
        
        # 3. 基于相似性的预测
        similarity_predictions = await self._predict_by_similarity(
            changed_files, tenant_id, repo_id
        )
        predictions.extend(similarity_predictions)
        
        # 4. 基于时间模式的预测
        temporal_predictions = await self._predict_by_temporal_patterns(
            changed_files, tenant_id, repo_id
        )
        predictions.extend(temporal_predictions)
        
        # 去重和排序
        unique_predictions = self._deduplicate_predictions(predictions)
        sorted_predictions = sorted(
            unique_predictions,
            key=lambda x: (x.predicted_score, x.priority),
            reverse=True
        )
        
        return sorted_predictions[:limit]
    
    async def preload_predicted_cache(
        self,
        predictions: List[CachePrediction],
        tenant_id: str,
        repo_id: str,
        rulepack_version: str = "default"
    ):
        """预加载预测的缓存项"""
        preload_tasks = []
        
        for prediction in predictions:
            if prediction.predicted_score < 0.3:  # 跳过低置信度预测
                continue
                
            task = asyncio.create_task(
                self._preload_single_cache(
                    prediction.file_path,
                    tenant_id,
                    repo_id,
                    rulepack_version,
                    prediction.estimated_ttl
                )
            )
            preload_tasks.append(task)
        
        # 并行预加载，限制并发数
        semaphore = asyncio.Semaphore(5)
        
        async def bounded_preload(task):
            async with semaphore:
                try:
                    await task
                except Exception as e:
                    logger.error(f"Failed to preload cache: {e}")
        
        await asyncio.gather(
            *[bounded_preload(task) for task in preload_tasks],
            return_exceptions=True
        )
    
    async def get_cache_hit_ratio_prediction(
        self,
        tenant_id: str,
        repo_id: str
    ) -> Dict[str, float]:
        """获取缓存命中率预测"""
        # 获取历史访问模式
        patterns = await self._get_access_patterns(tenant_id, repo_id)
        
        if not patterns:
            return {"predicted_hit_ratio": 0.0, "confidence": 0.0}
        
        # 计算预测指标
        total_files = len(patterns)
        frequently_accessed = len([
            p for p in patterns 
            if p.access_frequency > 0.1  # 每天访问超过0.1次
        ])
        
        # 基于历史数据预测命中率
        base_hit_ratio = frequently_accessed / total_files if total_files > 0 else 0
        
        # 应用时间衰减因子
        recent_access_weight = await self._calculate_recent_access_weight(patterns)
        predicted_ratio = base_hit_ratio * recent_access_weight
        
        # 计算置信度
        confidence = min(1.0, total_files / 100)  # 样本越多置信度越高
        
        return {
            "predicted_hit_ratio": min(0.95, predicted_ratio),  # 最高95%
            "confidence": confidence,
            "sample_size": total_files
        }
    
    async def _load_historical_patterns(self):
        """加载历史访问模式"""
        # 从Redis加载历史访问数据
        access_logs = await self.redis_client.lrange(
            self.ACCESS_LOG_KEY, 0, -1
        )
        
        if not access_logs:
            logger.info("No historical access patterns found")
            return
        
        # 解析访问日志
        parsed_logs = []
        for log_entry in access_logs:
            try:
                parsed_logs.append(json.loads(log_entry))
            except json.JSONDecodeError:
                continue
        
        # 按租户和仓库分组
        pattern_groups = defaultdict(list)
        for log in parsed_logs:
            key = f"{log['tenant_id']}:{log['repo_id']}"
            pattern_groups[key].append(log)
        
        # 为每个租户/仓库计算访问模式
        for key, logs in pattern_groups.items():
            tenant_id, repo_id = key.split(":")
            await self._calculate_access_patterns(logs, tenant_id, repo_id)
    
    async def _train_similarity_model(self):
        """训练相似性模型"""
        # 获取所有文件路径用于训练TF-IDF模型
        all_patterns = await self._get_all_access_patterns()
        
        if not all_patterns:
            return
        
        file_paths = [pattern.file_path for pattern in all_patterns]
        
        # 训练TF-IDF向量化器
        try:
            self.tfidf_vectorizer.fit(file_paths)
            logger.info(f"Similarity model trained on {len(file_paths)} file paths")
        except Exception as e:
            logger.error(f"Failed to train similarity model: {e}")
    
    async def _predict_by_dependencies(
        self,
        changed_files: List[str],
        tenant_id: str,
        repo_id: str
    ) -> List[CachePrediction]:
        """基于依赖关系预测"""
        predictions = []
        
        for changed_file in changed_files:
            try:
                # 获取依赖图中的反向闭包
                affected_files = await self.dependency_service.get_reverse_closure(
                    {changed_file}, tenant_id, repo_id
                )
                
                for affected_file in affected_files:
                    if affected_file in changed_files:
                        continue  # 跳过已变更的文件
                    
                    # 计算预测分数
                    distance = await self.dependency_service.get_dependency_distance(
                        changed_file, affected_file, tenant_id, repo_id
                    )
                    
                    score = 1.0 / (1.0 + distance) if distance else 1.0
                    
                    predictions.append(CachePrediction(
                        file_path=affected_file,
                        predicted_score=score * 0.8,  # 依赖关系权重
                        reason=f"Dependency of {changed_file} (distance: {distance})",
                        priority=1 if distance <= 2 else 2,
                        estimated_ttl=3600  # 1小时
                    ))
                    
            except Exception as e:
                logger.error(f"Failed to predict dependencies for {changed_file}: {e}")
        
        return predictions
    
    async def _predict_by_access_patterns(
        self,
        changed_files: List[str],
        tenant_id: str,
        repo_id: str
    ) -> List[CachePrediction]:
        """基于访问模式预测"""
        predictions = []
        patterns = await self._get_access_patterns(tenant_id, repo_id)
        
        pattern_map = {p.file_path: p for p in patterns}
        
        for changed_file in changed_files:
            if changed_file not in pattern_map:
                continue
            
            pattern = pattern_map[changed_file]
            
            # 预测经常一起访问的文件
            for co_accessed_file in pattern.co_accessed_files:
                if co_accessed_file in changed_files:
                    continue
                
                co_pattern = pattern_map.get(co_accessed_file)
                if not co_pattern:
                    continue
                
                # 计算共访问频率
                co_access_score = min(1.0, co_pattern.access_frequency * 10)
                
                predictions.append(CachePrediction(
                    file_path=co_accessed_file,
                    predicted_score=co_access_score * 0.6,  # 访问模式权重
                    reason=f"Co-accessed with {changed_file}",
                    priority=2,
                    estimated_ttl=1800  # 30分钟
                ))
        
        return predictions
    
    async def _predict_by_similarity(
        self,
        changed_files: List[str],
        tenant_id: str,
        repo_id: str
    ) -> List[CachePrediction]:
        """基于相似性预测"""
        predictions = []
        patterns = await self._get_access_patterns(tenant_id, repo_id)
        
        if not patterns or len(patterns) < 5:
            return predictions
        
        # 计算文件路径的TF-IDF向量
        all_paths = [p.file_path for p in patterns]
        
        try:
            tfidf_matrix = self.tfidf_vectorizer.transform(all_paths)
            
            for changed_file in changed_files:
                if changed_file not in all_paths:
                    continue
                
                # 找到变更文件的索引
                changed_idx = all_paths.index(changed_file)
                changed_vector = tfidf_matrix[changed_idx:changed_idx+1]
                
                # 计算相似度
                similarities = cosine_similarity(changed_vector, tfidf_matrix).flatten()
                
                # 获取最相似的文件（排除自身）
                similar_indices = np.argsort(similarities)[::-1][1:6]  # 前5个最相似的
                
                for idx in similar_indices:
                    similarity_score = similarities[idx]
                    if similarity_score < 0.3:  # 相似度阈值
                        continue
                    
                    similar_file = all_paths[idx]
                    if similar_file in changed_files:
                        continue
                    
                    predictions.append(CachePrediction(
                        file_path=similar_file,
                        predicted_score=similarity_score * 0.4,  # 相似性权重
                        reason=f"Similar to {changed_file} (similarity: {similarity_score:.2f})",
                        priority=3,
                        estimated_ttl=900  # 15分钟
                    ))
                    
        except Exception as e:
            logger.error(f"Failed to predict by similarity: {e}")
        
        return predictions
    
    async def _predict_by_temporal_patterns(
        self,
        changed_files: List[str],
        tenant_id: str,
        repo_id: str
    ) -> List[CachePrediction]:
        """基于时间模式预测"""
        predictions = []
        
        # 获取当前时间特征
        now = datetime.utcnow()
        hour = now.hour
        day_of_week = now.weekday()
        
        # 获取历史时间模式
        temporal_patterns = await self._get_temporal_patterns(tenant_id, repo_id)
        
        for file_path, pattern_data in temporal_patterns.items():
            if file_path in changed_files:
                continue
            
            # 计算时间匹配分数
            hour_score = pattern_data.get("hour_distribution", {}).get(str(hour), 0)
            dow_score = pattern_data.get("dow_distribution", {}).get(str(day_of_week), 0)
            
            temporal_score = (hour_score + dow_score) / 2
            
            if temporal_score > 0.1:  # 时间模式阈值
                predictions.append(CachePrediction(
                    file_path=file_path,
                    predicted_score=temporal_score * 0.3,  # 时间模式权重
                    reason=f"Temporal pattern match (hour: {hour}, dow: {day_of_week})",
                    priority=4,
                    estimated_ttl=600  # 10分钟
                ))
        
        return predictions
    
    def _deduplicate_predictions(
        self, 
        predictions: List[CachePrediction]
    ) -> List[CachePrediction]:
        """去重预测结果"""
        seen_files = {}
        
        for pred in predictions:
            if pred.file_path not in seen_files:
                seen_files[pred.file_path] = pred
            else:
                # 合并预测，取最高分数和优先级
                existing = seen_files[pred.file_path]
                if pred.predicted_score > existing.predicted_score:
                    existing.predicted_score = pred.predicted_score
                    existing.reason = pred.reason
                    existing.priority = min(existing.priority, pred.priority)
        
        return list(seen_files.values())
    
    async def _preload_single_cache(
        self,
        file_path: str,
        tenant_id: str,
        repo_id: str,
        rulepack_version: str,
        ttl: int
    ):
        """预加载单个缓存项"""
        try:
            # 检查是否已缓存
            existing_cache = await self.cache_manager.get(
                tenant_id, repo_id, rulepack_version, file_path
            )
            
            if existing_cache:
                return  # 已存在，无需预加载
            
            # 触发分析并缓存结果
            # 这里需要调用分析服务，具体实现取决于你的分析架构
            logger.info(f"Preloading cache for {file_path}")
            
            # TODO: 调用实际的分析服务
            # analysis_result = await analysis_service.analyze_file(
            #     file_path, tenant_id, repo_id, rulepack_version
            # )
            # await self.cache_manager.set(
            #     tenant_id, repo_id, rulepack_version, file_path,
            #     analysis_result, ttl=ttl
            # )
            
        except Exception as e:
            logger.error(f"Failed to preload cache for {file_path}: {e}")
    
    async def _update_access_pattern(
        self,
        file_path: str,
        tenant_id: str,
        repo_id: str,
        timestamp: datetime
    ):
        """更新访问模式"""
        pattern_key = f"{self.PATTERN_KEY_PREFIX}{tenant_id}:{repo_id}:{file_path}"
        
        # 获取现有模式
        pattern_data = await self.redis_client.hgetall(pattern_key)
        
        if not pattern_data:
            pattern_data = {
                "access_count": 0,
                "first_access": timestamp.isoformat(),
                "last_access": timestamp.isoformat(),
                "co_accessed_files": "[]"
            }
        else:
            pattern_data = {k.decode(): v.decode() for k, v in pattern_data.items()}
        
        # 更新访问计数
        pattern_data["access_count"] = int(pattern_data["access_count"]) + 1
        pattern_data["last_access"] = timestamp.isoformat()
        
        # 保存更新后的模式
        await self.redis_client.hset(pattern_key, mapping=pattern_data)
        await self.redis_client.expire(pattern_key, self.prediction_window_days * 24 * 3600)
    
    async def _get_access_patterns(
        self,
        tenant_id: str,
        repo_id: str
    ) -> List[AccessPattern]:
        """获取访问模式"""
        pattern_keys = await self.redis_client.keys(
            f"{self.PATTERN_KEY_PREFIX}{tenant_id}:{repo_id}:*"
        )
        
        patterns = []
        for key in pattern_keys:
            pattern_data = await self.redis_client.hgetall(key)
            if pattern_data:
                data = {k.decode(): v.decode() for k, v in pattern_data.items()}
                
                file_path = key.decode().split(":")[-1]
                pattern = AccessPattern(
                    file_path=file_path,
                    access_count=int(data.get("access_count", 0)),
                    last_access=datetime.fromisoformat(data.get("last_access")),
                    access_frequency=0.0,  # 需要计算
                    co_accessed_files=set(json.loads(data.get("co_accessed_files", "[]"))),
                    change_frequency=0.0  # 需要计算
                )
                
                # 计算访问频率
                days_since_first = (
                    datetime.utcnow() - datetime.fromisoformat(data.get("first_access"))
                ).days + 1
                pattern.access_frequency = pattern.access_count / days_since_first
                
                patterns.append(pattern)
        
        return patterns
    
    async def _get_temporal_patterns(
        self,
        tenant_id: str,
        repo_id: str
    ) -> Dict[str, Dict]:
        """获取时间模式"""
        # 从访问日志中分析时间模式
        access_logs = await self.redis_client.lrange(self.ACCESS_LOG_KEY, 0, -1)
        
        temporal_patterns = defaultdict(lambda: {
            "hour_distribution": defaultdict(int),
            "dow_distribution": defaultdict(int)
        })
        
        for log_entry in access_logs:
            try:
                log = json.loads(log_entry)
                if log["tenant_id"] == tenant_id and log["repo_id"] == repo_id:
                    timestamp = datetime.fromisoformat(log["timestamp"])
                    hour = timestamp.hour
                    dow = timestamp.weekday()
                    
                    temporal_patterns[log["file_path"]]["hour_distribution"][str(hour)] += 1
                    temporal_patterns[log["file_path"]]["dow_distribution"][str(dow)] += 1
                    
            except (json.JSONDecodeError, KeyError):
                continue
        
        # 转换为百分比分布
        for file_path, pattern_data in temporal_patterns.items():
            total_hour = sum(pattern_data["hour_distribution"].values())
            total_dow = sum(pattern_data["dow_distribution"].values())
            
            if total_hour > 0:
                pattern_data["hour_distribution"] = {
                    k: v / total_hour 
                    for k, v in pattern_data["hour_distribution"].items()
                }
            
            if total_dow > 0:
                pattern_data["dow_distribution"] = {
                    k: v / total_dow 
                    for k, v in pattern_data["dow_distribution"].items()
                }
        
        return dict(temporal_patterns)
    
    async def _cleanup_old_access_logs(self):
        """清理旧的访问日志"""
        cutoff_date = (datetime.utcnow() - timedelta(days=self.prediction_window_days)).isoformat()
        
        # 获取所有日志
        all_logs = await self.redis_client.lrange(self.ACCESS_LOG_KEY, 0, -1)
        
        # 过滤并保留新日志
        valid_logs = []
        for log_entry in all_logs:
            try:
                log = json.loads(log_entry)
                if log["timestamp"] >= cutoff_date:
                    valid_logs.append(log_entry)
            except (json.JSONDecodeError, KeyError):
                continue
        
        # 更新日志列表
        if len(valid_logs) != len(all_logs):
            await self.redis_client.delete(self.ACCESS_LOG_KEY)
            if valid_logs:
                await self.redis_client.lpush(self.ACCESS_LOG_KEY, *valid_logs)
    
    async def _calculate_recent_access_weight(self, patterns: List[AccessPattern]) -> float:
        """计算最近访问权重"""
        now = datetime.utcnow()
        recent_weight = 0.0
        
        for pattern in patterns:
            days_since_last = (now - pattern.last_access).days
            if days_since_last <= 7:  # 最近7天
                recent_weight += 1.0 / (1.0 + days_since_last)
        
        return min(1.5, recent_weight / len(patterns)) if patterns else 1.0
    
    async def _get_all_access_patterns(self) -> List[AccessPattern]:
        """获取所有访问模式（用于训练）"""
        pattern_keys = await self.redis_client.keys(f"{self.PATTERN_KEY_PREFIX}*")
        
        patterns = []
        for key in pattern_keys:
            pattern_data = await self.redis_client.hgetall(key)
            if pattern_data:
                data = {k.decode(): v.decode() for k, v in pattern_data.items()}
                file_path = key.decode().split(":")[-1]
                
                pattern = AccessPattern(
                    file_path=file_path,
                    access_count=int(data.get("access_count", 0)),
                    last_access=datetime.fromisoformat(data.get("last_access")),
                    access_frequency=0.0,
                    co_accessed_files=set(),
                    change_frequency=0.0
                )
                patterns.append(pattern)
        
        return patterns