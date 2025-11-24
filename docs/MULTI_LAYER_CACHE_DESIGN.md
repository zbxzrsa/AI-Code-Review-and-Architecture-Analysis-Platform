# 多层缓存系统设计文档

## 概述

多层缓存系统是增量分析的核心，通过缓存文件哈希、AST 指纹和规则结果，将增量分析场景的执行时间降低 ≥50%，并达到 60% 的缓存命中率。

---

## 架构设计

### 缓存层次

```
┌─────────────────────────────────────────────────────┐
│           应用层（上游：分析请求）                  │
└──────────────────┬──────────────────────────────────┘
                   │
        ┌──────────┴─────────┬───────────────┐
        │                    │               │
    ┌───▼──────┐  ┌──────────▼─┐  ┌─────────▼────┐
    │ L1 本地  │  │  L2 Redis  │  │ L3 数据库    │
    │ LRU 缓存 │  │ 分布式缓存 │  │ 持久化缓存   │
    │ 1-5min  │  │ 5-24h     │  │ 永久保存    │
    └──────────┘  └────────────┘  └──────────────┘
        │              │              │
        └──────────┬───┴──────────┬───┘
                   │              │
        ┌──────────▼────┐  ┌──────▼────────┐
        │ 缓存失效策略  │  │ 缓存热启动    │
        │ 版本 + 配置   │  │ 预加载 + 异步 │
        └───────────────┘  └───────────────┘
```

---

## 详细设计

### 1. 缓存键设计

所有缓存键遵循统一格式，便于故障排查和版本管理：

```
# 文件哈希缓存
FILE_HASH:{repo_id}:{file_path}:{commit_sha} -> sha256_hash

# AST 指纹缓存
AST_FINGERPRINT:{repo_id}:{file_path}:{ast_version} -> fingerprint

# 规则结果缓存
RULE_RESULT:{repo_id}:{commit_sha}:{rule_pack_version}:{file_path} -> findings[]

# 增量分析元数据
INCREMENTAL_META:{repo_id}:{commit_sha} -> {
  affected_files: [],
  dependency_graph: {},
  timestamp: iso8601,
  hit_ratio: 0.65
}
```

### 2. 缓存数据结构

#### 文件哈希缓存

```python
@dataclass
class FileHashCache:
    file_path: str
    commit_sha: str
    content_hash: str  # SHA256
    file_size: int
    created_at: datetime
    ttl_minutes: int = 5  # L1 本地缓存 TTL
    version: str = "1.0"
```

#### AST 指纹缓存

```python
@dataclass
class ASTFingerprintCache:
    file_path: str
    ast_version: str
    fingerprint: str  # 结构哈希（与内容无关）
    ast_depth: int
    ast_node_count: int
    created_at: datetime
    language: str
    parser_version: str
```

#### 规则结果缓存

```python
@dataclass
class RuleResultCache:
    repo_id: str
    commit_sha: str
    rule_pack_version: str
    file_path: str
    findings: List[Finding]
    execution_time_ms: int
    created_at: datetime
    ttl_hours: int = 24
```

### 3. 缓存管理器核心接口

```python
class CacheManager:
    """多层缓存管理器"""

    async def get_file_hash(self, repo_id: str, file_path: str, commit_sha: str) -> Optional[str]:
        """
        获取文件哈希
        优先级: L1 本地 -> L2 Redis -> 计算 + 写入
        """

    async def get_ast_fingerprint(self, file_path: str, content: str) -> Optional[str]:
        """获取 AST 指纹（结构敏感，内容不敏感）"""

    async def get_rule_results(self, repo_id: str, commit_sha: str, rule_pack_version: str, file_path: str) -> Optional[List[Finding]]:
        """获取规则执行结果"""

    async def get_affected_files(self, repo_id: str, prev_commit: str, curr_commit: str) -> List[str]:
        """获取变更文件列表"""

    async def get_dependency_subgraph(self, repo_id: str, files: List[str]) -> Dict:
        """获取依赖子图（用于分析范围扩展）"""

    async def invalidate_by_rule_version(self, repo_id: str, rule_pack_version: str) -> int:
        """按规则包版本失效缓存"""

    async def invalidate_by_config_change(self, config_hash: str) -> int:
        """按配置变更失效缓存"""

    def record_hit_ratio(self, hit: bool) -> None:
        """记录缓存命中率"""

    async def get_hit_ratio(self, window_minutes: int = 60) -> float:
        """获取增量缓存命中率"""
```

### 4. 缓存失效策略

#### 策略 1：规则包版本变更

```python
class RulePackVersionInvalidationStrategy:
    """规则包版本更新时，失效该包相关的所有规则结果缓存"""

    async def on_rule_pack_update(self, rule_pack_id: str, new_version: str):
        # 删除所有 RULE_RESULT:*:*:{new_version}:* 的缓存
        # 发布事件给各租户
        # 记录失效原因与数量
```

#### 策略 2：配置变更失效

```python
class ConfigurationInvalidationStrategy:
    """配置变更时，失效相关缓存"""

    async def on_config_update(self, config_key: str, new_hash: str):
        # 如果是全局配置，失效所有缓存
        # 如果是租户级配置，失效该租户缓存
        # 如果是规则级配置，失效相关规则结果缓存
```

#### 策略 3：时间过期失效

```python
# L1 本地缓存: 5分钟
# L2 Redis 缓存: 24小时
# L3 数据库: 永久保存（有手动清理机制）
```

#### 策略 4：手动清理

```python
class ManualCacheEviction:
    """手动清理过期或冗余缓存"""

    async def cleanup_old_commits(self, days: int = 7) -> int:
        """清理超过N天的提交缓存"""

    async def cleanup_by_pattern(self, pattern: str) -> int:
        """按模式清理缓存"""

    async def compact_database(self) -> int:
        """数据库碎片整理"""
```

---

## 实现细节

### L1 本地缓存 (LRU)

```python
from functools import lru_cache
from collections import OrderedDict

class LocalLRUCache:
    def __init__(self, max_size: int = 10000, ttl_seconds: int = 300):
        self.cache: OrderedDict = OrderedDict()
        self.ttl: Dict[str, float] = {}
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[Any]:
        self._cleanup_expired()
        if key in self.cache:
            self.cache.move_to_end(key)
            self.hits += 1
            return self.cache[key]
        self.misses += 1
        return None

    def put(self, key: str, value: Any, ttl_seconds: Optional[int] = None):
        if key in self.cache:
            self.cache.move_to_end(key)
        elif len(self.cache) >= self.max_size:
            self.cache.popitem(last=False)  # Remove oldest

        self.cache[key] = value
        self.ttl[key] = time.time() + (ttl_seconds or self.ttl_seconds)

    def _cleanup_expired(self):
        now = time.time()
        expired_keys = [k for k, t in self.ttl.items() if t < now]
        for k in expired_keys:
            self.cache.pop(k, None)
            self.ttl.pop(k, None)

    def hit_ratio(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
```

### L2 Redis 缓存

```python
import redis
from redis import RedisCluster

class RedisCache:
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client

    async def get(self, key: str) -> Optional[Any]:
        value = self.redis.get(key)
        return json.loads(value) if value else None

    async def put(self, key: str, value: Any, ttl_seconds: int = 86400):
        self.redis.setex(key, ttl_seconds, json.dumps(value))

    async def delete_by_pattern(self, pattern: str) -> int:
        """删除匹配的所有键"""
        pipe = self.redis.pipeline()
        for key in self.redis.scan_iter(match=pattern):
            pipe.delete(key)
        return pipe.execute()
```

### L3 数据库缓存

```python
from sqlalchemy import create_engine, Column, String, Integer, DateTime, JSON

class CacheRecord(Base):
    __tablename__ = 'cache_records'

    id = Column(Integer, primary_key=True)
    cache_key = Column(String(255), unique=True, index=True)
    cache_type = Column(String(50))  # 'file_hash', 'ast', 'rule_result'
    repo_id = Column(String(100), index=True)
    commit_sha = Column(String(40), index=True)
    value = Column(JSON)
    created_at = Column(DateTime, server_default=func.now())
    expires_at = Column(DateTime)  # NULL 表示永不过期

    __table_args__ = (
        Index('idx_repo_commit_type', 'repo_id', 'commit_sha', 'cache_type'),
    )

class DatabaseCache:
    async def get(self, key: str) -> Optional[Any]:
        record = await session.query(CacheRecord).filter(
            CacheRecord.cache_key == key,
            (CacheRecord.expires_at.is_(None)) | (CacheRecord.expires_at > datetime.now())
        ).first()
        return record.value if record else None

    async def put(self, key: str, value: Any, cache_type: str, repo_id: str, commit_sha: str, expires_at: Optional[datetime] = None):
        record = CacheRecord(
            cache_key=key,
            cache_type=cache_type,
            repo_id=repo_id,
            commit_sha=commit_sha,
            value=value,
            expires_at=expires_at
        )
        await session.merge(record)
        await session.commit()
```

---

## 增量分析优化

### 受影响文件分析

```python
class AffectedFilesAnalyzer:
    """分析提交中的受影响文件"""

    async def get_affected_files(self, repo_id: str, prev_commit: str, curr_commit: str) -> List[str]:
        """
        获取两个提交之间变更的文件
        - 新增文件
        - 删除文件
        - 修改文件（直接变更）
        """
        diff = await git_service.get_diff(repo_id, prev_commit, curr_commit)
        return [f.path for f in diff.files]

    async def get_dependency_subgraph(self, repo_id: str, affected_files: List[str]) -> Dict:
        """
        扩展分析范围：获取依赖于受影响文件的所有文件

        示例：
        - 修改了 utils/helper.py
        - 那么所有 import helper 的文件也需要重新分析
        """
        graph = await dependency_graph_service.get_graph(repo_id)
        subgraph = set(affected_files)

        # 递归收集依赖
        queue = list(affected_files)
        while queue:
            file = queue.pop(0)
            dependents = graph.get(file, [])
            for dep in dependents:
                if dep not in subgraph:
                    subgraph.add(dep)
                    queue.append(dep)

        return {"files": list(subgraph), "count": len(subgraph)}
```

### 优先队列调度

```python
from heapq import heappush, heappop
from datetime import datetime

class PriorityQueueScheduler:
    """使用优先队列调度分析任务，优先处理最近修改的文件"""

    def __init__(self):
        self.queue = []
        self.counter = 0  # 用于 FIFO 排序

    def add_file(self, file_path: str, last_modified_timestamp: int):
        """
        添加文件到优先队列
        优先级: -last_modified (最近修改的优先)
        """
        priority = (-last_modified_timestamp, self.counter)  # 负数表示降序
        heappush(self.queue, (priority, file_path))
        self.counter += 1

    def get_next_file(self) -> Optional[str]:
        """获取下一个需要分析的文件"""
        if self.queue:
            _, file_path = heappop(self.queue)
            return file_path
        return None

    def size(self) -> int:
        return len(self.queue)
```

---

## 监控指标

### 核心指标

```python
class CacheMetrics:
    """缓存性能指标"""

    # Counter: 缓存命中次数
    cache_hits_total = Counter('cache_hits_total', 'Total cache hits', ['cache_level', 'cache_type'])

    # Counter: 缓存未命中次数
    cache_misses_total = Counter('cache_misses_total', 'Total cache misses', ['cache_level', 'cache_type'])

    # Gauge: 缓存容量使用率
    cache_usage_ratio = Gauge('cache_usage_ratio', 'Cache usage ratio', ['cache_level'])

    # Gauge: 当前缓存条目数
    cache_entries_count = Gauge('cache_entries_count', 'Total cache entries', ['cache_level', 'cache_type'])

    # Gauge: 增量缓存命中率
    incremental_hit_ratio = Gauge('incremental_hit_ratio', 'Incremental analysis hit ratio', ['repo_id'])

    # Histogram: 缓存访问延迟
    cache_access_latency = Histogram('cache_access_latency_ms', 'Cache access latency', ['cache_level'])

    # Gauge: 缓存失效事件
    cache_invalidation_events = Counter('cache_invalidation_events_total', 'Cache invalidation events', ['reason'])
```

### 仪表盘指标

```json
{
    "dashboard": {
        "title": "多层缓存监控",
        "panels": [
            {
                "title": "缓存命中率 (全局)",
                "query": "rate(cache_hits_total[5m]) / (rate(cache_hits_total[5m]) + rate(cache_misses_total[5m]))"
            },
            {
                "title": "增量分析缓存命中率",
                "query": "incremental_hit_ratio"
            },
            {
                "title": "分层缓存使用情况",
                "query": "cache_usage_ratio by (cache_level)"
            },
            {
                "title": "缓存访问延迟 P95",
                "query": "histogram_quantile(0.95, cache_access_latency)"
            },
            {
                "title": "缓存失效原因分布",
                "query": "rate(cache_invalidation_events_total[1h]) by (reason)"
            }
        ]
    }
}
```

---

## 测试策略

### 单元测试

```python
@pytest.mark.asyncio
async def test_file_hash_cache_hit():
    """测试文件哈希缓存命中"""
    manager = CacheManager()

    # 第一次获取（cache miss，计算结果）
    hash1 = await manager.get_file_hash("repo1", "file.py", "commit1")
    assert hash1 is not None

    # 第二次获取（cache hit）
    hash2 = await manager.get_file_hash("repo1", "file.py", "commit1")
    assert hash1 == hash2

@pytest.mark.asyncio
async def test_ast_fingerprint_insensitive_to_content():
    """测试 AST 指纹对内容变化不敏感"""
    manager = CacheManager()

    code1 = """
def foo(x):
    return x + 1
"""
    code2 = """
def foo(x):
    return x + 2  # 只改变常数，结构相同
"""

    fp1 = await manager.get_ast_fingerprint("file.py", code1)
    fp2 = await manager.get_ast_fingerprint("file.py", code2)

    # 不同：AST 结构不同（返回值不同）
    assert fp1 != fp2
```

### 集成测试

```python
@pytest.mark.asyncio
async def test_incremental_analysis_with_cache():
    """测试增量分析缓存流程"""
    manager = CacheManager()
    analyzer = IncrementalAnalyzer(manager)

    # 模拟第一次全量扫描
    results1 = await analyzer.analyze_commit("repo1", "commit1")
    time1 = time.time()

    # 模拟第二次增量扫描（只修改了一个文件）
    results2 = await analyzer.analyze_commit("repo1", "commit2")
    time2 = time.time()

    # 验证性能提升
    assert (time1 - time2) / time1 >= 0.5  # 至少快 50%
    assert manager.hit_ratio() >= 0.6  # 缓存命中率 ≥60%
```

---

## 部署与配置

### 环境变量

```bash
# 缓存配置
CACHE_LEVEL_1_ENABLED=true
CACHE_LEVEL_1_MAX_SIZE=10000
CACHE_LEVEL_1_TTL_MINUTES=5

CACHE_LEVEL_2_ENABLED=true
CACHE_LEVEL_2_REDIS_URL=redis://redis:6379/0
CACHE_LEVEL_2_TTL_HOURS=24

CACHE_LEVEL_3_ENABLED=true
CACHE_LEVEL_3_DB_URL=postgresql://postgres:password@postgres:5432/cache

# 监控
PROMETHEUS_ENABLED=true
CACHE_METRICS_EXPORT_INTERVAL_SECONDS=60
```

### Docker Compose 配置

```yaml
version: "3.8"
services:
    redis:
        image: redis:7-alpine
        ports:
            - "6379:6379"
        volumes:
            - redis_data:/data

    postgres-cache:
        image: postgres:15
        environment:
            POSTGRES_DB: cache
            POSTGRES_PASSWORD: password
        ports:
            - "5433:5432"
        volumes:
            - postgres_cache_data:/var/lib/postgresql/data

volumes:
    redis_data:
    postgres_cache_data:
```

---

## 最佳实践

1. **缓存预热**: 在启动时，从数据库加载热数据到 Redis
2. **缓存穿透防护**: 使用 bloom filter 检测不存在的键
3. **缓存雪崩防护**: 使用随机 TTL，防止批量过期
4. **缓存击穿防护**: 使用互斥锁，防止并发计算
5. **定期清理**: 定期清理过期缓存，防止存储溢出
6. **观测与告警**: 监控缓存命中率、延迟、容量等指标

---

## 迁移计划

### Phase 1: 本地缓存 (Week 1)

-   [ ] 实现 LocalLRUCache
-   [ ] 集成到分析流程
-   [ ] 本地测试验证

### Phase 2: Redis 缓存 (Week 2)

-   [ ] Redis 连接与池化
-   [ ] RedisCache 实现
-   [ ] 集群部署测试

### Phase 3: 数据库缓存 (Week 2)

-   [ ] 数据库 schema 设计
-   [ ] DatabaseCache 实现
-   [ ] 迁移策略与脚本

### Phase 4: 监控与优化 (Week 3)

-   [ ] Prometheus 指标接入
-   [ ] 监控面板开发
-   [ ] 性能基准测试

---

**文档版本**: 1.0
**最后更新**: 2025-11-20
