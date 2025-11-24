"""
自动扩缩容系统 - 基于指标的动态扩展
"""

import asyncio
import time
import logging
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import statistics
import json

logger = logging.getLogger(__name__)


class ScalingDirection(Enum):
    """扩缩容方向"""
    UP = "up"
    DOWN = "down"
    NONE = "none"


class ScalingPolicy(Enum):
    """扩缩容策略"""
    TARGET_TRACKING = "target_tracking"  # 目标跟踪
    STEP_SCALING = "step_scaling"        # 阶梯扩缩容
    SIMPLE_SCALING = "simple_scaling"    # 简单扩缩容
    PREDICTIVE = "predictive"            # 预测性扩缩容


class MetricType(Enum):
    """指标类型"""
    CPU_UTILIZATION = "cpu_utilization"
    MEMORY_UTILIZATION = "memory_utilization"
    REQUEST_COUNT = "request_count"
    RESPONSE_TIME = "response_time"
    ERROR_RATE = "error_rate"
    QUEUE_LENGTH = "queue_length"
    CUSTOM = "custom"


@dataclass
class MetricData:
    """指标数据"""
    metric_type: MetricType
    value: float
    timestamp: float = field(default_factory=time.time)
    unit: str = ""
    tags: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'metric_type': self.metric_type.value,
            'value': self.value,
            'timestamp': self.timestamp,
            'unit': self.unit,
            'tags': self.tags
        }


@dataclass
class ScalingRule:
    """扩缩容规则"""
    name: str
    metric_type: MetricType
    threshold_up: float
    threshold_down: float
    scale_up_adjustment: int
    scale_down_adjustment: int
    cooldown_period: int = 300  # 冷却期（秒）
    evaluation_periods: int = 2  # 评估周期数
    datapoints_to_alarm: int = 2  # 触发告警的数据点数
    enabled: bool = True
    
    def should_scale_up(self, values: List[float]) -> bool:
        """判断是否应该扩容"""
        if not self.enabled or len(values) < self.datapoints_to_alarm:
            return False
        
        # 检查最近的数据点是否超过阈值
        recent_values = values[-self.datapoints_to_alarm:]
        return all(v > self.threshold_up for v in recent_values)
    
    def should_scale_down(self, values: List[float]) -> bool:
        """判断是否应该缩容"""
        if not self.enabled or len(values) < self.datapoints_to_alarm:
            return False
        
        # 检查最近的数据点是否低于阈值
        recent_values = values[-self.datapoints_to_alarm:]
        return all(v < self.threshold_down for v in recent_values)


@dataclass
class ScalingAction:
    """扩缩容动作"""
    service_name: str
    direction: ScalingDirection
    adjustment: int
    reason: str
    timestamp: float = field(default_factory=time.time)
    rule_name: str = ""
    current_capacity: int = 0
    target_capacity: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'service_name': self.service_name,
            'direction': self.direction.value,
            'adjustment': self.adjustment,
            'reason': self.reason,
            'timestamp': self.timestamp,
            'rule_name': self.rule_name,
            'current_capacity': self.current_capacity,
            'target_capacity': self.target_capacity
        }


class MetricCollector(ABC):
    """指标收集器抽象基类"""
    
    @abstractmethod
    async def collect_metrics(self, service_name: str) -> List[MetricData]:
        """收集指标"""
        pass


class SystemMetricCollector(MetricCollector):
    """系统指标收集器"""
    
    def __init__(self):
        self.service_metrics: Dict[str, List[MetricData]] = {}
    
    async def collect_metrics(self, service_name: str) -> List[MetricData]:
        """收集系统指标"""
        metrics = []
        
        try:
            # 模拟收集CPU使用率
            import random
            cpu_usage = random.uniform(20, 90)
            metrics.append(MetricData(
                metric_type=MetricType.CPU_UTILIZATION,
                value=cpu_usage,
                unit="percent"
            ))
            
            # 模拟收集内存使用率
            memory_usage = random.uniform(30, 85)
            metrics.append(MetricData(
                metric_type=MetricType.MEMORY_UTILIZATION,
                value=memory_usage,
                unit="percent"
            ))
            
            # 模拟收集请求数量
            request_count = random.randint(50, 500)
            metrics.append(MetricData(
                metric_type=MetricType.REQUEST_COUNT,
                value=request_count,
                unit="count"
            ))
            
            # 模拟收集响应时间
            response_time = random.uniform(50, 300)
            metrics.append(MetricData(
                metric_type=MetricType.RESPONSE_TIME,
                value=response_time,
                unit="ms"
            ))
            
            # 存储历史数据
            if service_name not in self.service_metrics:
                self.service_metrics[service_name] = []
            
            self.service_metrics[service_name].extend(metrics)
            
            # 保留最近1000个数据点
            if len(self.service_metrics[service_name]) > 1000:
                self.service_metrics[service_name] = self.service_metrics[service_name][-1000:]
        
        except Exception as e:
            logger.error(f"Error collecting metrics for {service_name}: {e}")
        
        return metrics
    
    def get_metric_history(self, service_name: str, 
                          metric_type: MetricType, 
                          duration: int = 300) -> List[float]:
        """获取指标历史数据"""
        if service_name not in self.service_metrics:
            return []
        
        current_time = time.time()
        cutoff_time = current_time - duration
        
        values = []
        for metric in self.service_metrics[service_name]:
            if (metric.metric_type == metric_type and 
                metric.timestamp >= cutoff_time):
                values.append(metric.value)
        
        return values


class ServiceScaler(ABC):
    """服务扩缩容器抽象基类"""
    
    @abstractmethod
    async def scale_service(self, service_name: str, target_capacity: int) -> bool:
        """扩缩容服务"""
        pass
    
    @abstractmethod
    async def get_current_capacity(self, service_name: str) -> int:
        """获取当前容量"""
        pass


class MockServiceScaler(ServiceScaler):
    """模拟服务扩缩容器"""
    
    def __init__(self):
        self.service_capacities: Dict[str, int] = {}
        self.min_capacity = 1
        self.max_capacity = 10
    
    async def scale_service(self, service_name: str, target_capacity: int) -> bool:
        """扩缩容服务"""
        try:
            # 限制容量范围
            target_capacity = max(self.min_capacity, 
                                min(self.max_capacity, target_capacity))
            
            current_capacity = self.service_capacities.get(service_name, 1)
            
            if target_capacity != current_capacity:
                logger.info(f"Scaling {service_name} from {current_capacity} to {target_capacity}")
                
                # 模拟扩缩容延迟
                await asyncio.sleep(0.1)
                
                self.service_capacities[service_name] = target_capacity
                return True
            
            return False
        
        except Exception as e:
            logger.error(f"Error scaling service {service_name}: {e}")
            return False
    
    async def get_current_capacity(self, service_name: str) -> int:
        """获取当前容量"""
        return self.service_capacities.get(service_name, 1)
    
    def set_capacity_limits(self, min_capacity: int, max_capacity: int):
        """设置容量限制"""
        self.min_capacity = min_capacity
        self.max_capacity = max_capacity


class AutoScaler:
    """自动扩缩容器"""
    
    def __init__(self, metric_collector: MetricCollector, 
                 service_scaler: ServiceScaler):
        self.metric_collector = metric_collector
        self.service_scaler = service_scaler
        self.scaling_rules: Dict[str, List[ScalingRule]] = {}
        self.scaling_history: List[ScalingAction] = []
        self.last_scaling_time: Dict[str, float] = {}
        self.is_running = False
        self.scaling_task = None
        self.evaluation_interval = 60  # 评估间隔（秒）
        self.event_handlers: List[Callable] = []
    
    def add_scaling_rule(self, service_name: str, rule: ScalingRule):
        """添加扩缩容规则"""
        if service_name not in self.scaling_rules:
            self.scaling_rules[service_name] = []
        
        self.scaling_rules[service_name].append(rule)
        logger.info(f"Added scaling rule '{rule.name}' for service '{service_name}'")
    
    def remove_scaling_rule(self, service_name: str, rule_name: str) -> bool:
        """移除扩缩容规则"""
        if service_name not in self.scaling_rules:
            return False
        
        rules = self.scaling_rules[service_name]
        for i, rule in enumerate(rules):
            if rule.name == rule_name:
                del rules[i]
                logger.info(f"Removed scaling rule '{rule_name}' for service '{service_name}'")
                return True
        
        return False
    
    def add_event_handler(self, handler: Callable):
        """添加事件处理器"""
        self.event_handlers.append(handler)
    
    async def start(self):
        """启动自动扩缩容"""
        if self.is_running:
            return
        
        self.is_running = True
        self.scaling_task = asyncio.create_task(self._scaling_loop())
        logger.info("Auto scaler started")
    
    async def stop(self):
        """停止自动扩缩容"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        if self.scaling_task:
            self.scaling_task.cancel()
            try:
                await self.scaling_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Auto scaler stopped")
    
    async def _scaling_loop(self):
        """扩缩容循环"""
        while self.is_running:
            try:
                await self._evaluate_scaling()
                await asyncio.sleep(self.evaluation_interval)
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in scaling loop: {e}")
                await asyncio.sleep(self.evaluation_interval)
    
    async def _evaluate_scaling(self):
        """评估扩缩容"""
        for service_name, rules in self.scaling_rules.items():
            try:
                await self._evaluate_service_scaling(service_name, rules)
            except Exception as e:
                logger.error(f"Error evaluating scaling for {service_name}: {e}")
    
    async def _evaluate_service_scaling(self, service_name: str, rules: List[ScalingRule]):
        """评估单个服务的扩缩容"""
        # 收集指标
        metrics = await self.metric_collector.collect_metrics(service_name)
        
        if not metrics:
            return
        
        # 按规则评估
        for rule in rules:
            if not rule.enabled:
                continue
            
            # 检查冷却期
            last_scaling = self.last_scaling_time.get(f"{service_name}_{rule.name}", 0)
            if time.time() - last_scaling < rule.cooldown_period:
                continue
            
            # 获取指标历史数据
            if hasattr(self.metric_collector, 'get_metric_history'):
                metric_values = self.metric_collector.get_metric_history(
                    service_name, rule.metric_type, rule.evaluation_periods * 60
                )
            else:
                # 从当前指标中获取值
                metric_values = [
                    m.value for m in metrics 
                    if m.metric_type == rule.metric_type
                ]
            
            if not metric_values:
                continue
            
            # 判断是否需要扩缩容
            current_capacity = await self.service_scaler.get_current_capacity(service_name)
            
            if rule.should_scale_up(metric_values):
                await self._execute_scaling(
                    service_name, rule, ScalingDirection.UP, 
                    current_capacity, metric_values
                )
            elif rule.should_scale_down(metric_values):
                await self._execute_scaling(
                    service_name, rule, ScalingDirection.DOWN, 
                    current_capacity, metric_values
                )
    
    async def _execute_scaling(self, service_name: str, rule: ScalingRule, 
                             direction: ScalingDirection, current_capacity: int,
                             metric_values: List[float]):
        """执行扩缩容"""
        if direction == ScalingDirection.UP:
            adjustment = rule.scale_up_adjustment
            reason = f"Metric {rule.metric_type.value} above threshold {rule.threshold_up}"
        else:
            adjustment = -rule.scale_down_adjustment
            reason = f"Metric {rule.metric_type.value} below threshold {rule.threshold_down}"
        
        target_capacity = current_capacity + adjustment
        
        # 执行扩缩容
        success = await self.service_scaler.scale_service(service_name, target_capacity)
        
        if success:
            # 记录扩缩容动作
            action = ScalingAction(
                service_name=service_name,
                direction=direction,
                adjustment=abs(adjustment),
                reason=reason,
                rule_name=rule.name,
                current_capacity=current_capacity,
                target_capacity=target_capacity
            )
            
            self.scaling_history.append(action)
            self.last_scaling_time[f"{service_name}_{rule.name}"] = time.time()
            
            # 限制历史记录数量
            if len(self.scaling_history) > 1000:
                self.scaling_history = self.scaling_history[-1000:]
            
            # 触发事件处理器
            await self._trigger_event_handlers(action)
            
            logger.info(f"Scaling executed: {action.to_dict()}")
    
    async def _trigger_event_handlers(self, action: ScalingAction):
        """触发事件处理器"""
        for handler in self.event_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(action)
                else:
                    handler(action)
            except Exception as e:
                logger.error(f"Error in event handler: {e}")
    
    def get_scaling_history(self, service_name: Optional[str] = None, 
                          limit: int = 100) -> List[Dict[str, Any]]:
        """获取扩缩容历史"""
        history = self.scaling_history
        
        if service_name:
            history = [a for a in history if a.service_name == service_name]
        
        # 按时间倒序排列
        history = sorted(history, key=lambda x: x.timestamp, reverse=True)
        
        return [action.to_dict() for action in history[:limit]]
    
    def get_service_rules(self, service_name: str) -> List[Dict[str, Any]]:
        """获取服务的扩缩容规则"""
        rules = self.scaling_rules.get(service_name, [])
        
        return [
            {
                'name': rule.name,
                'metric_type': rule.metric_type.value,
                'threshold_up': rule.threshold_up,
                'threshold_down': rule.threshold_down,
                'scale_up_adjustment': rule.scale_up_adjustment,
                'scale_down_adjustment': rule.scale_down_adjustment,
                'cooldown_period': rule.cooldown_period,
                'evaluation_periods': rule.evaluation_periods,
                'datapoints_to_alarm': rule.datapoints_to_alarm,
                'enabled': rule.enabled
            }
            for rule in rules
        ]
    
    async def get_service_status(self, service_name: str) -> Dict[str, Any]:
        """获取服务状态"""
        current_capacity = await self.service_scaler.get_current_capacity(service_name)
        
        # 获取最近的指标
        recent_metrics = await self.metric_collector.collect_metrics(service_name)
        
        # 获取最近的扩缩容动作
        recent_actions = [
            a for a in self.scaling_history[-10:] 
            if a.service_name == service_name
        ]
        
        return {
            'service_name': service_name,
            'current_capacity': current_capacity,
            'recent_metrics': [m.to_dict() for m in recent_metrics],
            'recent_actions': [a.to_dict() for a in recent_actions],
            'rules_count': len(self.scaling_rules.get(service_name, [])),
            'last_scaling_time': self.last_scaling_time.get(service_name, 0)
        }


class PredictiveScaler:
    """预测性扩缩容器"""
    
    def __init__(self, auto_scaler: AutoScaler):
        self.auto_scaler = auto_scaler
        self.prediction_window = 300  # 预测窗口（秒）
        self.history_window = 3600   # 历史数据窗口（秒）
    
    async def predict_scaling_needs(self, service_name: str) -> List[Dict[str, Any]]:
        """预测扩缩容需求"""
        predictions = []
        
        try:
            # 获取历史指标数据
            if hasattr(self.auto_scaler.metric_collector, 'get_metric_history'):
                for metric_type in [MetricType.CPU_UTILIZATION, MetricType.MEMORY_UTILIZATION]:
                    history = self.auto_scaler.metric_collector.get_metric_history(
                        service_name, metric_type, self.history_window
                    )
                    
                    if len(history) >= 10:  # 需要足够的历史数据
                        predicted_value = self._simple_linear_prediction(history)
                        
                        predictions.append({
                            'metric_type': metric_type.value,
                            'current_value': history[-1] if history else 0,
                            'predicted_value': predicted_value,
                            'prediction_time': time.time() + self.prediction_window,
                            'confidence': self._calculate_confidence(history)
                        })
        
        except Exception as e:
            logger.error(f"Error predicting scaling needs for {service_name}: {e}")
        
        return predictions
    
    def _simple_linear_prediction(self, values: List[float]) -> float:
        """简单线性预测"""
        if len(values) < 2:
            return values[0] if values else 0
        
        # 使用最近的数据点进行线性回归
        recent_values = values[-10:]  # 使用最近10个数据点
        
        n = len(recent_values)
        x_values = list(range(n))
        
        # 计算线性回归参数
        x_mean = statistics.mean(x_values)
        y_mean = statistics.mean(recent_values)
        
        numerator = sum((x - x_mean) * (y - y_mean) 
                       for x, y in zip(x_values, recent_values))
        denominator = sum((x - x_mean) ** 2 for x in x_values)
        
        if denominator == 0:
            return recent_values[-1]
        
        slope = numerator / denominator
        intercept = y_mean - slope * x_mean
        
        # 预测下一个值
        next_x = n
        predicted_value = slope * next_x + intercept
        
        return max(0, predicted_value)  # 确保预测值非负
    
    def _calculate_confidence(self, values: List[float]) -> float:
        """计算预测置信度"""
        if len(values) < 3:
            return 0.5
        
        # 基于数据的稳定性计算置信度
        variance = statistics.variance(values[-10:])  # 使用最近10个数据点
        mean_value = statistics.mean(values[-10:])
        
        if mean_value == 0:
            return 0.5
        
        # 变异系数越小，置信度越高
        cv = (variance ** 0.5) / mean_value
        confidence = max(0.1, min(0.9, 1 - cv))
        
        return confidence


# 事件处理器示例
async def scaling_event_handler(action: ScalingAction):
    """扩缩容事件处理器"""
    logger.info(f"Scaling event: {action.service_name} scaled {action.direction.value} "
               f"by {action.adjustment} instances. Reason: {action.reason}")


def scaling_alert_handler(action: ScalingAction):
    """扩缩容告警处理器"""
    if action.direction == ScalingDirection.UP:
        logger.warning(f"Service {action.service_name} scaled up due to high load")
    elif action.direction == ScalingDirection.DOWN:
        logger.info(f"Service {action.service_name} scaled down due to low load")


# 工厂函数
async def create_auto_scaler(metric_collector: Optional[MetricCollector] = None,
                           service_scaler: Optional[ServiceScaler] = None) -> AutoScaler:
    """创建自动扩缩容器"""
    if metric_collector is None:
        metric_collector = SystemMetricCollector()
    
    if service_scaler is None:
        service_scaler = MockServiceScaler()
    
    auto_scaler = AutoScaler(metric_collector, service_scaler)
    
    # 添加默认事件处理器
    auto_scaler.add_event_handler(scaling_event_handler)
    auto_scaler.add_event_handler(scaling_alert_handler)
    
    return auto_scaler


def create_default_scaling_rules() -> List[ScalingRule]:
    """创建默认扩缩容规则"""
    return [
        ScalingRule(
            name="cpu_scaling",
            metric_type=MetricType.CPU_UTILIZATION,
            threshold_up=70.0,
            threshold_down=30.0,
            scale_up_adjustment=1,
            scale_down_adjustment=1,
            cooldown_period=300,
            evaluation_periods=2,
            datapoints_to_alarm=2
        ),
        ScalingRule(
            name="memory_scaling",
            metric_type=MetricType.MEMORY_UTILIZATION,
            threshold_up=80.0,
            threshold_down=40.0,
            scale_up_adjustment=1,
            scale_down_adjustment=1,
            cooldown_period=300,
            evaluation_periods=2,
            datapoints_to_alarm=2
        ),
        ScalingRule(
            name="response_time_scaling",
            metric_type=MetricType.RESPONSE_TIME,
            threshold_up=200.0,  # 200ms
            threshold_down=50.0,   # 50ms
            scale_up_adjustment=2,
            scale_down_adjustment=1,
            cooldown_period=180,
            evaluation_periods=1,
            datapoints_to_alarm=1
        )
    ]