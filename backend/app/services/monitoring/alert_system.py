"""
智能告警系统 - 支持多种告警规则和通知方式
"""

import asyncio
import time
import json
import logging
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """告警严重程度"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class AlertStatus(Enum):
    """告警状态"""
    ACTIVE = "active"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"
    ACKNOWLEDGED = "acknowledged"


class ComparisonOperator(Enum):
    """比较操作符"""
    GREATER_THAN = "gt"
    GREATER_EQUAL = "gte"
    LESS_THAN = "lt"
    LESS_EQUAL = "lte"
    EQUAL = "eq"
    NOT_EQUAL = "ne"


class NotificationChannel(Enum):
    """通知渠道"""
    EMAIL = "email"
    WEBHOOK = "webhook"
    SLACK = "slack"
    SMS = "sms"
    CONSOLE = "console"


@dataclass
class AlertRule:
    """告警规则"""
    name: str
    metric_name: str
    operator: ComparisonOperator
    threshold: float
    severity: AlertSeverity
    description: str = ""
    tags: Dict[str, str] = field(default_factory=dict)
    evaluation_window: int = 300  # 评估窗口（秒）
    evaluation_count: int = 1     # 连续触发次数
    cooldown_period: int = 600    # 冷却期（秒）
    enabled: bool = True
    
    def evaluate(self, value: float) -> bool:
        """评估告警条件"""
        if not self.enabled:
            return False
        
        if self.operator == ComparisonOperator.GREATER_THAN:
            return value > self.threshold
        elif self.operator == ComparisonOperator.GREATER_EQUAL:
            return value >= self.threshold
        elif self.operator == ComparisonOperator.LESS_THAN:
            return value < self.threshold
        elif self.operator == ComparisonOperator.LESS_EQUAL:
            return value <= self.threshold
        elif self.operator == ComparisonOperator.EQUAL:
            return value == self.threshold
        elif self.operator == ComparisonOperator.NOT_EQUAL:
            return value != self.threshold
        
        return False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'metric_name': self.metric_name,
            'operator': self.operator.value,
            'threshold': self.threshold,
            'severity': self.severity.value,
            'description': self.description,
            'tags': self.tags,
            'evaluation_window': self.evaluation_window,
            'evaluation_count': self.evaluation_count,
            'cooldown_period': self.cooldown_period,
            'enabled': self.enabled
        }


@dataclass
class Alert:
    """告警"""
    id: str
    rule_name: str
    metric_name: str
    current_value: float
    threshold: float
    severity: AlertSeverity
    status: AlertStatus
    message: str
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    resolved_at: Optional[float] = None
    acknowledged_at: Optional[float] = None
    acknowledged_by: Optional[str] = None
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def acknowledge(self, user: str):
        """确认告警"""
        self.status = AlertStatus.ACKNOWLEDGED
        self.acknowledged_at = time.time()
        self.acknowledged_by = user
        self.updated_at = time.time()
    
    def resolve(self):
        """解决告警"""
        self.status = AlertStatus.RESOLVED
        self.resolved_at = time.time()
        self.updated_at = time.time()
    
    def suppress(self):
        """抑制告警"""
        self.status = AlertStatus.SUPPRESSED
        self.updated_at = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'rule_name': self.rule_name,
            'metric_name': self.metric_name,
            'current_value': self.current_value,
            'threshold': self.threshold,
            'severity': self.severity.value,
            'status': self.status.value,
            'message': self.message,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'resolved_at': self.resolved_at,
            'acknowledged_at': self.acknowledged_at,
            'acknowledged_by': self.acknowledged_by,
            'tags': self.tags,
            'metadata': self.metadata
        }


class NotificationProvider(ABC):
    """通知提供者抽象基类"""
    
    @abstractmethod
    async def send_notification(self, alert: Alert, recipients: List[str]) -> bool:
        """发送通知"""
        pass


class EmailNotificationProvider(NotificationProvider):
    """邮件通知提供者"""
    
    def __init__(self, smtp_host: str, smtp_port: int, 
                 username: str, password: str, use_tls: bool = True):
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.use_tls = use_tls
    
    async def send_notification(self, alert: Alert, recipients: List[str]) -> bool:
        """发送邮件通知"""
        try:
            # 创建邮件内容
            msg = MimeMultipart()
            msg['From'] = self.username
            msg['To'] = ', '.join(recipients)
            msg['Subject'] = f"[{alert.severity.value.upper()}] {alert.rule_name}"
            
            # 邮件正文
            body = self._create_email_body(alert)
            msg.attach(MimeText(body, 'html'))
            
            # 发送邮件
            server = smtplib.SMTP(self.smtp_host, self.smtp_port)
            
            if self.use_tls:
                server.starttls()
            
            server.login(self.username, self.password)
            server.send_message(msg)
            server.quit()
            
            logger.info(f"Email notification sent for alert {alert.id}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to send email notification: {e}")
            return False
    
    def _create_email_body(self, alert: Alert) -> str:
        """创建邮件正文"""
        return f"""
        <html>
        <body>
            <h2>告警通知</h2>
            <table border="1" cellpadding="5" cellspacing="0">
                <tr><td><strong>告警ID</strong></td><td>{alert.id}</td></tr>
                <tr><td><strong>规则名称</strong></td><td>{alert.rule_name}</td></tr>
                <tr><td><strong>指标名称</strong></td><td>{alert.metric_name}</td></tr>
                <tr><td><strong>当前值</strong></td><td>{alert.current_value}</td></tr>
                <tr><td><strong>阈值</strong></td><td>{alert.threshold}</td></tr>
                <tr><td><strong>严重程度</strong></td><td>{alert.severity.value}</td></tr>
                <tr><td><strong>状态</strong></td><td>{alert.status.value}</td></tr>
                <tr><td><strong>消息</strong></td><td>{alert.message}</td></tr>
                <tr><td><strong>创建时间</strong></td><td>{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(alert.created_at))}</td></tr>
            </table>
        </body>
        </html>
        """


class WebhookNotificationProvider(NotificationProvider):
    """Webhook通知提供者"""
    
    def __init__(self, webhook_url: str, headers: Optional[Dict[str, str]] = None):
        self.webhook_url = webhook_url
        self.headers = headers or {}
    
    async def send_notification(self, alert: Alert, recipients: List[str]) -> bool:
        """发送Webhook通知"""
        try:
            import aiohttp
            
            payload = {
                'alert': alert.to_dict(),
                'recipients': recipients,
                'timestamp': time.time()
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.webhook_url,
                    json=payload,
                    headers=self.headers,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        logger.info(f"Webhook notification sent for alert {alert.id}")
                        return True
                    else:
                        logger.error(f"Webhook notification failed with status {response.status}")
                        return False
        
        except Exception as e:
            logger.error(f"Failed to send webhook notification: {e}")
            return False


class ConsoleNotificationProvider(NotificationProvider):
    """控制台通知提供者"""
    
    async def send_notification(self, alert: Alert, recipients: List[str]) -> bool:
        """发送控制台通知"""
        try:
            severity_colors = {
                AlertSeverity.CRITICAL: '\033[91m',  # 红色
                AlertSeverity.HIGH: '\033[93m',      # 黄色
                AlertSeverity.MEDIUM: '\033[94m',    # 蓝色
                AlertSeverity.LOW: '\033[92m',       # 绿色
                AlertSeverity.INFO: '\033[96m'       # 青色
            }
            
            color = severity_colors.get(alert.severity, '\033[0m')
            reset_color = '\033[0m'
            
            message = f"""
{color}[ALERT] {alert.severity.value.upper()}{reset_color}
Rule: {alert.rule_name}
Metric: {alert.metric_name}
Current Value: {alert.current_value}
Threshold: {alert.threshold}
Message: {alert.message}
Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(alert.created_at))}
Recipients: {', '.join(recipients)}
{'-' * 50}
            """
            
            print(message)
            logger.info(f"Console notification sent for alert {alert.id}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to send console notification: {e}")
            return False


@dataclass
class NotificationConfig:
    """通知配置"""
    channel: NotificationChannel
    provider: NotificationProvider
    recipients: List[str]
    severity_filter: List[AlertSeverity] = field(default_factory=lambda: list(AlertSeverity))
    enabled: bool = True
    
    def should_notify(self, alert: Alert) -> bool:
        """判断是否应该发送通知"""
        return (self.enabled and 
                alert.severity in self.severity_filter and
                alert.status == AlertStatus.ACTIVE)


class AlertManager:
    """告警管理器"""
    
    def __init__(self):
        self.rules: Dict[str, AlertRule] = {}
        self.alerts: Dict[str, Alert] = {}
        self.notification_configs: List[NotificationConfig] = []
        self.rule_states: Dict[str, Dict[str, Any]] = {}  # 规则状态跟踪
        self.is_running = False
        self.evaluation_task = None
        self.evaluation_interval = 30  # 评估间隔（秒）
        self.metric_collector = None
        self.alert_history: List[Alert] = []
        self.max_history_size = 10000
    
    def set_metric_collector(self, collector):
        """设置指标收集器"""
        self.metric_collector = collector
    
    def add_rule(self, rule: AlertRule):
        """添加告警规则"""
        self.rules[rule.name] = rule
        self.rule_states[rule.name] = {
            'consecutive_triggers': 0,
            'last_trigger_time': 0,
            'last_alert_time': 0
        }
        logger.info(f"Added alert rule: {rule.name}")
    
    def remove_rule(self, rule_name: str) -> bool:
        """移除告警规则"""
        if rule_name in self.rules:
            del self.rules[rule_name]
            if rule_name in self.rule_states:
                del self.rule_states[rule_name]
            logger.info(f"Removed alert rule: {rule_name}")
            return True
        return False
    
    def add_notification_config(self, config: NotificationConfig):
        """添加通知配置"""
        self.notification_configs.append(config)
        logger.info(f"Added notification config for {config.channel.value}")
    
    def remove_notification_config(self, channel: NotificationChannel) -> bool:
        """移除通知配置"""
        for i, config in enumerate(self.notification_configs):
            if config.channel == channel:
                del self.notification_configs[i]
                logger.info(f"Removed notification config for {channel.value}")
                return True
        return False
    
    async def start(self):
        """启动告警管理器"""
        if self.is_running:
            return
        
        self.is_running = True
        self.evaluation_task = asyncio.create_task(self._evaluation_loop())
        logger.info("Alert manager started")
    
    async def stop(self):
        """停止告警管理器"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        if self.evaluation_task:
            self.evaluation_task.cancel()
            try:
                await self.evaluation_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Alert manager stopped")
    
    async def _evaluation_loop(self):
        """评估循环"""
        while self.is_running:
            try:
                await self._evaluate_rules()
                await asyncio.sleep(self.evaluation_interval)
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in alert evaluation loop: {e}")
                await asyncio.sleep(self.evaluation_interval)
    
    async def _evaluate_rules(self):
        """评估所有规则"""
        if not self.metric_collector:
            return
        
        current_time = time.time()
        
        for rule_name, rule in self.rules.items():
            try:
                await self._evaluate_rule(rule, current_time)
            except Exception as e:
                logger.error(f"Error evaluating rule {rule_name}: {e}")
    
    async def _evaluate_rule(self, rule: AlertRule, current_time: float):
        """评估单个规则"""
        if not rule.enabled:
            return
        
        # 获取指标数据
        start_time = current_time - rule.evaluation_window
        metrics = self.metric_collector.get_metrics(
            [rule.metric_name], start_time, current_time, rule.tags
        )
        
        if not metrics:
            return
        
        # 计算当前值（使用最新值或平均值）
        current_value = metrics[-1].value
        
        # 获取规则状态
        state = self.rule_states[rule.name]
        
        # 评估条件
        condition_met = rule.evaluate(current_value)
        
        if condition_met:
            state['consecutive_triggers'] += 1
            state['last_trigger_time'] = current_time
            
            # 检查是否达到触发条件
            if (state['consecutive_triggers'] >= rule.evaluation_count and
                current_time - state['last_alert_time'] >= rule.cooldown_period):
                
                await self._create_alert(rule, current_value, current_time)
                state['last_alert_time'] = current_time
        else:
            # 重置连续触发计数
            state['consecutive_triggers'] = 0
            
            # 检查是否需要解决现有告警
            await self._resolve_alerts_for_rule(rule.name)
    
    async def _create_alert(self, rule: AlertRule, current_value: float, timestamp: float):
        """创建告警"""
        alert_id = f"{rule.name}_{int(timestamp)}"
        
        # 检查是否已存在相同的活跃告警
        existing_alert = self._find_active_alert_for_rule(rule.name)
        if existing_alert:
            # 更新现有告警
            existing_alert.current_value = current_value
            existing_alert.updated_at = timestamp
            return existing_alert
        
        # 创建新告警
        message = f"{rule.description or rule.name}: {rule.metric_name} is {current_value}, threshold is {rule.threshold}"
        
        alert = Alert(
            id=alert_id,
            rule_name=rule.name,
            metric_name=rule.metric_name,
            current_value=current_value,
            threshold=rule.threshold,
            severity=rule.severity,
            status=AlertStatus.ACTIVE,
            message=message,
            created_at=timestamp,
            tags=rule.tags.copy()
        )
        
        self.alerts[alert_id] = alert
        self.alert_history.append(alert)
        
        # 限制历史记录大小
        if len(self.alert_history) > self.max_history_size:
            self.alert_history = self.alert_history[-self.max_history_size:]
        
        # 发送通知
        await self._send_notifications(alert)
        
        logger.warning(f"Alert created: {alert.id} - {alert.message}")
        return alert
    
    async def _resolve_alerts_for_rule(self, rule_name: str):
        """解决规则相关的告警"""
        for alert in self.alerts.values():
            if (alert.rule_name == rule_name and 
                alert.status == AlertStatus.ACTIVE):
                alert.resolve()
                logger.info(f"Alert resolved: {alert.id}")
    
    def _find_active_alert_for_rule(self, rule_name: str) -> Optional[Alert]:
        """查找规则的活跃告警"""
        for alert in self.alerts.values():
            if (alert.rule_name == rule_name and 
                alert.status == AlertStatus.ACTIVE):
                return alert
        return None
    
    async def _send_notifications(self, alert: Alert):
        """发送通知"""
        for config in self.notification_configs:
            if config.should_notify(alert):
                try:
                    success = await config.provider.send_notification(
                        alert, config.recipients
                    )
                    if success:
                        logger.info(f"Notification sent via {config.channel.value} for alert {alert.id}")
                    else:
                        logger.error(f"Failed to send notification via {config.channel.value} for alert {alert.id}")
                
                except Exception as e:
                    logger.error(f"Error sending notification via {config.channel.value}: {e}")
    
    def acknowledge_alert(self, alert_id: str, user: str) -> bool:
        """确认告警"""
        if alert_id in self.alerts:
            self.alerts[alert_id].acknowledge(user)
            logger.info(f"Alert {alert_id} acknowledged by {user}")
            return True
        return False
    
    def resolve_alert(self, alert_id: str) -> bool:
        """解决告警"""
        if alert_id in self.alerts:
            self.alerts[alert_id].resolve()
            logger.info(f"Alert {alert_id} resolved")
            return True
        return False
    
    def suppress_alert(self, alert_id: str) -> bool:
        """抑制告警"""
        if alert_id in self.alerts:
            self.alerts[alert_id].suppress()
            logger.info(f"Alert {alert_id} suppressed")
            return True
        return False
    
    def get_active_alerts(self, severity_filter: Optional[List[AlertSeverity]] = None) -> List[Alert]:
        """获取活跃告警"""
        alerts = [alert for alert in self.alerts.values() 
                 if alert.status == AlertStatus.ACTIVE]
        
        if severity_filter:
            alerts = [alert for alert in alerts 
                     if alert.severity in severity_filter]
        
        return sorted(alerts, key=lambda x: x.created_at, reverse=True)
    
    def get_alert_history(self, limit: int = 100, 
                         severity_filter: Optional[List[AlertSeverity]] = None) -> List[Alert]:
        """获取告警历史"""
        alerts = self.alert_history
        
        if severity_filter:
            alerts = [alert for alert in alerts 
                     if alert.severity in severity_filter]
        
        return sorted(alerts, key=lambda x: x.created_at, reverse=True)[:limit]
    
    def get_alert_statistics(self) -> Dict[str, Any]:
        """获取告警统计"""
        total_alerts = len(self.alert_history)
        active_alerts = len(self.get_active_alerts())
        
        # 按严重程度统计
        severity_stats = {}
        for severity in AlertSeverity:
            count = len([alert for alert in self.alert_history 
                        if alert.severity == severity])
            severity_stats[severity.value] = count
        
        # 按状态统计
        status_stats = {}
        for status in AlertStatus:
            count = len([alert for alert in self.alerts.values() 
                        if alert.status == status])
            status_stats[status.value] = count
        
        return {
            'total_alerts': total_alerts,
            'active_alerts': active_alerts,
            'severity_distribution': severity_stats,
            'status_distribution': status_stats,
            'rules_count': len(self.rules),
            'notification_configs_count': len(self.notification_configs)
        }


# 全局实例
_alert_manager = None


def get_alert_manager() -> AlertManager:
    """获取告警管理器实例"""
    global _alert_manager
    if _alert_manager is None:
        _alert_manager = AlertManager()
    return _alert_manager


def create_default_alert_rules() -> List[AlertRule]:
    """创建默认告警规则"""
    return [
        AlertRule(
            name="high_cpu_usage",
            metric_name="cpu_usage",
            operator=ComparisonOperator.GREATER_THAN,
            threshold=80.0,
            severity=AlertSeverity.HIGH,
            description="CPU使用率过高",
            evaluation_window=300,
            evaluation_count=2,
            cooldown_period=600
        ),
        AlertRule(
            name="high_memory_usage",
            metric_name="memory_usage",
            operator=ComparisonOperator.GREATER_THAN,
            threshold=85.0,
            severity=AlertSeverity.HIGH,
            description="内存使用率过高",
            evaluation_window=300,
            evaluation_count=2,
            cooldown_period=600
        ),
        AlertRule(
            name="high_response_time",
            metric_name="response_time",
            operator=ComparisonOperator.GREATER_THAN,
            threshold=1000.0,  # 1秒
            severity=AlertSeverity.MEDIUM,
            description="响应时间过长",
            evaluation_window=180,
            evaluation_count=3,
            cooldown_period=300
        ),
        AlertRule(
            name="high_error_rate",
            metric_name="error_rate",
            operator=ComparisonOperator.GREATER_THAN,
            threshold=5.0,  # 5%
            severity=AlertSeverity.CRITICAL,
            description="错误率过高",
            evaluation_window=120,
            evaluation_count=1,
            cooldown_period=300
        ),
        AlertRule(
            name="low_cache_hit_rate",
            metric_name="cache_hit_rate",
            operator=ComparisonOperator.LESS_THAN,
            threshold=70.0,  # 70%
            severity=AlertSeverity.MEDIUM,
            description="缓存命中率过低",
            evaluation_window=600,
            evaluation_count=3,
            cooldown_period=900
        )
    ]


async def initialize_alert_system(metric_collector):
    """初始化告警系统"""
    alert_manager = get_alert_manager()
    alert_manager.set_metric_collector(metric_collector)
    
    # 添加默认规则
    for rule in create_default_alert_rules():
        alert_manager.add_rule(rule)
    
    # 添加控制台通知
    console_config = NotificationConfig(
        channel=NotificationChannel.CONSOLE,
        provider=ConsoleNotificationProvider(),
        recipients=["admin"],
        severity_filter=list(AlertSeverity)
    )
    alert_manager.add_notification_config(console_config)
    
    # 启动告警管理器
    await alert_manager.start()
    
    logger.info("Alert system initialized")
    return alert_manager