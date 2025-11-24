"""
实时监控仪表板 - 提供可视化的监控界面
"""

import asyncio
import json
import time
import logging
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime, timedelta
import statistics

logger = logging.getLogger(__name__)


class ChartType(Enum):
    """图表类型"""
    LINE = "line"
    BAR = "bar"
    PIE = "pie"
    GAUGE = "gauge"
    HEATMAP = "heatmap"
    SCATTER = "scatter"
    AREA = "area"


class TimeRange(Enum):
    """时间范围"""
    LAST_5_MINUTES = "5m"
    LAST_15_MINUTES = "15m"
    LAST_30_MINUTES = "30m"
    LAST_1_HOUR = "1h"
    LAST_6_HOURS = "6h"
    LAST_24_HOURS = "24h"
    LAST_7_DAYS = "7d"
    LAST_30_DAYS = "30d"


class RefreshInterval(Enum):
    """刷新间隔"""
    REAL_TIME = 1      # 1秒
    FAST = 5           # 5秒
    NORMAL = 30        # 30秒
    SLOW = 60          # 1分钟
    VERY_SLOW = 300    # 5分钟


@dataclass
class ChartData:
    """图表数据"""
    labels: List[str]
    datasets: List[Dict[str, Any]]
    title: str = ""
    subtitle: str = ""
    unit: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Widget:
    """仪表板组件"""
    id: str
    title: str
    chart_type: ChartType
    metric_names: List[str]
    time_range: TimeRange
    refresh_interval: RefreshInterval
    position: Dict[str, int]  # x, y, width, height
    config: Dict[str, Any] = field(default_factory=dict)
    filters: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'title': self.title,
            'chart_type': self.chart_type.value,
            'metric_names': self.metric_names,
            'time_range': self.time_range.value,
            'refresh_interval': self.refresh_interval.value,
            'position': self.position,
            'config': self.config,
            'filters': self.filters,
            'enabled': self.enabled
        }


@dataclass
class Dashboard:
    """仪表板"""
    id: str
    name: str
    description: str
    widgets: List[Widget]
    layout: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    
    def add_widget(self, widget: Widget):
        """添加组件"""
        self.widgets.append(widget)
        self.updated_at = time.time()
    
    def remove_widget(self, widget_id: str) -> bool:
        """移除组件"""
        for i, widget in enumerate(self.widgets):
            if widget.id == widget_id:
                del self.widgets[i]
                self.updated_at = time.time()
                return True
        return False
    
    def get_widget(self, widget_id: str) -> Optional[Widget]:
        """获取组件"""
        for widget in self.widgets:
            if widget.id == widget_id:
                return widget
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'widgets': [widget.to_dict() for widget in self.widgets],
            'layout': self.layout,
            'tags': self.tags,
            'created_at': self.created_at,
            'updated_at': self.updated_at
        }


class DataProcessor:
    """数据处理器"""
    
    @staticmethod
    def get_time_range_seconds(time_range: TimeRange) -> int:
        """获取时间范围的秒数"""
        mapping = {
            TimeRange.LAST_5_MINUTES: 300,
            TimeRange.LAST_15_MINUTES: 900,
            TimeRange.LAST_30_MINUTES: 1800,
            TimeRange.LAST_1_HOUR: 3600,
            TimeRange.LAST_6_HOURS: 21600,
            TimeRange.LAST_24_HOURS: 86400,
            TimeRange.LAST_7_DAYS: 604800,
            TimeRange.LAST_30_DAYS: 2592000
        }
        return mapping.get(time_range, 3600)
    
    @staticmethod
    def format_timestamp(timestamp: float, time_range: TimeRange) -> str:
        """格式化时间戳"""
        dt = datetime.fromtimestamp(timestamp)
        
        if time_range in [TimeRange.LAST_5_MINUTES, TimeRange.LAST_15_MINUTES, TimeRange.LAST_30_MINUTES]:
            return dt.strftime('%H:%M:%S')
        elif time_range in [TimeRange.LAST_1_HOUR, TimeRange.LAST_6_HOURS]:
            return dt.strftime('%H:%M')
        elif time_range == TimeRange.LAST_24_HOURS:
            return dt.strftime('%m-%d %H:%M')
        else:
            return dt.strftime('%m-%d')
    
    @staticmethod
    def aggregate_data(data: List[Dict[str, Any]], interval_seconds: int) -> List[Dict[str, Any]]:
        """聚合数据"""
        if not data:
            return []
        
        # 按时间间隔分组
        groups = {}
        for point in data:
            timestamp = point['timestamp']
            group_key = int(timestamp // interval_seconds) * interval_seconds
            
            if group_key not in groups:
                groups[group_key] = []
            groups[group_key].append(point)
        
        # 聚合每个组
        result = []
        for group_timestamp, group_data in sorted(groups.items()):
            values = [point['value'] for point in group_data]
            
            aggregated = {
                'timestamp': group_timestamp,
                'value': statistics.mean(values),
                'min': min(values),
                'max': max(values),
                'count': len(values)
            }
            result.append(aggregated)
        
        return result
    
    @staticmethod
    def create_line_chart_data(metrics: List[Dict[str, Any]], 
                              time_range: TimeRange,
                              title: str = "",
                              unit: str = "") -> ChartData:
        """创建折线图数据"""
        if not metrics:
            return ChartData(labels=[], datasets=[], title=title, unit=unit)
        
        # 按指标名称分组
        metric_groups = {}
        for metric in metrics:
            name = metric.get('name', 'Unknown')
            if name not in metric_groups:
                metric_groups[name] = []
            metric_groups[name].append(metric)
        
        # 生成时间标签
        labels = []
        datasets = []
        
        for metric_name, metric_data in metric_groups.items():
            data_points = []
            time_labels = []
            
            for point in sorted(metric_data, key=lambda x: x['timestamp']):
                time_labels.append(DataProcessor.format_timestamp(point['timestamp'], time_range))
                data_points.append(point['value'])
            
            if not labels:
                labels = time_labels
            
            datasets.append({
                'label': metric_name,
                'data': data_points,
                'borderColor': DataProcessor._get_color(len(datasets)),
                'backgroundColor': DataProcessor._get_color(len(datasets), alpha=0.2),
                'fill': False,
                'tension': 0.1
            })
        
        return ChartData(
            labels=labels,
            datasets=datasets,
            title=title,
            unit=unit
        )
    
    @staticmethod
    def create_bar_chart_data(metrics: List[Dict[str, Any]], 
                             title: str = "",
                             unit: str = "") -> ChartData:
        """创建柱状图数据"""
        if not metrics:
            return ChartData(labels=[], datasets=[], title=title, unit=unit)
        
        # 按指标名称聚合
        metric_aggregates = {}
        for metric in metrics:
            name = metric.get('name', 'Unknown')
            if name not in metric_aggregates:
                metric_aggregates[name] = []
            metric_aggregates[name].append(metric['value'])
        
        labels = list(metric_aggregates.keys())
        data = [statistics.mean(values) for values in metric_aggregates.values()]
        
        datasets = [{
            'label': title or 'Metrics',
            'data': data,
            'backgroundColor': [DataProcessor._get_color(i, alpha=0.7) for i in range(len(data))],
            'borderColor': [DataProcessor._get_color(i) for i in range(len(data))],
            'borderWidth': 1
        }]
        
        return ChartData(
            labels=labels,
            datasets=datasets,
            title=title,
            unit=unit
        )
    
    @staticmethod
    def create_pie_chart_data(metrics: List[Dict[str, Any]], 
                             title: str = "",
                             unit: str = "") -> ChartData:
        """创建饼图数据"""
        if not metrics:
            return ChartData(labels=[], datasets=[], title=title, unit=unit)
        
        # 按指标名称聚合
        metric_aggregates = {}
        for metric in metrics:
            name = metric.get('name', 'Unknown')
            if name not in metric_aggregates:
                metric_aggregates[name] = 0
            metric_aggregates[name] += metric['value']
        
        labels = list(metric_aggregates.keys())
        data = list(metric_aggregates.values())
        
        datasets = [{
            'data': data,
            'backgroundColor': [DataProcessor._get_color(i, alpha=0.7) for i in range(len(data))],
            'borderColor': [DataProcessor._get_color(i) for i in range(len(data))],
            'borderWidth': 1
        }]
        
        return ChartData(
            labels=labels,
            datasets=datasets,
            title=title,
            unit=unit
        )
    
    @staticmethod
    def create_gauge_chart_data(metrics: List[Dict[str, Any]], 
                               max_value: float = 100,
                               title: str = "",
                               unit: str = "") -> ChartData:
        """创建仪表盘数据"""
        if not metrics:
            return ChartData(labels=[], datasets=[], title=title, unit=unit)
        
        # 使用最新值
        latest_metric = max(metrics, key=lambda x: x['timestamp'])
        current_value = latest_metric['value']
        
        # 计算百分比
        percentage = min((current_value / max_value) * 100, 100)
        
        datasets = [{
            'data': [percentage, 100 - percentage],
            'backgroundColor': [
                DataProcessor._get_gauge_color(percentage),
                '#e0e0e0'
            ],
            'borderWidth': 0,
            'cutout': '70%'
        }]
        
        return ChartData(
            labels=['Current', 'Remaining'],
            datasets=datasets,
            title=title,
            subtitle=f"{current_value:.2f} {unit}",
            unit=unit
        )
    
    @staticmethod
    def _get_color(index: int, alpha: float = 1.0) -> str:
        """获取颜色"""
        colors = [
            '#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0',
            '#9966FF', '#FF9F40', '#FF6384', '#C9CBCF'
        ]
        base_color = colors[index % len(colors)]
        
        if alpha < 1.0:
            # 转换为RGBA
            hex_color = base_color.lstrip('#')
            rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
            return f'rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {alpha})'
        
        return base_color
    
    @staticmethod
    def _get_gauge_color(percentage: float) -> str:
        """获取仪表盘颜色"""
        if percentage >= 90:
            return '#FF4444'  # 红色
        elif percentage >= 70:
            return '#FFA500'  # 橙色
        elif percentage >= 50:
            return '#FFFF00'  # 黄色
        else:
            return '#00FF00'  # 绿色


class DashboardManager:
    """仪表板管理器"""
    
    def __init__(self):
        self.dashboards: Dict[str, Dashboard] = {}
        self.metric_collector = None
        self.alert_manager = None
        self.data_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_ttl = 30  # 缓存TTL（秒）
        self.is_running = False
        self.update_task = None
    
    def set_metric_collector(self, collector):
        """设置指标收集器"""
        self.metric_collector = collector
    
    def set_alert_manager(self, alert_manager):
        """设置告警管理器"""
        self.alert_manager = alert_manager
    
    def create_dashboard(self, dashboard_id: str, name: str, description: str = "") -> Dashboard:
        """创建仪表板"""
        dashboard = Dashboard(
            id=dashboard_id,
            name=name,
            description=description,
            widgets=[]
        )
        
        self.dashboards[dashboard_id] = dashboard
        logger.info(f"Created dashboard: {dashboard_id}")
        return dashboard
    
    def delete_dashboard(self, dashboard_id: str) -> bool:
        """删除仪表板"""
        if dashboard_id in self.dashboards:
            del self.dashboards[dashboard_id]
            logger.info(f"Deleted dashboard: {dashboard_id}")
            return True
        return False
    
    def get_dashboard(self, dashboard_id: str) -> Optional[Dashboard]:
        """获取仪表板"""
        return self.dashboards.get(dashboard_id)
    
    def list_dashboards(self) -> List[Dashboard]:
        """列出所有仪表板"""
        return list(self.dashboards.values())
    
    def add_widget_to_dashboard(self, dashboard_id: str, widget: Widget) -> bool:
        """添加组件到仪表板"""
        dashboard = self.get_dashboard(dashboard_id)
        if dashboard:
            dashboard.add_widget(widget)
            return True
        return False
    
    def remove_widget_from_dashboard(self, dashboard_id: str, widget_id: str) -> bool:
        """从仪表板移除组件"""
        dashboard = self.get_dashboard(dashboard_id)
        if dashboard:
            return dashboard.remove_widget(widget_id)
        return False
    
    async def get_widget_data(self, dashboard_id: str, widget_id: str) -> Optional[ChartData]:
        """获取组件数据"""
        dashboard = self.get_dashboard(dashboard_id)
        if not dashboard:
            return None
        
        widget = dashboard.get_widget(widget_id)
        if not widget or not widget.enabled:
            return None
        
        # 检查缓存
        cache_key = f"{dashboard_id}_{widget_id}"
        if cache_key in self.data_cache:
            cache_data = self.data_cache[cache_key]
            if time.time() - cache_data['timestamp'] < self.cache_ttl:
                return cache_data['data']
        
        # 获取数据
        chart_data = await self._fetch_widget_data(widget)
        
        # 更新缓存
        self.data_cache[cache_key] = {
            'data': chart_data,
            'timestamp': time.time()
        }
        
        return chart_data
    
    async def _fetch_widget_data(self, widget: Widget) -> ChartData:
        """获取组件数据"""
        if not self.metric_collector:
            return ChartData(labels=[], datasets=[], title=widget.title)
        
        # 计算时间范围
        current_time = time.time()
        time_range_seconds = DataProcessor.get_time_range_seconds(widget.time_range)
        start_time = current_time - time_range_seconds
        
        # 获取指标数据
        metrics = []
        for metric_name in widget.metric_names:
            metric_data = self.metric_collector.get_metrics(
                [metric_name], start_time, current_time, widget.filters
            )
            
            for point in metric_data:
                metrics.append({
                    'name': metric_name,
                    'timestamp': point.timestamp,
                    'value': point.value,
                    'tags': point.tags
                })
        
        # 根据图表类型生成数据
        if widget.chart_type == ChartType.LINE:
            return DataProcessor.create_line_chart_data(
                metrics, widget.time_range, widget.title, widget.config.get('unit', '')
            )
        elif widget.chart_type == ChartType.BAR:
            return DataProcessor.create_bar_chart_data(
                metrics, widget.title, widget.config.get('unit', '')
            )
        elif widget.chart_type == ChartType.PIE:
            return DataProcessor.create_pie_chart_data(
                metrics, widget.title, widget.config.get('unit', '')
            )
        elif widget.chart_type == ChartType.GAUGE:
            max_value = widget.config.get('max_value', 100)
            return DataProcessor.create_gauge_chart_data(
                metrics, max_value, widget.title, widget.config.get('unit', '')
            )
        else:
            return DataProcessor.create_line_chart_data(
                metrics, widget.time_range, widget.title, widget.config.get('unit', '')
            )
    
    async def get_dashboard_data(self, dashboard_id: str) -> Dict[str, Any]:
        """获取仪表板完整数据"""
        dashboard = self.get_dashboard(dashboard_id)
        if not dashboard:
            return {}
        
        dashboard_data = dashboard.to_dict()
        
        # 获取所有组件数据
        widget_data = {}
        for widget in dashboard.widgets:
            if widget.enabled:
                chart_data = await self.get_widget_data(dashboard_id, widget.id)
                if chart_data:
                    widget_data[widget.id] = chart_data.to_dict()
        
        dashboard_data['widget_data'] = widget_data
        
        # 添加告警信息
        if self.alert_manager:
            active_alerts = self.alert_manager.get_active_alerts()
            dashboard_data['alerts'] = [alert.to_dict() for alert in active_alerts[:10]]
            dashboard_data['alert_count'] = len(active_alerts)
        
        return dashboard_data
    
    async def start(self):
        """启动仪表板管理器"""
        if self.is_running:
            return
        
        self.is_running = True
        self.update_task = asyncio.create_task(self._update_loop())
        logger.info("Dashboard manager started")
    
    async def stop(self):
        """停止仪表板管理器"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        if self.update_task:
            self.update_task.cancel()
            try:
                await self.update_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Dashboard manager stopped")
    
    async def _update_loop(self):
        """更新循环"""
        while self.is_running:
            try:
                # 清理过期缓存
                current_time = time.time()
                expired_keys = [
                    key for key, data in self.data_cache.items()
                    if current_time - data['timestamp'] > self.cache_ttl * 2
                ]
                
                for key in expired_keys:
                    del self.data_cache[key]
                
                await asyncio.sleep(60)  # 每分钟清理一次
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in dashboard update loop: {e}")
                await asyncio.sleep(60)


def create_default_dashboards() -> List[Dashboard]:
    """创建默认仪表板"""
    dashboards = []
    
    # 系统性能仪表板
    performance_dashboard = Dashboard(
        id="system_performance",
        name="系统性能监控",
        description="监控系统CPU、内存、网络等性能指标",
        widgets=[
            Widget(
                id="cpu_usage_chart",
                title="CPU使用率",
                chart_type=ChartType.LINE,
                metric_names=["cpu_usage"],
                time_range=TimeRange.LAST_1_HOUR,
                refresh_interval=RefreshInterval.NORMAL,
                position={"x": 0, "y": 0, "width": 6, "height": 4},
                config={"unit": "%", "max_value": 100}
            ),
            Widget(
                id="memory_usage_chart",
                title="内存使用率",
                chart_type=ChartType.LINE,
                metric_names=["memory_usage"],
                time_range=TimeRange.LAST_1_HOUR,
                refresh_interval=RefreshInterval.NORMAL,
                position={"x": 6, "y": 0, "width": 6, "height": 4},
                config={"unit": "%", "max_value": 100}
            ),
            Widget(
                id="cpu_gauge",
                title="当前CPU使用率",
                chart_type=ChartType.GAUGE,
                metric_names=["cpu_usage"],
                time_range=TimeRange.LAST_5_MINUTES,
                refresh_interval=RefreshInterval.FAST,
                position={"x": 0, "y": 4, "width": 3, "height": 3},
                config={"unit": "%", "max_value": 100}
            ),
            Widget(
                id="memory_gauge",
                title="当前内存使用率",
                chart_type=ChartType.GAUGE,
                metric_names=["memory_usage"],
                time_range=TimeRange.LAST_5_MINUTES,
                refresh_interval=RefreshInterval.FAST,
                position={"x": 3, "y": 4, "width": 3, "height": 3},
                config={"unit": "%", "max_value": 100}
            )
        ]
    )
    dashboards.append(performance_dashboard)
    
    # 翻译服务仪表板
    translation_dashboard = Dashboard(
        id="translation_service",
        name="翻译服务监控",
        description="监控翻译服务的性能和质量指标",
        widgets=[
            Widget(
                id="response_time_chart",
                title="响应时间趋势",
                chart_type=ChartType.LINE,
                metric_names=["response_time"],
                time_range=TimeRange.LAST_6_HOURS,
                refresh_interval=RefreshInterval.NORMAL,
                position={"x": 0, "y": 0, "width": 8, "height": 4},
                config={"unit": "ms"}
            ),
            Widget(
                id="translation_count_chart",
                title="翻译请求数量",
                chart_type=ChartType.BAR,
                metric_names=["translation_count"],
                time_range=TimeRange.LAST_24_HOURS,
                refresh_interval=RefreshInterval.SLOW,
                position={"x": 8, "y": 0, "width": 4, "height": 4},
                config={"unit": "次"}
            ),
            Widget(
                id="error_rate_chart",
                title="错误率",
                chart_type=ChartType.LINE,
                metric_names=["error_rate"],
                time_range=TimeRange.LAST_6_HOURS,
                refresh_interval=RefreshInterval.NORMAL,
                position={"x": 0, "y": 4, "width": 6, "height": 3},
                config={"unit": "%"}
            ),
            Widget(
                id="cache_hit_rate_gauge",
                title="缓存命中率",
                chart_type=ChartType.GAUGE,
                metric_names=["cache_hit_rate"],
                time_range=TimeRange.LAST_15_MINUTES,
                refresh_interval=RefreshInterval.NORMAL,
                position={"x": 6, "y": 4, "width": 3, "height": 3},
                config={"unit": "%", "max_value": 100}
            ),
            Widget(
                id="language_distribution",
                title="语言对分布",
                chart_type=ChartType.PIE,
                metric_names=["language_pair_usage"],
                time_range=TimeRange.LAST_24_HOURS,
                refresh_interval=RefreshInterval.SLOW,
                position={"x": 9, "y": 4, "width": 3, "height": 3},
                config={"unit": "次"}
            )
        ]
    )
    dashboards.append(translation_dashboard)
    
    # 业务指标仪表板
    business_dashboard = Dashboard(
        id="business_metrics",
        name="业务指标监控",
        description="监控业务相关的关键指标",
        widgets=[
            Widget(
                id="user_activity_chart",
                title="用户活跃度",
                chart_type=ChartType.LINE,
                metric_names=["active_users"],
                time_range=TimeRange.LAST_7_DAYS,
                refresh_interval=RefreshInterval.VERY_SLOW,
                position={"x": 0, "y": 0, "width": 6, "height": 4},
                config={"unit": "人"}
            ),
            Widget(
                id="satisfaction_score_chart",
                title="用户满意度评分",
                chart_type=ChartType.LINE,
                metric_names=["satisfaction_score"],
                time_range=TimeRange.LAST_7_DAYS,
                refresh_interval=RefreshInterval.VERY_SLOW,
                position={"x": 6, "y": 0, "width": 6, "height": 4},
                config={"unit": "分", "max_value": 5}
            ),
            Widget(
                id="feature_usage_pie",
                title="功能使用分布",
                chart_type=ChartType.PIE,
                metric_names=["feature_usage"],
                time_range=TimeRange.LAST_24_HOURS,
                refresh_interval=RefreshInterval.SLOW,
                position={"x": 0, "y": 4, "width": 4, "height": 3},
                config={"unit": "次"}
            ),
            Widget(
                id="retention_rate_gauge",
                title="用户留存率",
                chart_type=ChartType.GAUGE,
                metric_names=["retention_rate"],
                time_range=TimeRange.LAST_30_DAYS,
                refresh_interval=RefreshInterval.VERY_SLOW,
                position={"x": 4, "y": 4, "width": 4, "height": 3},
                config={"unit": "%", "max_value": 100}
            ),
            Widget(
                id="revenue_chart",
                title="收入趋势",
                chart_type=ChartType.AREA,
                metric_names=["revenue"],
                time_range=TimeRange.LAST_30_DAYS,
                refresh_interval=RefreshInterval.VERY_SLOW,
                position={"x": 8, "y": 4, "width": 4, "height": 3},
                config={"unit": "元"}
            )
        ]
    )
    dashboards.append(business_dashboard)
    
    return dashboards


# 全局实例
_dashboard_manager = None


def get_dashboard_manager() -> DashboardManager:
    """获取仪表板管理器实例"""
    global _dashboard_manager
    if _dashboard_manager is None:
        _dashboard_manager = DashboardManager()
    return _dashboard_manager


async def initialize_dashboard_system(metric_collector, alert_manager):
    """初始化仪表板系统"""
    dashboard_manager = get_dashboard_manager()
    dashboard_manager.set_metric_collector(metric_collector)
    dashboard_manager.set_alert_manager(alert_manager)
    
    # 创建默认仪表板
    for dashboard in create_default_dashboards():
        dashboard_manager.dashboards[dashboard.id] = dashboard
    
    # 启动仪表板管理器
    await dashboard_manager.start()
    
    logger.info("Dashboard system initialized")
    return dashboard_manager