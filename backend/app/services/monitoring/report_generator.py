"""
自动化报告生成系统 - 支持多种报告类型和格式
"""

import asyncio
import json
import time
import logging
import statistics
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime, timedelta
import io
import base64

logger = logging.getLogger(__name__)


class ReportType(Enum):
    """报告类型"""
    PERFORMANCE = "performance"
    QUALITY = "quality"
    BUSINESS = "business"
    SYSTEM_HEALTH = "system_health"
    USER_BEHAVIOR = "user_behavior"
    COST_ANALYSIS = "cost_analysis"
    CUSTOM = "custom"


class ReportFormat(Enum):
    """报告格式"""
    JSON = "json"
    HTML = "html"
    PDF = "pdf"
    CSV = "csv"
    EXCEL = "excel"


class ReportFrequency(Enum):
    """报告频率"""
    REAL_TIME = "real_time"
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"
    ON_DEMAND = "on_demand"


class ReportStatus(Enum):
    """报告状态"""
    PENDING = "pending"
    GENERATING = "generating"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ReportMetric:
    """报告指标"""
    name: str
    value: Union[float, int, str]
    unit: str = ""
    description: str = ""
    trend: Optional[str] = None  # "up", "down", "stable"
    change_percentage: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ReportSection:
    """报告章节"""
    title: str
    content: str
    metrics: List[ReportMetric] = field(default_factory=list)
    charts: List[Dict[str, Any]] = field(default_factory=list)
    tables: List[Dict[str, Any]] = field(default_factory=list)
    order: int = 0
    
    def add_metric(self, metric: ReportMetric):
        """添加指标"""
        self.metrics.append(metric)
    
    def add_chart(self, chart_data: Dict[str, Any]):
        """添加图表"""
        self.charts.append(chart_data)
    
    def add_table(self, table_data: Dict[str, Any]):
        """添加表格"""
        self.tables.append(table_data)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'title': self.title,
            'content': self.content,
            'metrics': [metric.to_dict() for metric in self.metrics],
            'charts': self.charts,
            'tables': self.tables,
            'order': self.order
        }


@dataclass
class ReportTemplate:
    """报告模板"""
    id: str
    name: str
    report_type: ReportType
    description: str
    sections: List[str]  # 章节名称列表
    metrics: List[str]   # 需要的指标列表
    time_range: str = "24h"
    filters: Dict[str, Any] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'name': self.name,
            'report_type': self.report_type.value,
            'description': self.description,
            'sections': self.sections,
            'metrics': self.metrics,
            'time_range': self.time_range,
            'filters': self.filters,
            'config': self.config
        }


@dataclass
class ReportSchedule:
    """报告调度"""
    id: str
    template_id: str
    name: str
    frequency: ReportFrequency
    format: ReportFormat
    recipients: List[str]
    enabled: bool = True
    next_run: Optional[float] = None
    last_run: Optional[float] = None
    created_at: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'template_id': self.template_id,
            'name': self.name,
            'frequency': self.frequency.value,
            'format': self.format.value,
            'recipients': self.recipients,
            'enabled': self.enabled,
            'next_run': self.next_run,
            'last_run': self.last_run,
            'created_at': self.created_at
        }


@dataclass
class Report:
    """报告"""
    id: str
    template_id: str
    name: str
    report_type: ReportType
    format: ReportFormat
    status: ReportStatus
    sections: List[ReportSection] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    file_path: Optional[str] = None
    file_size: Optional[int] = None
    error_message: Optional[str] = None
    
    def add_section(self, section: ReportSection):
        """添加章节"""
        self.sections.append(section)
        self.sections.sort(key=lambda x: x.order)
    
    def set_completed(self, file_path: str = None, file_size: int = None):
        """设置完成状态"""
        self.status = ReportStatus.COMPLETED
        self.completed_at = time.time()
        self.file_path = file_path
        self.file_size = file_size
    
    def set_failed(self, error_message: str):
        """设置失败状态"""
        self.status = ReportStatus.FAILED
        self.error_message = error_message
        self.completed_at = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'template_id': self.template_id,
            'name': self.name,
            'report_type': self.report_type.value,
            'format': self.format.value,
            'status': self.status.value,
            'sections': [section.to_dict() for section in self.sections],
            'summary': self.summary,
            'metadata': self.metadata,
            'created_at': self.created_at,
            'completed_at': self.completed_at,
            'file_path': self.file_path,
            'file_size': self.file_size,
            'error_message': self.error_message
        }


class DataAnalyzer:
    """数据分析器"""
    
    @staticmethod
    def calculate_trend(values: List[float], threshold: float = 0.05) -> str:
        """计算趋势"""
        if len(values) < 2:
            return "stable"
        
        # 计算线性回归斜率
        n = len(values)
        x = list(range(n))
        
        x_mean = statistics.mean(x)
        y_mean = statistics.mean(values)
        
        numerator = sum((x[i] - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            return "stable"
        
        slope = numerator / denominator
        
        # 根据斜率判断趋势
        if abs(slope) < threshold:
            return "stable"
        elif slope > 0:
            return "up"
        else:
            return "down"
    
    @staticmethod
    def calculate_change_percentage(current: float, previous: float) -> float:
        """计算变化百分比"""
        if previous == 0:
            return 0.0
        return ((current - previous) / previous) * 100
    
    @staticmethod
    def calculate_statistics(values: List[float]) -> Dict[str, float]:
        """计算统计信息"""
        if not values:
            return {}
        
        return {
            'count': len(values),
            'sum': sum(values),
            'mean': statistics.mean(values),
            'median': statistics.median(values),
            'min': min(values),
            'max': max(values),
            'std_dev': statistics.stdev(values) if len(values) > 1 else 0,
            'variance': statistics.variance(values) if len(values) > 1 else 0
        }
    
    @staticmethod
    def calculate_percentiles(values: List[float], percentiles: List[float] = None) -> Dict[str, float]:
        """计算百分位数"""
        if not values:
            return {}
        
        if percentiles is None:
            percentiles = [25, 50, 75, 90, 95, 99]
        
        sorted_values = sorted(values)
        n = len(sorted_values)
        
        result = {}
        for p in percentiles:
            index = (p / 100) * (n - 1)
            if index.is_integer():
                result[f'p{int(p)}'] = sorted_values[int(index)]
            else:
                lower = int(index)
                upper = lower + 1
                weight = index - lower
                result[f'p{int(p)}'] = sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight
        
        return result


class ReportGenerator:
    """报告生成器"""
    
    def __init__(self):
        self.templates: Dict[str, ReportTemplate] = {}
        self.schedules: Dict[str, ReportSchedule] = {}
        self.reports: Dict[str, Report] = {}
        self.metric_collector = None
        self.alert_manager = None
        self.dashboard_manager = None
        self.is_running = False
        self.scheduler_task = None
        self.report_history: List[Report] = []
        self.max_history_size = 1000
        self.output_directory = "reports"
    
    def set_metric_collector(self, collector):
        """设置指标收集器"""
        self.metric_collector = collector
    
    def set_alert_manager(self, alert_manager):
        """设置告警管理器"""
        self.alert_manager = alert_manager
    
    def set_dashboard_manager(self, dashboard_manager):
        """设置仪表板管理器"""
        self.dashboard_manager = dashboard_manager
    
    def add_template(self, template: ReportTemplate):
        """添加报告模板"""
        self.templates[template.id] = template
        logger.info(f"Added report template: {template.id}")
    
    def remove_template(self, template_id: str) -> bool:
        """移除报告模板"""
        if template_id in self.templates:
            del self.templates[template_id]
            logger.info(f"Removed report template: {template_id}")
            return True
        return False
    
    def add_schedule(self, schedule: ReportSchedule):
        """添加报告调度"""
        # 计算下次运行时间
        schedule.next_run = self._calculate_next_run(schedule.frequency)
        self.schedules[schedule.id] = schedule
        logger.info(f"Added report schedule: {schedule.id}")
    
    def remove_schedule(self, schedule_id: str) -> bool:
        """移除报告调度"""
        if schedule_id in self.schedules:
            del self.schedules[schedule_id]
            logger.info(f"Removed report schedule: {schedule_id}")
            return True
        return False
    
    def _calculate_next_run(self, frequency: ReportFrequency) -> float:
        """计算下次运行时间"""
        now = datetime.now()
        
        if frequency == ReportFrequency.HOURLY:
            next_run = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        elif frequency == ReportFrequency.DAILY:
            next_run = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
        elif frequency == ReportFrequency.WEEKLY:
            days_ahead = 6 - now.weekday()  # 周日
            next_run = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=days_ahead)
        elif frequency == ReportFrequency.MONTHLY:
            if now.month == 12:
                next_run = now.replace(year=now.year + 1, month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
            else:
                next_run = now.replace(month=now.month + 1, day=1, hour=0, minute=0, second=0, microsecond=0)
        else:
            next_run = now + timedelta(hours=1)  # 默认1小时后
        
        return next_run.timestamp()
    
    async def generate_report(self, template_id: str, format: ReportFormat = ReportFormat.JSON,
                            custom_params: Dict[str, Any] = None) -> Report:
        """生成报告"""
        template = self.templates.get(template_id)
        if not template:
            raise ValueError(f"Template not found: {template_id}")
        
        # 创建报告实例
        report_id = f"{template_id}_{int(time.time())}"
        report = Report(
            id=report_id,
            template_id=template_id,
            name=f"{template.name} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            report_type=template.report_type,
            format=format,
            status=ReportStatus.GENERATING
        )
        
        self.reports[report_id] = report
        
        try:
            # 生成报告内容
            await self._generate_report_content(report, template, custom_params or {})
            
            # 生成文件
            file_path = await self._generate_report_file(report)
            
            # 设置完成状态
            import os
            file_size = os.path.getsize(file_path) if file_path and os.path.exists(file_path) else 0
            report.set_completed(file_path, file_size)
            
            # 添加到历史记录
            self.report_history.append(report)
            if len(self.report_history) > self.max_history_size:
                self.report_history = self.report_history[-self.max_history_size:]
            
            logger.info(f"Report generated successfully: {report_id}")
            
        except Exception as e:
            report.set_failed(str(e))
            logger.error(f"Failed to generate report {report_id}: {e}")
        
        return report
    
    async def _generate_report_content(self, report: Report, template: ReportTemplate, 
                                     custom_params: Dict[str, Any]):
        """生成报告内容"""
        # 计算时间范围
        time_range = custom_params.get('time_range', template.time_range)
        end_time = time.time()
        start_time = end_time - self._parse_time_range(time_range)
        
        # 获取指标数据
        metrics_data = {}
        if self.metric_collector:
            for metric_name in template.metrics:
                data = self.metric_collector.get_metrics(
                    [metric_name], start_time, end_time, template.filters
                )
                metrics_data[metric_name] = data
        
        # 生成各个章节
        for section_name in template.sections:
            section = await self._generate_section(
                section_name, template, metrics_data, start_time, end_time
            )
            report.add_section(section)
        
        # 生成摘要
        report.summary = await self._generate_summary(report, template, metrics_data)
        
        # 设置元数据
        report.metadata = {
            'time_range': time_range,
            'start_time': start_time,
            'end_time': end_time,
            'metrics_count': len(template.metrics),
            'sections_count': len(report.sections),
            'generation_time': time.time() - report.created_at
        }
    
    async def _generate_section(self, section_name: str, template: ReportTemplate,
                              metrics_data: Dict[str, List], start_time: float, 
                              end_time: float) -> ReportSection:
        """生成报告章节"""
        section = ReportSection(title=section_name, content="", order=template.sections.index(section_name))
        
        if section_name == "概述":
            section.content = "本报告提供了系统在指定时间范围内的关键指标分析和性能评估。"
            
            # 添加关键指标
            for metric_name, data in metrics_data.items():
                if data:
                    values = [point.value for point in data]
                    stats = DataAnalyzer.calculate_statistics(values)
                    trend = DataAnalyzer.calculate_trend(values)
                    
                    metric = ReportMetric(
                        name=metric_name,
                        value=stats.get('mean', 0),
                        unit=self._get_metric_unit(metric_name),
                        description=f"{metric_name}的平均值",
                        trend=trend
                    )
                    section.add_metric(metric)
        
        elif section_name == "性能分析":
            section.content = "系统性能指标的详细分析，包括响应时间、吞吐量等关键性能指标。"
            
            # 性能相关指标
            performance_metrics = ['response_time', 'throughput', 'cpu_usage', 'memory_usage']
            for metric_name in performance_metrics:
                if metric_name in metrics_data and metrics_data[metric_name]:
                    data = metrics_data[metric_name]
                    values = [point.value for point in data]
                    
                    stats = DataAnalyzer.calculate_statistics(values)
                    percentiles = DataAnalyzer.calculate_percentiles(values)
                    
                    # 添加统计指标
                    section.add_metric(ReportMetric(
                        name=f"{metric_name}_avg",
                        value=stats.get('mean', 0),
                        unit=self._get_metric_unit(metric_name),
                        description=f"{metric_name}平均值"
                    ))
                    
                    section.add_metric(ReportMetric(
                        name=f"{metric_name}_p95",
                        value=percentiles.get('p95', 0),
                        unit=self._get_metric_unit(metric_name),
                        description=f"{metric_name} 95百分位"
                    ))
        
        elif section_name == "质量分析":
            section.content = "翻译质量相关指标的分析，包括准确率、用户满意度等。"
            
            quality_metrics = ['accuracy_rate', 'satisfaction_score', 'error_rate']
            for metric_name in quality_metrics:
                if metric_name in metrics_data and metrics_data[metric_name]:
                    data = metrics_data[metric_name]
                    values = [point.value for point in data]
                    
                    current_value = values[-1] if values else 0
                    previous_value = values[-2] if len(values) > 1 else current_value
                    change_pct = DataAnalyzer.calculate_change_percentage(current_value, previous_value)
                    
                    section.add_metric(ReportMetric(
                        name=metric_name,
                        value=current_value,
                        unit=self._get_metric_unit(metric_name),
                        description=f"当前{metric_name}",
                        change_percentage=change_pct
                    ))
        
        elif section_name == "业务指标":
            section.content = "业务相关指标的分析，包括用户活跃度、功能使用情况等。"
            
            business_metrics = ['active_users', 'translation_count', 'feature_usage', 'retention_rate']
            for metric_name in business_metrics:
                if metric_name in metrics_data and metrics_data[metric_name]:
                    data = metrics_data[metric_name]
                    values = [point.value for point in data]
                    
                    total_value = sum(values)
                    avg_value = statistics.mean(values) if values else 0
                    
                    section.add_metric(ReportMetric(
                        name=f"{metric_name}_total",
                        value=total_value,
                        unit=self._get_metric_unit(metric_name),
                        description=f"{metric_name}总计"
                    ))
                    
                    section.add_metric(ReportMetric(
                        name=f"{metric_name}_avg",
                        value=avg_value,
                        unit=self._get_metric_unit(metric_name),
                        description=f"{metric_name}平均值"
                    ))
        
        elif section_name == "告警分析":
            section.content = "系统告警情况的分析，包括告警数量、严重程度分布等。"
            
            if self.alert_manager:
                alert_stats = self.alert_manager.get_alert_statistics()
                
                section.add_metric(ReportMetric(
                    name="total_alerts",
                    value=alert_stats.get('total_alerts', 0),
                    unit="个",
                    description="总告警数量"
                ))
                
                section.add_metric(ReportMetric(
                    name="active_alerts",
                    value=alert_stats.get('active_alerts', 0),
                    unit="个",
                    description="活跃告警数量"
                ))
        
        return section
    
    async def _generate_summary(self, report: Report, template: ReportTemplate,
                              metrics_data: Dict[str, List]) -> Dict[str, Any]:
        """生成报告摘要"""
        summary = {
            'report_type': template.report_type.value,
            'metrics_analyzed': len(template.metrics),
            'sections_generated': len(report.sections),
            'key_findings': [],
            'recommendations': []
        }
        
        # 分析关键发现
        for metric_name, data in metrics_data.items():
            if data:
                values = [point.value for point in data]
                trend = DataAnalyzer.calculate_trend(values)
                
                if trend == "up":
                    summary['key_findings'].append(f"{metric_name}呈上升趋势")
                elif trend == "down":
                    summary['key_findings'].append(f"{metric_name}呈下降趋势")
        
        # 生成建议
        if self.alert_manager:
            active_alerts = self.alert_manager.get_active_alerts()
            if active_alerts:
                summary['recommendations'].append(f"当前有{len(active_alerts)}个活跃告警需要处理")
        
        return summary
    
    async def _generate_report_file(self, report: Report) -> str:
        """生成报告文件"""
        import os
        
        # 确保输出目录存在
        os.makedirs(self.output_directory, exist_ok=True)
        
        # 生成文件名
        timestamp = datetime.fromtimestamp(report.created_at).strftime('%Y%m%d_%H%M%S')
        filename = f"{report.template_id}_{timestamp}.{report.format.value}"
        file_path = os.path.join(self.output_directory, filename)
        
        if report.format == ReportFormat.JSON:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)
        
        elif report.format == ReportFormat.HTML:
            html_content = self._generate_html_report(report)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
        
        elif report.format == ReportFormat.CSV:
            csv_content = self._generate_csv_report(report)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(csv_content)
        
        return file_path
    
    def _generate_html_report(self, report: Report) -> str:
        """生成HTML报告"""
        html = f"""
        <!DOCTYPE html>
        <html lang="zh-CN">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{report.name}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f8f9fa; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                .metric {{ display: inline-block; margin: 10px; padding: 10px; background-color: #e9ecef; border-radius: 3px; }}
                .summary {{ background-color: #d4edda; padding: 15px; border-radius: 5px; }}
                table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>{report.name}</h1>
                <p>报告类型: {report.report_type.value}</p>
                <p>生成时间: {datetime.fromtimestamp(report.created_at).strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
        """
        
        # 添加摘要
        if report.summary:
            html += f"""
            <div class="summary">
                <h2>摘要</h2>
                <p>分析指标数量: {report.summary.get('metrics_analyzed', 0)}</p>
                <p>生成章节数量: {report.summary.get('sections_generated', 0)}</p>
            </div>
            """
        
        # 添加各个章节
        for section in report.sections:
            html += f"""
            <div class="section">
                <h2>{section.title}</h2>
                <p>{section.content}</p>
            """
            
            # 添加指标
            if section.metrics:
                html += "<h3>关键指标</h3>"
                for metric in section.metrics:
                    trend_icon = "↑" if metric.trend == "up" else "↓" if metric.trend == "down" else "→"
                    html += f"""
                    <div class="metric">
                        <strong>{metric.name}:</strong> {metric.value} {metric.unit} {trend_icon}
                        <br><small>{metric.description}</small>
                    </div>
                    """
            
            html += "</div>"
        
        html += """
        </body>
        </html>
        """
        
        return html
    
    def _generate_csv_report(self, report: Report) -> str:
        """生成CSV报告"""
        import csv
        import io
        
        output = io.StringIO()
        writer = csv.writer(output)
        
        # 写入标题行
        writer.writerow(['Section', 'Metric Name', 'Value', 'Unit', 'Description', 'Trend'])
        
        # 写入数据行
        for section in report.sections:
            for metric in section.metrics:
                writer.writerow([
                    section.title,
                    metric.name,
                    metric.value,
                    metric.unit,
                    metric.description,
                    metric.trend or ''
                ])
        
        return output.getvalue()
    
    def _parse_time_range(self, time_range: str) -> int:
        """解析时间范围字符串，返回秒数"""
        if time_range.endswith('m'):
            return int(time_range[:-1]) * 60
        elif time_range.endswith('h'):
            return int(time_range[:-1]) * 3600
        elif time_range.endswith('d'):
            return int(time_range[:-1]) * 86400
        else:
            return 3600  # 默认1小时
    
    def _get_metric_unit(self, metric_name: str) -> str:
        """获取指标单位"""
        unit_mapping = {
            'response_time': 'ms',
            'cpu_usage': '%',
            'memory_usage': '%',
            'error_rate': '%',
            'cache_hit_rate': '%',
            'accuracy_rate': '%',
            'satisfaction_score': '分',
            'active_users': '人',
            'translation_count': '次',
            'retention_rate': '%',
            'throughput': 'req/s'
        }
        return unit_mapping.get(metric_name, '')
    
    async def start(self):
        """启动报告生成器"""
        if self.is_running:
            return
        
        self.is_running = True
        self.scheduler_task = asyncio.create_task(self._scheduler_loop())
        logger.info("Report generator started")
    
    async def stop(self):
        """停止报告生成器"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        if self.scheduler_task:
            self.scheduler_task.cancel()
            try:
                await self.scheduler_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Report generator stopped")
    
    async def _scheduler_loop(self):
        """调度循环"""
        while self.is_running:
            try:
                current_time = time.time()
                
                for schedule in self.schedules.values():
                    if (schedule.enabled and 
                        schedule.next_run and 
                        current_time >= schedule.next_run):
                        
                        # 生成报告
                        try:
                            await self.generate_report(schedule.template_id, schedule.format)
                            schedule.last_run = current_time
                            logger.info(f"Scheduled report generated: {schedule.id}")
                        except Exception as e:
                            logger.error(f"Failed to generate scheduled report {schedule.id}: {e}")
                        
                        # 计算下次运行时间
                        schedule.next_run = self._calculate_next_run(schedule.frequency)
                
                await asyncio.sleep(60)  # 每分钟检查一次
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in report scheduler loop: {e}")
                await asyncio.sleep(60)
    
    def get_report_history(self, limit: int = 50) -> List[Report]:
        """获取报告历史"""
        return sorted(self.report_history, key=lambda x: x.created_at, reverse=True)[:limit]
    
    def get_report_statistics(self) -> Dict[str, Any]:
        """获取报告统计"""
        total_reports = len(self.report_history)
        completed_reports = len([r for r in self.report_history if r.status == ReportStatus.COMPLETED])
        failed_reports = len([r for r in self.report_history if r.status == ReportStatus.FAILED])
        
        return {
            'total_reports': total_reports,
            'completed_reports': completed_reports,
            'failed_reports': failed_reports,
            'success_rate': (completed_reports / total_reports * 100) if total_reports > 0 else 0,
            'templates_count': len(self.templates),
            'schedules_count': len(self.schedules),
            'active_schedules': len([s for s in self.schedules.values() if s.enabled])
        }


def create_default_report_templates() -> List[ReportTemplate]:
    """创建默认报告模板"""
    return [
        ReportTemplate(
            id="daily_performance",
            name="日常性能报告",
            report_type=ReportType.PERFORMANCE,
            description="每日系统性能分析报告",
            sections=["概述", "性能分析", "告警分析"],
            metrics=["response_time", "throughput", "cpu_usage", "memory_usage", "error_rate"],
            time_range="24h"
        ),
        ReportTemplate(
            id="weekly_quality",
            name="周度质量报告",
            report_type=ReportType.QUALITY,
            description="每周翻译质量分析报告",
            sections=["概述", "质量分析", "用户反馈"],
            metrics=["accuracy_rate", "satisfaction_score", "error_rate", "quality_score"],
            time_range="7d"
        ),
        ReportTemplate(
            id="monthly_business",
            name="月度业务报告",
            report_type=ReportType.BUSINESS,
            description="每月业务指标分析报告",
            sections=["概述", "业务指标", "用户分析", "收入分析"],
            metrics=["active_users", "translation_count", "retention_rate", "revenue", "feature_usage"],
            time_range="30d"
        ),
        ReportTemplate(
            id="system_health",
            name="系统健康报告",
            report_type=ReportType.SYSTEM_HEALTH,
            description="系统整体健康状况报告",
            sections=["概述", "性能分析", "质量分析", "告警分析"],
            metrics=["cpu_usage", "memory_usage", "response_time", "error_rate", "cache_hit_rate"],
            time_range="24h"
        )
    ]


# 全局实例
_report_generator = None


def get_report_generator() -> ReportGenerator:
    """获取报告生成器实例"""
    global _report_generator
    if _report_generator is None:
        _report_generator = ReportGenerator()
    return _report_generator


async def initialize_report_system(metric_collector, alert_manager, dashboard_manager):
    """初始化报告系统"""
    report_generator = get_report_generator()
    report_generator.set_metric_collector(metric_collector)
    report_generator.set_alert_manager(alert_manager)
    report_generator.set_dashboard_manager(dashboard_manager)
    
    # 添加默认模板
    for template in create_default_report_templates():
        report_generator.add_template(template)
    
    # 添加默认调度
    daily_schedule = ReportSchedule(
        id="daily_performance_schedule",
        template_id="daily_performance",
        name="每日性能报告",
        frequency=ReportFrequency.DAILY,
        format=ReportFormat.HTML,
        recipients=["admin@example.com"]
    )
    report_generator.add_schedule(daily_schedule)
    
    # 启动报告生成器
    await report_generator.start()
    
    logger.info("Report system initialized")
    return report_generator