#!/usr/bin/env python3
"""
系统监控工具
用于监控系统性能和可用性
"""
import os
import sys
import json
import time
import argparse
import subprocess
import logging
import socket
import platform
import psutil
from datetime import datetime
from typing import Dict, List, Any, Optional
import requests
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('system_monitor.log')
    ]
)
logger = logging.getLogger('system_monitor')

class SystemMonitor:
    """系统监控器"""
    
    def __init__(self, config_file: str):
        """初始化监控器"""
        self.config_file = config_file
        self.config = self._load_config()
        self.alerts = []
        self.metrics = {}
        
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            return {
                "monitoring": {
                    "interval": 60,
                    "endpoints": [],
                    "system_metrics": True,
                    "process_metrics": True,
                    "disk_metrics": True,
                    "network_metrics": True
                },
                "alerting": {
                    "enabled": False,
                    "thresholds": {
                        "cpu_usage": 80,
                        "memory_usage": 80,
                        "disk_usage": 80,
                        "response_time": 2000
                    },
                    "channels": {
                        "email": {
                            "enabled": False,
                            "smtp_server": "",
                            "smtp_port": 587,
                            "username": "",
                            "password": "",
                            "from_email": "",
                            "to_emails": []
                        },
                        "slack": {
                            "enabled": False,
                            "webhook_url": ""
                        }
                    }
                },
                "backup": {
                    "enabled": False,
                    "schedule": "daily",
                    "retention_days": 7,
                    "paths": [],
                    "destination": ""
                }
            }
    
    def start_monitoring(self) -> None:
        """开始监控"""
        logger.info("开始系统监控...")
        
        interval = self.config.get("monitoring", {}).get("interval", 60)
        
        try:
            while True:
                self._collect_metrics()
                self._check_alerts()
                self._save_metrics()
                
                logger.info(f"等待 {interval} 秒后进行下一次监控...")
                time.sleep(interval)
        except KeyboardInterrupt:
            logger.info("监控已停止")
        except Exception as e:
            logger.error(f"监控过程中发生错误: {e}")
    
    def _collect_metrics(self) -> None:
        """收集系统指标"""
        logger.info("收集系统指标...")
        
        timestamp = datetime.now().isoformat()
        self.metrics = {
            "timestamp": timestamp,
            "hostname": socket.gethostname(),
            "platform": platform.platform()
        }
        
        # 收集系统指标
        if self.config.get("monitoring", {}).get("system_metrics", True):
            self._collect_system_metrics()
        
        # 收集进程指标
        if self.config.get("monitoring", {}).get("process_metrics", True):
            self._collect_process_metrics()
        
        # 收集磁盘指标
        if self.config.get("monitoring", {}).get("disk_metrics", True):
            self._collect_disk_metrics()
        
        # 收集网络指标
        if self.config.get("monitoring", {}).get("network_metrics", True):
            self._collect_network_metrics()
        
        # 检查端点可用性
        self._check_endpoints()
    
    def _collect_system_metrics(self) -> None:
        """收集系统指标"""
        # CPU 使用率
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # 内存使用率
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        
        # 系统负载
        load_avg = psutil.getloadavg()
        
        # 系统启动时间
        boot_time = datetime.fromtimestamp(psutil.boot_time()).isoformat()
        
        self.metrics["system"] = {
            "cpu_percent": cpu_percent,
            "memory_percent": memory_percent,
            "memory_total": memory.total,
            "memory_available": memory.available,
            "load_avg_1min": load_avg[0],
            "load_avg_5min": load_avg[1],
            "load_avg_15min": load_avg[2],
            "boot_time": boot_time
        }
    
    def _collect_process_metrics(self) -> None:
        """收集进程指标"""
        processes = []
        
        for proc in psutil.process_iter(['pid', 'name', 'username', 'cpu_percent', 'memory_percent', 'create_time']):
            try:
                process_info = proc.info
                process_info['create_time'] = datetime.fromtimestamp(process_info['create_time']).isoformat()
                processes.append(process_info)
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
        
        # 按 CPU 使用率排序，只保留前 10 个进程
        processes.sort(key=lambda x: x['cpu_percent'], reverse=True)
        top_processes = processes[:10]
        
        self.metrics["processes"] = top_processes
    
    def _collect_disk_metrics(self) -> None:
        """收集磁盘指标"""
        disks = []
        
        for partition in psutil.disk_partitions():
            try:
                usage = psutil.disk_usage(partition.mountpoint)
                disk_info = {
                    "device": partition.device,
                    "mountpoint": partition.mountpoint,
                    "fstype": partition.fstype,
                    "total": usage.total,
                    "used": usage.used,
                    "free": usage.free,
                    "percent": usage.percent
                }
                disks.append(disk_info)
            except (PermissionError, FileNotFoundError):
                pass
        
        # 磁盘 IO 统计
        disk_io = psutil.disk_io_counters()
        
        self.metrics["disks"] = {
            "partitions": disks,
            "io": {
                "read_count": disk_io.read_count,
                "write_count": disk_io.write_count,
                "read_bytes": disk_io.read_bytes,
                "write_bytes": disk_io.write_bytes,
                "read_time": disk_io.read_time,
                "write_time": disk_io.write_time
            } if disk_io else {}
        }
    
    def _collect_network_metrics(self) -> None:
        """收集网络指标"""
        # 网络连接数
        connections = len(psutil.net_connections())
        
        # 网络 IO 统计
        net_io = psutil.net_io_counters()
        
        # 网络接口信息
        interfaces = []
        for name, stats in psutil.net_if_stats().items():
            interfaces.append({
                "name": name,
                "isup": stats.isup,
                "speed": stats.speed,
                "mtu": stats.mtu
            })
        
        self.metrics["network"] = {
            "connections": connections,
            "io": {
                "bytes_sent": net_io.bytes_sent,
                "bytes_recv": net_io.bytes_recv,
                "packets_sent": net_io.packets_sent,
                "packets_recv": net_io.packets_recv,
                "errin": net_io.errin,
                "errout": net_io.errout,
                "dropin": net_io.dropin,
                "dropout": net_io.dropout
            },
            "interfaces": interfaces
        }
    
    def _check_endpoints(self) -> None:
        """检查端点可用性"""
        endpoints = self.config.get("monitoring", {}).get("endpoints", [])
        results = []
        
        for endpoint in endpoints:
            url = endpoint.get("url")
            name = endpoint.get("name", url)
            
            try:
                start_time = time.time()
                response = requests.get(url, timeout=10)
                response_time = (time.time() - start_time) * 1000  # 毫秒
                
                result = {
                    "name": name,
                    "url": url,
                    "status_code": response.status_code,
                    "response_time": response_time,
                    "is_up": response.status_code < 400
                }
            except Exception as e:
                result = {
                    "name": name,
                    "url": url,
                    "error": str(e),
                    "is_up": False
                }
            
            results.append(result)
        
        self.metrics["endpoints"] = results
    
    def _check_alerts(self) -> None:
        """检查是否需要发送告警"""
        if not self.config.get("alerting", {}).get("enabled", False):
            return
        
        thresholds = self.config.get("alerting", {}).get("thresholds", {})
        
        # 检查 CPU 使用率
        cpu_percent = self.metrics.get("system", {}).get("cpu_percent", 0)
        cpu_threshold = thresholds.get("cpu_usage", 80)
        if cpu_percent > cpu_threshold:
            self._add_alert("CPU 使用率过高", f"CPU 使用率为 {cpu_percent}%，超过阈值 {cpu_threshold}%", "high")
        
        # 检查内存使用率
        memory_percent = self.metrics.get("system", {}).get("memory_percent", 0)
        memory_threshold = thresholds.get("memory_usage", 80)
        if memory_percent > memory_threshold:
            self._add_alert("内存使用率过高", f"内存使用率为 {memory_percent}%，超过阈值 {memory_threshold}%", "high")
        
        # 检查磁盘使用率
        for partition in self.metrics.get("disks", {}).get("partitions", []):
            disk_percent = partition.get("percent", 0)
            disk_threshold = thresholds.get("disk_usage", 80)
            if disk_percent > disk_threshold:
                self._add_alert("磁盘使用率过高", f"磁盘 {partition.get('mountpoint')} 使用率为 {disk_percent}%，超过阈值 {disk_threshold}%", "medium")
        
        # 检查端点响应时间
        for endpoint in self.metrics.get("endpoints", []):
            if not endpoint.get("is_up", True):
                self._add_alert("端点不可用", f"端点 {endpoint.get('name')} ({endpoint.get('url')}) 不可用", "high")
            else:
                response_time = endpoint.get("response_time", 0)
                response_time_threshold = thresholds.get("response_time", 2000)
                if response_time > response_time_threshold:
                    self._add_alert("端点响应时间过长", f"端点 {endpoint.get('name')} ({endpoint.get('url')}) 响应时间为 {response_time:.2f} ms，超过阈值 {response_time_threshold} ms", "medium")
        
        # 发送告警
        if self.alerts:
            self._send_alerts()
    
    def _add_alert(self, title: str, message: str, severity: str) -> None:
        """添加告警"""
        self.alerts.append({
            "timestamp": datetime.now().isoformat(),
            "title": title,
            "message": message,
            "severity": severity
        })
    
    def _send_alerts(self) -> None:
        """发送告警"""
        logger.info(f"发送 {len(self.alerts)} 个告警...")
        
        # 发送邮件告警
        if self.config.get("alerting", {}).get("channels", {}).get("email", {}).get("enabled", False):
            self._send_email_alerts()
        
        # 发送 Slack 告警
        if self.config.get("alerting", {}).get("channels", {}).get("slack", {}).get("enabled", False):
            self._send_slack_alerts()
        
        # 清空告警列表
        self.alerts = []
    
    def _send_email_alerts(self) -> None:
        """发送邮件告警"""
        email_config = self.config.get("alerting", {}).get("channels", {}).get("email", {})
        
        smtp_server = email_config.get("smtp_server")
        smtp_port = email_config.get("smtp_port", 587)
        username = email_config.get("username")
        password = email_config.get("password")
        from_email = email_config.get("from_email")
        to_emails = email_config.get("to_emails", [])
        
        if not (smtp_server and username and password and from_email and to_emails):
            logger.error("邮件配置不完整，无法发送邮件告警")
            return
        
        # 构建邮件内容
        subject = f"系统监控告警: {len(self.alerts)} 个告警"
        
        body = "<html><body>"
        body += f"<h2>系统监控告警</h2>"
        body += f"<p>主机: {socket.gethostname()}</p>"
        body += f"<p>时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>"
        body += f"<p>共有 {len(self.alerts)} 个告警:</p>"
        
        for alert in self.alerts:
            severity_color = {
                "high": "red",
                "medium": "orange",
                "low": "yellow"
            }.get(alert.get("severity", "low"), "black")
            
            body += f"<div style='margin-bottom: 10px; padding: 10px; border: 1px solid {severity_color};'>"
            body += f"<h3 style='color: {severity_color};'>{alert.get('title')}</h3>"
            body += f"<p>{alert.get('message')}</p>"
            body += f"<p>时间: {alert.get('timestamp')}</p>"
            body += f"</div>"
        
        body += "</body></html>"
        
        # 发送邮件
        try:
            msg = MIMEMultipart()
            msg['From'] = from_email
            msg['To'] = ", ".join(to_emails)
            msg['Subject'] = subject
            
            msg.attach(MIMEText(body, 'html'))
            
            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()
            server.login(username, password)
            server.send_message(msg)
            server.quit()
            
            logger.info(f"已发送邮件告警到 {', '.join(to_emails)}")
        except Exception as e:
            logger.error(f"发送邮件告警失败: {e}")
    
    def _send_slack_alerts(self) -> None:
        """发送 Slack 告警"""
        slack_config = self.config.get("alerting", {}).get("channels", {}).get("slack", {})
        
        webhook_url = slack_config.get("webhook_url")
        
        if not webhook_url:
            logger.error("Slack 配置不完整，无法发送 Slack 告警")
            return
        
        # 构建 Slack 消息
        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"系统监控告警: {len(self.alerts)} 个告警"
                }
            },
            {
                "type": "section",
                "fields": [
                    {
                        "type": "mrkdwn",
                        "text": f"*主机:*\n{socket.gethostname()}"
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*时间:*\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                    }
                ]
            },
            {
                "type": "divider"
            }
        ]
        
        for alert in self.alerts:
            severity_emoji = {
                "high": ":red_circle:",
                "medium": ":large_orange_circle:",
                "low": ":large_yellow_circle:"
            }.get(alert.get("severity", "low"), ":white_circle:")
            
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"{severity_emoji} *{alert.get('title')}*\n{alert.get('message')}\n_时间: {alert.get('timestamp')}_"
                }
            })
        
        # 发送 Slack 消息
        try:
            response = requests.post(
                webhook_url,
                json={
                    "blocks": blocks
                }
            )
            
            if response.status_code == 200:
                logger.info("已发送 Slack 告警")
            else:
                logger.error(f"发送 Slack 告警失败: {response.status_code} {response.text}")
        except Exception as e:
            logger.error(f"发送 Slack 告警失败: {e}")
    
    def _save_metrics(self) -> None:
        """保存指标数据"""
        metrics_dir = self.config.get("monitoring", {}).get("metrics_dir", "metrics")
        
        if not os.path.exists(metrics_dir):
            os.makedirs(metrics_dir)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(metrics_dir, f"metrics_{timestamp}.json")
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.metrics, f, indent=2)
            
            logger.info(f"指标数据已保存到 {filename}")
        except Exception as e:
            logger.error(f"保存指标数据失败: {e}")
    
    def perform_backup(self) -> None:
        """执行备份"""
        if not self.config.get("backup", {}).get("enabled", False):
            logger.info("备份功能未启用")
            return
        
        logger.info("开始执行备份...")
        
        backup_paths = self.config.get("backup", {}).get("paths", [])
        backup_destination = self.config.get("backup", {}).get("destination", "")
        
        if not (backup_paths and backup_destination):
            logger.error("备份配置不完整，无法执行备份")
            return
        
        # 创建备份目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = os.path.join(backup_destination, f"backup_{timestamp}")
        
        try:
            os.makedirs(backup_dir, exist_ok=True)
            
            # 执行备份
            for path in backup_paths:
                path_name = os.path.basename(path)
                dest_path = os.path.join(backup_dir, path_name)
                
                if os.path.isdir(path):
                    shutil.copytree(path, dest_path)
                else:
                    shutil.copy2(path, dest_path)
            
            logger.info(f"备份已完成，保存到 {backup_dir}")
            
            # 清理旧备份
            self._cleanup_old_backups()
        except Exception as e:
            logger.error(f"执行备份失败: {e}")
    
    def _cleanup_old_backups(self) -> None:
        """清理旧备份"""
        backup_destination = self.config.get("backup", {}).get("destination", "")
        retention_days = self.config.get("backup", {}).get("retention_days", 7)
        
        if not backup_destination:
            return
        
        try:
            # 获取所有备份目录
            backup_dirs = []
            for item in os.listdir(backup_destination):
                item_path = os.path.join(backup_destination, item)
                if os.path.isdir(item_path) and item.startswith("backup_"):
                    backup_dirs.append(item_path)
            
            # 按修改时间排序
            backup_dirs.sort(key=lambda x: os.path.getmtime(x))
            
            # 计算需要保留的备份数量
            now = time.time()
            retention_seconds = retention_days * 24 * 60 * 60
            
            # 删除过期备份
            for backup_dir in backup_dirs:
                mtime = os.path.getmtime(backup_dir)
                if now - mtime > retention_seconds:
                    shutil.rmtree(backup_dir)
                    logger.info(f"已删除过期备份: {backup_dir}")
        except Exception as e:
            logger.error(f"清理旧备份失败: {e}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="系统监控工具")
    parser.add_argument("--config", "-c", default="monitor_config.json", help="配置文件路径")
    parser.add_argument("--backup", "-b", action="store_true", help="执行备份")
    
    args = parser.parse_args()
    
    monitor = SystemMonitor(args.config)
    
    if args.backup:
        monitor.perform_backup()
    else:
        monitor.start_monitoring()

if __name__ == "__main__":
    main()