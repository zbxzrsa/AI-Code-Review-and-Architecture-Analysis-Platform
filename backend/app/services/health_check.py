import requests
import time
import subprocess
import socket
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

class ServiceHealthChecker:
    def __init__(self):
        self.services = {
            'database': {'port': 5432, 'timeout': 30, 'type': 'tcp'},
            'redis': {'port': 6379, 'timeout': 10, 'type': 'tcp'},
            'ai_service': {'port': 8001, 'timeout': 60, 'type': 'http', 'endpoint': '/health'},
            'web_ui': {'port': 3000, 'timeout': 120, 'type': 'http', 'endpoint': '/'}
        }
    
    def check_port(self, port: int) -> bool:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1)
                return s.connect_ex(('localhost', port)) == 0
        except Exception as e:
            logger.error(f"端口检查失败: {e}")
            return False
    
    def check_service_health(self) -> Dict[str, bool]:
        status = {}
        for name, config in self.services.items():
            if config['type'] == 'http':
                try:
                    response = requests.get(f"http://localhost:{config['port']}{config.get('endpoint', '/')}", timeout=2)
                    status[name] = response.status_code == 200
                except Exception as e:
                    status[name] = False
                    logger.warning(f"服务 {name} 健康检查失败: {e}")
            else:
                status[name] = self.check_port(config['port'])
        return status
    
    def wait_for_services(self) -> bool:
        start_time = time.time()
        all_ready = False
        while not all_ready and (time.time() - start_time) < max(
            [config['timeout'] for config in self.services.values()]):
            all_ready = True
            status = self.check_service_health()
            for name, is_ready in status.items():
                if not is_ready:
                    logger.info(f"等待服务 {name} 就绪...")
                    all_ready = False
            time.sleep(3)
        return all_ready

    def auto_recovery(self, max_retries=3):
        """自动恢复故障服务"""
        for retry in range(max_retries):
            logger.info(f"开始第 {retry+1} 次故障恢复尝试")
            failed_services = [
                name for name, status in self.check_service_health().items() 
                if not status
            ]
            
            if not failed_services:
                logger.info("所有服务已恢复")
                return True
            
            for name in failed_services:
                logger.warning(f"尝试重启服务 {name}")
                try:
                    subprocess.run([
                        'docker-compose', 'restart', 
                        self._map_to_service_name(name)
                    ], check=True, timeout=60)
                except Exception as e:
                    logger.error(f"服务 {name} 重启失败: {e}")
            
            time.sleep(5)
            
        logger.error(f"故障恢复失败，共尝试 {max_retries} 次")
        return False
    
    def _map_to_service_name(self, config_name):
        """将配置名称映射到docker-compose服务名"""
        mapping = {
            'database': 'postgres',
            'redis': 'redis',
            'ai_service': 'ai-service',
            'web_ui': 'frontend'
        }
        return mapping.get(config_name)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Codeinsight服务健康监测')
    parser.add_argument('--interval', type=int, default=60, 
                        help='健康检查间隔时间(秒)')
    parser.add_argument('--daemon', action='store_true',
                        help='以守护进程模式运行')
    args = parser.parse_args()
    
    checker = ServiceHealthChecker()
    
    try:
        while True:
            status = checker.check_service_health()
            if not all(status.values()):
                print(f"检测到服务异常: {status}")
                if not checker.auto_recovery():
                    print("自动恢复失败，请手动检查")
            else:
                print(f"服务状态正常: {status}")
                
            if not args.daemon:
                break
                
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\n健康检查已终止")