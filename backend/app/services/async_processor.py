"""
异步处理优化服务
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
import json
from enum import Enum

logger = logging.getLogger(__name__)


class ProcessingPriority(Enum):
    """处理优先级"""
    LOW = 1
    NORMAL = 5
    HIGH = 10
    CRITICAL = 20


class BatchProcessor:
    """批处理器"""
    
    def __init__(self, batch_size: int = 100, max_concurrent: int = 5):
        self.batch_size = batch_size
        self.max_concurrent = max_concurrent
        self.processing_tasks = set()
        self.completed_tasks = set()
        
    async def process_batch(self, files: List[str], processor: Callable, priority: ProcessingPriority = ProcessingPriority.NORMAL) -> List[Dict[str, Any]]:
        """
        处理单个批次
        
        Args:
            files: 文件列表
            processor: 处理函数
            priority: 优先级
            
        Returns:
            处理结果列表
        """
        start_time = datetime.utcnow()
        
        try:
            # 限制并发数
            semaphore = asyncio.Semaphore(self.max_concurrent)
            
            # 创建处理任务
            tasks = []
            for file_path in files:
                task = self._process_single_file(file_path, processor, priority, semaphore)
                tasks.append(task)
                self.processing_tasks.add(task.id)
            
            # 等待所有任务完成
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 统计结果
            successful_count = sum(1 for r in results if r.get("success", False))
            total_time = (datetime.utcnow() - start_time).total_seconds()
            
            logger.info(f"Batch processed: {successful_count}/{len(files)} files in {total_time:.2f}s")
            
            return results
            
        except Exception as e:
            logger.error(f"Batch processing error: {str(e)}")
            return []
    
    async def _process_single_file(self, file_path: str, processor: Callable, priority: ProcessingPriority, semaphore: asyncio.Semaphore) -> Dict[str, Any]:
        """
        处理单个文件
        
        Args:
            file_path: 文件路径
            processor: 处理函数
            priority: 优先级
            semaphore: 并发控制信号量
            
        Returns:
            处理结果
        """
        task_id = f"task_{file_path}_{priority.value}_{datetime.utcnow().timestamp()}"
        
        async with semaphore:
            try:
                start_time = datetime.utcnow()
                
                # 调用处理函数
                result = await processor(file_path)
                
                duration = (datetime.utcnow() - start_time).total_seconds()
                
                return {
                    "task_id": task_id,
                    "file_path": file_path,
                    "success": True,
                    "result": result,
                    "priority": priority.value,
                    "duration_seconds": duration,
                    "processed_at": datetime.utcnow().isoformat()
                }
                
            except Exception as e:
                logger.error(f"Error processing {file_path}: {str(e)}")
                return {
                    "task_id": task_id,
                    "file_path": file_path,
                    "success": False,
                    "error": str(e),
                    "priority": priority.value,
                    "duration_seconds": 0,
                    "processed_at": datetime.utcnow().isoformat()
                }
            finally:
                self.processing_tasks.discard(task_id)
                self.completed_tasks.add(task_id)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取处理统计"""
        return {
            "processing_tasks": len(self.processing_tasks),
            "completed_tasks": len(self.completed_tasks),
            "batch_size": self.batch_size,
            "max_concurrent": self.max_concurrent,
            "timestamp": datetime.utcnow().isoformat()
        }


class StreamingProcessor:
    """流式处理器"""
    
    def __init__(self):
        self.active_streams = {}
        
    async def process_stream(self, files: List[str], processor: Callable, chunk_size: int = 50) -> AsyncResult:
        """
        流式处理文件
        
        Args:
            files: 文件列表
            processor: 处理函数
            chunk_size: 每批文件数
            
        Returns:
            流式处理结果
        """
        start_time = datetime.utcnow()
        processed_files = []
        total_files = len(files)
        
        try:
            # 分批处理
            for i in range(0, total_files, chunk_size):
                chunk = files[i:i + chunk_size]
                logger.info(f"Processing chunk {i//chunk_size + 1}/{(total_files//chunk_size) + 1} ({len(chunk)} files)")
                
                # 处理当前批次
                chunk_results = await self._process_chunk(chunk, processor)
                processed_files.extend([r.get("file_path") for r in chunk_results if r.get("success")])
            
            total_time = (datetime.utcnow() - start_time).total_seconds()
            
            return {
                "success": True,
                "total_files": total_files,
                "processed_files": len(processed_files),
                "total_time": total_time,
                "chunks_processed": (total_files // chunk_size) + (1 if total_files % chunk_size else 0),
                "processed_files": processed_files
            }
            
        except Exception as e:
            logger.error(f"Stream processing error: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "total_files": total_files
            }
    
    async def _process_chunk(self, chunk: List[str], processor: Callable) -> List[Dict[str, Any]]:
        """处理单个批次"""
        tasks = []
        for file_path in chunk:
            task = asyncio.create_task(self._process_single_file_for_chunk(file_path, processor))
            tasks.append(task)
        
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _process_single_file_for_chunk(self, file_path: str, processor: Callable) -> Dict[str, Any]:
        """为批次处理优化的单文件处理"""
        try:
            start_time = datetime.utcnow()
            result = await processor(file_path)
            duration = (datetime.utcnow() - start_time).total_seconds()
            
            return {
                "file_path": file_path,
                "success": True,
                "result": result,
                "duration_seconds": duration,
                "processed_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")
            return {
                "file_path": file_path,
                "success": False,
                "error": str(e),
                "duration_seconds": 0,
                "processed_at": datetime.utcnow().isoformat()
            }


class PriorityTaskQueue:
    """优先级任务队列"""
    
    def __init__(self):
        self.queues = {
            ProcessingPriority.CRITICAL: asyncio.Queue(maxsize=50),
            ProcessingPriority.HIGH: asyncio.Queue(maxsize=200),
            ProcessingPriority.NORMAL: asyncio.Queue(maxsize=500),
            ProcessingPriority.LOW: asyncio.Queue(maxsize=1000)
        }
        self.workers = {}
        
    async def add_task(self, task: Dict[str, Any], priority: ProcessingPriority = ProcessingPriority.NORMAL):
        """添加任务到优先级队列"""
        queue = self.queues[priority]
        await queue.put(task)
        logger.info(f"Added task to {priority.name} queue: {task.get('file_path', 'unknown')}")
    
    async def get_task(self, priority: ProcessingPriority) -> Optional[Dict[str, Any]]:
        """从优先级队列获取任务"""
        queue = self.queues[priority]
        try:
            return await asyncio.wait_for(queue.get(), timeout=1.0)
        except asyncio.TimeoutError:
            return None
    
    async def start_workers(self):
        """启动工作线程"""
        for priority in ProcessingPriority:
            if priority != ProcessingPriority.LOW:
                worker = asyncio.create_task(
                    self._worker_loop(priority, self.queues[priority])
                )
                self.workers[priority] = worker
                logger.info(f"Started {priority.name} worker")
    
    async def _worker_loop(self, priority: ProcessingPriority, queue: asyncio.Queue):
        """工作线程循环"""
        while True:
            task = await queue.get()
            if task is None:  # 队列关闭信号
                break
                
            try:
                # 处理任务
                logger.info(f"Processing {priority.name} task: {task.get('file_path', 'unknown')}")
                
                # 这里应该调用实际的处理逻辑
                # 模拟处理
                await asyncio.sleep(0.1)  # 模拟处理时间
                
                logger.info(f"Completed {priority.name} task: {task.get('file_path', 'unknown')}")
                
            except Exception as e:
                logger.error(f"Error in {priority.name} worker: {str(e)}")


# 全局实例
batch_processor = BatchProcessor()
streaming_processor = StreamingProcessor()
priority_queue = PriorityTaskQueue()


def get_batch_processor() -> BatchProcessor:
    """获取批处理器实例"""
    return batch_processor


def get_streaming_processor() -> StreamingProcessor:
    """获取流处理器实例"""
    return streaming_processor


def get_priority_queue() -> PriorityTaskQueue:
    """获取优先级任务队列实例"""
    return priority_queue


# 使用示例
async def example_usage():
    """使用示例"""
    
    # 模拟文件处理函数
    async def sample_processor(file_path: str) -> Dict[str, Any]:
        return {
            "file_path": file_path,
            "analysis_result": f"Analysis of {file_path}",
            "issues_found": 2,
            "processing_time": 0.5
        }
    
    # 批处理示例
    files = ["file1.py", "file2.py", "file3.py", "file4.py", "file5.py"]
    results = await batch_processor.process_batch(files, sample_processor, ProcessingPriority.HIGH)
    
    print(f"Batch processing completed: {len(results)} files")
    
    # 流式处理示例
    large_file_list = [f"file_{i}.py" for i in range(1, 201)]  # 201个文件
    stream_results = await streaming_processor.process_stream(large_file_list, sample_processor)
    
    print(f"Stream processing completed: {stream_results['processed_files']} files")


if __name__ == "__main__":
    asyncio.run(example_usage())