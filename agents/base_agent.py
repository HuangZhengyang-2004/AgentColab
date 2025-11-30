"""
基础Agent类
所有具体Agent的父类，提供通用功能
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from datetime import datetime

from utils.logger import logger
from utils.file_manager import file_manager


class BaseAgent(ABC):
    """Agent基类"""
    
    def __init__(self, agent_name: str):
        """
        初始化Agent
        
        Args:
            agent_name: Agent名称
        """
        self.agent_name = agent_name
        self.logger = logger
        self.file_manager = file_manager
        self.created_at = datetime.now()
        
        self.logger.info(f"初始化Agent: {self.agent_name}")
    
    @abstractmethod
    def run(self, *args, **kwargs) -> Any:
        """
        执行Agent的主要任务
        子类必须实现此方法
        
        Returns:
            任务执行结果
        """
        pass
    
    def log_start(self, task_description: str):
        """
        记录任务开始
        
        Args:
            task_description: 任务描述
        """
        self.logger.info(f"[{self.agent_name}] 开始任务: {task_description}")
    
    def log_end(self, task_description: str, success: bool = True):
        """
        记录任务结束
        
        Args:
            task_description: 任务描述
            success: 是否成功
        """
        status = "成功" if success else "失败"
        self.logger.info(f"[{self.agent_name}] 任务{status}: {task_description}")
    
    def log_error(self, error_message: str):
        """
        记录错误信息
        
        Args:
            error_message: 错误信息
        """
        self.logger.error(f"[{self.agent_name}] 错误: {error_message}")
    
    def save_result(self, result: Any, filename: str, dir_name: str, 
                   format: str = 'text') -> str:
        """
        保存任务结果
        
        Args:
            result: 要保存的结果
            filename: 文件名
            dir_name: 目录名称
            format: 保存格式 ('text' 或 'json')
            
        Returns:
            保存的文件路径
        """
        try:
            if format == 'text':
                file_path = self.file_manager.save_text(str(result), filename, dir_name)
            elif format == 'json':
                file_path = self.file_manager.save_json(result, filename, dir_name)
            else:
                raise ValueError(f"不支持的保存格式: {format}")
            
            self.logger.info(f"[{self.agent_name}] 结果已保存到: {file_path}")
            return str(file_path)
            
        except Exception as e:
            self.log_error(f"保存结果失败: {str(e)}")
            raise
    
    def load_data(self, filename: str, dir_name: str, format: str = 'text') -> Any:
        """
        加载数据
        
        Args:
            filename: 文件名
            dir_name: 目录名称
            format: 数据格式 ('text' 或 'json')
            
        Returns:
            加载的数据
        """
        try:
            if format == 'text':
                return self.file_manager.load_text(filename, dir_name)
            elif format == 'json':
                return self.file_manager.load_json(filename, dir_name)
            else:
                raise ValueError(f"不支持的数据格式: {format}")
                
        except Exception as e:
            self.log_error(f"加载数据失败: {str(e)}")
            raise
    
    def __str__(self) -> str:
        """返回Agent的字符串表示"""
        return f"{self.agent_name} (创建时间: {self.created_at.strftime('%Y-%m-%d %H:%M:%S')})"

