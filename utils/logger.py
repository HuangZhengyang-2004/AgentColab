"""
日志管理模块
提供统一的日志记录功能
"""

import os
import logging
from datetime import datetime
from pathlib import Path


class LoggerManager:
    """日志管理器"""
    
    def __init__(self, log_dir: str = "logs"):
        """
        初始化日志管理器
        
        Args:
            log_dir: 日志文件目录
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # 创建日志文件名（按日期）
        log_filename = f"agentcolab_{datetime.now().strftime('%Y%m%d')}.log"
        self.log_file = self.log_dir / log_filename
        
        # 配置日志
        self._setup_logger()
    
    def _setup_logger(self):
        """配置日志格式和处理器"""
        # 创建logger
        self.logger = logging.getLogger('AgentColab')
        self.logger.setLevel(logging.DEBUG)
        
        # 清除已有的处理器
        self.logger.handlers.clear()
        
        # 创建格式器
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # 文件处理器
        file_handler = logging.FileHandler(self.log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        
        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        
        # 添加处理器
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def debug(self, message: str):
        """记录DEBUG级别日志"""
        self.logger.debug(message)
    
    def info(self, message: str):
        """记录INFO级别日志"""
        self.logger.info(message)
    
    def warning(self, message: str):
        """记录WARNING级别日志"""
        self.logger.warning(message)
    
    def error(self, message: str):
        """记录ERROR级别日志"""
        self.logger.error(message)
    
    def critical(self, message: str):
        """记录CRITICAL级别日志"""
        self.logger.critical(message)


# 全局日志管理器实例
logger = LoggerManager()

