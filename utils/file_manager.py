"""
文件管理模块
提供统一的文件读写和管理功能
"""

import os
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime


class FileManager:
    """文件管理器"""
    
    def __init__(self, base_dir: str = "data"):
        """
        初始化文件管理器
        
        Args:
            base_dir: 数据目录基础路径
        """
        self.base_dir = Path(base_dir)
        
        # 定义各个子目录
        self.dirs = {
            'input': self.base_dir / 'input',
            'extracted': self.base_dir / 'extracted',
            'cleaned': self.base_dir / 'cleaned',
            'analyzed': self.base_dir / 'analyzed',
            'ideas': self.base_dir / 'ideas',
            'code': self.base_dir / 'code',
        }
        
        # 确保所有目录存在
        for dir_path in self.dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def get_dir(self, dir_name: str) -> Path:
        """
        获取指定目录路径
        
        Args:
            dir_name: 目录名称
            
        Returns:
            目录路径
        """
        if dir_name not in self.dirs:
            raise ValueError(f"未知的目录名称: {dir_name}")
        return self.dirs[dir_name]
    
    def save_text(self, content: str, filename: str, dir_name: str) -> Path:
        """
        保存文本内容到文件
        
        Args:
            content: 文本内容
            filename: 文件名
            dir_name: 目录名称
            
        Returns:
            保存的文件路径
        """
        file_path = self.dirs[dir_name] / filename
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return file_path
    
    def load_text(self, filename: str, dir_name: str) -> str:
        """
        从文件加载文本内容
        
        Args:
            filename: 文件名
            dir_name: 目录名称
            
        Returns:
            文本内容
        """
        file_path = self.dirs[dir_name] / filename
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def save_json(self, data: Any, filename: str, dir_name: str) -> Path:
        """
        保存JSON数据到文件
        
        Args:
            data: 要保存的数据
            filename: 文件名
            dir_name: 目录名称
            
        Returns:
            保存的文件路径
        """
        file_path = self.dirs[dir_name] / filename
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return file_path
    
    def load_json(self, filename: str, dir_name: str) -> Any:
        """
        从文件加载JSON数据
        
        Args:
            filename: 文件名
            dir_name: 目录名称
            
        Returns:
            加载的数据
        """
        file_path = self.dirs[dir_name] / filename
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def list_files(self, dir_name: str, pattern: str = "*") -> List[Path]:
        """
        列出指定目录中的文件
        
        Args:
            dir_name: 目录名称
            pattern: 文件匹配模式（如 "*.txt"）
            
        Returns:
            文件路径列表
        """
        dir_path = self.dirs[dir_name]
        return sorted(dir_path.glob(pattern))
    
    def file_exists(self, filename: str, dir_name: str) -> bool:
        """
        检查文件是否存在
        
        Args:
            filename: 文件名
            dir_name: 目录名称
            
        Returns:
            文件是否存在
        """
        file_path = self.dirs[dir_name] / filename
        return file_path.exists()
    
    def generate_filename(self, prefix: str, extension: str = "txt") -> str:
        """
        生成带时间戳的文件名
        
        Args:
            prefix: 文件名前缀
            extension: 文件扩展名
            
        Returns:
            生成的文件名
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return f"{prefix}_{timestamp}.{extension}"


# 全局文件管理器实例
file_manager = FileManager()

