"""
配置加载模块
从config.yaml加载配置信息
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict


class ConfigLoader:
    """配置加载器"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        初始化配置加载器
        
        Args:
            config_path: 配置文件路径
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """
        加载配置文件
        
        Returns:
            配置字典
        """
        if not self.config_path.exists():
            print(f"警告: 配置文件 {self.config_path} 不存在，使用默认配置")
            return self._get_default_config()
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config if config else self._get_default_config()
        except Exception as e:
            print(f"警告: 加载配置文件失败 ({str(e)})，使用默认配置")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """
        获取默认配置
        
        Returns:
            默认配置字典
        """
        return {
            'api_keys': {
                'google_api_key': '',
                'deepseek_api_key': '',
                'anthropic_api_key': '',
                'mineru_api_key': '',
            },
            'api': {
                'gemini': {
                    'model': 'gemini-2.5-flash',
                    'temperature': 0.7,
                    'top_p': 1,
                    'top_k': 1,
                    'max_output_tokens': 8192,
                },
                'deepseek': {
                    'base_url': 'https://api.deepseek.com',
                    'model': 'deepseek-chat',
                    'temperature': 0.7,
                    'max_tokens': 4096,
                },
                'claude': {
                    'model': 'claude-3-5-sonnet-20241022',
                    'temperature': 0.7,
                    'max_tokens': 4096,
                },
                'mineru': {
                    'base_url': 'https://api.mineru.com',
                    'timeout': 300,
                }
            },
            'directories': {
                'base_dir': 'data',
                'input': 'data/input',
                'extracted': 'data/extracted',
                'cleaned': 'data/cleaned',
                'analyzed': 'data/analyzed',
                'ideas': 'data/ideas',
                'code': 'data/code',
                'logs': 'logs',
            },
            'logging': {
                'level': 'INFO',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'date_format': '%Y-%m-%d %H:%M:%S',
                'file_prefix': 'agentcolab',
            },
            'pipeline': {
                'auto_save': True,
                'continue_on_error': True,
                'pdf_extraction': {
                    'use_mineru': False,
                    'fallback_to_pypdf2': True,
                },
                'paper_cleaning': {
                    'enabled': True,
                },
                'paper_analysis': {
                    'do_translation': True,
                    'do_summary': True,
                },
                'idea_generation': {
                    'min_ideas': 3,
                    'score_threshold': 60,
                }
            }
        }
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        获取配置值（支持点号路径）
        
        Args:
            key_path: 配置键路径，如 "api.gemini.model"
            default: 默认值
            
        Returns:
            配置值
        """
        keys = key_path.split('.')
        value = self.config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def get_api_config(self, api_name: str) -> Dict[str, Any]:
        """
        获取指定API的配置
        
        Args:
            api_name: API名称
            
        Returns:
            API配置字典
        """
        api_config = self.get(f'api.{api_name}', {})
        
        # 添加API密钥（优先级：环境变量 > 配置文件）
        env_key_map = {
            'gemini': ('GOOGLE_API_KEY', 'google_api_key'),
            'deepseek': ('DEEPSEEK_API_KEY', 'deepseek_api_key'),
            'claude': ('ANTHROPIC_API_KEY', 'anthropic_api_key'),
            'mineru': ('MINERU_API_KEY', 'mineru_api_key'),
        }
        
        if api_name in env_key_map:
            env_var, config_key = env_key_map[api_name]
            
            # 优先使用环境变量
            api_key = os.getenv(env_var, '')
            
            # 如果环境变量为空，尝试从配置文件读取
            if not api_key:
                api_key = self.get(f'api_keys.{config_key}', '')
            
            api_config['api_key'] = api_key
        
        return api_config
    
    def get_directory(self, dir_name: str) -> str:
        """
        获取目录路径
        
        Args:
            dir_name: 目录名称
            
        Returns:
            目录路径
        """
        return self.get(f'directories.{dir_name}', f'data/{dir_name}')
    
    def reload(self):
        """重新加载配置文件"""
        self.config = self._load_config()


# 全局配置加载器实例
config_loader = ConfigLoader()

