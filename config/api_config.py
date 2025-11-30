"""
API配置管理模块
管理所有API的密钥和配置参数
"""

import os
from typing import Dict, Any


class APIConfig:
    """API配置管理类"""
    
    def __init__(self, config_loader=None):
        """
        初始化API配置
        
        Args:
            config_loader: 配置加载器实例，如果为None则使用默认配置
        """
        self.config_loader = config_loader
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """
        加载配置
        
        Returns:
            配置字典
        """
        # 如果有配置加载器，从中读取配置
        if self.config_loader:
            return {
                'gemini': self.config_loader.get_api_config('gemini'),
                'deepseek': self.config_loader.get_api_config('deepseek'),
                'claude': self.config_loader.get_api_config('claude'),
                'mineru': self.config_loader.get_api_config('mineru'),
            }
        
        # 否则使用默认配置
        return {
            # Google Gemini API
            'gemini': {
                'api_key': os.getenv('GOOGLE_API_KEY', ''),
                'model': 'gemini-2.5-flash',
                'temperature': 0.7,
                'top_p': 1,
                'top_k': 1,
                'max_output_tokens': 8192,
            },
            
            # DeepSeek API
            'deepseek': {
                'api_key': os.getenv('DEEPSEEK_API_KEY', ''),
                'base_url': 'https://api.deepseek.com',
                'model': 'deepseek-chat',
                'temperature': 0.7,
                'max_tokens': 4096,
            },
            
            # Claude API (Anthropic)
            'claude': {
                'api_key': os.getenv('ANTHROPIC_API_KEY', ''),
                'model': 'claude-3-5-sonnet-20241022',
                'temperature': 0.7,
                'max_tokens': 4096,
            },
            
            # MinerU API
            'mineru': {
                'api_key': os.getenv('MINERU_API_KEY', ''),
                'base_url': 'https://api.mineru.com',
                'timeout': 300,  # 5分钟超时
            }
        }
    
    def get_api_config(self, api_name: str) -> Dict[str, Any]:
        """
        获取指定API的配置
        
        Args:
            api_name: API名称 (gemini/deepseek/claude/mineru)
            
        Returns:
            API配置字典
        """
        if api_name not in self.config:
            raise ValueError(f"未知的API名称: {api_name}")
        return self.config[api_name].copy()
    
    def validate_api_keys(self) -> Dict[str, bool]:
        """
        验证所有API密钥是否已设置
        
        Returns:
            各API密钥的验证状态字典
        """
        status = {}
        for api_name, config in self.config.items():
            status[api_name] = bool(config.get('api_key'))
        return status
    
    def update_config(self, api_name: str, updates: Dict[str, Any]) -> None:
        """
        更新指定API的配置
        
        Args:
            api_name: API名称
            updates: 要更新的配置项字典
        """
        if api_name not in self.config:
            raise ValueError(f"未知的API名称: {api_name}")
        self.config[api_name].update(updates)
    
    def reload(self):
        """重新加载配置"""
        self.config = self._load_config()


# 全局配置实例（延迟初始化，在主程序中会用config_loader重新初始化）
api_config = APIConfig()

