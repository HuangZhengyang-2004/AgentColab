"""
API客户端封装模块
提供统一的API调用接口
"""

import os
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from anthropic import Anthropic
from openai import OpenAI
from typing import Optional, Dict, Any, List

from utils.config_loader import config_loader
from utils.logger import logger


class GeminiClient:
    """Gemini API客户端"""
    
    def __init__(self, model: str = None, temperature: float = None, max_tokens: int = None):
        """
        初始化Gemini客户端
        
        Args:
            model: 模型名称（可选，使用API配置的默认值）
            temperature: 温度参数（可选）
            max_tokens: 最大输出token数（可选）
        """
        config = config_loader.get_api_config('gemini')
        if not config.get('api_key'):
            raise ValueError("未设置GOOGLE_API_KEY，请在config.yaml中配置或设置环境变量")
        
        genai.configure(api_key=config['api_key'])
        self.model_name = model or config.get('model', 'gemini-1.5-flash')
        self.generation_config = {
            'temperature': temperature if temperature is not None else config.get('temperature', 0.7),
            'top_p': config.get('top_p', 1),
            'top_k': config.get('top_k', 1),
            'max_output_tokens': max_tokens or config.get('max_output_tokens', 8192),
        }
    
    def generate(self, prompt: str, stream: bool = False) -> str:
        """
        生成文本
        
        Args:
            prompt: 输入prompt
            stream: 是否使用流式输出
            
        Returns:
            生成的文本
        """
        try:
            # 强制关闭安全过滤（使用正确的枚举类型）
            safety_settings = {
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }
            
            model = genai.GenerativeModel(
                self.model_name,
                generation_config=self.generation_config,
                safety_settings=safety_settings  # 强制关闭安全过滤
            )
            
            logger.info(f"调用Gemini API，模型: {self.model_name}，安全过滤: 已强制关闭（BLOCK_NONE）")
            
            if stream:
                response = model.generate_content(prompt, stream=True)
                full_text = ""
                for chunk in response:
                    full_text += chunk.text
                return full_text
            else:
                response = model.generate_content(prompt)
                return response.text
                
        except Exception as e:
            logger.error(f"Gemini API调用失败: {str(e)}")
            raise
    
    def chat(self, messages: List[Dict[str, str]]) -> str:
        """
        多轮对话
        
        Args:
            messages: 对话历史，格式为 [{"role": "user", "content": "..."}]
            
        Returns:
            AI回复
        """
        try:
            model = genai.GenerativeModel(
                self.model_name,
                generation_config=self.generation_config
            )
            
            # 构建对话历史
            history = []
            for msg in messages[:-1]:  # 除了最后一条
                history.append({
                    'role': 'user' if msg['role'] == 'user' else 'model',
                    'parts': [msg['content']]
                })
            
            chat = model.start_chat(history=history)
            
            # 发送最后一条消息
            response = chat.send_message(messages[-1]['content'])
            return response.text
            
        except Exception as e:
            logger.error(f"Gemini对话失败: {str(e)}")
            raise


class DeepSeekClient:
    """DeepSeek API客户端"""
    
    def __init__(self, model: str = None, temperature: float = None, max_tokens: int = None):
        """
        初始化DeepSeek客户端
        
        Args:
            model: 模型名称（可选，使用API配置的默认值）
            temperature: 温度参数（可选）
            max_tokens: 最大输出token数（可选）
        """
        config = config_loader.get_api_config('deepseek')
        if not config.get('api_key'):
            raise ValueError("未设置DEEPSEEK_API_KEY，请在config.yaml中配置或设置环境变量")
        
        self.client = OpenAI(
            api_key=config['api_key'],
            base_url=config.get('base_url', 'https://api.deepseek.com')
        )
        self.model = model or config.get('model', 'deepseek-chat')
        self.temperature = temperature if temperature is not None else config.get('temperature', 0.7)
        self.max_tokens = max_tokens or config.get('max_tokens', 4096)
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        生成文本
        
        Args:
            prompt: 用户输入
            system_prompt: 系统提示（可选）
            
        Returns:
            生成的文本
        """
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            logger.info(f"调用DeepSeek API，模型: {self.model}")
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"DeepSeek API调用失败: {str(e)}")
            raise


class GptsApiClient:
    """GptsApi客户端 - 使用OpenAI兼容接口访问第三方Gemini服务"""
    
    def __init__(self, model: str = None, temperature: float = None, max_tokens: int = None):
        """
        初始化GptsApi客户端
        
        Args:
            model: 模型名称（可选，使用API配置的默认值）
            temperature: 温度参数（可选）
            max_tokens: 最大输出token数（可选）
        """
        config = config_loader.get_api_config('gptsapi')
        if not config.get('api_key'):
            raise ValueError("未设置GPTSAPI_KEY，请在config.yaml中配置或设置环境变量")
        
        # 使用OpenAI客户端，只需改变base_url
        self.client = OpenAI(
            api_key=config['api_key'],
            base_url=config.get('base_url', 'https://api.gptsapi.net/v1')
        )
        self.model = model or config.get('model', 'gemini-3-pro-preview')
        self.temperature = temperature if temperature is not None else config.get('temperature', 0.7)
        self.max_tokens = max_tokens or config.get('max_tokens', 8192)
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        生成文本
        
        Args:
            prompt: 用户输入
            system_prompt: 系统提示（可选）
            
        Returns:
            生成的文本
        """
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            logger.info(f"调用GptsApi (Gemini)，模型: {self.model}")
            
            # 构建请求参数
            request_params = {
                "model": self.model,
                "messages": messages,
                "max_tokens": self.max_tokens
            }
            
            # 某些模型（如gpt-5）只支持默认temperature=1，不支持自定义
            # 如果temperature不是1，则尝试传递；如果是1或接近1，则不传递（使用默认值）
            if self.temperature is not None and abs(self.temperature - 1.0) > 0.01:
                request_params["temperature"] = self.temperature
            
            response = self.client.chat.completions.create(**request_params)
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"GptsApi调用失败: {str(e)}")
            raise


class ClaudeClient:
    """Claude API客户端"""
    
    def __init__(self, model: str = None, temperature: float = None, max_tokens: int = None):
        """
        初始化Claude客户端
        
        Args:
            model: 模型名称（可选，使用API配置的默认值）
            temperature: 温度参数（可选）
            max_tokens: 最大输出token数（可选）
        """
        config = config_loader.get_api_config('claude')
        if not config.get('api_key'):
            raise ValueError("未设置ANTHROPIC_API_KEY，请在config.yaml中配置或设置环境变量")
        
        self.client = Anthropic(api_key=config['api_key'])
        self.model = model or config.get('model', 'claude-3-5-sonnet-20241022')
        self.temperature = temperature if temperature is not None else config.get('temperature', 0.7)
        self.max_tokens = max_tokens or config.get('max_tokens', 4096)
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        生成文本
        
        Args:
            prompt: 用户输入
            system_prompt: 系统提示（可选）
            
        Returns:
            生成的文本
        """
        try:
            logger.info(f"调用Claude API，模型: {self.model}")
            
            kwargs = {
                'model': self.model,
                'max_tokens': self.max_tokens,
                'temperature': self.temperature,
                'messages': [{"role": "user", "content": prompt}]
            }
            
            if system_prompt:
                kwargs['system'] = system_prompt
            
            response = self.client.messages.create(**kwargs)
            return response.content[0].text
            
        except Exception as e:
            logger.error(f"Claude API调用失败: {str(e)}")
            raise


# 创建全局客户端实例的工厂函数
def get_llm_client(api_provider: str, model: str = None, temperature: float = None, max_tokens: int = None):
    """
    根据配置获取对应的LLM客户端
    
    Args:
        api_provider: API提供商名称 (deepseek, gemini, claude, gptsapi)
        model: 模型名称（可选）
        temperature: 温度参数（可选）
        max_tokens: 最大token数（可选）
        
    Returns:
        对应的客户端实例
    """
    api_provider = api_provider.lower()
    
    if api_provider == "deepseek":
        return DeepSeekClient(model, temperature, max_tokens)
    elif api_provider == "gemini":
        return GeminiClient(model, temperature, max_tokens)
    elif api_provider == "claude":
        return ClaudeClient(model, temperature, max_tokens)
    elif api_provider == "gptsapi":
        return GptsApiClient(model, temperature, max_tokens)
    else:
        raise ValueError(f"不支持的API提供商: {api_provider}。支持的选项: deepseek, gemini, claude, gptsapi")


class UnifiedLLMClient:
    """
    统一的LLM客户端接口
    根据配置自动选择合适的API提供商
    """
    
    def __init__(self, api_provider: str, model: str = None, temperature: float = 0.7, max_tokens: int = 4096):
        """
        初始化统一客户端
        
        Args:
            api_provider: API提供商 (deepseek, gemini, claude, gptsapi)
            model: 模型名称（必须指定）
            temperature: 温度参数
            max_tokens: 最大token数
        """
        self.api_provider = api_provider.lower()
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # 传递参数给底层客户端
        self.client = get_llm_client(self.api_provider, model, temperature, max_tokens)
        
        logger.info(f"初始化LLM客户端: {self.api_provider}, 模型: {model or '默认'}")
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        生成文本（统一接口）
        
        Args:
            prompt: 用户输入
            system_prompt: 系统提示（可选）
            
        Returns:
            生成的文本
        """
        if self.api_provider == "gemini":
            # Gemini不支持system_prompt参数，需要合并到prompt中
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"
            else:
                full_prompt = prompt
            return self.client.generate(full_prompt)
        else:
            # DeepSeek, Claude, GptsApi都支持system_prompt
            return self.client.generate(prompt, system_prompt)

