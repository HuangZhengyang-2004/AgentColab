"""
代码生成Agent
负责调用Claude API根据详细化的idea生成Python代码实现
"""

from agents.base_agent import BaseAgent
from utils.api_client import UnifiedLLMClient
from config.prompts import prompts


class CodeGeneratorAgent(BaseAgent):
    """代码生成Agent - 可配置使用不同的LLM"""
    
    def __init__(self, api_provider: str = None, model: str = None):
        """
        初始化代码生成Agent
        
        Args:
            api_provider: API提供商，None则从配置读取
            model: 模型名称，None则从配置读取
        """
        super().__init__("代码生成Agent")
        
        # 从配置读取API设置
        if api_provider is None:
            api_provider = self.config_loader.get('pipeline.code_generation.api_provider', 'claude')
        if model is None:
            model = self.config_loader.get('pipeline.code_generation.model', 'claude-3-5-sonnet-20241022')
        
        temperature = self.config_loader.get('pipeline.code_generation.temperature', 0.3)
        max_tokens = self.config_loader.get('pipeline.code_generation.max_tokens', 4096)
        
        self.llm_client = UnifiedLLMClient(
            api_provider=api_provider,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        self.logger.info(f"使用 {api_provider} API, 模型: {model}")
    
    def run(self, detailed_idea: str = None) -> str:
        """
        生成Python代码
        
        Args:
            detailed_idea: 详细化的idea文章，如果为None则从文件读取
            
        Returns:
            生成的Python代码
        """
        self.log_start("生成Python代码实现")
        
        try:
            # 如果未提供详细idea，则从文件读取
            if detailed_idea is None:
                detailed_idea = self._load_detailed_idea()
            
            if not detailed_idea:
                raise ValueError("未找到详细化的idea")
            
            # 调用Claude API生成代码
            self.logger.info("正在生成代码...")
            code = self._generate_code(detailed_idea)
            
            # 保存代码
            self.save_result(
                code,
                'generated_implementation.py',
                'code',
                format='text'
            )
            
            self.logger.info(f"✓ 代码生成完成，长度: {len(code)} 字符")
            self.log_end("生成Python代码实现")
            
            return code
            
        except Exception as e:
            self.log_error(f"生成代码失败: {str(e)}")
            raise
    
    def _load_detailed_idea(self) -> str:
        """
        加载详细化的idea
        
        Returns:
            详细化的idea文本
        """
        try:
            detailed_idea = self.file_manager.load_text('detailed_idea.txt', 'ideas')
            self.logger.info(f"加载详细化idea，长度: {len(detailed_idea)} 字符")
            return detailed_idea
        except Exception as e:
            self.log_error(f"加载详细化idea失败: {str(e)}")
            return ""
    
    def _generate_code(self, idea_detail: str) -> str:
        """
        调用Claude生成代码
        
        Args:
            idea_detail: 详细化的idea
            
        Returns:
            生成的代码
        """
        prompt = prompts.format_prompt(
            prompts.CODE_GENERATION,
            idea_detail=idea_detail
        )
        
        system_prompt = """你是一个专业的Python开发工程师，擅长将学术论文中的算法转化为高质量的代码实现。
你的代码应该：
1. 结构清晰，模块化设计
2. 包含详细的注释和文档字符串
3. 遵循PEP 8编码规范
4. 使用类型提示
5. 包含错误处理
6. 可以直接运行
"""
        
        result = self.llm_client.generate(
            prompt=prompt,
            system_prompt=system_prompt
        )
        
        return result

