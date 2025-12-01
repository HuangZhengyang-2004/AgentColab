"""
想法详细化Agent
负责将最优想法结合相关论文进行详细化
"""

from typing import Dict, Optional
import json
from pathlib import Path

from agents.base_agent import BaseAgent
from agents.idea_selector_agent import IdeaSelectorAgent
from utils.api_client import UnifiedLLMClient


class IdeaDetailerAgent(BaseAgent):
    """想法详细化Agent - 可配置使用不同的LLM"""
    
    def __init__(self, api_provider: str = None, model: str = None):
        """
        初始化想法详细化Agent
        
        Args:
            api_provider: API提供商，None则从配置读取
            model: 模型名称，None则从配置读取
        """
        super().__init__("想法详细化Agent")
        
        # 从配置读取API设置
        if api_provider is None:
            api_provider = self.config_loader.get('pipeline.idea_detailing.api_provider', 'gemini')
        if model is None:
            model = self.config_loader.get('pipeline.idea_detailing.model', 'gemini-2.5-flash')
        
        temperature = self.config_loader.get('pipeline.idea_detailing.temperature', 0.7)
        max_tokens = self.config_loader.get('pipeline.idea_detailing.max_tokens', 8192)
        
        self.llm_client = UnifiedLLMClient(
            api_provider=api_provider,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        self.logger.info(f"使用 {api_provider} API, 模型: {model}")
        
        # 初始化筛选Agent
        self.selector = IdeaSelectorAgent()
    
    def run(self, best_idea: Dict = None, source_papers: Dict = None) -> str:
        """
        详细化最优想法
        
        Args:
            best_idea: 最优想法信息，None则自动筛选
            source_papers: 相关论文内容，None则自动加载
            
        Returns:
            详细化后的想法文本
        """
        self.log_start("详细化想法")
        
        try:
            # 如果未提供最优想法，则自动筛选
            if best_idea is None:
                self.logger.info("自动筛选最优想法...")
                best_idea = self.selector.run()
            
            if not best_idea:
                self.logger.warning("未找到最优想法")
                return ""
            
            # 如果未提供相关论文，则自动加载
            if source_papers is None:
                paper_keys = best_idea.get('source_papers', [])
                self.logger.info(f"加载相关论文: {paper_keys}")
                source_papers = self.selector.get_source_papers_content(paper_keys)
            
            if not source_papers:
                self.logger.warning("未找到相关论文")
                return ""
            
            # 构建输入文本
            input_text = self._format_input_for_llm(source_papers, best_idea)
            
            # 调用LLM详细化
            self.logger.info("正在详细化想法...")
            detailed_idea = self._detail_idea(input_text, len(source_papers))
            
            # 保存结果
            self.save_result(
                detailed_idea,
                'detailed_idea.md',
                'ideas',
                format='text'
            )
            
            # 同时保存JSON格式（包含元数据）
            result_data = {
                'original_idea': best_idea,
                'source_papers': list(source_papers.keys()),
                'detailed_content': detailed_idea
            }
            self.save_result(
                result_data,
                'detailed_idea.json',
                'ideas',
                format='json'
            )
            
            self.log_end("想法详细化完成")
            return detailed_idea
            
        except Exception as e:
            self.log_error(f"详细化想法失败: {str(e)}")
            raise
    
    def _format_input_for_llm(self, source_papers: Dict[str, dict], best_idea: Dict) -> str:
        """
        格式化输入给LLM
        
        格式:
        【文章1】论文名1
        内容1...
        
        【文章2】论文名2
        内容2...
        
        【基于以上文章的创新想法】
        想法内容...
        
        Args:
            source_papers: {paper_key: {'name': ..., 'content': ...}}
            best_idea: 最优想法信息
            
        Returns:
            格式化的文本
        """
        formatted_text = ""
        
        # 按paper_key排序
        sorted_keys = sorted(source_papers.keys(), 
                           key=lambda x: int(x.split('_')[1]) if '_' in x else 0)
        
        # 添加论文内容
        for i, paper_key in enumerate(sorted_keys, 1):
            paper_data = source_papers[paper_key]
            name = paper_data.get('name', paper_key)
            content = paper_data.get('content', '')
            
            formatted_text += f"【文章{i}】{name}\n\n"
            formatted_text += f"{content}\n\n"
            formatted_text += "=" * 80 + "\n\n"
        
        # 添加想法
        formatted_text += "【基于以上文章的创新想法】\n\n"
        formatted_text += f"**标题**: {best_idea.get('title', '')}\n\n"
        formatted_text += f"**创新性评分**: {best_idea.get('score', 0)}/100\n\n"
        formatted_text += f"**想法描述**:\n{best_idea.get('full_content', '')}\n\n"
        
        return formatted_text
    
    def _detail_idea(self, input_text: str, n_papers: int) -> str:
        """
        调用LLM详细化想法
        
        Args:
            input_text: 格式化的输入文本
            n_papers: 论文数量
            
        Returns:
            详细化后的想法
        """
        # 构建prompt
        prompt = f"""{input_text}

我先给你{n_papers}篇文章，然后再给你根据这{n_papers}篇文章结合产生的idea，最后你把这个idea详细化。

要求：
1. 直接输出详细化的内容，不要开场白
2. 详细化应包含：
   - 研究背景与动机
   - 核心创新点的深入阐述
   - 详细的技术实现方案（算法步骤、数学模型）
   - 实验设计与验证方案
   - 预期贡献与影响
   - 可能的挑战与解决方案
3. 使用Markdown格式
4. 对于涉及的公式，使用$$公式$$格式
5. 确保内容详实、逻辑清晰、可操作性强
"""
        
        # 调用LLM
        try:
            result = self.llm_client.generate(
                prompt=prompt,
                system_prompt="你是研究方案详细化助手。基于提供的论文和创新想法，生成详细的研究方案。直接输出内容，不要客套话。"
            )
            return result
            
        except Exception as e:
            self.log_error(f"LLM调用失败: {str(e)}")
            raise
