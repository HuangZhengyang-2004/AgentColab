"""
创新想法生成Agent
负责汇总所有论文分析结果，调用LLM生成创新ideas并打分
"""

from typing import Dict, Optional
import json
from pathlib import Path

from agents.base_agent import BaseAgent
from utils.api_client import UnifiedLLMClient


class IdeaGeneratorAgent(BaseAgent):
    """创新想法生成Agent - 可配置使用不同的LLM"""
    
    def __init__(self, api_provider: str = None, model: str = None):
        """
        初始化创新想法生成Agent
        
        Args:
            api_provider: API提供商，None则从配置读取
            model: 模型名称，None则从配置读取
        """
        super().__init__("创新想法生成Agent")
        
        # 从配置读取API设置
        if api_provider is None:
            api_provider = self.config_loader.get('pipeline.idea_generation.api_provider', 'deepseek')
        if model is None:
            model = self.config_loader.get('pipeline.idea_generation.model', 'deepseek-chat')
        
        temperature = self.config_loader.get('pipeline.idea_generation.temperature', 0.8)
        max_tokens = self.config_loader.get('pipeline.idea_generation.max_tokens', 8192)
        
        self.llm_client = UnifiedLLMClient(
            api_provider=api_provider,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        self.logger.info(f"使用 {api_provider} API, 模型: {model}")
    
    def run(self, paper_analyses: Dict[str, str] = None) -> str:
        """
        生成创新ideas
        
        Args:
            paper_analyses: 论文分析字典 {paper_key: analysis_content}
                          如果为None则从analyzed集合读取
            
        Returns:
            生成的ideas文本（包含打分）
        """
        self.log_start("生成创新想法")
        
        try:
            # 如果未提供论文分析，则从analyzed集合读取
            if paper_analyses is None:
                paper_analyses = self._load_paper_analyses()
            
            if not paper_analyses:
                self.logger.warning("未找到任何论文分析")
                return ""
            
            self.logger.info(f"加载了 {len(paper_analyses)} 篇论文的分析")
            
            # 构建输入文本：【Paper_i】论文名：分析内容
            papers_text = self._format_papers_for_llm(paper_analyses)
            
            # 调用LLM生成ideas
            self.logger.info("正在生成创新想法...")
            ideas_text = self._generate_ideas(papers_text)
            
            # 保存ideas（Markdown格式）
            self.save_result(
                ideas_text,
                'generated_ideas.md',
                'ideas',
                format='text'
            )
            
            self.log_end(f"创新想法生成完成，基于 {len(paper_analyses)} 篇论文")
            return ideas_text
            
        except Exception as e:
            self.log_error(f"生成想法失败: {str(e)}")
            raise
    
    def _load_paper_analyses(self) -> Dict[str, dict]:
        """
        从analyzed集合加载论文分析
        
        Returns:
            {paper_key: {'name': ..., 'analysis': ...}}
        """
        collection_path = Path("data/collections/all_papers_analyzed.json")
        
        if not collection_path.exists():
            self.logger.warning(f"未找到分析集合: {collection_path}")
            return {}
        
        try:
            with open(collection_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            papers = data.get('papers', {})
            
            # 提取每篇论文的分析内容
            analyses = {}
            for paper_key, paper_data in papers.items():
                name = paper_data.get('name', paper_key)
                analysis = paper_data.get('analysis', '')
                analyses[paper_key] = {
                    'name': name,
                    'analysis': analysis
                }
            
            self.logger.info(f"从集合中加载了 {len(analyses)} 篇论文分析")
            return analyses
            
        except Exception as e:
            self.log_error(f"加载论文分析失败: {str(e)}")
            return {}
    
    def _format_papers_for_llm(self, paper_analyses: Dict[str, dict]) -> str:
        """
        格式化论文分析为LLM输入格式
        
        格式: 【Paper_i】论文名：分析内容
        
        Args:
            paper_analyses: {paper_key: {'name': ..., 'analysis': ...}}
            
        Returns:
            格式化的文本
        """
        formatted_text = ""
        
        # 按paper_key排序（paper_1, paper_2, ...）
        sorted_keys = sorted(paper_analyses.keys(), 
                           key=lambda x: int(x.split('_')[1]) if '_' in x else 0)
        
        for i, paper_key in enumerate(sorted_keys, 1):
            paper_data = paper_analyses[paper_key]
            name = paper_data.get('name', paper_key)
            analysis = paper_data.get('analysis', '')
            
            # 清理论文名称（去掉_analyzed后缀）
            if name.endswith('_analyzed'):
                name = name[:-9]
            
            formatted_text += f"【Paper_{i}】{name}：\n\n{analysis}\n\n"
            formatted_text += "=" * 80 + "\n\n"
        
        return formatted_text
    
    def _generate_ideas(self, papers_text: str) -> str:
        """
        调用LLM生成创新想法
        
        Args:
            papers_text: 格式化的论文分析文本
            
        Returns:
            生成的ideas文本
        """
        # 构建prompt
        prompt = f"""{papers_text}

这是我最近看的几篇文章，请尽量只根据这几篇文章的思路，帮我想几个创新性比较强的idea(尽量详细一些)，同时按照创新性对这几个idea进行打分。

要求：
1. 直接输出idea内容，不要开场白
2. 每个idea包含：
   - 标题
   - 创新性评分（0-100分）
   - 详细描述（包括核心思路、技术方案、预期效果）
3. 使用Markdown格式
4. 按创新性从高到低排序
"""
        
        # 调用LLM
        try:
            result = self.llm_client.generate(
                prompt=prompt,
                system_prompt="你是研究创新助手。基于提供的论文分析，生成创新性强的研究想法。直接输出内容，不要客套话。"
            )
            return result
            
        except Exception as e:
            self.log_error(f"LLM调用失败: {str(e)}")
            raise
