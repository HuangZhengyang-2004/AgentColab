"""
创新想法生成Agent
负责汇总所有论文分析结果，调用Gemini API生成创新ideas并打分
"""

from typing import Dict, List
import re

from agents.base_agent import BaseAgent
from utils.api_client import get_gemini_client
from config.prompts import prompts


class IdeaGeneratorAgent(BaseAgent):
    """创新想法生成Agent"""
    
    def __init__(self):
        """初始化创新想法生成Agent"""
        super().__init__("创新想法生成Agent")
        self.gemini_client = get_gemini_client()
    
    def run(self, paper_summaries: Dict[str, str] = None) -> List[Dict[str, any]]:
        """
        生成创新ideas
        
        Args:
            paper_summaries: 论文总结字典 {paper_name: summary}
                           如果为None则从analyzed目录读取
            
        Returns:
            ideas列表，每个idea包含: title, score, source_papers, description
        """
        self.log_start("生成创新想法")
        
        try:
            # 如果未提供论文总结，则从analyzed目录读取
            if paper_summaries is None:
                paper_summaries = self._load_paper_summaries()
            
            if not paper_summaries:
                self.logger.warning("未找到任何论文总结")
                return []
            
            # 构建汇总的论文内容
            papers_summary_text = self._format_papers_summary(paper_summaries)
            
            # 调用Gemini API生成ideas
            self.logger.info("正在生成创新想法...")
            ideas_text = self._generate_ideas(papers_summary_text)
            
            # 解析ideas
            ideas = self._parse_ideas(ideas_text)
            
            # 保存ideas
            self.save_result(
                ideas,
                'generated_ideas.json',
                'ideas',
                format='json'
            )
            
            # 同时保存原始文本
            self.save_result(
                ideas_text,
                'generated_ideas_raw.txt',
                'ideas',
                format='text'
            )
            
            self.logger.info(f"✓ 成功生成 {len(ideas)} 个创新想法")
            self.log_end("生成创新想法")
            
            return ideas
            
        except Exception as e:
            self.log_error(f"生成想法失败: {str(e)}")
            raise
    
    def _load_paper_summaries(self) -> Dict[str, str]:
        """
        从analyzed目录加载论文总结
        
        Returns:
            论文总结字典
        """
        analyzed_files = self.file_manager.list_files('analyzed', '*.json')
        summaries = {}
        
        for file_path in analyzed_files:
            paper_name = file_path.stem.replace('_analysis', '')
            analysis = self.file_manager.load_json(file_path.name, 'analyzed')
            
            # 只取summary部分
            if 'summary' in analysis:
                summaries[paper_name] = analysis['summary']
        
        self.logger.info(f"加载了 {len(summaries)} 篇论文的总结")
        return summaries
    
    def _format_papers_summary(self, paper_summaries: Dict[str, str]) -> str:
        """
        格式化论文总结为统一的文本
        
        Args:
            paper_summaries: 论文总结字典
            
        Returns:
            格式化的文本
        """
        formatted_parts = []
        
        for i, (paper_name, summary) in enumerate(paper_summaries.items(), 1):
            formatted_parts.append(f"【Paper {i}: {paper_name}】\n{summary}\n")
        
        return "\n".join(formatted_parts)
    
    def _generate_ideas(self, papers_summary: str) -> str:
        """
        调用Gemini生成ideas
        
        Args:
            papers_summary: 格式化的论文总结
            
        Returns:
            生成的ideas文本
        """
        prompt = prompts.format_prompt(
            prompts.IDEA_GENERATION,
            papers_summary=papers_summary
        )
        
        result = self.gemini_client.generate(prompt)
        return result
    
    def _parse_ideas(self, ideas_text: str) -> List[Dict[str, any]]:
        """
        解析ideas文本为结构化数据
        
        Args:
            ideas_text: AI生成的ideas文本
            
        Returns:
            ideas列表
        """
        ideas = []
        
        # 使用正则表达式匹配每个idea
        pattern = r'【Idea \d+】\s*标题：(.+?)\s*创新性评分：(\d+)\s*来源论文：(.+?)\s*详细描述：(.+?)(?=【Idea \d+】|$)'
        matches = re.finditer(pattern, ideas_text, re.DOTALL)
        
        for match in matches:
            title = match.group(1).strip()
            score = int(match.group(2).strip())
            source_papers = [p.strip() for p in match.group(3).split(',')]
            description = match.group(4).strip()
            
            ideas.append({
                'title': title,
                'score': score,
                'source_papers': source_papers,
                'description': description
            })
        
        # 如果解析失败，尝试备用方案
        if not ideas:
            self.logger.warning("无法解析ideas，使用备用方案")
            ideas = self._parse_ideas_fallback(ideas_text)
        
        # 按分数排序
        ideas.sort(key=lambda x: x['score'], reverse=True)
        
        return ideas
    
    def _parse_ideas_fallback(self, ideas_text: str) -> List[Dict[str, any]]:
        """
        备用的ideas解析方法
        
        Args:
            ideas_text: ideas文本
            
        Returns:
            ideas列表
        """
        # 简单的分块处理
        ideas = []
        idea_blocks = ideas_text.split('【Idea')
        
        for i, block in enumerate(idea_blocks[1:], 1):  # 跳过第一个空块
            ideas.append({
                'title': f"Idea {i}",
                'score': 50,  # 默认分数
                'source_papers': [],
                'description': block.strip()
            })
        
        return ideas

