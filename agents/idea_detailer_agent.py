"""
想法详细化Agent
负责将最优idea和对应论文输入Gemini API进行详细阐述
"""

from typing import Dict, List

from agents.base_agent import BaseAgent
from utils.api_client import get_gemini_client
from config.prompts import prompts


class IdeaDetailerAgent(BaseAgent):
    """想法详细化Agent"""
    
    def __init__(self):
        """初始化想法详细化Agent"""
        super().__init__("想法详细化Agent")
        self.gemini_client = get_gemini_client()
    
    def run(self, best_idea: Dict[str, any] = None, 
            paper_contents: Dict[str, str] = None) -> str:
        """
        详细化idea
        
        Args:
            best_idea: 最优idea字典，如果为None则从文件读取
            paper_contents: 相关论文内容字典，如果为None则自动加载
            
        Returns:
            详细化的idea文章
        """
        self.log_start("详细化创新想法")
        
        try:
            # 如果未提供best_idea，则从文件读取
            if best_idea is None:
                best_idea = self._load_best_idea()
            
            if not best_idea:
                raise ValueError("未找到最优idea")
            
            # 获取来源论文内容
            source_papers = best_idea.get('source_papers', [])
            if paper_contents is None:
                paper_contents = self._load_source_papers(source_papers)
            
            # 格式化论文内容
            papers_text = self._format_papers(paper_contents)
            
            # 格式化idea内容
            idea_text = self._format_idea(best_idea)
            
            # 调用Gemini API详细化
            self.logger.info("正在详细化想法...")
            detailed_idea = self._detail_idea(papers_text, idea_text)
            
            # 保存详细化结果
            self.save_result(
                detailed_idea,
                'detailed_idea.txt',
                'ideas',
                format='text'
            )
            
            self.logger.info(f"✓ 想法详细化完成，长度: {len(detailed_idea)} 字符")
            self.log_end("详细化创新想法")
            
            return detailed_idea
            
        except Exception as e:
            self.log_error(f"详细化想法失败: {str(e)}")
            raise
    
    def _load_best_idea(self) -> Dict[str, any]:
        """
        加载最优idea
        
        Returns:
            最优idea字典
        """
        try:
            best_idea = self.file_manager.load_json('best_idea.json', 'ideas')
            self.logger.info(f"加载最优idea: {best_idea.get('title', 'Unknown')}")
            return best_idea
        except Exception as e:
            self.log_error(f"加载最优idea失败: {str(e)}")
            return {}
    
    def _load_source_papers(self, source_papers: List[str]) -> Dict[str, str]:
        """
        加载来源论文的内容
        
        Args:
            source_papers: 来源论文列表（如 ["Paper 1", "Paper 3"]）
            
        Returns:
            论文内容字典
        """
        papers = {}
        
        # 从cleaned目录加载所有清洗后的论文
        cleaned_files = self.file_manager.list_files('cleaned', '*.txt')
        
        for file_path in cleaned_files:
            paper_name = file_path.stem.replace('_cleaned', '')
            
            # 检查是否是来源论文
            # 支持 "Paper 1: name" 或 "name" 格式
            is_source = any(
                paper_name in source or source.split(':')[-1].strip() in paper_name
                for source in source_papers
            )
            
            if is_source or not source_papers:  # 如果没有指定来源，加载所有论文
                text = self.file_manager.load_text(file_path.name, 'cleaned')
                papers[paper_name] = text
        
        self.logger.info(f"加载了 {len(papers)} 篇相关论文")
        return papers
    
    def _format_papers(self, paper_contents: Dict[str, str]) -> str:
        """
        格式化论文内容
        
        Args:
            paper_contents: 论文内容字典
            
        Returns:
            格式化的文本
        """
        formatted_parts = []
        
        for i, (paper_name, content) in enumerate(paper_contents.items(), 1):
            formatted_parts.append(f"【论文 {i}: {paper_name}】\n{content}\n")
        
        return "\n" + "="*80 + "\n\n".join(formatted_parts)
    
    def _format_idea(self, idea: Dict[str, any]) -> str:
        """
        格式化idea
        
        Args:
            idea: idea字典
            
        Returns:
            格式化的文本
        """
        return f"""
标题: {idea.get('title', 'Unknown')}
创新性评分: {idea.get('score', 0)}
来源论文: {', '.join(idea.get('source_papers', []))}

详细描述:
{idea.get('description', '')}
"""
    
    def _detail_idea(self, papers_content: str, idea_content: str) -> str:
        """
        调用Gemini详细化idea
        
        Args:
            papers_content: 格式化的论文内容
            idea_content: 格式化的idea内容
            
        Returns:
            详细化的idea文章
        """
        prompt = prompts.format_prompt(
            prompts.IDEA_DETAILING,
            papers_content=papers_content,
            idea_content=idea_content
        )
        
        result = self.gemini_client.generate(prompt)
        return result

