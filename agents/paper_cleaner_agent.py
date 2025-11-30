"""
论文清洗Agent
负责调用DeepSeek API清理论文中的附录、参考文献等无关内容
"""

from typing import Dict, List
from pathlib import Path

from agents.base_agent import BaseAgent
from utils.api_client import get_deepseek_client
from config.prompts import prompts


class PaperCleanerAgent(BaseAgent):
    """论文清洗Agent"""
    
    def __init__(self):
        """初始化论文清洗Agent"""
        super().__init__("论文清洗Agent")
        self.deepseek_client = get_deepseek_client()
    
    def run(self, paper_texts: Dict[str, str] = None) -> Dict[str, str]:
        """
        清洗论文内容
        
        Args:
            paper_texts: 论文文本字典 {paper_name: text}
                        如果为None则从extracted目录读取
            
        Returns:
            清洗后的论文字典 {paper_name: cleaned_text}
        """
        self.log_start("批量清洗论文内容")
        
        try:
            # 如果未提供论文文本，则从extracted目录读取
            if paper_texts is None:
                paper_texts = self._load_extracted_papers()
            
            if not paper_texts:
                self.logger.warning("未找到任何论文文本")
                return {}
            
            results = {}
            
            # 逐个清洗论文
            for paper_name, text in paper_texts.items():
                self.logger.info(f"正在清洗论文: {paper_name}")
                
                try:
                    cleaned_text = self._clean_paper(text)
                    
                    # 保存清洗结果
                    output_filename = f"{paper_name}_cleaned.txt"
                    self.save_result(
                        cleaned_text,
                        output_filename,
                        'cleaned',
                        format='text'
                    )
                    
                    results[paper_name] = cleaned_text
                    self.logger.info(f"✓ {paper_name} 清洗完成")
                    
                except Exception as e:
                    self.log_error(f"清洗 {paper_name} 失败: {str(e)}")
                    continue
            
            self.log_end(f"批量清洗论文，成功: {len(results)}/{len(paper_texts)}")
            return results
            
        except Exception as e:
            self.log_error(f"批量清洗失败: {str(e)}")
            raise
    
    def _load_extracted_papers(self) -> Dict[str, str]:
        """
        从extracted目录加载提取的论文
        
        Returns:
            论文字典
        """
        extracted_files = self.file_manager.list_files('extracted', '*.txt')
        papers = {}
        
        for file_path in extracted_files:
            paper_name = file_path.stem.replace('_extracted', '')
            text = self.file_manager.load_text(file_path.name, 'extracted')
            papers[paper_name] = text
        
        self.logger.info(f"加载了 {len(papers)} 篇提取的论文")
        return papers
    
    def _clean_paper(self, paper_text: str) -> str:
        """
        清洗单篇论文
        
        Args:
            paper_text: 论文原始文本
            
        Returns:
            清洗后的文本
        """
        # 构建prompt
        prompt = prompts.format_prompt(
            prompts.PAPER_CLEANING,
            paper_content=paper_text
        )
        
        # 调用DeepSeek API
        cleaned_text = self.deepseek_client.generate(
            prompt=prompt,
            system_prompt="你是一个专业的论文处理助手，擅长清理和整理学术论文。"
        )
        
        return cleaned_text
    
    def clean_single(self, paper_name: str, paper_text: str) -> str:
        """
        清洗单篇论文
        
        Args:
            paper_name: 论文名称
            paper_text: 论文文本
            
        Returns:
            清洗后的文本
        """
        self.log_start(f"清洗单篇论文: {paper_name}")
        
        result = self.run({paper_name: paper_text})
        
        return result.get(paper_name, "")

