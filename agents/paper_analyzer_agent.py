"""
论文分析Agent
负责调用Gemini API对论文进行翻译、分析和总结
"""

from typing import Dict, List
from pathlib import Path

from agents.base_agent import BaseAgent
from utils.api_client import get_gemini_client
from config.prompts import prompts


class PaperAnalyzerAgent(BaseAgent):
    """论文分析Agent"""
    
    def __init__(self):
        """初始化论文分析Agent"""
        super().__init__("论文分析Agent")
        self.gemini_client = get_gemini_client()
    
    def run(self, paper_texts: Dict[str, str] = None) -> Dict[str, Dict[str, str]]:
        """
        分析论文内容
        
        Args:
            paper_texts: 论文文本字典 {paper_name: text}
                        如果为None则从cleaned目录读取
            
        Returns:
            分析结果字典 {paper_name: {"translation": ..., "summary": ...}}
        """
        self.log_start("批量分析论文内容")
        
        try:
            # 如果未提供论文文本，则从cleaned目录读取
            if paper_texts is None:
                paper_texts = self._load_cleaned_papers()
            
            if not paper_texts:
                self.logger.warning("未找到任何清洗后的论文")
                return {}
            
            results = {}
            
            # 逐个分析论文
            for paper_name, text in paper_texts.items():
                self.logger.info(f"正在分析论文: {paper_name}")
                
                try:
                    # 步骤1: 翻译和分析
                    translation = self._translate_and_analyze(text)
                    self.logger.info(f"  ✓ 翻译分析完成")
                    
                    # 步骤2: 总结核心
                    summary = self._summarize_core(text)
                    self.logger.info(f"  ✓ 核心总结完成")
                    
                    # 保存分析结果
                    analysis_result = {
                        "translation": translation,
                        "summary": summary
                    }
                    
                    output_filename = f"{paper_name}_analysis.json"
                    self.save_result(
                        analysis_result,
                        output_filename,
                        'analyzed',
                        format='json'
                    )
                    
                    results[paper_name] = analysis_result
                    self.logger.info(f"✓ {paper_name} 分析完成")
                    
                except Exception as e:
                    self.log_error(f"分析 {paper_name} 失败: {str(e)}")
                    continue
            
            self.log_end(f"批量分析论文，成功: {len(results)}/{len(paper_texts)}")
            return results
            
        except Exception as e:
            self.log_error(f"批量分析失败: {str(e)}")
            raise
    
    def _load_cleaned_papers(self) -> Dict[str, str]:
        """
        从cleaned目录加载清洗后的论文
        
        Returns:
            论文字典
        """
        cleaned_files = self.file_manager.list_files('cleaned', '*.txt')
        papers = {}
        
        for file_path in cleaned_files:
            paper_name = file_path.stem.replace('_cleaned', '')
            text = self.file_manager.load_text(file_path.name, 'cleaned')
            papers[paper_name] = text
        
        self.logger.info(f"加载了 {len(papers)} 篇清洗后的论文")
        return papers
    
    def _translate_and_analyze(self, paper_text: str) -> str:
        """
        翻译和分析论文
        
        Args:
            paper_text: 论文文本
            
        Returns:
            翻译分析结果
        """
        prompt = prompts.format_prompt(
            prompts.PAPER_TRANSLATION_AND_ANALYSIS,
            paper_content=paper_text
        )
        
        result = self.gemini_client.generate(prompt)
        return result
    
    def _summarize_core(self, paper_text: str) -> str:
        """
        总结论文核心
        
        Args:
            paper_text: 论文文本
            
        Returns:
            核心总结
        """
        prompt = prompts.format_prompt(
            prompts.PAPER_CORE_SUMMARY,
            paper_content=paper_text
        )
        
        result = self.gemini_client.generate(prompt)
        return result
    
    def analyze_single(self, paper_name: str, paper_text: str) -> Dict[str, str]:
        """
        分析单篇论文
        
        Args:
            paper_name: 论文名称
            paper_text: 论文文本
            
        Returns:
            分析结果字典
        """
        self.log_start(f"分析单篇论文: {paper_name}")
        
        result = self.run({paper_name: paper_text})
        
        return result.get(paper_name, {})

