"""
论文清洗Agent
使用Python代码清洗论文，删除参考文献、致谢等无关内容
"""

import re
from pathlib import Path
from typing import Dict, Optional, List
from agents.base_agent import BaseAgent
from utils import PaperCollection


class PaperCleanerAgent(BaseAgent):
    """论文清洗Agent - 基于规则的文本清洗"""
    
    def __init__(self):
        super().__init__("论文清洗Agent")
        
        # 定义要删除的章节关键词（支持中英文）
        self.section_keywords = {
            'references': [
                'references', 'reference', 'bibliography', 'works cited',
                '参考文献', '引用文献', '文献引用'
            ],
            'acknowledgments': [
                'acknowledgments', 'acknowledgement', 'acknowledgements',
                'acknowledgment', '致谢', '鸣谢', '感谢'
            ],
            'appendix': [
                'appendix', 'appendices', 'supplementary material',
                'supplementary information', 'supplemental material',
                '附录', '补充材料', '附加材料'
            ],
            'funding': [
                'funding', 'financial support', 'grant',
                '资助', '基金', '经费支持'
            ],
            'author_contributions': [
                'author contributions', 'authors contributions',
                'contribution', 'contributions',
                '作者贡献'
            ],
            'conflict_of_interest': [
                'conflict of interest', 'conflicts of interest',
                'competing interests', 'declaration of interests',
                '利益冲突', '利益声明'
            ]
        }
    
    def _find_section_start(self, text: str, keywords: List[str]) -> Optional[int]:
        """
        查找章节开始位置
        
        Args:
            text: 论文文本
            keywords: 关键词列表
        
        Returns:
            章节开始位置（字符索引），未找到返回None
        """
        # 查找最早出现的位置
        earliest_pos = None
        
        for keyword in keywords:
            # 1. 尝试匹配独立成行的标题（前后都有换行）
            pattern1 = rf'\n\s*{re.escape(keyword)}\s*\n'
            match = re.search(pattern1, text, re.IGNORECASE)
            if match:
                pos = match.start()
                if earliest_pos is None or pos < earliest_pos:
                    earliest_pos = pos
                    continue
            
            # 2. 尝试匹配带编号的标题
            pattern2 = rf'\n\s*[\dIVXivx]+[\.\)]\s*{re.escape(keyword)}\s*\n'
            match = re.search(pattern2, text, re.IGNORECASE)
            if match:
                pos = match.start()
                if earliest_pos is None or pos < earliest_pos:
                    earliest_pos = pos
                    continue
            
            # 3. 尝试匹配带#号的markdown标题
            pattern3 = rf'\n\s*#+\s*{re.escape(keyword)}\s*\n'
            match = re.search(pattern3, text, re.IGNORECASE)
            if match:
                pos = match.start()
                if earliest_pos is None or pos < earliest_pos:
                    earliest_pos = pos
        
        return earliest_pos
    
    def _remove_section(self, text: str, section_name: str) -> str:
        """
        删除特定章节
        
        Args:
            text: 论文文本
            section_name: 章节名称（如'references'）
        
        Returns:
            删除章节后的文本
        """
        if section_name not in self.section_keywords:
            return text
        
        keywords = self.section_keywords[section_name]
        start_pos = self._find_section_start(text, keywords)
        
        if start_pos is not None:
            # 删除从找到的位置到文本末尾的所有内容
            cleaned_text = text[:start_pos].strip()
            self.logger.info(f"✓ 删除了 {section_name} 章节（从位置 {start_pos} 开始）")
            return cleaned_text
        else:
            self.logger.debug(f"未找到 {section_name} 章节")
            return text
    
    def _remove_citations(self, text: str) -> str:
        """
        删除行内引用标记
        
        Args:
            text: 论文文本
        
        Returns:
            删除引用后的文本
        """
        # 删除各种格式的引用标记
        patterns = [
            r'\[\d+\]',  # [1], [2], [123]
            r'\[\d+[-,]\d+\]',  # [1-3], [1,2]
            r'\[\d+(?:,\s*\d+)*\]',  # [1, 2, 3]
            r'\(\s*\w+\s+et al\.\s*,?\s*\d{4}\s*\)',  # (Smith et al., 2020)
            r'\(\s*\w+\s+and\s+\w+\s*,?\s*\d{4}\s*\)',  # (Smith and Jones, 2020)
            r'\(\s*\w+\s*,?\s*\d{4}\s*\)',  # (Smith, 2020)
        ]
        
        cleaned_text = text
        total_removed = 0
        
        for pattern in patterns:
            matches = re.findall(pattern, cleaned_text)
            count = len(matches)
            if count > 0:
                cleaned_text = re.sub(pattern, '', cleaned_text)
                total_removed += count
        
        if total_removed > 0:
            self.logger.info(f"✓ 删除了 {total_removed} 个引用标记")
        
        return cleaned_text
    
    def _remove_urls_and_emails(self, text: str) -> str:
        """删除URL和邮箱"""
        # 删除URL
        text = re.sub(r'https?://\S+', '', text)
        text = re.sub(r'www\.\S+', '', text)
        
        # 删除邮箱
        text = re.sub(r'\S+@\S+\.\S+', '', text)
        
        return text
    
    def _remove_extra_whitespace(self, text: str) -> str:
        """删除多余的空白字符"""
        # 删除多个连续空格
        text = re.sub(r' {2,}', ' ', text)
        
        # 删除多个连续换行（保留最多2个）
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # 删除行首行尾空格
        lines = [line.strip() for line in text.split('\n')]
        text = '\n'.join(lines)
        
        return text.strip()
    
    def _remove_page_numbers(self, text: str) -> str:
        """删除页码"""
        # 删除独立成行的数字（可能是页码）
        text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
        
        return text
    
    def clean_paper(self, text: str, remove_citations: bool = True) -> str:
        """
        清洗单篇论文
        
        Args:
            text: 论文原始文本
            remove_citations: 是否删除行内引用标记
        
        Returns:
            清洗后的文本
        """
        if not text or not text.strip():
            self.logger.warning("输入文本为空")
            return ""
        
        original_length = len(text)
        cleaned_text = text
        
        # 1. 删除各个章节（按顺序，从最后面的开始删除）
        sections_to_remove = [
            'conflict_of_interest',
            'author_contributions',
            'funding',
            'acknowledgments',
            'appendix',
            'references'
        ]
        
        for section in sections_to_remove:
            cleaned_text = self._remove_section(cleaned_text, section)
        
        # 2. 删除行内引用（可选）
        if remove_citations:
            cleaned_text = self._remove_citations(cleaned_text)
        
        # 3. 删除URL和邮箱
        cleaned_text = self._remove_urls_and_emails(cleaned_text)
        
        # 4. 删除页码
        cleaned_text = self._remove_page_numbers(cleaned_text)
        
        # 5. 清理多余空白
        cleaned_text = self._remove_extra_whitespace(cleaned_text)
        
        # 统计
        cleaned_length = len(cleaned_text)
        removed_length = original_length - cleaned_length
        removal_rate = (removed_length / original_length * 100) if original_length > 0 else 0
        
        self.logger.info(f"清洗完成：{original_length} → {cleaned_length} 字符（删除 {removal_rate:.1f}%）")
        
        return cleaned_text
    
    def run(self, papers: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """
        运行论文清洗
        
        Args:
            papers: 论文字典 {paper_key: content}，如果为None则从集合加载
        
        Returns:
            清洗后的论文字典 {paper_key: cleaned_content}
        """
        self.log_start("开始清洗论文")
        
        try:
            # 如果没有提供论文，从集合加载
            if papers is None:
                collection_path = "data/collections/all_papers.json"
                if not Path(collection_path).exists():
                    self.logger.error(f"集合文件不存在: {collection_path}")
                    return {}
                
                collection = PaperCollection.load_from_json(collection_path)
                papers = collection.get_all_contents()
                self.logger.info(f"从集合加载了 {len(papers)} 篇论文")
            
            if not papers:
                self.logger.warning("没有论文需要清洗")
                return {}
            
            # 清洗每篇论文
            cleaned_papers = {}
            
            for paper_key, content in papers.items():
                self.logger.info(f"\n{'='*60}")
                self.logger.info(f"清洗 {paper_key}")
                self.logger.info(f"{'='*60}")
                
                cleaned_content = self.clean_paper(content)
                cleaned_papers[paper_key] = cleaned_content
                
                # 保存清洗后的论文
                output_filename = f"{paper_key}_cleaned.txt"
                self.save_result(
                    cleaned_content,
                    output_filename,
                    'cleaned',
                    format='text'
                )
            
            # 创建清洗后的集合
            cleaned_collection = PaperCollection()
            for paper_key, content in cleaned_papers.items():
                # 获取原始论文名称
                if papers:
                    original_collection = PaperCollection.load_from_json("data/collections/all_papers.json")
                    paper_info = original_collection.get_paper(paper_key)
                    paper_name = paper_info['name'] if paper_info else paper_key
                else:
                    paper_name = paper_key
                
                cleaned_collection.add_paper(
                    paper_name=f"{paper_name}_cleaned",
                    content=content,
                    index=int(paper_key.split('_')[1])
                )
            
            # 保存清洗后的集合
            output_path = "data/collections/all_papers_cleaned.json"
            cleaned_collection.save_to_json(output_path)
            self.logger.info(f"\n✓ 清洗后的集合已保存: {output_path}")
            
            self.log_end("论文清洗完成")
            return cleaned_papers
            
        except Exception as e:
            self.logger.error(f"清洗失败: {str(e)}", exc_info=True)
            return {}
