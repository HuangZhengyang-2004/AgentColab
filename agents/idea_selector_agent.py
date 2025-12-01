"""
最优想法筛选Agent
负责从生成的ideas中找出最高分的idea及其来源论文
"""

from typing import Dict, List, Optional
import re
import json
from pathlib import Path

from agents.base_agent import BaseAgent


class IdeaSelectorAgent(BaseAgent):
    """最优想法筛选Agent - 解析Markdown格式的ideas"""
    
    def __init__(self):
        """初始化最优想法筛选Agent"""
        super().__init__("最优想法筛选Agent")
    
    def run(self) -> Dict[str, any]:
        """
        筛选最优idea并识别使用的论文
        
        Returns:
            {
                'title': str,
                'score': int,
                'description': str,
                'source_papers': List[str],  # ['paper_1', 'paper_2', ...]
                'full_content': str
            }
        """
        self.log_start("筛选最优想法")
        
        try:
            # 从Markdown文件读取ideas
            ideas = self._load_ideas_from_markdown()
            
            if not ideas:
                self.logger.warning("未找到任何ideas")
                return {}
            
            # 找出最高分的idea
            best_idea = max(ideas, key=lambda x: x.get('score', 0))
            
            self.logger.info(f"最优Idea: {best_idea['title']}")
            self.logger.info(f"创新性评分: {best_idea['score']}")
            self.logger.info(f"来源论文: {', '.join(best_idea.get('source_papers', []))}")
            
            # 保存结果
            self.save_result(
                best_idea,
                'best_idea.json',
                'ideas',
                format='json'
            )
            
            self.log_end("筛选最优想法")
            return best_idea
            
        except Exception as e:
            self.log_error(f"筛选想法失败: {str(e)}")
            raise
    
    def _load_ideas_from_markdown(self) -> List[Dict[str, any]]:
        """
        从Markdown文件解析ideas
        
        Returns:
            ideas列表
        """
        ideas_file = Path("data/ideas/generated_ideas.md")
        
        if not ideas_file.exists():
            self.logger.warning(f"未找到ideas文件: {ideas_file}")
            return []
        
        try:
            with open(ideas_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 解析Markdown格式的ideas
            ideas = self._parse_markdown_ideas(content)
            
            self.logger.info(f"从Markdown中解析了 {len(ideas)} 个ideas")
            return ideas
            
        except Exception as e:
            self.log_error(f"读取ideas文件失败: {str(e)}")
            return []
    
    def _parse_markdown_ideas(self, content: str) -> List[Dict[str, any]]:
        """
        解析Markdown格式的ideas
        
        格式示例:
        ### Idea 1: 标题
        - **创新性评分：** 92
        - **详细描述：**
          ...
        
        Args:
            content: Markdown内容
            
        Returns:
            解析后的ideas列表
        """
        ideas = []
        
        # 按 ### Idea 或 ### **Idea 分割（兼容不同格式）
        idea_blocks = re.split(r'###\s+\*{0,2}\s*Idea\s+\d+:', content)
        
        for block in idea_blocks[1:]:  # 跳过第一个空块
            if not block.strip():
                continue
            
            idea = {}
            
            # 提取标题（第一行）
            lines = block.strip().split('\n')
            idea['title'] = lines[0].strip()
            
            # 提取评分
            score_match = re.search(r'创新性评分[：:]\s*[*\s]*(\d+)', block)
            if score_match:
                idea['score'] = int(score_match.group(1))
            else:
                idea['score'] = 0
            
            # 提取完整描述
            idea['description'] = block.strip()
            idea['full_content'] = block.strip()
            
            # 识别使用的论文（Paper_1, Paper_2, Paper_3等）
            paper_mentions = re.findall(r'Paper[_\s]*(\d+)', block)
            # 去重并排序
            paper_numbers = sorted(set(int(p) for p in paper_mentions))
            idea['source_papers'] = [f'paper_{n}' for n in paper_numbers]
            
            ideas.append(idea)
        
        return ideas
    
    def get_source_papers_content(self, paper_keys: List[str]) -> Dict[str, dict]:
        """
        获取指定论文的清洗后内容
        
        Args:
            paper_keys: ['paper_1', 'paper_2', ...]
            
        Returns:
            {
                'paper_1': {'name': ..., 'content': ...},
                'paper_2': {'name': ..., 'content': ...},
                ...
            }
        """
        collection_path = Path("data/collections/all_papers_cleaned.json")
        
        if not collection_path.exists():
            self.logger.warning(f"未找到清洗集合: {collection_path}")
            return {}
        
        try:
            with open(collection_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            papers = data.get('papers', {})
            
            # 提取指定的论文
            result = {}
            for paper_key in paper_keys:
                if paper_key in papers:
                    result[paper_key] = {
                        'name': papers[paper_key].get('name', paper_key),
                        'content': papers[paper_key].get('content', '')
                    }
                    self.logger.info(f"加载论文: {paper_key} ({result[paper_key]['name']})")
                else:
                    self.logger.warning(f"未找到论文: {paper_key}")
            
            return result
            
        except Exception as e:
            self.log_error(f"加载论文内容失败: {str(e)}")
            return {}
