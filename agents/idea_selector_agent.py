"""
最优想法筛选Agent
负责从生成的ideas中找出最高分的idea及其来源论文
"""

from typing import Dict, List, Optional

from agents.base_agent import BaseAgent


class IdeaSelectorAgent(BaseAgent):
    """最优想法筛选Agent"""
    
    def __init__(self):
        """初始化最优想法筛选Agent"""
        super().__init__("最优想法筛选Agent")
    
    def run(self, ideas: List[Dict[str, any]] = None) -> Dict[str, any]:
        """
        筛选最优idea
        
        Args:
            ideas: ideas列表，如果为None则从ideas目录读取
            
        Returns:
            最优idea的详细信息
        """
        self.log_start("筛选最优想法")
        
        try:
            # 如果未提供ideas，则从文件读取
            if ideas is None:
                ideas = self._load_ideas()
            
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
    
    def _load_ideas(self) -> List[Dict[str, any]]:
        """
        从ideas目录加载生成的ideas
        
        Returns:
            ideas列表
        """
        try:
            ideas = self.file_manager.load_json('generated_ideas.json', 'ideas')
            self.logger.info(f"加载了 {len(ideas)} 个ideas")
            return ideas
        except Exception as e:
            self.log_error(f"加载ideas失败: {str(e)}")
            return []
    
    def get_top_n_ideas(self, ideas: List[Dict[str, any]] = None, n: int = 3) -> List[Dict[str, any]]:
        """
        获取评分最高的前N个ideas
        
        Args:
            ideas: ideas列表
            n: 要获取的数量
            
        Returns:
            前N个ideas
        """
        self.log_start(f"获取评分最高的前{n}个想法")
        
        if ideas is None:
            ideas = self._load_ideas()
        
        if not ideas:
            return []
        
        # 按分数排序
        sorted_ideas = sorted(ideas, key=lambda x: x.get('score', 0), reverse=True)
        top_ideas = sorted_ideas[:n]
        
        self.logger.info(f"返回前{len(top_ideas)}个ideas")
        for i, idea in enumerate(top_ideas, 1):
            self.logger.info(f"  {i}. {idea['title']} (分数: {idea['score']})")
        
        return top_ideas
    
    def filter_ideas_by_score(self, ideas: List[Dict[str, any]] = None, 
                             min_score: int = 70) -> List[Dict[str, any]]:
        """
        筛选评分高于阈值的ideas
        
        Args:
            ideas: ideas列表
            min_score: 最低分数阈值
            
        Returns:
            符合条件的ideas
        """
        self.log_start(f"筛选评分高于{min_score}的想法")
        
        if ideas is None:
            ideas = self._load_ideas()
        
        filtered_ideas = [idea for idea in ideas if idea.get('score', 0) >= min_score]
        
        self.logger.info(f"找到 {len(filtered_ideas)} 个符合条件的ideas")
        
        return filtered_ideas

