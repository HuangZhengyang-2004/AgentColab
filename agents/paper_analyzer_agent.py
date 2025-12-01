"""
论文分析Agent
负责调用DeepSeek API对论文进行分析和总结
"""

from typing import Dict, Optional
from pathlib import Path
import json

from agents.base_agent import BaseAgent
from utils.api_client import UnifiedLLMClient
from utils import PaperCollection


class PaperAnalyzerAgent(BaseAgent):
    """论文分析Agent - 可配置使用不同的LLM进行论文核心内容和算法分析"""
    
    def __init__(self, api_provider: str = None, model: str = None):
        """
        初始化论文分析Agent
        
        Args:
            api_provider: API提供商 (deepseek, gemini, claude)，None则从配置读取
            model: 模型名称，None则从配置读取
        """
        super().__init__("论文分析Agent")
        
        # 从配置读取API设置
        if api_provider is None:
            api_provider = self.config_loader.get('pipeline.paper_analysis.api_provider', 'deepseek')
        if model is None:
            model = self.config_loader.get('pipeline.paper_analysis.model', 'deepseek-chat')
        
        temperature = self.config_loader.get('pipeline.paper_analysis.temperature', 0.7)
        max_tokens = self.config_loader.get('pipeline.paper_analysis.max_tokens', 4096)
        
        # 创建统一客户端
        self.llm_client = UnifiedLLMClient(
            api_provider=api_provider,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        self.logger.info(f"使用 {api_provider} API, 模型: {model}")
        
        self.analysis_prompt = """请总结这篇论文的核心内容和算法实现逻辑。

要求：
1. 直接输出分析内容，不要任何开场白、客套话或角色扮演
2. 不要说"作为...专家"、"我将..."等表述
3. 使用Markdown格式组织内容
4. 包含以下部分：
   - 论文核心内容（主要研究问题、创新点）
   - 核心算法实现逻辑（算法原理、关键步骤）
   - 技术亮点和贡献
   - 分析和推导公式。涉及矩阵和向量请全部展开。请把公式渲染完整，行间公式两边用两个美元符号显示公式。
5. 直接从"## 论文核心内容"开始

论文内容：
{paper_content}"""
    
    def run(self, papers: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """
        分析论文内容
        
        Args:
            papers: 论文文本字典 {paper_key: content}
                   如果为None则从 data/collections/all_papers_cleaned.json 加载
            
        Returns:
            分析结果字典 {paper_key: analysis_markdown}
        """
        self.log_start("开始批量分析论文")
        
        try:
            # 1. 加载论文
            if papers is None:
                from pathlib import Path
                collection_path = Path("data/collections/all_papers_cleaned.json")
                
                if not collection_path.exists():
                    self.logger.error("未找到清洗后的论文集合，请先执行清洗步骤")
                    return {}
                
                collection = PaperCollection.load_from_json(str(collection_path))
                papers = collection.get_all_contents()
                self.logger.info(f"从集合中加载了 {len(papers)} 篇论文")
            
            if not papers:
                self.logger.warning("没有论文需要分析")
                return {}
            
            # 2. 逐个分析论文
            analysis_results = {}
            total = len(papers)
            
            for idx, (paper_key, content) in enumerate(papers.items(), 1):
                try:
                    self.logger.info(f"[{idx}/{total}] 正在分析: {paper_key}")
                    
                    # 调用DeepSeek API
                    analysis = self._analyze_paper(content)
                    
                    # 保存结果
                    analysis_results[paper_key] = analysis
                    
                    # 保存单个分析文件（Markdown格式）
                    self.save_result(
                        analysis,
                        f"{paper_key}_analysis.md",
                        'analyzed',
                        format='text'
                    )
                    
                    self.logger.info(f"  ✓ {paper_key} 分析完成")
                    
                except Exception as e:
                    self.log_error(f"分析 {paper_key} 失败: {str(e)}")
                    continue
            
            # 3. 创建分析集合
            self._create_analysis_collection(analysis_results, papers)
            
            # 4. 保存统计信息
            stats = {
                "total_papers": total,
                "successful": len(analysis_results),
                "failed": total - len(analysis_results),
                "papers": {
                    key: {
                        "length": len(analysis),
                        "status": "success"
                    } for key, analysis in analysis_results.items()
                }
            }
            
            self.save_result(
                stats,
                'analysis_stats.json',
                'analyzed',
                format='json'
            )
            
            self.log_end(f"批量分析完成，成功: {len(analysis_results)}/{total}")
            return analysis_results
            
        except Exception as e:
            self.log_error(f"批量分析失败: {str(e)}")
            raise
    
    def _analyze_paper(self, paper_content: str) -> str:
        """
        分析单篇论文
        
        Args:
            paper_content: 论文内容
            
        Returns:
            分析结果（Markdown格式）
        """
        # 构建prompt
        prompt = self.analysis_prompt.format(paper_content=paper_content)
        
        # 调用LLM API
        try:
            analysis = self.llm_client.generate(
                prompt=prompt,
                system_prompt="你是论文分析助手。直接输出分析内容，不要开场白、客套话或角色扮演。"
            )
            return analysis
            
        except Exception as e:
            self.log_error(f"LLM API调用失败: {str(e)}")
            raise
    
    def _create_analysis_collection(
        self, 
        analysis_results: Dict[str, str],
        original_papers: Dict[str, str]
    ) -> None:
        """
        创建分析结果集合
        
        Args:
            analysis_results: 分析结果字典
            original_papers: 原始论文字典
        """
        try:
            from datetime import datetime
            
            collection_data = {
                "metadata": {
                    "total_papers": len(analysis_results),
                    "created_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    "source": "PaperAnalyzerAgent",
                    "description": "DeepSeek分析的论文核心内容和算法逻辑"
                },
                "papers": {}
            }
            
            for paper_key, analysis in analysis_results.items():
                # 从paper_key尝试获取原始论文名称
                original_name = original_papers.get(paper_key, "Unknown")
                if isinstance(original_name, str) and len(original_name) > 100:
                    # 如果是内容而不是名字，尝试从cleaned collection获取
                    try:
                        from pathlib import Path
                        cleaned_collection_path = Path("data/collections/all_papers_cleaned.json")
                        
                        if cleaned_collection_path.exists():
                            cleaned_coll = PaperCollection.load_from_json(
                                str(cleaned_collection_path)
                            )
                            paper_data = cleaned_coll.get_paper(paper_key)
                            if paper_data:
                                original_name = paper_data.get('name', paper_key)
                    except:
                        original_name = paper_key
                
                collection_data["papers"][paper_key] = {
                    "name": f"{original_name}_analyzed",
                    "analysis": analysis,
                    "analysis_length": len(analysis),
                    "analyzed_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
            
            # 保存集合
            from pathlib import Path
            collections_dir = Path("data/collections")
            collections_dir.mkdir(parents=True, exist_ok=True)
            output_path = collections_dir / 'all_papers_analyzed.json'
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(collection_data, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"分析集合已保存: {output_path}")
            
        except Exception as e:
            self.log_error(f"创建分析集合失败: {str(e)}")
    
    def analyze_single(self, paper_name: str, paper_content: str) -> str:
        """
        分析单篇论文（便捷方法）
        
        Args:
            paper_name: 论文名称
            paper_content: 论文内容
            
        Returns:
            分析结果
        """
        self.log_start(f"分析单篇论文: {paper_name}")
        
        result = self.run({paper_name: paper_content})
        
        return result.get(paper_name, "")

