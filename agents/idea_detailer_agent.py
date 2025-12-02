"""
想法详细化Agent
负责将最优想法结合相关论文进行详细化
"""

from typing import Dict, Optional, List
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
            api_provider = self.config_loader.get('pipeline.idea_detailing.api_provider', 'deepseek')
        if model is None:
            model = self.config_loader.get('pipeline.idea_detailing.model', 'deepseek-chat')
        
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
            
            # 如果未提供相关论文，则自动加载（使用分析内容）
            if source_papers is None:
                paper_keys = best_idea.get('source_papers', [])
                self.logger.info(f"加载相关论文的分析: {paper_keys}")
                source_papers = self._get_source_papers_analysis(paper_keys)
            
            if not source_papers:
                self.logger.warning("未找到相关论文")
                return ""
            
            # 调用LLM详细化（分步输入）
            self.logger.info("正在详细化想法（分步输入模式）...")
            detailed_idea = self._detail_idea(source_papers, best_idea, len(source_papers))
            
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
    
    def _get_source_papers_analysis(self, paper_keys: List[str]) -> Dict[str, dict]:
        """
        获取指定论文的分析内容（而不是清洗后的内容）
        
        Args:
            paper_keys: ['paper_1', 'paper_2', ...]
            
        Returns:
            {
                'paper_1': {'name': ..., 'analysis': ...},
                'paper_2': {'name': ..., 'analysis': ...},
                ...
            }
        """
        collection_path = Path("data/collections/all_papers_analyzed.json")
        
        if not collection_path.exists():
            self.logger.warning(f"未找到分析集合: {collection_path}")
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
                        'analysis': papers[paper_key].get('analysis', '')
                    }
                    self.logger.info(f"加载论文分析: {paper_key} ({result[paper_key]['name']})")
                else:
                    self.logger.warning(f"未找到论文: {paper_key}")
            
            return result
            
        except Exception as e:
            self.log_error(f"加载论文分析失败: {str(e)}")
            return {}
    
    def _format_input_for_llm(self, source_papers: Dict[str, dict], best_idea: Dict) -> str:
        """
        格式化输入给LLM
        
        格式:
        【Paper_1】论文名1：分析内容...
        【Paper_2】论文名2：分析内容...
        【Paper_3】论文名3：分析内容...
        
        【基于以上文章的创新想法】
        想法内容...
        
        Args:
            source_papers: {paper_key: {'name': ..., 'analysis': ...}}
            best_idea: 最优想法信息
            
        Returns:
            格式化的文本
        """
        self.logger.info("=" * 80)
        self.logger.info("📝 开始格式化输入给LLM")
        self.logger.info("=" * 80)
        
        formatted_text = ""
        
        # 按paper_key排序
        sorted_keys = sorted(source_papers.keys(), 
                           key=lambda x: int(x.split('_')[1]) if '_' in x else 0)
        
        self.logger.info(f"📚 论文数量: {len(sorted_keys)}")
        self.logger.info(f"📚 论文列表: {sorted_keys}")
        self.logger.info("")
        
        # 添加论文分析内容（按【Paper_i】格式）
        for paper_key in sorted_keys:
            paper_data = source_papers[paper_key]
            name = paper_data.get('name', paper_key)
            analysis = paper_data.get('analysis', '')
            
            # 提取paper编号
            paper_num = paper_key.split('_')[1] if '_' in paper_key else '1'
            
            self.logger.info(f"📄 添加 Paper_{paper_num}: {name}")
            self.logger.info(f"   分析内容长度: {len(analysis)} 字符")
            self.logger.info(f"   分析内容预览: {analysis[:200]}...")
            self.logger.info("")
            
            formatted_text += f"【Paper_{paper_num}】{name}：\n\n{analysis}\n\n"
            formatted_text += "=" * 80 + "\n\n"
        
        # 添加想法
        idea_content = best_idea.get('full_content', '')
        self.logger.info("💡 添加最优想法")
        self.logger.info(f"   想法标题: {best_idea.get('title', 'N/A')}")
        self.logger.info(f"   想法评分: {best_idea.get('score', 'N/A')}")
        self.logger.info(f"   想法内容长度: {len(idea_content)} 字符")
        self.logger.info(f"   想法内容预览: {idea_content[:200]}...")
        self.logger.info("")
        
        formatted_text += "【基于以上文章的创新想法】\n\n"
        formatted_text += idea_content
        
        self.logger.info("=" * 80)
        self.logger.info("📊 格式化完成统计")
        self.logger.info("=" * 80)
        self.logger.info(f"总字符数: {len(formatted_text)}")
        self.logger.info(f"总行数: {formatted_text.count(chr(10))}")
        self.logger.info("")
        self.logger.info("📋 完整输入内容预览（前500字符）:")
        self.logger.info("-" * 80)
        self.logger.info(formatted_text[:500])
        self.logger.info("-" * 80)
        self.logger.info("")
        
        return formatted_text
    
    def _detail_idea(self, formatted_papers: Dict[str, dict], best_idea: Dict, n_papers: int) -> str:
        """
        调用LLM详细化想法（分步输入）
        
        Args:
            formatted_papers: {paper_key: {'name': ..., 'analysis': ...}}
            best_idea: 最优想法信息
            n_papers: 论文数量
            
        Returns:
            详细化后的想法
        """
        self.logger.info("=" * 80)
        self.logger.info("🚀 准备分步调用 LLM")
        self.logger.info("=" * 80)
        self.logger.info(f"API提供商: {self.llm_client.api_provider}")
        self.logger.info(f"模型: {self.llm_client.model}")
        self.logger.info(f"温度: {self.llm_client.temperature}")
        self.logger.info(f"最大tokens: {self.llm_client.max_tokens}")
        self.logger.info(f"输入方式: 分步输入（多轮对话）")
        self.logger.info("")
        
        system_prompt = "你是研究方案详细化助手。我会分步给你提供论文分析和创新想法，请记住所有内容，最后根据我的要求生成详细的研究方案。"
        
        try:
            # 步骤1: 任务说明
            self.logger.info("=" * 80)
            self.logger.info("📝 步骤1: 发送任务说明")
            self.logger.info("=" * 80)
            
            task_prompt = f"""我先给你{n_papers}篇文章的分析内容，然后再给你根据这{n_papers}篇文章结合产生的创新想法，最后你把这个想法详细化，涉及公式和理论要进行推导。

现在开始，我会分{n_papers + 1}步给你内容：
- 第1-{n_papers}步：逐篇给你论文分析
- 第{n_papers + 1}步：给你创新想法并要求详细化

请先回复"好的，我准备好了，请开始"。"""
            
            self.logger.info(f"任务说明内容:")
            self.logger.info("-" * 80)
            self.logger.info(task_prompt)
            self.logger.info("-" * 80)
            self.logger.info("")
            
            response1 = self.llm_client.generate(
                prompt=task_prompt,
                system_prompt=system_prompt
            )
            
            self.logger.info(f"✅ LLM响应: {response1[:200]}")
            self.logger.info("")
            
            # 步骤2-n+1: 逐篇输入论文
            sorted_keys = sorted(formatted_papers.keys(), 
                               key=lambda x: int(x.split('_')[1]) if '_' in x else 0)
            
            for step_idx, paper_key in enumerate(sorted_keys, 1):
                # 提取原始论文编号（保持与idea中一致）
                original_paper_num = int(paper_key.split('_')[1]) if '_' in paper_key else step_idx
                
                self.logger.info("=" * 80)
                self.logger.info(f"📝 步骤{step_idx + 1}: 发送 Paper_{original_paper_num}")
                self.logger.info("=" * 80)
                
                paper_data = formatted_papers[paper_key]
                name = paper_data.get('name', paper_key)
                analysis = paper_data.get('analysis', '')
                
                paper_prompt = f"""【Paper_{original_paper_num}】{name}：

{analysis}

请回复"已收到Paper_{original_paper_num}"。"""
                
                self.logger.info(f"论文Key: {paper_key} → Paper_{original_paper_num}")
                self.logger.info(f"论文名称: {name}")
                self.logger.info(f"分析内容长度: {len(analysis)} 字符")
                self.logger.info(f"分析内容预览: {analysis[:200]}...")
                self.logger.info("")
                
                # 添加重试机制（针对500错误）
                max_retries = 3
                retry_delay = 2  # 秒
                
                for attempt in range(max_retries):
                    try:
                        response = self.llm_client.generate(
                            prompt=paper_prompt,
                            system_prompt=system_prompt
                        )
                        
                        self.logger.info(f"✅ LLM响应: {response[:200]}")
                        self.logger.info("")
                        break  # 成功则跳出重试循环
                        
                    except Exception as e:
                        error_msg = str(e)
                        if "500" in error_msg and attempt < max_retries - 1:
                            self.logger.warning(f"⚠️  遇到500错误，{retry_delay}秒后重试 (第{attempt + 1}/{max_retries}次)")
                            import time
                            time.sleep(retry_delay)
                            retry_delay *= 2  # 指数退避
                        else:
                            raise  # 最后一次重试失败或非500错误，抛出异常
            
            # 最后一步: 输入想法并要求详细化
            self.logger.info("=" * 80)
            self.logger.info(f"📝 步骤{n_papers + 2}: 发送创新想法并要求详细化")
            self.logger.info("=" * 80)
            
            idea_content = best_idea.get('full_content', '')
            
            final_prompt = f"""【基于以上{n_papers}篇文章的创新想法】

{idea_content}

现在请你把这个想法详细化，要求：
1. 直接输出详细化的内容，不要开场白
2. 详细化应包含：
   - 研究背景与动机
   - 核心创新点的深入阐述
   - 详细的技术实现方案（算法步骤、数学模型）
   - 公式推导（涉及矩阵和向量请全部展开）
   - 实验设计与验证方案
   - 预期贡献与影响
   - 可能的挑战与解决方案
3. 使用Markdown格式
4. 行间公式两边用两个美元符号显示公式（$$公式$$）
5. 确保内容详实、逻辑清晰、可操作性强"""
            
            self.logger.info(f"想法标题: {best_idea.get('title', 'N/A')}")
            self.logger.info(f"想法评分: {best_idea.get('score', 'N/A')}")
            self.logger.info(f"想法内容长度: {len(idea_content)} 字符")
            self.logger.info(f"想法内容预览: {idea_content[:200]}...")
            self.logger.info("")
            
            # 添加重试机制（针对500错误）
            max_retries = 3
            retry_delay = 2
            
            for attempt in range(max_retries):
                try:
                    self.logger.info(f"⏳ 正在调用 LLM 生成详细化内容... (尝试 {attempt + 1}/{max_retries})")
                    result = self.llm_client.generate(
                        prompt=final_prompt,
                        system_prompt=system_prompt
                    )
                    break  # 成功则跳出重试循环
                    
                except Exception as e:
                    error_msg = str(e)
                    if "500" in error_msg and attempt < max_retries - 1:
                        self.logger.warning(f"⚠️  遇到500错误，{retry_delay}秒后重试")
                        import time
                        time.sleep(retry_delay)
                        retry_delay *= 2
                    else:
                        raise
            
            self.logger.info("=" * 80)
            self.logger.info("✅ LLM 详细化完成")
            self.logger.info("=" * 80)
            self.logger.info(f"响应长度: {len(result)} 字符")
            self.logger.info(f"响应预览（前500字符）:")
            self.logger.info("-" * 80)
            self.logger.info(result[:500])
            self.logger.info("-" * 80)
            self.logger.info("")
            
            return result
            
        except Exception as e:
            self.logger.error("=" * 80)
            self.logger.error("❌ LLM 调用失败")
            self.logger.error("=" * 80)
            self.log_error(f"错误信息: {str(e)}")
            self.logger.error("")
            raise
