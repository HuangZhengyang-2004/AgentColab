"""
监督Agent - 评价生成代码的指标质量
"""
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

from agents.base_agent import BaseAgent
from utils.api_client import get_llm_client
from utils.config_loader import config_loader


class SupervisorAgent(BaseAgent):
    """
    监督Agent
    
    功能：
    1. 读取代码生成的指标
    2. 使用LLM评价指标的合理性
    3. 生成评价报告
    4. 留下详细日志
    """
    
    def __init__(self):
        super().__init__("监督Agent")
        
        # 加载配置（使用与代码生成相同的配置）
        pipeline_config = self.config_loader.config.get('pipeline', {})
        supervisor_config = pipeline_config.get('code_generation', {})
        
        self.api_provider = supervisor_config.get('api_provider', 'deepseek')
        self.model = supervisor_config.get('model', 'deepseek-chat')
        self.temperature = supervisor_config.get('temperature', 0.7)
        self.max_tokens = supervisor_config.get('max_tokens', 4096)
        
        # 初始化LLM客户端
        self.llm_client = get_llm_client(
            api_provider=self.api_provider,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        
        # 目录配置
        dirs = self.config_loader.config.get('directories', {})
        self.code_dir = Path(dirs.get('code', 'data/code'))
        self.logs_dir = Path('logs')
        
        # 创建必要的目录
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        # 评价提示词
        self.evaluation_prompt = """你是一个资深的科研评审专家，负责评价实验结果的质量和合理性。

请根据以下信息，对实验指标进行评价：

【实验指标】
{metrics}

【评价要求】
1. 指标的合理性：指标值是否在合理范围内
2. 指标的完整性：是否包含了必要的评价指标
3. 指标的可信度：结果是否可信，是否有异常值
4. 改进建议：如何改进实验设计或指标选择

请直接输出评价内容，使用Markdown格式，包含以下部分：

## 总体评价
（简要总结实验质量，给出1-10分的评分）

## 指标分析
（逐个分析各项指标）

## 存在的问题
（指出潜在的问题和异常）

## 改进建议
（提供具体的改进建议）

请开始评价。
"""
    
    def run(self, metrics_file: str = None, code_file: str = None) -> Dict:
        """
        运行监督评价流程
        
        Args:
            metrics_file: 指标文件路径
            code_file: 代码文件路径（用于上下文）
            
        Returns:
            评价结果字典
        """
        self.log_start("监督评价")
        
        try:
            # 1. 加载指标
            if not metrics_file:
                # 查找最新的metrics文件
                metrics_file = self._find_latest_metrics()
            
            if not metrics_file or not Path(metrics_file).exists():
                self.logger.warning("未找到指标文件，跳过评价")
                return {
                    'success': False,
                    'error': '未找到指标文件'
                }
            
            self.logger.info(f"加载指标文件: {metrics_file}")
            metrics = self._load_metrics(metrics_file)
            
            # 2. 使用LLM评价指标
            self.logger.info("使用LLM评价指标...")
            evaluation = self._evaluate_metrics(metrics)
            
            # 3. 保存评价结果
            result = {
                'success': True,
                'metrics_file': str(metrics_file),
                'code_file': str(code_file) if code_file else None,
                'metrics': metrics,
                'evaluation': evaluation,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            self._save_evaluation(result)
            
            self.log_end("监督评价完成")
            return result
            
        except Exception as e:
            self.log_error(f"监督评价失败: {str(e)}")
            raise
    
    def _find_latest_metrics(self) -> Optional[str]:
        """查找最新的metrics文件"""
        metrics_files = list(self.code_dir.glob("metrics.json"))
        if not metrics_files:
            return None
        
        # 返回最新的文件
        latest = max(metrics_files, key=lambda p: p.stat().st_mtime)
        return str(latest)
    
    def _load_metrics(self, metrics_file: str) -> Dict:
        """加载指标文件"""
        with open(metrics_file, 'r', encoding='utf-8') as f:
            metrics = json.load(f)
        
        self.logger.info(f"成功加载指标: {list(metrics.keys())}")
        return metrics
    
    def _evaluate_metrics(self, metrics: Dict) -> str:
        """
        使用LLM评价指标
        
        Args:
            metrics: 指标字典
            
        Returns:
            评价内容（Markdown格式）
        """
        # 格式化指标
        metrics_str = json.dumps(metrics, ensure_ascii=False, indent=2)
        
        # 准备提示词
        prompt = self.evaluation_prompt.format(metrics=metrics_str)
        
        # 调用LLM
        self.logger.info(f"调用LLM评价指标（API: {self.api_provider}, Model: {self.model}）")
        
        system_prompt = "你是资深的科研评审专家。直接输出评价内容，不要客套话。"
        
        evaluation = self.llm_client.generate(
            prompt=prompt,
            system_prompt=system_prompt
        )
        
        self.logger.info(f"✅ 评价完成，长度: {len(evaluation)} 字符")
        return evaluation
    
    def _save_evaluation(self, result: Dict):
        """保存评价结果"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 保存JSON格式
        json_file = self.code_dir / f"evaluation_{timestamp}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        # 保存Markdown格式（仅评价内容）
        md_file = self.code_dir / f"evaluation_{timestamp}.md"
        with open(md_file, 'w', encoding='utf-8') as f:
            f.write(f"# 实验评价报告\n\n")
            f.write(f"**时间**: {result['timestamp']}\n\n")
            f.write(f"**指标文件**: {result['metrics_file']}\n\n")
            if result.get('code_file'):
                f.write(f"**代码文件**: {result['code_file']}\n\n")
            f.write(f"---\n\n")
            f.write(result['evaluation'])
        
        self.logger.info(f"评价结果已保存:")
        self.logger.info(f"  - JSON: {json_file}")
        self.logger.info(f"  - Markdown: {md_file}")

