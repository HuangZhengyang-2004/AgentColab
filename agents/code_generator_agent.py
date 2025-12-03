"""
代码生成Agent - 使用Aider自动生成和调试代码
"""
import os
import sys
import subprocess
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple
import traceback

from agents.base_agent import BaseAgent
from utils.config_loader import config_loader


class CodeGeneratorAgent(BaseAgent):
    """
    代码生成Agent
    
    功能：
    1. 读取详细化的idea
    2. 使用aider-chat生成Python代码
    3. 自动运行代码
    4. 如果报错 -> 提交到GitHub并记录
    5. 如果成功 -> 生成指标表和图表
    """
    
    def __init__(self):
        super().__init__("代码生成Agent")
        
        # 加载配置
        pipeline_config = self.config_loader.config.get('pipeline', {})
        code_gen_config = pipeline_config.get('code_generation', {})
        
        self.api_provider = code_gen_config.get('api_provider', 'deepseek')
        self.model = code_gen_config.get('model', 'deepseek-chat')
        self.temperature = code_gen_config.get('temperature', 0.7)
        self.max_tokens = code_gen_config.get('max_tokens', 4096)
        
        # 目录配置
        dirs = self.config_loader.config.get('directories', {})
        self.code_dir = Path(dirs.get('code', 'data/code'))
        self.ideas_dir = Path(dirs.get('ideas', 'data/ideas'))
        self.logs_dir = Path('logs')
        
        # 创建必要的目录
        self.code_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Aider配置
        self.aider_model_map = {
            'deepseek': 'deepseek/deepseek-chat',
            'gemini': 'gemini/gemini-2.5-flash',
            'claude': 'claude-3-5-sonnet-20241022',
            'gptsapi': 'gpt-5'
        }
    
    def run(self, detailed_idea_path: str = None, max_iterations: int = 3) -> Dict:
        """
        运行代码生成流程（支持自动调试迭代）
        
        Args:
            detailed_idea_path: 详细化idea的文件路径
            max_iterations: 最大迭代次数（默认3次）
            
        Returns:
            包含生成结果的字典
        """
        self.log_start("代码生成")
        
        try:
            # 1. 加载详细化的idea
            if not detailed_idea_path:
                detailed_idea_path = self.ideas_dir / "detailed_idea.md"
            
            self.logger.info(f"加载详细化idea: {detailed_idea_path}")
            idea_content = self._load_detailed_idea(detailed_idea_path)
            
            # 迭代历史
            iterations = []
            current_prompt = self._prepare_aider_prompt(idea_content)
            
            # 2. 迭代生成和调试
            for iteration in range(1, max_iterations + 1):
                self.logger.info(f"\n{'='*60}")
                self.logger.info(f"🔄 迭代 {iteration}/{max_iterations}")
                self.logger.info(f"{'='*60}")
                
                # 生成代码
                self.logger.info("使用LLM生成代码...")
                code_result = self._generate_code_with_aider(current_prompt)
                
                if not code_result['success']:
                    self.logger.error(f"代码生成失败: {code_result['error']}")
                    iterations.append({
                        'iteration': iteration,
                        'stage': 'generation_failed',
                        'error': code_result['error']
                    })
                    break
                
                # 运行代码
                self.logger.info("运行生成的代码...")
                run_result = self._run_generated_code(code_result['code_file'])
                
                # 记录本次迭代
                iteration_info = {
                    'iteration': iteration,
                    'code_file': str(code_result['code_file']),
                    'success': run_result['success']
                }
                
                if run_result['success']:
                    # 成功！
                    self.logger.info(f"✅ 迭代{iteration}: 代码运行成功！")
                    iteration_info.update({
                        'stage': 'completed',
                        'output': run_result['output'],
                        'metrics_file': run_result.get('metrics_file'),
                        'figures': run_result.get('figures', [])
                    })
                    iterations.append(iteration_info)
                    
                    # 保存结果
                    result = {
                        'success': True,
                        'iterations': iterations,
                        'final_iteration': iteration,
                        **iteration_info
                    }
                    self._save_result(result)
                    self.log_end("代码生成完成")
                    return result
                
                else:
                    # 失败，尝试调试
                    self.logger.error(f"❌ 迭代{iteration}: 代码运行失败")
                    error_output = run_result['error']
                    
                    iteration_info.update({
                        'stage': 'execution_failed',
                        'error': error_output
                    })
                    
                    # 记录错误日志
                    log_result = self._submit_to_github(
                        code_result['code_file'],
                        error_output
                    )
                    iteration_info['error_log'] = log_result.get('error_log')
                    
                    iterations.append(iteration_info)
                    
                    # 如果还有迭代次数，尝试调试修复
                    if iteration < max_iterations:
                        self.logger.info(f"🔧 尝试自动修复（剩余{max_iterations - iteration}次机会）...")
                        
                        # 使用DebugAgent分析错误
                        from agents.debug_agent import DebugAgent
                        debug_agent = DebugAgent()
                        
                        error_analysis = debug_agent.analyze_error(
                            code_result['code_file'],
                            error_output
                        )
                        
                        # 如果可自动修复，生成新的prompt
                        if error_analysis['auto_fixable']:
                            self.logger.info(f"✅ 错误可自动修复: {error_analysis['fix_strategy']}")
                            
                            # 准备修复提示词
                            current_prompt = self._prepare_fix_prompt(
                                idea_content,
                                code_result['code_file'],
                                error_analysis
                            )
                            
                            # 增加温度参数以获得更多样化的输出
                            self.temperature = min(0.9, self.temperature + 0.1 * iteration)
                            self.logger.info(f"🌡️  调整温度参数: {self.temperature}")
                        else:
                            self.logger.warning("⚠️  错误需要人工介入，停止迭代")
                            break
                    else:
                        self.logger.warning(f"⚠️  已达到最大迭代次数({max_iterations})，停止尝试")
            
            # 所有迭代都失败了
            result = {
                'success': False,
                'iterations': iterations,
                'final_iteration': len(iterations),
                'error': '所有迭代都未能成功运行代码',
                'stage': 'max_iterations_reached'
            }
            self._save_result(result)
            self.log_end("代码生成完成（未成功）")
            return result
            
        except Exception as e:
            self.log_error(f"代码生成失败: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise
    
    def _load_detailed_idea(self, idea_path: Path) -> str:
        """加载详细化的idea"""
        if not Path(idea_path).exists():
            raise FileNotFoundError(f"未找到详细化idea文件: {idea_path}")
        
        with open(idea_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        self.logger.info(f"成功加载idea，长度: {len(content)} 字符")
        return content
    
    def _prepare_aider_prompt(self, idea_content: str) -> str:
        """
        准备Aider提示词
        
        Args:
            idea_content: 详细化的idea内容
            
        Returns:
            完整的提示词
        """
        prompt = f"""这是我的一个idea产生的文章，请根据这篇文章帮我用python完整复现一下去产生相应可运行的代码。

要求：
1. 代码必须是完整的、可运行的Python脚本
2. 包含所有必要的import语句
3. 实现文章中描述的核心算法和方法
4. 生成评价指标（如准确率、F1分数、MSE等）
5. 使用matplotlib绘制评价指标的图表
6. 将指标保存到JSON文件（metrics.json）
7. 将图表保存为PNG文件（figure_*.png）
8. 代码要有详细的注释
9. 使用try-except处理可能的错误
10. 在最后打印"实验完成！"

请直接生成代码，不要有多余的解释。

---

【Idea内容】

{idea_content}

---

请开始生成代码。
"""
        return prompt
    
    def _prepare_fix_prompt(self, idea_content: str, failed_code_file: Path, error_analysis: Dict) -> str:
        """
        准备修复提示词（优化版：只提供错误行附近代码）
        
        Args:
            idea_content: 详细化的idea内容
            failed_code_file: 失败的代码文件
            error_analysis: 错误分析结果
            
        Returns:
            修复提示词
        """
        # 读取失败的代码
        with open(failed_code_file, 'r', encoding='utf-8') as f:
            failed_code_lines = f.readlines()
        
        error_type = error_analysis['error_type']
        error_details = error_analysis['error_details']
        error_line = error_details.get('error_line', 0)
        error_message = error_details.get('error_message', '未知')
        
        # 提取错误行附近的代码（±10行）
        context_lines = 10
        start_line = max(0, error_line - context_lines - 1)
        end_line = min(len(failed_code_lines), error_line + context_lines)
        
        error_context = ''.join(failed_code_lines[start_line:end_line])
        
        # 标记错误行
        error_line_content = failed_code_lines[error_line - 1].rstrip() if error_line > 0 else "未知"
        
        prompt = f"""⚠️ 代码存在{error_type}错误，必须修复！

【错误详情】
错误类型: {error_type}
错误行号: 第{error_line}行
错误消息: {error_message}
错误代码: {error_line_content}

【错误位置的代码上下文】（第{start_line + 1}行 到 第{end_line}行）
```python
{error_context}
```

【问题分析】
第{error_line}行的代码有语法错误：{error_message}

【修复要求】
1. ⚠️ 必须修复第{error_line}行的{error_type}错误
2. ⚠️ 不要只是复制原代码，必须真正修改错误部分
3. 生成完整的、可运行的Python代码
4. 保持原有功能和逻辑不变
5. 包含所有必要的import语句
6. 实现以下功能：
   - 实现论文中的核心算法
   - 生成评价指标并保存到metrics.json
   - 使用matplotlib绘制图表并保存为PNG文件
   - 在最后打印"实验完成！"

【原始需求（简要）】
{idea_content[:500]}...

⚠️ 重要提示：
- 这是第N次尝试，之前的代码都有相同的错误
- 请仔细检查第{error_line}行，确保语法正确
- 如果是括号/引号未闭合，请补全
- 如果是缺少冒号，请添加
- 如果是try块缺少except，请添加except块

请直接输出修复后的完整代码，不要有任何解释文字。
"""
        return prompt
    
    def _generate_code_with_aider(self, prompt: str) -> Dict:
        """
        使用LLM直接生成代码（不使用Aider CLI，因为其在某些环境下不稳定）
        
        Args:
            prompt: 提示词
            
        Returns:
            生成结果字典
        """
        try:
            from utils.api_client import get_llm_client
            
            # 准备输出文件
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            code_file = self.code_dir / f"generated_code_{timestamp}.py"
            
            # 初始化LLM客户端（每次都重新创建以使用最新的温度参数）
            self.logger.info(f"使用LLM生成代码: {self.api_provider}/{self.model}, 温度: {self.temperature}")
            llm_client = get_llm_client(
                api_provider=self.api_provider,
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            # 系统提示词
            system_prompt = """你是一个专业的Python代码生成专家。
根据用户提供的研究方案，生成完整的、可运行的Python代码。

要求：
1. 只输出Python代码，不要有任何解释性文字
2. 代码必须完整且可直接运行
3. 包含所有必要的import语句
4. 使用try-except处理错误
5. 生成评价指标并保存到metrics.json
6. 使用matplotlib绘制图表并保存为PNG
7. 在最后打印"实验完成！"

直接输出代码，以```python开始，以```结束。"""
            
            # 调用LLM生成代码
            self.logger.info("正在调用LLM生成代码...")
            response = llm_client.generate(
                prompt=prompt,
                system_prompt=system_prompt
            )
            
            # 提取代码块
            code = self._extract_code_from_response(response)
            
            if not code:
                return {
                    'success': False,
                    'error': 'LLM未生成有效的代码块'
                }
            
            # 保存代码到文件
            with open(code_file, 'w', encoding='utf-8') as f:
                f.write(code)
            
            self.logger.info(f"✅ 代码生成成功: {code_file}")
            self.logger.info(f"代码长度: {len(code)} 字符")
            
            return {
                'success': True,
                'code_file': code_file,
                'llm_response': response[:500]  # 保存前500字符
            }
            
        except Exception as e:
            self.logger.error(f"代码生成异常: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {
                'success': False,
                'error': str(e)
            }
    
    def _extract_code_from_response(self, response: str) -> str:
        """
        从LLM响应中提取代码块
        
        Args:
            response: LLM的响应文本
            
        Returns:
            提取的Python代码（不含markdown标记）
        """
        import re
        
        # 尝试提取```python ... ```代码块
        code_blocks = re.findall(r'```python\n(.*?)```', response, re.DOTALL)
        if code_blocks:
            code = code_blocks[0].strip()
            self.logger.info(f"✅ 从```python块提取代码，长度: {len(code)}")
            return code
        
        # 尝试提取``` ... ```代码块（不带语言标记）
        code_blocks = re.findall(r'```\n(.*?)```', response, re.DOTALL)
        if code_blocks:
            code = code_blocks[0].strip()
            self.logger.info(f"✅ 从```块提取代码，长度: {len(code)}")
            return code
        
        # 如果响应以```python开头但没有结束标记，移除开头的标记
        if response.startswith('```python'):
            code = response.replace('```python\n', '', 1).replace('```python', '', 1)
            # 移除可能的结尾```
            if code.endswith('```'):
                code = code.rsplit('```', 1)[0]
            code = code.strip()
            self.logger.info(f"✅ 移除markdown标记后提取代码，长度: {len(code)}")
            return code
        
        # 如果响应以```开头，移除标记
        if response.startswith('```'):
            code = response.replace('```\n', '', 1).replace('```', '', 1)
            if code.endswith('```'):
                code = code.rsplit('```', 1)[0]
            code = code.strip()
            self.logger.info(f"✅ 移除```标记后提取代码，长度: {len(code)}")
            return code
        
        # 如果没有代码块标记，检查是否整个响应都是代码
        if 'import ' in response and ('def ' in response or 'class ' in response or 'if __name__' in response):
            code = response.strip()
            self.logger.info(f"✅ 直接使用响应作为代码，长度: {len(code)}")
            return code
        
        self.logger.warning("⚠️ 无法从响应中提取有效代码")
        return ""
    
    def _run_generated_code(self, code_file: Path) -> Dict:
        """
        运行生成的代码
        
        Args:
            code_file: 代码文件路径
            
        Returns:
            运行结果字典
        """
        try:
            self.logger.info(f"运行代码: {code_file}")
            
            # 转换为绝对路径
            abs_code_file = Path(code_file).resolve()
            abs_code_dir = abs_code_file.parent
            
            self.logger.info(f"工作目录: {abs_code_dir}")
            self.logger.info(f"代码文件（绝对路径）: {abs_code_file}")
            
            # 运行代码
            result = subprocess.run(
                [sys.executable, str(abs_code_file)],
                capture_output=True,
                text=True,
                timeout=300,  # 5分钟超时
                cwd=str(abs_code_dir)  # 在代码目录下运行
            )
            
            # 记录输出
            self.logger.info(f"代码输出:\n{result.stdout}")
            
            if result.returncode != 0:
                self.logger.error(f"代码执行失败:\n{result.stderr}")
                return {
                    'success': False,
                    'error': result.stderr,
                    'output': result.stdout
                }
            
            # 检查生成的文件
            metrics_file = code_file.parent / "metrics.json"
            figures = list(code_file.parent.glob("figure_*.png"))
            
            return {
                'success': True,
                'output': result.stdout,
                'metrics_file': str(metrics_file) if metrics_file.exists() else None,
                'figures': [str(f) for f in figures]
            }
            
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'error': '代码执行超时（5分钟）'
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def _submit_to_github(self, code_file: Path, error: str) -> Dict:
        """
        记录错误日志（不提交到GitHub，只保存本地）
        
        Args:
            code_file: 代码文件路径
            error: 错误信息
            
        Returns:
            日志记录结果
        """
        try:
            # 创建错误日志文件
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            error_log = self.logs_dir / f"code_error_{timestamp}.log"
            
            with open(error_log, 'w', encoding='utf-8') as f:
                f.write(f"代码文件: {code_file}\n")
                f.write(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"\n错误信息:\n{error}\n")
            
            self.logger.info(f"✅ 错误日志已保存: {error_log}")
            
            return {
                'success': True,
                'error_log': str(error_log),
                'message': '错误日志已保存到本地'
            }
                
        except Exception as e:
            self.logger.error(f"保存错误日志失败: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _save_result(self, result: Dict):
        """保存生成结果"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        result_file = self.code_dir / f"generation_result_{timestamp}.json"
        
        # 添加时间戳
        result['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"结果已保存: {result_file}")
