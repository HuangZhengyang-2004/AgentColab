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
        
        # 独立的代码运行环境（venv）
        self.code_venv_dir = self.code_dir / ".code_venv"
        self.code_venv_python = None  # 将在初始化时设置
        
        # Aider配置
        self.aider_model_map = {
            'deepseek': 'deepseek/deepseek-chat',
            'gemini': 'gemini/gemini-2.5-flash',
            'claude': 'claude-3-5-sonnet-20241022',
            'gptsapi': 'gpt-5'
        }
        
        # 初始化代码运行环境
        self._setup_code_venv()
    
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
                
                # 生成 requirements.txt
                self.logger.info("生成依赖文件 requirements.txt...")
                requirements_file = self._generate_requirements(code_result['code_file'])
                if requirements_file:
                    self.logger.info(f"✅ 依赖文件已生成: {requirements_file}")
                    self.logger.info(f"📦 请运行以下命令安装依赖:")
                    self.logger.info(f"   pip install -r {requirements_file}")
                    code_result['requirements_file'] = requirements_file
                else:
                    self.logger.warning("⚠️ 未能生成 requirements.txt，请手动检查代码依赖")
                
                # 运行代码
                self.logger.info("运行生成的代码...")
                run_result = self._run_generated_code(code_result['code_file'])
                
                # 记录本次迭代
                iteration_info = {
                    'iteration': iteration,
                    'code_file': str(code_result['code_file']),
                    'requirements_file': str(code_result.get('requirements_file', '')),
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
    
    def _setup_code_venv(self):
        """
        设置独立的代码运行环境（venv）
        这个环境专门用于运行生成的代码，不会影响用户的conda环境
        """
        try:
            # 确定 venv 的 Python 路径
            if sys.platform == 'win32':
                venv_python = self.code_venv_dir / "Scripts" / "python.exe"
            else:
                venv_python = self.code_venv_dir / "bin" / "python"
            
            # 如果 venv 不存在，创建它
            if not self.code_venv_dir.exists() or not venv_python.exists():
                self.logger.info(f"🔧 创建独立的代码运行环境: {self.code_venv_dir}")
                self.logger.info(f"   这个环境专门用于运行生成的代码，不会影响您的conda环境")
                
                # 使用系统 Python3 创建 venv（不依赖当前环境）
                import shutil
                python3_cmd = shutil.which('python3') or shutil.which('python')
                
                if not python3_cmd:
                    raise RuntimeError("未找到 python3 命令，无法创建虚拟环境")
                
                # 创建 venv
                result = subprocess.run(
                    [python3_cmd, '-m', 'venv', str(self.code_venv_dir)],
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                
                if result.returncode != 0:
                    raise RuntimeError(f"创建虚拟环境失败: {result.stderr}")
                
                self.logger.info(f"✅ 虚拟环境创建成功")
            
            # 验证 venv Python 是否存在
            if not venv_python.exists():
                raise RuntimeError(f"虚拟环境Python不存在: {venv_python}")
            
            self.code_venv_python = venv_python.resolve()
            self.logger.info(f"✅ 代码运行环境: {self.code_venv_python}")
            
        except Exception as e:
            self.logger.error(f"❌ 设置代码运行环境失败: {str(e)}")
            self.logger.warning(f"⚠️  将使用当前Python环境运行代码（可能影响您的conda环境）")
            self.code_venv_python = sys.executable
    
    def _install_dependencies_to_venv(self, requirements_file: Path) -> bool:
        """
        将依赖安装到独立的 venv 环境中
        
        Args:
            requirements_file: requirements.txt 文件路径
            
        Returns:
            是否安装成功
        """
        if not self.code_venv_python or not self.code_venv_python.exists():
            self.logger.error("代码运行环境未正确设置")
            return False
        
        try:
            self.logger.info(f"📦 正在安装依赖到代码运行环境...")
            self.logger.info(f"   环境: {self.code_venv_python}")
            self.logger.info(f"   依赖文件: {requirements_file}")
            
            # 确定 pip 路径
            if sys.platform == 'win32':
                venv_pip = self.code_venv_dir / "Scripts" / "pip"
            else:
                venv_pip = self.code_venv_dir / "bin" / "pip"
            
            # 先升级 pip
            upgrade_result = subprocess.run(
                [str(self.code_venv_python), '-m', 'pip', 'install', '--upgrade', 'pip', '-q'],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            # 安装依赖
            install_result = subprocess.run(
                [str(self.code_venv_python), '-m', 'pip', 'install', '-r', str(requirements_file)],
                capture_output=True,
                text=True,
                timeout=300  # 5分钟超时
            )
            
            if install_result.returncode != 0:
                self.logger.error(f"❌ 依赖安装失败:")
                self.logger.error(install_result.stderr)
                return False
            
            self.logger.info(f"✅ 依赖安装成功")
            return True
            
        except subprocess.TimeoutExpired:
            self.logger.error(f"❌ 依赖安装超时（超过5分钟）")
            return False
        except Exception as e:
            self.logger.error(f"❌ 安装依赖时出错: {str(e)}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return False
    
    def _get_python_executable(self) -> str:
        """
        获取用于运行代码的Python解释器路径
        优先使用独立的代码运行环境（venv）
        
        Returns:
            Python解释器的绝对路径
        """
        # 优先使用独立的代码运行环境
        if self.code_venv_python and self.code_venv_python.exists():
            python_exe = str(self.code_venv_python)
            self.logger.info(f"✅ 使用独立的代码运行环境: {python_exe}")
            self.logger.info(f"   此环境不会影响您的conda环境")
        else:
            # 回退到当前环境
            python_exe = sys.executable
            self.logger.warning(f"⚠️  使用当前Python环境: {python_exe}")
            if 'conda' in python_exe or 'envs' in python_exe:
                self.logger.warning(f"   注意：这可能会在您的conda环境中安装依赖")
        
        return python_exe
    
    def _check_module_installed(self, module_name: str, python_exe: str) -> bool:
        """
        检查模块是否已安装
        
        Args:
            module_name: 模块名称
            python_exe: Python解释器路径
            
        Returns:
            是否已安装
        """
        try:
            result = subprocess.run(
                [python_exe, '-c', f'import {module_name}'],
                capture_output=True,
                timeout=5
            )
            return result.returncode == 0
        except:
            return False
    
    def _run_generated_code(self, code_file: Path) -> Dict:
        """
        运行生成的代码（在当前的Python环境中运行）
        
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
            
            # 获取Python解释器（使用当前环境的Python）
            python_exe = self._get_python_executable()
            
            # 检查是否有requirements.txt，如果有则自动安装到独立的venv
            requirements_file = code_file.parent / "requirements.txt"
            if requirements_file.exists():
                self.logger.info(f"📦 检测到依赖文件: {requirements_file}")
                
                # 自动安装依赖到独立的venv环境
                if self.code_venv_python and self.code_venv_python.exists():
                    # 检查依赖是否已安装
                    with open(requirements_file, 'r', encoding='utf-8') as f:
                        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
                    
                    missing_deps = []
                    for req in requirements:
                        pkg_name = req.split('>=')[0].split('==')[0].split('<')[0].split('>')[0].strip()
                        if not self._check_module_installed(pkg_name, python_exe):
                            missing_deps.append(pkg_name)
                    
                    if missing_deps:
                        self.logger.info(f"🔧 检测到缺失的依赖: {', '.join(missing_deps)}")
                        self.logger.info(f"   正在自动安装到代码运行环境...")
                        
                        # 自动安装
                        if self._install_dependencies_to_venv(requirements_file):
                            self.logger.info(f"✅ 依赖安装完成，可以运行代码了")
                        else:
                            self.logger.error(f"❌ 依赖安装失败，代码可能无法运行")
                            self.logger.error(f"   请手动运行: {python_exe} -m pip install -r {requirements_file}")
                    else:
                        self.logger.info(f"✅ 所有依赖已安装")
                else:
                    self.logger.warning(f"⚠️  代码运行环境未设置，无法自动安装依赖")
                    self.logger.warning(f"   请手动运行: pip install -r {requirements_file}")
            
            # 运行代码（使用当前Python环境）
            result = subprocess.run(
                [python_exe, str(abs_code_file)],
                capture_output=True,
                text=True,
                timeout=300,  # 5分钟超时
                cwd=str(abs_code_dir),  # 在代码目录下运行
                env=os.environ.copy()  # 继承当前环境变量（包括虚拟环境）
            )
            
            # 记录输出
            self.logger.info(f"代码输出:\n{result.stdout}")
            
            if result.returncode != 0:
                error_msg = result.stderr
                self.logger.error(f"代码执行失败:\n{error_msg}")
                
                # 检查是否是模块缺失错误
                if 'ModuleNotFoundError' in error_msg or 'ImportError' in error_msg:
                    # 提取缺失的模块名
                    import re
                    module_match = re.search(r"No module named ['\"]([^'\"]+)['\"]", error_msg)
                    if module_match:
                        missing_module = module_match.group(1)
                        self.logger.error(f"\n{'='*60}")
                        self.logger.error(f"❌ 缺少依赖模块: {missing_module}")
                        self.logger.error(f"{'='*60}")
                        self.logger.error(f"📦 请运行以下命令安装依赖:")
                        self.logger.error(f"   pip install {missing_module}")
                        
                        # 检查是否有requirements.txt
                        requirements_file = code_file.parent / "requirements.txt"
                        if requirements_file.exists():
                            self.logger.error(f"   或安装所有依赖:")
                            self.logger.error(f"   pip install -r {requirements_file}")
                        self.logger.error(f"{'='*60}\n")
                
                return {
                    'success': False,
                    'error': error_msg,
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
    
    def _generate_requirements(self, code_file: Path) -> Optional[Path]:
        """
        分析生成的代码，生成 requirements.txt 文件
        
        Args:
            code_file: 生成的代码文件路径
            
        Returns:
            requirements.txt 文件路径，如果生成失败则返回 None
        """
        try:
            # 读取代码文件
            with open(code_file, 'r', encoding='utf-8') as f:
                code_content = f.read()
            
            # 使用LLM分析代码依赖
            from utils.api_client import get_llm_client
            
            llm_client = get_llm_client(
                api_provider=self.api_provider,
                model=self.model,
                temperature=0.3,  # 降低温度以获得更准确的依赖列表
                max_tokens=1024
            )
            
            prompt = f"""分析以下Python代码，提取所有需要安装的第三方库（不包括标准库）。

代码：
```python
{code_content[:5000]}  # 限制长度避免token过多
```

要求：
1. 只列出第三方库（不是Python标准库）
2. 使用标准的包名（例如：numpy, pandas, matplotlib）
3. 每行一个包名
4. 如果可能，指定最低版本号（例如：numpy>=1.20.0）
5. 不要包含任何解释性文字，只输出包名列表

直接输出requirements.txt格式的内容："""
            
            self.logger.info("正在分析代码依赖...")
            response = llm_client.generate(prompt=prompt)
            
            # 提取requirements内容
            requirements_content = self._extract_requirements_from_response(response)
            
            if not requirements_content:
                # 如果LLM分析失败，尝试简单的正则提取
                requirements_content = self._extract_imports_simple(code_content)
            
            if not requirements_content:
                self.logger.warning("无法提取依赖信息")
                return None
            
            # 保存 requirements.txt
            requirements_file = code_file.parent / "requirements.txt"
            with open(requirements_file, 'w', encoding='utf-8') as f:
                f.write(requirements_content)
            
            return requirements_file
            
        except Exception as e:
            self.logger.error(f"生成requirements.txt失败: {str(e)}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return None
    
    def _extract_requirements_from_response(self, response: str) -> str:
        """
        从LLM响应中提取requirements内容
        
        Args:
            response: LLM的响应文本
            
        Returns:
            提取的requirements内容
        """
        import re
        
        # 尝试提取代码块中的内容
        code_blocks = re.findall(r'```(?:txt|text|requirements)?\n(.*?)```', response, re.DOTALL)
        if code_blocks:
            content = code_blocks[0].strip()
            return content
        
        # 如果没有代码块，尝试提取所有非空行（可能是直接的包列表）
        lines = [line.strip() for line in response.split('\n') if line.strip()]
        # 过滤掉明显的解释性文字
        valid_lines = []
        for line in lines:
            # 跳过明显的解释性文字
            if any(skip in line.lower() for skip in ['要求', '注意', '说明', '以下是', '如下', '：', ':', '```']):
                continue
            # 如果行看起来像包名（字母、数字、下划线、连字符、点、>=、==）
            if re.match(r'^[a-zA-Z0-9_\-\.]+(?:[>=<]+[0-9\.]+)?$', line):
                valid_lines.append(line)
        
        if valid_lines:
            return '\n'.join(valid_lines)
        
        return ""
    
    def _extract_imports_simple(self, code_content: str) -> str:
        """
        使用简单方法从代码中提取import语句（备用方案）
        
        Args:
            code_content: 代码内容
            
        Returns:
            requirements格式的字符串
        """
        import re
        
        # 标准库列表（不需要安装）
        stdlib_modules = {
            'os', 'sys', 'json', 'datetime', 'time', 'random', 'math', 'collections',
            'itertools', 'functools', 'operator', 'string', 're', 'pathlib', 'io',
            'csv', 'urllib', 'http', 'email', 'base64', 'hashlib', 'pickle', 'copy',
            'typing', 'dataclasses', 'enum', 'abc', 'contextlib', 'threading', 'multiprocessing',
            'subprocess', 'logging', 'warnings', 'traceback', 'inspect', 'pdb'
        }
        
        # 提取所有 import 语句
        import_pattern = r'^(?:from\s+(\S+)\s+)?import\s+(\S+)'
        imports = set()
        
        for line in code_content.split('\n'):
            match = re.match(import_pattern, line.strip())
            if match:
                module = match.group(1) or match.group(2)
                if module:
                    # 提取主包名（例如：numpy -> numpy, matplotlib.pyplot -> matplotlib）
                    main_package = module.split('.')[0]
                    if main_package not in stdlib_modules:
                        imports.add(main_package)
        
        if imports:
            # 按字母顺序排序
            sorted_imports = sorted(imports)
            return '\n'.join(sorted_imports)
        
        return ""
    
    def _save_result(self, result: Dict):
        """保存生成结果"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        result_file = self.code_dir / f"generation_result_{timestamp}.json"
        
        # 添加时间戳
        result['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"结果已保存: {result_file}")
