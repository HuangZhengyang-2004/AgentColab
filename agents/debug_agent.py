"""
调试Agent - 分析代码错误并自动修复
"""
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional, List
from datetime import datetime

from agents.base_agent import BaseAgent
from utils.api_client import get_llm_client


class DebugAgent(BaseAgent):
    """
    调试Agent
    
    功能：
    1. 分析Python代码错误
    2. 识别错误类型（语法/导入/运行时/超时）
    3. 生成修复建议
    4. 自动修复简单错误
    5. 调用LLM修复复杂错误
    """
    
    def __init__(self):
        super().__init__("调试Agent")
        
        # 加载配置
        pipeline_config = self.config_loader.config.get('pipeline', {})
        code_gen_config = pipeline_config.get('code_generation', {})
        
        self.api_provider = code_gen_config.get('api_provider', 'deepseek')
        self.model = code_gen_config.get('model', 'deepseek-chat')
        self.temperature = code_gen_config.get('temperature', 0.7)
        self.max_tokens = code_gen_config.get('max_tokens', 4096)
        
        # 初始化LLM客户端
        self.llm_client = get_llm_client(
            api_provider=self.api_provider,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
    
    def run(self, code_file: Path, error_output: str, stdout: str = "") -> Dict:
        """
        运行调试流程：分析错误 → 修复错误
        
        Args:
            code_file: 代码文件路径
            error_output: stderr错误输出
            stdout: stdout标准输出
            
        Returns:
            修复结果
        """
        # 分析错误
        error_analysis = self.analyze_error(code_file, error_output, stdout)
        
        # 如果可自动修复，则尝试修复
        if error_analysis['auto_fixable']:
            fix_result = self.fix_error(code_file, error_analysis)
            return {
                **error_analysis,
                'fix_result': fix_result
            }
        else:
            return {
                **error_analysis,
                'fix_result': {
                    'success': False,
                    'error': '需要人工介入'
                }
            }
    
    def analyze_error(self, 
                     code_file: Path, 
                     error_output: str, 
                     stdout: str = "") -> Dict:
        """
        分析代码错误
        
        Args:
            code_file: 代码文件路径
            error_output: stderr错误输出
            stdout: stdout标准输出
            
        Returns:
            错误分析结果
        """
        self.log_start("错误分析")
        
        try:
            # 读取代码
            with open(code_file, 'r', encoding='utf-8') as f:
                code = f.read()
            
            # 识别错误类型
            error_type = self._identify_error_type(error_output)
            
            # 提取错误详情
            error_details = self._extract_error_details(error_output, error_type)
            
            # 判断是否可自动修复
            auto_fixable, fix_strategy = self._is_auto_fixable(error_type, error_details)
            
            result = {
                'error_type': error_type,
                'error_details': error_details,
                'auto_fixable': auto_fixable,
                'fix_strategy': fix_strategy,
                'code_file': str(code_file),
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            self.logger.info(f"错误类型: {error_type}")
            self.logger.info(f"可自动修复: {auto_fixable}")
            self.logger.info(f"修复策略: {fix_strategy}")
            
            self.log_end("错误分析完成")
            return result
            
        except Exception as e:
            self.log_error(f"错误分析失败: {str(e)}")
            raise
    
    def fix_error(self, 
                  code_file: Path, 
                  error_analysis: Dict) -> Dict:
        """
        修复代码错误
        
        Args:
            code_file: 代码文件路径
            error_analysis: 错误分析结果
            
        Returns:
            修复结果
        """
        self.log_start("错误修复")
        
        try:
            error_type = error_analysis['error_type']
            fix_strategy = error_analysis['fix_strategy']
            
            if fix_strategy == 'install_dependency':
                # 自动安装依赖
                result = self._fix_dependency(error_analysis['error_details'])
            
            elif fix_strategy == 'fix_syntax':
                # 使用LLM修复语法错误
                result = self._fix_with_llm(code_file, error_analysis)
            
            elif fix_strategy == 'fix_runtime':
                # 使用LLM修复运行时错误
                result = self._fix_with_llm(code_file, error_analysis)
            
            elif fix_strategy == 'optimize_timeout':
                # 使用LLM优化超时代码
                result = self._optimize_with_llm(code_file, error_analysis)
            
            else:
                result = {
                    'success': False,
                    'error': f'不支持的修复策略: {fix_strategy}'
                }
            
            self.log_end("错误修复完成")
            return result
            
        except Exception as e:
            self.log_error(f"错误修复失败: {str(e)}")
            raise
    
    def _identify_error_type(self, error_output: str) -> str:
        """识别错误类型"""
        error_lower = error_output.lower()
        
        if 'syntaxerror' in error_lower:
            return 'SyntaxError'
        elif 'importerror' in error_lower or 'modulenotfounderror' in error_lower:
            return 'ImportError'
        elif 'nameerror' in error_lower:
            return 'NameError'
        elif 'typeerror' in error_lower:
            return 'TypeError'
        elif 'valueerror' in error_lower:
            return 'ValueError'
        elif 'indexerror' in error_lower:
            return 'IndexError'
        elif 'keyerror' in error_lower:
            return 'KeyError'
        elif 'attributeerror' in error_lower:
            return 'AttributeError'
        elif 'linalgerror' in error_lower or 'incompatible dimensions' in error_lower:
            return 'LinAlgError'
        elif 'timeout' in error_lower or 'timed out' in error_lower:
            return 'TimeoutError'
        else:
            return 'RuntimeError'
    
    def _extract_error_details(self, error_output: str, error_type: str) -> Dict:
        """提取错误详情"""
        details = {
            'full_message': error_output,
            'error_line': None,
            'error_message': None,
            'missing_module': None
        }
        
        # 提取错误行号
        line_match = re.search(r'line (\d+)', error_output)
        if line_match:
            details['error_line'] = int(line_match.group(1))
        
        # 提取错误消息（最后一行通常是错误消息）
        lines = error_output.strip().split('\n')
        if lines:
            details['error_message'] = lines[-1].strip()
        
        # 如果是ImportError，提取缺失的模块名
        if error_type == 'ImportError':
            module_match = re.search(r"No module named '([^']+)'", error_output)
            if module_match:
                details['missing_module'] = module_match.group(1)
        
        return details
    
    def _is_auto_fixable(self, error_type: str, error_details: Dict) -> tuple:
        """
        判断是否可自动修复
        
        Returns:
            (auto_fixable, fix_strategy)
        """
        if error_type == 'ImportError' and error_details.get('missing_module'):
            return True, 'install_dependency'
        
        elif error_type == 'SyntaxError':
            return True, 'fix_syntax'
        
        elif error_type in ['NameError', 'TypeError', 'ValueError', 'IndexError', 'KeyError', 'AttributeError', 'LinAlgError']:
            return True, 'fix_runtime'
        
        elif error_type == 'TimeoutError':
            return True, 'optimize_timeout'
        
        # 对于其他RuntimeError，也尝试自动修复（可能是算法逻辑问题）
        elif error_type == 'RuntimeError':
            # 检查错误消息，如果是明显的算法问题，尝试修复
            error_msg = error_details.get('error_message', '').lower()
            if any(keyword in error_msg for keyword in ['dimension', 'shape', 'size', 'incompatible', 'mismatch']):
                return True, 'fix_runtime'
            # 默认也尝试修复（给LLM一个机会）
            return True, 'fix_runtime'
        
        else:
            return False, 'manual_intervention'
    
    def _fix_dependency(self, error_details: Dict) -> Dict:
        """自动安装缺失的依赖"""
        missing_module = error_details.get('missing_module')
        
        if not missing_module:
            return {
                'success': False,
                'error': '无法识别缺失的模块'
            }
        
        # 模块名映射（有些pip包名与import名不同）
        module_map = {
            'sklearn': 'scikit-learn',
            'cv2': 'opencv-python',
            'PIL': 'Pillow',
        }
        
        pip_package = module_map.get(missing_module, missing_module)
        
        self.logger.info(f"尝试安装依赖: {pip_package}")
        
        try:
            result = subprocess.run(
                [sys.executable, '-m', 'pip', 'install', pip_package],
                capture_output=True,
                text=True,
                timeout=120
            )
            
            if result.returncode == 0:
                self.logger.info(f"✅ 成功安装: {pip_package}")
                return {
                    'success': True,
                    'installed_package': pip_package,
                    'message': f'已安装 {pip_package}'
                }
            else:
                self.logger.error(f"安装失败: {result.stderr}")
                return {
                    'success': False,
                    'error': result.stderr
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def _fix_with_llm(self, code_file: Path, error_analysis: Dict) -> Dict:
        """使用LLM修复代码错误"""
        self.logger.info("使用LLM修复代码...")
        
        # 读取原始代码
        with open(code_file, 'r', encoding='utf-8') as f:
            original_code = f.read()
        
        # 准备提示词
        error_type = error_analysis['error_type']
        error_details = error_analysis['error_details']
        error_line = error_details.get('error_line', '未知')
        error_message = error_details.get('error_message', '未知')
        
        prompt = f"""以下Python代码存在{error_type}错误，请修复它。

错误信息:
- 错误类型: {error_type}
- 错误行号: {error_line}
- 错误消息: {error_message}

原始代码:
```python
{original_code}
```

要求:
1. 只输出修复后的完整代码
2. 不要有任何解释性文字
3. 代码以```python开始，以```结束
4. 确保代码可以正常运行

请直接输出修复后的代码。
"""
        
        system_prompt = "你是Python代码调试专家。直接输出修复后的代码，不要有任何解释。"
        
        try:
            # 调用LLM
            response = self.llm_client.generate(
                prompt=prompt,
                system_prompt=system_prompt
            )
            
            # 提取代码
            fixed_code = self._extract_code_from_response(response)
            
            if not fixed_code:
                return {
                    'success': False,
                    'error': 'LLM未生成有效的代码'
                }
            
            # 保存修复后的代码
            fixed_file = code_file.parent / f"{code_file.stem}_fixed_v{datetime.now().strftime('%H%M%S')}.py"
            with open(fixed_file, 'w', encoding='utf-8') as f:
                f.write(fixed_code)
            
            self.logger.info(f"✅ 代码已修复: {fixed_file}")
            
            return {
                'success': True,
                'fixed_file': str(fixed_file),
                'fixed_code': fixed_code
            }
            
        except Exception as e:
            self.logger.error(f"LLM修复失败: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _optimize_with_llm(self, code_file: Path, error_analysis: Dict) -> Dict:
        """使用LLM优化超时代码"""
        self.logger.info("使用LLM优化代码...")
        
        # 读取原始代码
        with open(code_file, 'r', encoding='utf-8') as f:
            original_code = f.read()
        
        prompt = f"""以下Python代码执行超时，请优化它以提高执行效率。

原始代码:
```python
{original_code}
```

优化要求:
1. 减少计算复杂度
2. 优化循环和数据结构
3. 考虑使用向量化操作
4. 减少不必要的计算
5. 保持功能不变

只输出优化后的完整代码，以```python开始，以```结束。
"""
        
        system_prompt = "你是Python性能优化专家。直接输出优化后的代码，不要有任何解释。"
        
        try:
            response = self.llm_client.generate(
                prompt=prompt,
                system_prompt=system_prompt
            )
            
            optimized_code = self._extract_code_from_response(response)
            
            if not optimized_code:
                return {
                    'success': False,
                    'error': 'LLM未生成有效的代码'
                }
            
            # 保存优化后的代码
            optimized_file = code_file.parent / f"{code_file.stem}_optimized_v{datetime.now().strftime('%H%M%S')}.py"
            with open(optimized_file, 'w', encoding='utf-8') as f:
                f.write(optimized_code)
            
            self.logger.info(f"✅ 代码已优化: {optimized_file}")
            
            return {
                'success': True,
                'fixed_file': str(optimized_file),
                'fixed_code': optimized_code
            }
            
        except Exception as e:
            self.logger.error(f"LLM优化失败: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _extract_code_from_response(self, response: str) -> str:
        """从LLM响应中提取代码块"""
        import re
        
        # 尝试提取```python ... ```代码块
        code_blocks = re.findall(r'```python\n(.*?)```', response, re.DOTALL)
        if code_blocks:
            return code_blocks[0].strip()
        
        # 尝试提取``` ... ```代码块
        code_blocks = re.findall(r'```\n(.*?)```', response, re.DOTALL)
        if code_blocks:
            return code_blocks[0].strip()
        
        # 如果没有代码块标记，检查是否整个响应都是代码
        if 'import ' in response and 'def ' in response:
            return response.strip()
        
        return ""

