"""
AgentColab 环境测试脚本
用于验证环境配置是否正确
"""

import os
import sys
from pathlib import Path


def print_header(title):
    """打印标题"""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)


def test_python_version():
    """测试Python版本"""
    print_header("Python版本检查")
    version = sys.version
    print(f"Python版本: {version}")
    
    if sys.version_info < (3, 8):
        print("❌ Python版本过低，需要3.8或更高版本")
        return False
    else:
        print("✓ Python版本符合要求")
        return True


def test_imports():
    """测试依赖包导入"""
    print_header("依赖包检查")
    
    packages = {
        'yaml': 'pyyaml',
        'google.generativeai': 'google-generativeai',
        'anthropic': 'anthropic',
        'openai': 'openai',
        'PyPDF2': 'PyPDF2',
        'numpy': 'numpy',
        'scipy': 'scipy',
    }
    
    all_ok = True
    for module_name, package_name in packages.items():
        try:
            __import__(module_name)
            print(f"✓ {package_name}")
        except ImportError:
            print(f"❌ {package_name} 未安装")
            all_ok = False
    
    if not all_ok:
        print("\n请运行: pip install -r requirements.txt")
    
    return all_ok


def test_project_structure():
    """测试项目结构"""
    print_header("项目结构检查")
    
    required_dirs = [
        'agents', 'config', 'utils', 'data',
        'data/input', 'data/extracted', 'data/cleaned',
        'data/analyzed', 'data/ideas', 'data/code', 'logs'
    ]
    
    required_files = [
        'main.py', 'config.yaml', 'requirements.txt',
        'agents/__init__.py', 'config/__init__.py', 'utils/__init__.py'
    ]
    
    all_ok = True
    
    print("\n目录检查:")
    for dir_path in required_dirs:
        if Path(dir_path).is_dir():
            print(f"✓ {dir_path}/")
        else:
            print(f"❌ {dir_path}/ 不存在")
            all_ok = False
    
    print("\n文件检查:")
    for file_path in required_files:
        if Path(file_path).is_file():
            print(f"✓ {file_path}")
        else:
            print(f"❌ {file_path} 不存在")
            all_ok = False
    
    return all_ok


def test_config_file():
    """测试配置文件"""
    print_header("配置文件检查")
    
    try:
        import yaml
        
        if not Path('config.yaml').exists():
            print("❌ config.yaml 不存在")
            return False
        
        with open('config.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        print("✓ config.yaml 格式正确")
        
        # 检查必需的配置项
        required_keys = ['api', 'directories', 'logging', 'pipeline']
        for key in required_keys:
            if key in config:
                print(f"✓ 配置项 '{key}' 存在")
            else:
                print(f"❌ 配置项 '{key}' 缺失")
        
        return True
        
    except Exception as e:
        print(f"❌ 配置文件解析失败: {str(e)}")
        return False


def test_api_keys():
    """测试API密钥"""
    print_header("API密钥检查")
    
    from utils.config_loader import config_loader
    
    api_configs = {
        'gemini': ('Gemini API', False),
        'deepseek': ('DeepSeek API', False),
        'claude': ('Claude API', False),
        'mineru': ('MinerU API', True),  # 可选
    }
    
    all_set = True
    
    for api_name, (display_name, optional) in api_configs.items():
        config = config_loader.get_api_config(api_name)
        api_key = config.get('api_key', '')
        
        # 检查是从环境变量还是配置文件读取
        env_var_map = {
            'gemini': 'GOOGLE_API_KEY',
            'deepseek': 'DEEPSEEK_API_KEY',
            'claude': 'ANTHROPIC_API_KEY',
            'mineru': 'MINERU_API_KEY',
        }
        
        env_var = env_var_map.get(api_name)
        source = ""
        
        if api_key:
            # 判断来源
            if os.getenv(env_var):
                source = " [环境变量]"
            else:
                source = " [配置文件]"
            
            masked_key = api_key[:8] + "..." + api_key[-4:] if len(api_key) > 12 else "***"
            print(f"✓ {display_name}: {masked_key}{source}")
        else:
            if optional:
                print(f"⚠ {display_name}: 未设置 (可选)")
            else:
                print(f"❌ {display_name}: 未设置")
                all_set = False
    
    if not all_set:
        print("\n请设置必需的API密钥")
        print("方式1: 环境变量 (推荐)")
        print("  export GOOGLE_API_KEY='your_key'")
        print("\n方式2: 配置文件 config.yaml")
        print("  api_keys:")
        print("    google_api_key: 'your_key'")
        print("\n详见: docs/API_KEY_CONFIG.md")
    
    return all_set


def test_modules_import():
    """测试项目模块导入"""
    print_header("项目模块检查")
    
    modules = [
        'config.api_config',
        'config.prompts',
        'utils.logger',
        'utils.file_manager',
        'utils.config_loader',
        'utils.api_client',
        'agents.base_agent',
        'agents.pdf_extractor_agent',
        'agents.paper_cleaner_agent',
        'agents.paper_analyzer_agent',
        'agents.idea_generator_agent',
        'agents.idea_selector_agent',
        'agents.idea_detailer_agent',
        'agents.code_generator_agent',
    ]
    
    all_ok = True
    for module_name in modules:
        try:
            __import__(module_name)
            print(f"✓ {module_name}")
        except Exception as e:
            print(f"❌ {module_name}: {str(e)}")
            all_ok = False
    
    return all_ok


def test_input_files():
    """测试输入文件"""
    print_header("输入文件检查")
    
    input_dir = Path('data/input')
    pdf_files = list(input_dir.glob('*.pdf'))
    
    if pdf_files:
        print(f"✓ 找到 {len(pdf_files)} 个PDF文件:")
        for pdf in pdf_files:
            print(f"  - {pdf.name}")
        return True
    else:
        print("⚠ data/input 目录中没有PDF文件")
        print("  请将PDF论文放入该目录后再运行程序")
        return False


def main():
    """主函数"""
    print("""
╔════════════════════════════════════════════════════════════╗
║              AgentColab 环境测试程序                        ║
╚════════════════════════════════════════════════════════════╝
""")
    
    results = {
        'Python版本': test_python_version(),
        '依赖包': test_imports(),
        '项目结构': test_project_structure(),
        '配置文件': test_config_file(),
        'API密钥': test_api_keys(),
        '项目模块': test_modules_import(),
        '输入文件': test_input_files(),
    }
    
    # 汇总结果
    print_header("测试结果汇总")
    
    for test_name, result in results.items():
        status = "✓ 通过" if result else "❌ 失败"
        print(f"{test_name:12} {status}")
    
    print("\n" + "="*60)
    
    passed = sum(results.values())
    total = len(results)
    
    if passed == total:
        print(f"\n🎉 所有测试通过！({passed}/{total})")
        print("\n你可以开始使用AgentColab了:")
        print("  ./run.sh full     # 运行完整流程")
        print("  python main.py full")
    else:
        print(f"\n⚠️  部分测试失败 ({passed}/{total})")
        print("\n请根据上面的提示修复问题后重试")
        print("  ./run.sh setup    # 重新初始化环境")
        print("  ./run.sh check    # 快速检查环境")


if __name__ == "__main__":
    main()

