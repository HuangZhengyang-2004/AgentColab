"""
AgentColab主程序
协调各个Agent完成从PDF到代码生成的完整流程
"""

import sys
from typing import Optional

from agents import (
    PDFExtractorAgent,
    PaperCleanerAgent,
    PaperAnalyzerAgent,
    IdeaGeneratorAgent,
    IdeaSelectorAgent,
    IdeaDetailerAgent,
    CodeGeneratorAgent
)
from utils import logger
from config import api_config


class AgentColab:
    """AgentColab主控制类"""
    
    def __init__(self):
        """初始化AgentColab"""
        logger.info("="*80)
        logger.info("AgentColab 系统启动")
        logger.info("="*80)
        
        # 验证API密钥
        self._validate_api_keys()
        
        # 初始化所有Agents
        self.agents = {}
        self._init_agents()
    
    def _validate_api_keys(self):
        """验证API密钥设置"""
        logger.info("验证API密钥...")
        
        api_status = api_config.validate_api_keys()
        
        for api_name, is_valid in api_status.items():
            status = "✓" if is_valid else "✗"
            logger.info(f"  {status} {api_name}: {'已设置' if is_valid else '未设置'}")
        
        # 检查必需的API
        required_apis = ['gemini', 'deepseek', 'claude']
        missing_apis = [api for api in required_apis if not api_status.get(api)]
        
        if missing_apis:
            logger.warning(f"警告: 以下必需的API密钥未设置: {', '.join(missing_apis)}")
            logger.warning("请设置相应的环境变量后再运行")
    
    def _init_agents(self):
        """初始化所有Agent"""
        logger.info("初始化Agents...")
        
        agent_classes = {
            'pdf_extractor': PDFExtractorAgent,
            'paper_cleaner': PaperCleanerAgent,
            'paper_analyzer': PaperAnalyzerAgent,
            'idea_generator': IdeaGeneratorAgent,
            'idea_selector': IdeaSelectorAgent,
            'idea_detailer': IdeaDetailerAgent,
            'code_generator': CodeGeneratorAgent,
        }
        
        for name, AgentClass in agent_classes.items():
            try:
                self.agents[name] = AgentClass()
                logger.info(f"  ✓ {name} 初始化成功")
            except Exception as e:
                logger.error(f"  ✗ {name} 初始化失败: {str(e)}")
    
    def run_full_pipeline(self, pdf_files: Optional[list] = None):
        """
        运行完整的流程
        
        Args:
            pdf_files: PDF文件列表，如果为None则从data/input目录读取
        """
        logger.info("="*80)
        logger.info("开始执行完整流程")
        logger.info("="*80)
        
        try:
            # 模块1: PDF提取
            logger.info("\n【模块1】PDF文档提取")
            extracted_papers = self.agents['pdf_extractor'].run(pdf_files)
            logger.info(f"✓ 成功提取 {len(extracted_papers)} 篇论文")
            
            # 模块2.1: 论文清洗
            logger.info("\n【模块2.1】论文内容清洗")
            cleaned_papers = self.agents['paper_cleaner'].run(extracted_papers)
            logger.info(f"✓ 成功清洗 {len(cleaned_papers)} 篇论文")
            
            # 模块2.2-2.3: 论文分析
            logger.info("\n【模块2.2-2.3】论文分析与总结")
            analyzed_papers = self.agents['paper_analyzer'].run(cleaned_papers)
            logger.info(f"✓ 成功分析 {len(analyzed_papers)} 篇论文")
            
            # 模块3: 生成创新想法
            logger.info("\n【模块3】生成创新想法")
            ideas = self.agents['idea_generator'].run()
            logger.info(f"✓ 成功生成 {len(ideas)} 个创新想法")
            
            # 模块4: 筛选最优想法
            logger.info("\n【模块4】筛选最优想法")
            best_idea = self.agents['idea_selector'].run(ideas)
            logger.info(f"✓ 最优想法: {best_idea.get('title', 'Unknown')}")
            logger.info(f"  创新性评分: {best_idea.get('score', 0)}")
            
            # 模块5: 详细化想法
            logger.info("\n【模块5】详细化最优想法")
            detailed_idea = self.agents['idea_detailer'].run(best_idea)
            logger.info(f"✓ 想法详细化完成")
            
            # 模块6: 生成代码
            logger.info("\n【模块6】生成Python代码实现")
            code = self.agents['code_generator'].run(detailed_idea)
            logger.info(f"✓ 代码生成完成")
            
            logger.info("\n" + "="*80)
            logger.info("完整流程执行成功！")
            logger.info("="*80)
            
            return {
                'extracted_papers': extracted_papers,
                'cleaned_papers': cleaned_papers,
                'analyzed_papers': analyzed_papers,
                'ideas': ideas,
                'best_idea': best_idea,
                'detailed_idea': detailed_idea,
                'code': code
            }
            
        except Exception as e:
            logger.error(f"流程执行失败: {str(e)}")
            raise
    
    def run_module(self, module_name: str, **kwargs):
        """
        运行单个模块
        
        Args:
            module_name: 模块名称
            **kwargs: 传递给模块的参数
        """
        logger.info(f"运行模块: {module_name}")
        
        module_map = {
            'pdf_extract': 'pdf_extractor',
            'paper_clean': 'paper_cleaner',
            'paper_analyze': 'paper_analyzer',
            'idea_generate': 'idea_generator',
            'idea_select': 'idea_selector',
            'idea_detail': 'idea_detailer',
            'code_generate': 'code_generator',
        }
        
        agent_name = module_map.get(module_name)
        
        if not agent_name or agent_name not in self.agents:
            logger.error(f"未知的模块名称: {module_name}")
            logger.info(f"可用的模块: {', '.join(module_map.keys())}")
            return None
        
        try:
            result = self.agents[agent_name].run(**kwargs)
            logger.info(f"✓ 模块 {module_name} 执行成功")
            return result
        except Exception as e:
            logger.error(f"模块 {module_name} 执行失败: {str(e)}")
            raise


def main():
    """主函数"""
    print("""
╔════════════════════════════════════════════════════════════╗
║                      AgentColab 系统                        ║
║          自动论文处理与创新想法生成系统                      ║
╚════════════════════════════════════════════════════════════╝
""")
    
    try:
        # 创建AgentColab实例
        agentcolab = AgentColab()
        
        # 检查命令行参数
        if len(sys.argv) > 1:
            command = sys.argv[1]
            
            if command == 'full':
                # 运行完整流程
                agentcolab.run_full_pipeline()
            
            elif command in ['pdf', 'clean', 'analyze', 'idea', 'select', 'detail', 'code']:
                # 运行单个模块
                module_map = {
                    'pdf': 'pdf_extract',
                    'clean': 'paper_clean',
                    'analyze': 'paper_analyze',
                    'idea': 'idea_generate',
                    'select': 'idea_select',
                    'detail': 'idea_detail',
                    'code': 'code_generate',
                }
                agentcolab.run_module(module_map[command])
            
            else:
                print(f"未知的命令: {command}")
                print_usage()
        
        else:
            print_usage()
    
    except KeyboardInterrupt:
        logger.info("\n用户中断执行")
    except Exception as e:
        logger.error(f"程序异常: {str(e)}")
        import traceback
        traceback.print_exc()


def print_usage():
    """打印使用说明"""
    print("""
使用方法:
    python main.py <command>

命令列表:
    full        - 运行完整流程（从PDF到代码生成）
    pdf         - 仅运行PDF提取模块
    clean       - 仅运行论文清洗模块
    analyze     - 仅运行论文分析模块
    idea        - 仅运行想法生成模块
    select      - 仅运行想法筛选模块
    detail      - 仅运行想法详细化模块
    code        - 仅运行代码生成模块

示例:
    python main.py full          # 运行完整流程
    python main.py pdf           # 只提取PDF
    python main.py analyze       # 只分析论文

注意:
    1. 请先将PDF文件放入 data/input/ 目录
    2. 确保已设置必要的API密钥环境变量:
       - GOOGLE_API_KEY (Gemini)
       - DEEPSEEK_API_KEY (DeepSeek)
       - ANTHROPIC_API_KEY (Claude)
       - MINERU_API_KEY (MinerU, 可选)
""")


if __name__ == "__main__":
    main()

