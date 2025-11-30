"""
Prompt模板管理模块
存储所有Agent使用的Prompt模板
"""


class PromptTemplates:
    """Prompt模板类"""
    
    # ==================== 模块2: 论文分析相关 ====================
    
    PAPER_TRANSLATION_AND_ANALYSIS = """我现在在读论文(阵列信号处理DOA估计方向)请跟我一起阅读分析，先翻译成中文后分析和推导公式。涉及矩阵和向量请全部展开。请把公式渲染完整，行间公式两边用两个美元符号显示公式。

论文内容如下：

{paper_content}
"""
    
    PAPER_CORE_SUMMARY = """请总结一下这篇文章的核心，以及核心算法实现逻辑。

论文内容如下：

{paper_content}
"""
    
    PAPER_CLEANING = """请帮我清理这篇论文的内容，删除以下部分：
1. 附录（Appendix）
2. 参考文献（References）
3. 致谢（Acknowledgments）
4. 作者信息和affiliations
5. 页眉页脚
6. 图表标题中的"Figure"、"Table"等标记

只保留论文的核心内容：摘要、引言、方法、实验、结论等主要章节。

论文内容如下：

{paper_content}
"""
    
    # ==================== 模块3: 创新想法生成 ====================
    
    IDEA_GENERATION = """这是我最近看的几篇文章，请尽量只根据这几篇文章的思路，帮我想几个创新性比较强的idea(尽量详细一些)，同时按照创新性对这几个idea进行打分。

请按照以下格式输出：

【Idea 1】
标题：[idea标题]
创新性评分：[0-100分]
来源论文：[论文编号，如：Paper 1, Paper 3]
详细描述：[详细的idea描述]

【Idea 2】
...

论文总结如下：

{papers_summary}
"""
    
    # ==================== 模块5: 想法详细化 ====================
    
    IDEA_DETAILING = """我先给你几篇文章，然后再给你根据这几篇文章结合产生的idea，最后你把这个idea详细化。

论文内容：
{papers_content}

创新想法：
{idea_content}

请详细阐述这个idea，包括：
1. 研究背景和动机
2. 具体的技术方案和实现思路
3. 理论推导和数学公式
4. 预期的创新点和优势
5. 可能的实验设计方案
"""
    
    # ==================== 模块6: 代码生成 ====================
    
    CODE_GENERATION = """这是我的一个idea产生的文章，请根据这篇文章帮我用python完整复现一下。

要求：
1. 代码要完整可运行
2. 包含必要的注释说明
3. 使用numpy、scipy等科学计算库
4. 如果涉及信号处理，使用合适的库
5. 代码要模块化，结构清晰
6. 包含简单的测试示例

Idea详细内容：

{idea_detail}
"""
    
    @staticmethod
    def format_prompt(template: str, **kwargs) -> str:
        """
        格式化prompt模板
        
        Args:
            template: prompt模板
            **kwargs: 要填充的变量
            
        Returns:
            格式化后的prompt
        """
        return template.format(**kwargs)


# 全局prompt模板实例
prompts = PromptTemplates()

