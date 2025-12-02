"""
Agents模块初始化文件
"""

from agents.base_agent import BaseAgent
from agents.pdf_extractor_agent import PDFExtractorAgent
from agents.paper_cleaner_agent import PaperCleanerAgent
from agents.paper_analyzer_agent import PaperAnalyzerAgent
from agents.idea_generator_agent import IdeaGeneratorAgent
from agents.idea_selector_agent import IdeaSelectorAgent
from agents.idea_detailer_agent import IdeaDetailerAgent
from agents.code_generator_agent import CodeGeneratorAgent
from agents.supervisor_agent import SupervisorAgent

__all__ = [
    'BaseAgent',
    'PDFExtractorAgent',
    'PaperCleanerAgent',
    'PaperAnalyzerAgent',
    'IdeaGeneratorAgent',
    'IdeaSelectorAgent',
    'IdeaDetailerAgent',
    'CodeGeneratorAgent',
    'SupervisorAgent'
]

