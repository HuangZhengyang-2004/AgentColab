"""
AgentColab Web UI
基于Gradio的Web用户界面
"""

import os
import gradio as gr
from pathlib import Path
import json
from datetime import datetime

from agents import (
    PDFExtractorAgent,
    PaperCleanerAgent,
    PaperAnalyzerAgent,
    IdeaGeneratorAgent,
    IdeaSelectorAgent,
    IdeaDetailerAgent,
    CodeGeneratorAgent
)
from utils.config_loader import config_loader
from utils.logger import logger
from utils.collection_ui import load_collection_info, view_paper_content, export_collection_summary


# ==================== 配置管理 ====================

def get_current_config():
    """获取当前配置"""
    config = {
        'google_api_key': os.getenv('GOOGLE_API_KEY', '') or config_loader.get('api_keys.google_api_key', ''),
        'deepseek_api_key': os.getenv('DEEPSEEK_API_KEY', '') or config_loader.get('api_keys.deepseek_api_key', ''),
        'anthropic_api_key': os.getenv('ANTHROPIC_API_KEY', '') or config_loader.get('api_keys.anthropic_api_key', ''),
        'mineru_api_key': os.getenv('MINERU_API_KEY', '') or config_loader.get('api_keys.mineru_api_key', ''),
        'use_mineru': config_loader.get('pipeline.pdf_extraction.use_mineru', False),
    }
    return config


def save_api_keys(google_key, deepseek_key, anthropic_key, mineru_key):
    """保存API密钥到环境变量"""
    if google_key:
        os.environ['GOOGLE_API_KEY'] = google_key
    if deepseek_key:
        os.environ['DEEPSEEK_API_KEY'] = deepseek_key
    if anthropic_key:
        os.environ['ANTHROPIC_API_KEY'] = anthropic_key
    if mineru_key:
        os.environ['MINERU_API_KEY'] = mineru_key
    
    return "✅ API密钥已保存到当前会话"


# ==================== PDF提取 ====================

def extract_pdf_from_upload(pdf_file, pdf_name, use_mineru, progress=gr.Progress()):
    """从上传的文件提取PDF"""
    try:
        if pdf_file is None:
            return "❌ 请先上传PDF文件"
        
        progress(0.1, desc="初始化PDF提取Agent...")
        
        # 获取上传文件的路径
        file_path = pdf_file.name if hasattr(pdf_file, 'name') else pdf_file
        
        if not pdf_name:
            pdf_name = Path(file_path).stem
        
        if use_mineru:
            # 使用MinerU上传并提取
            try:
                from utils.mineru_client import get_mineru_client
                
                progress(0.2, desc="连接MinerU服务...")
                client = get_mineru_client()
                
                progress(0.3, desc="上传文件到MinerU服务器...")
                content = client.upload_and_extract_file(
                    file_path=file_path,
                    data_id=pdf_name,
                    model_version="vlm"
                )
                
                # 保存提取结果
                from utils.file_manager import file_manager
                output_filename = f"{pdf_name}_extracted.txt"
                file_manager.save_text(content, output_filename, 'extracted')
                
                progress(1.0, desc="提取完成！")
                
                return f"✅ MinerU提取成功！\n\n文件名: {pdf_name}\n内容长度: {len(content)} 字符\n\n{'='*50}\n内容预览:\n{'='*50}\n\n{content[:1000]}..."
                
            except Exception as e:
                return f"❌ MinerU提取失败: {str(e)}\n\n💡 提示：可以取消勾选'使用MinerU'改用PyPDF2"
        
        else:
            # 使用PyPDF2提取
            agent = PDFExtractorAgent(use_mineru=False)
            
            progress(0.3, desc="使用PyPDF2提取PDF...")
            content = agent._extract_with_pypdf2(file_path)
            
            # 保存提取结果
            output_filename = f"{pdf_name}_extracted.txt"
            agent.save_result(content, output_filename, 'extracted', format='text')
            
            progress(1.0, desc="提取完成！")
            
            return f"✅ PyPDF2提取成功！\n\n文件名: {pdf_name}\n内容长度: {len(content)} 字符\n\n{'='*50}\n内容预览:\n{'='*50}\n\n{content[:1000]}..."
        
    except Exception as e:
        return f"❌ 提取失败: {str(e)}"


def extract_pdf_from_url(pdf_url, pdf_name, use_mineru, progress=gr.Progress()):
    """从URL提取PDF"""
    try:
        if not pdf_url or not pdf_url.strip():
            return "❌ 请输入PDF的URL"
        
        progress(0.1, desc="初始化PDF提取Agent...")
        agent = PDFExtractorAgent(use_mineru=use_mineru)
        
        if use_mineru and not pdf_url.startswith('http'):
            return "❌ MinerU需要PDF的公开URL（http://或https://开头）"
        
        progress(0.3, desc="开始提取PDF...")
        
        if use_mineru:
            content = agent.extract_from_url(
                pdf_url=pdf_url,
                pdf_name=pdf_name or "unnamed_pdf"
            )
        else:
            return "❌ URL模式下请使用MinerU，或下载后使用文件上传方式"
        
        progress(1.0, desc="提取完成！")
        
        return f"✅ 提取成功！\n\n内容长度: {len(content)} 字符\n\n{'='*50}\n内容预览:\n{'='*50}\n\n{content[:1000]}..."
        
    except Exception as e:
        return f"❌ 提取失败: {str(e)}"


def batch_extract_pdfs_upload(pdf_files, use_mineru, progress=gr.Progress()):
    """批量提取上传的PDF文件"""
    try:
        if not pdf_files:
            return "❌ 请先上传PDF文件"
        
        progress(0.1, desc=f"准备批量提取 {len(pdf_files)} 个PDF...")
        
        if use_mineru:
            # 使用MinerU批量处理
            try:
                from utils.mineru_client import get_mineru_client
                from utils.file_manager import file_manager
                
                client = get_mineru_client()
                
                results = {}
                for i, pdf_file in enumerate(pdf_files, 1):
                    file_path = pdf_file.name if hasattr(pdf_file, 'name') else pdf_file
                    pdf_name = Path(file_path).stem
                    
                    progress((i-0.5) / len(pdf_files), desc=f"处理第 {i}/{len(pdf_files)} 个文件（MinerU）...")
                    
                    try:
                        content = client.upload_and_extract_file(
                            file_path=file_path,
                            data_id=pdf_name,
                            model_version="vlm"
                        )
                        
                        # 保存结果
                        output_filename = f"{pdf_name}_extracted.txt"
                        file_manager.save_text(content, output_filename, 'extracted')
                        
                        results[pdf_name] = content
                        
                    except Exception as e:
                        results[f"error_{pdf_name}"] = str(e)
                
                progress(1.0, desc="批量提取完成！")
                
                success_count = len([k for k in results.keys() if not k.startswith("error_")])
                summary = f"✅ MinerU批量提取完成！成功: {success_count}/{len(pdf_files)}\n\n"
                for name, content in results.items():
                    if not name.startswith("error_"):
                        summary += f"• {name}: {len(content)} 字符\n"
                    else:
                        summary += f"• {name.replace('error_', '')}: ❌ {content}\n"
                
                return summary
                
            except Exception as e:
                return f"❌ MinerU批量提取失败: {str(e)}"
        
        else:
            # 使用PyPDF2批量处理
            agent = PDFExtractorAgent(use_mineru=False)
            
            results = {}
            for i, pdf_file in enumerate(pdf_files, 1):
                progress(i / len(pdf_files), desc=f"处理第 {i}/{len(pdf_files)} 个文件（PyPDF2）...")
                
                try:
                    file_path = pdf_file.name if hasattr(pdf_file, 'name') else pdf_file
                    pdf_name = Path(file_path).stem
                    
                    # 提取内容
                    content = agent._extract_with_pypdf2(file_path)
                    
                    # 保存结果
                    output_filename = f"{pdf_name}_extracted.txt"
                    agent.save_result(content, output_filename, 'extracted', format='text')
                    
                    results[pdf_name] = content
                    
                except Exception as e:
                    results[f"error_{i}"] = f"失败: {str(e)}"
            
            progress(1.0, desc="批量提取完成！")
            
            summary = f"✅ PyPDF2批量提取完成！成功: {len(results)}/{len(pdf_files)}\n\n"
            for name, content in results.items():
                if not name.startswith("error_"):
                    summary += f"• {name}: {len(content)} 字符\n"
                else:
                    summary += f"• {name}: {content}\n"
            
            return summary
        
    except Exception as e:
        return f"❌ 批量提取失败: {str(e)}"


def batch_extract_pdfs_url(pdf_urls_text, use_mineru, progress=gr.Progress()):
    """批量提取PDF（从URL）"""
    try:
        # 解析URL列表
        urls = [url.strip() for url in pdf_urls_text.split('\n') if url.strip()]
        
        if not urls:
            return "❌ 请输入至少一个PDF URL"
        
        if not use_mineru:
            return "❌ URL批量提取需要使用MinerU"
        
        progress(0.1, desc=f"准备批量提取 {len(urls)} 个PDF...")
        agent = PDFExtractorAgent(use_mineru=True)
        
        names = [f"paper_{i+1}" for i in range(len(urls))]
        
        progress(0.3, desc="批量提取中...")
        results = agent.extract_from_urls(pdf_urls=urls, pdf_names=names)
        
        progress(1.0, desc="批量提取完成！")
        
        summary = f"✅ 批量提取完成！成功: {len(results)}/{len(urls)}\n\n"
        for name, content in results.items():
            summary += f"• {name}: {len(content)} 字符\n"
        
        return summary
        
    except Exception as e:
        return f"❌ 批量提取失败: {str(e)}"


# ==================== 论文清洗 ====================

def clean_papers(progress=gr.Progress()):
    """清洗论文"""
    try:
        progress(0.1, desc="初始化论文清洗Agent...")
        agent = PaperCleanerAgent()
        
        progress(0.3, desc="清洗论文中...")
        results = agent.run()
        
        progress(1.0, desc="清洗完成！")
        
        if not results:
            return "❌ 没有找到需要清洗的论文\n请先在'论文集合'Tab中创建集合"
        
        # 生成报告
        report = f"✅ 清洗完成！共处理 {len(results)} 篇论文\n\n"
        report += "清洗统计:\n"
        report += "=" * 60 + "\n\n"
        
        # 加载原始集合对比
        from utils import PaperCollection
        try:
            original_collection = PaperCollection.load_from_json("data/collections/all_papers.json")
            original_papers = original_collection.get_all_contents()
            
            for paper_key in results.keys():
                original_len = len(original_papers.get(paper_key, ""))
                cleaned_len = len(results[paper_key])
                removal_rate = (1 - cleaned_len/original_len)*100 if original_len > 0 else 0
                
                report += f"{paper_key}:\n"
                report += f"  原始: {original_len:,} 字符\n"
                report += f"  清洗: {cleaned_len:,} 字符\n"
                report += f"  删除: {removal_rate:.1f}%\n\n"
        except:
            for paper_key, content in results.items():
                report += f"  • {paper_key}: {len(content):,} 字符\n"
        
        report += "\n" + "=" * 60 + "\n"
        report += "保存位置:\n"
        report += "  • 文本文件: data/cleaned/paper_*_cleaned.txt\n"
        report += "  • 集合文件: data/collections/all_papers_cleaned.json\n"
        
        return report
        
    except Exception as e:
        import traceback
        return f"❌ 清洗失败: {str(e)}\n\n{traceback.format_exc()}"


# ==================== 论文分析 ====================

def analyze_papers(progress=gr.Progress()):
    """分析论文 - 使用DeepSeek分析核心内容和算法"""
    try:
        progress(0.1, desc="初始化论文分析Agent...")
        agent = PaperAnalyzerAgent()
        
        progress(0.3, desc="使用DeepSeek分析论文中（这可能需要较长时间）...")
        results = agent.run()
        
        progress(1.0, desc="分析完成！")
        
        if not results:
            return "❌ 没有找到需要分析的论文\n请先执行清洗步骤"
        
        # 生成报告
        report = f"✅ 分析完成！共处理 {len(results)} 篇论文\n\n"
        report += "分析统计:\n"
        report += "=" * 60 + "\n\n"
        
        for paper_key, analysis in results.items():
            analysis_len = len(analysis)
            # 计算Markdown标题数量
            title_count = analysis.count('\n#')
            
            report += f"{paper_key}:\n"
            report += f"  分析长度: {analysis_len:,} 字符\n"
            report += f"  章节数: {title_count}\n"
            report += f"  预览: {analysis[:100].replace(chr(10), ' ')}...\n\n"
        
        report += "=" * 60 + "\n"
        report += "保存位置:\n"
        report += "  • Markdown文件: data/analyzed/paper_*_analysis.md\n"
        report += "  • 集合文件: data/collections/all_papers_analyzed.json\n"
        report += "  • 统计信息: data/analyzed/analysis_stats.json\n"
        
        return report
        
    except Exception as e:
        import traceback
        return f"❌ 分析失败: {str(e)}\n\n{traceback.format_exc()}"


def view_analysis(paper_key):
    """查看分析结果"""
    try:
        from pathlib import Path
        
        # 尝试读取Markdown文件
        analysis_file = Path(f"data/analyzed/{paper_key}_analysis.md")
        
        if not analysis_file.exists():
            return f"❌ 未找到 {paper_key} 的分析结果\n\n请先执行论文分析"
        
        with open(analysis_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        return f"# 📊 {paper_key} 分析结果\n\n" + content
        
    except Exception as e:
        return f"❌ 读取失败: {str(e)}"


# ==================== 想法生成 ====================

def generate_ideas(progress=gr.Progress()):
    """生成创新想法 - 使用DeepSeek基于论文分析生成ideas"""
    try:
        progress(0.1, desc="初始化想法生成Agent...")
        
        # 生成想法
        agent = IdeaGeneratorAgent()
        progress(0.3, desc="使用DeepSeek生成创新想法中（可能需要1-2分钟）...")
        ideas_text = agent.run()
        
        progress(1.0, desc="完成！")
        
        if not ideas_text:
            return "❌ 没有找到论文分析结果\n请先执行论文分析步骤"
        
        # 生成报告
        report = f"✅ 创新想法生成完成！\n\n"
        report += "=" * 60 + "\n"
        report += "📊 生成统计\n"
        report += "=" * 60 + "\n\n"
        report += f"输出长度: {len(ideas_text):,} 字符\n"
        report += f"包含评分: {'是' if '评分' in ideas_text or '分' in ideas_text else '否'}\n\n"
        report += "=" * 60 + "\n"
        report += "保存位置:\n"
        report += "  • 想法文件: data/ideas/generated_ideas.md\n\n"
        report += "=" * 60 + "\n"
        report += "📋 生成的想法预览\n"
        report += "=" * 60 + "\n\n"
        
        # 显示前500字符
        preview_length = 500
        if len(ideas_text) > preview_length:
            report += ideas_text[:preview_length] + "\n\n... (还有更多内容)"
        else:
            report += ideas_text
        
        return report
        
    except Exception as e:
        import traceback
        return f"❌ 生成想法失败: {str(e)}\n\n{traceback.format_exc()}"


def view_ideas():
    """查看生成的想法"""
    try:
        from pathlib import Path
        
        ideas_file = Path("data/ideas/generated_ideas.md")
        
        if not ideas_file.exists():
            return "❌ 未找到生成的想法\n\n请先执行想法生成"
        
        with open(ideas_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        return f"# 💡 生成的创新想法\n\n{content}"
        
    except Exception as e:
        return f"❌ 读取失败: {str(e)}"


# ==================== 想法详细化 ====================

def detail_idea(progress=gr.Progress()):
    """详细化最优想法"""
    try:
        progress(0.1, desc="初始化想法详细化Agent...")
        agent = IdeaDetailerAgent()
        
        progress(0.3, desc="详细化想法中...")
        detailed = agent.run()
        
        progress(1.0, desc="详细化完成！")
        
        return f"✅ 想法详细化完成！\n\n{'='*60}\n\n{detailed}"
        
    except Exception as e:
        return f"❌ 详细化失败: {str(e)}"


# ==================== 代码生成 ====================

def generate_code(progress=gr.Progress()):
    """生成代码实现"""
    try:
        progress(0.1, desc="初始化代码生成Agent...")
        agent = CodeGeneratorAgent()
        
        progress(0.3, desc="生成代码中...")
        code = agent.run()
        
        progress(1.0, desc="代码生成完成！")
        
        return code
        
    except Exception as e:
        return f"❌ 代码生成失败: {str(e)}"


# ==================== 完整流程 ====================

def run_full_pipeline(progress=gr.Progress()):
    """运行完整流程"""
    try:
        steps = [
            (0.1, "初始化系统..."),
            (0.2, "提取PDF..."),
            (0.3, "清洗论文..."),
            (0.5, "分析论文..."),
            (0.7, "生成想法..."),
            (0.8, "筛选最优想法..."),
            (0.9, "详细化想法..."),
            (0.95, "生成代码..."),
        ]
        
        output = "🚀 开始执行完整流程\n\n"
        
        # PDF提取
        progress(0.2, desc="提取PDF...")
        agent = PDFExtractorAgent(use_mineru=False)
        extracted = agent.run()
        output += f"✓ PDF提取: {len(extracted)} 篇\n"
        
        # 清洗
        progress(0.3, desc="清洗论文...")
        agent = PaperCleanerAgent()
        cleaned = agent.run()
        output += f"✓ 论文清洗: {len(cleaned)} 篇\n"
        
        # 分析
        progress(0.5, desc="分析论文...")
        agent = PaperAnalyzerAgent()
        analyzed = agent.run()
        output += f"✓ 论文分析: {len(analyzed)} 篇\n"
        
        # 生成想法
        progress(0.7, desc="生成想法...")
        agent = IdeaGeneratorAgent()
        ideas = agent.run()
        output += f"✓ 想法生成: {len(ideas)} 个\n"
        
        # 筛选
        progress(0.8, desc="筛选想法...")
        agent = IdeaSelectorAgent()
        best = agent.run(ideas)
        output += f"✓ 最优想法: {best['title']}\n"
        
        # 详细化
        progress(0.9, desc="详细化...")
        agent = IdeaDetailerAgent()
        detailed = agent.run()
        output += f"✓ 想法详细化完成\n"
        
        # 生成代码
        progress(0.95, desc="生成代码...")
        agent = CodeGeneratorAgent()
        code = agent.run()
        output += f"✓ 代码生成完成\n"
        
        progress(1.0, desc="完成！")
        output += "\n" + "="*60 + "\n"
        output += "🎉 完整流程执行成功！\n"
        output += "="*60 + "\n"
        
        return output
        
    except Exception as e:
        return f"❌ 流程执行失败: {str(e)}"


# ==================== UI界面 ====================

def create_ui():
    """创建Gradio UI"""
    
    with gr.Blocks(title="AgentColab - 自动论文处理系统") as app:
        
        gr.Markdown("""
        # 🎓 AgentColab - 自动论文处理与创新想法生成系统
        
        自动从PDF论文中提取内容、分析总结、生成创新想法，并自动生成代码实现。
        """)
        
        # ==================== Tab 1: 配置 ====================
        with gr.Tab("⚙️ 配置"):
            gr.Markdown("## API密钥配置")
            gr.Markdown("请输入你的API密钥（留空则使用环境变量或配置文件）")
            
            with gr.Row():
                with gr.Column():
                    google_key = gr.Textbox(
                        label="Google Gemini API Key",
                        type="password",
                        placeholder="sk-...",
                        value=get_current_config()['google_api_key']
                    )
                    deepseek_key = gr.Textbox(
                        label="DeepSeek API Key",
                        type="password",
                        placeholder="sk-...",
                        value=get_current_config()['deepseek_api_key']
                    )
                
                with gr.Column():
                    anthropic_key = gr.Textbox(
                        label="Anthropic Claude API Key",
                        type="password",
                        placeholder="sk-...",
                        value=get_current_config()['anthropic_api_key']
                    )
                    mineru_key = gr.Textbox(
                        label="MinerU API Key (可选)",
                        type="password",
                        placeholder="...",
                        value=get_current_config()['mineru_api_key']
                    )
            
            save_btn = gr.Button("💾 保存配置", variant="primary")
            config_output = gr.Textbox(label="状态", interactive=False)
            
            save_btn.click(
                fn=save_api_keys,
                inputs=[google_key, deepseek_key, anthropic_key, mineru_key],
                outputs=config_output
            )
            
            gr.Markdown("""
            ### 📚 获取API密钥
            - **Gemini**: https://makersuite.google.com/app/apikey
            - **DeepSeek**: https://platform.deepseek.com/
            - **Claude**: https://console.anthropic.com/
            - **MinerU**: https://mineru.net/ (每天2000页免费)
            """)
        
        # ==================== Tab 2: PDF提取 ====================
        with gr.Tab("📄 PDF提取"):
            gr.Markdown("## PDF文档提取")
            
            with gr.Row():
                use_mineru_pdf = gr.Checkbox(
                    label="使用MinerU（高精度，仅支持URL方式）",
                    value=get_current_config()['use_mineru']
                )
            
            gr.Markdown("""
            ### 📝 使用说明
            - **上传文件**：直接上传PDF，支持MinerU和PyPDF2
            - **URL方式**：输入公开URL，推荐使用MinerU
            - **MinerU优势**：高精度识别公式、表格、图片
            - **PyPDF2优势**：完全免费，速度快
            - **推荐**：学术论文用MinerU，普通文档用PyPDF2
            """)
            
            with gr.Tab("📤 上传文件"):
                gr.Markdown("""
                **直接上传PDF文件进行提取**
                - ✅ 支持MinerU高精度提取
                - ✅ 支持PyPDF2快速提取
                - 💡 勾选上方的"使用MinerU"可切换提取方式
                """)
                pdf_file = gr.File(
                    label="上传PDF文件",
                    file_types=[".pdf"],
                    type="filepath"
                )
                pdf_name_upload = gr.Textbox(
                    label="文件名称（可选，默认使用原文件名）",
                    placeholder="my_paper"
                )
                extract_upload_btn = gr.Button("🚀 开始提取", variant="primary", size="lg")
                extract_upload_output = gr.Textbox(label="提取结果", lines=15)
                
                extract_upload_btn.click(
                    fn=extract_pdf_from_upload,
                    inputs=[pdf_file, pdf_name_upload, use_mineru_pdf],
                    outputs=extract_upload_output
                )
            
            with gr.Tab("🔗 URL方式"):
                gr.Markdown("**从URL提取PDF（推荐使用MinerU）**")
                pdf_url = gr.Textbox(
                    label="PDF URL",
                    placeholder="https://example.com/paper.pdf",
                    lines=1
                )
                pdf_name_url = gr.Textbox(
                    label="PDF名称（可选）",
                    placeholder="my_paper"
                )
                extract_url_btn = gr.Button("🚀 开始提取", variant="primary", size="lg")
                extract_url_output = gr.Textbox(label="提取结果", lines=15)
                
                extract_url_btn.click(
                    fn=extract_pdf_from_url,
                    inputs=[pdf_url, pdf_name_url, use_mineru_pdf],
                    outputs=extract_url_output
                )
            
            with gr.Tab("📦 批量上传"):
                gr.Markdown("""
                **批量上传多个PDF文件**
                - ✅ 支持MinerU批量高精度提取
                - ✅ 支持PyPDF2批量快速提取
                - 💡 MinerU适合学术论文，PyPDF2适合普通文档
                """)
                batch_files = gr.File(
                    label="上传多个PDF文件",
                    file_count="multiple",
                    file_types=[".pdf"],
                    type="filepath"
                )
                batch_upload_btn = gr.Button("🚀 批量提取", variant="primary", size="lg")
                batch_upload_output = gr.Textbox(label="批量提取结果", lines=10)
                
                batch_upload_btn.click(
                    fn=batch_extract_pdfs_upload,
                    inputs=[batch_files, use_mineru_pdf],
                    outputs=batch_upload_output
                )
            
            with gr.Tab("🔗 批量URL"):
                gr.Markdown("**批量从URL提取PDF（使用MinerU）**")
                batch_urls = gr.Textbox(
                    label="PDF URLs（每行一个）",
                    placeholder="https://example.com/paper1.pdf\nhttps://example.com/paper2.pdf",
                    lines=5
                )
                batch_url_btn = gr.Button("🚀 批量提取", variant="primary", size="lg")
                batch_url_output = gr.Textbox(label="批量提取结果", lines=10)
                
                batch_url_btn.click(
                    fn=batch_extract_pdfs_url,
                    inputs=[batch_urls, use_mineru_pdf],
                    outputs=batch_url_output
                )
        
        # ==================== Tab 2.5: 论文集合管理 ====================
        with gr.Tab("📚 论文集合"):
            gr.Markdown("## 论文集合管理")
            gr.Markdown("""
            管理提取的论文，将多篇论文组织成统一格式（paper_1, paper_2, ...）
            """)
            
            with gr.Tab("📊 查看集合"):
                gr.Markdown("### 查看已提取的论文集合")
                
                with gr.Row():
                    collection_path = gr.Textbox(
                        label="集合文件路径",
                        placeholder="data/collections/all_papers.json",
                        value="data/collections/all_papers.json"
                    )
                    load_collection_btn = gr.Button("📂 加载集合", variant="secondary")
                
                collection_info = gr.Textbox(label="集合信息", lines=15)
                
                gr.Markdown("### 查看特定论文")
                with gr.Row():
                    paper_key = gr.Textbox(
                        label="论文键名",
                        placeholder="paper_1",
                        value="paper_1"
                    )
                    view_paper_btn = gr.Button("👁️ 查看内容", variant="secondary")
                
                paper_content = gr.Textbox(label="论文内容", lines=15)
                
                # 绑定事件
                load_collection_btn.click(
                    fn=lambda path: load_collection_info(path),
                    inputs=[collection_path],
                    outputs=[collection_info]
                )
                
                view_paper_btn.click(
                    fn=lambda path, key: view_paper_content(path, key),
                    inputs=[collection_path, paper_key],
                    outputs=[paper_content]
                )
            
            with gr.Tab("🔄 创建集合"):
                gr.Markdown("### 从extracted目录创建论文集合")
                gr.Markdown("""
                自动加载 `data/extracted/` 目录下所有已提取的论文，
                创建统一格式的集合文件。
                """)
                
                create_collection_btn = gr.Button("📦 创建集合", variant="primary", size="lg")
                create_output = gr.Textbox(label="创建结果", lines=10)
                
                def create_collection_from_extracted():
                    try:
                        from utils.paper_collection import PaperCollection
                        
                        # 从extracted目录加载
                        collection = PaperCollection.from_extracted_dir("data/extracted")
                        
                        if len(collection) == 0:
                            return "❌ data/extracted/ 目录中没有找到论文"
                        
                        # 保存集合
                        output_path = "data/collections/all_papers.json"
                        collection.save_to_json(output_path)
                        
                        # 生成报告
                        summary = collection.get_summary()
                        report = f"✓ 成功创建论文集合！\n\n"
                        report += f"📊 统计信息:\n"
                        report += f"  • 总论文数: {summary['total_papers']}\n"
                        report += f"  • 总字符数: {summary['total_characters']:,}\n"
                        report += f"  • 保存位置: {output_path}\n\n"
                        report += f"📚 论文列表:\n"
                        
                        for p in summary['papers']:
                            name = p['name'][:50] + "..." if len(p['name']) > 50 else p['name']
                            report += f"  {p['key']}: {name}\n"
                            report += f"          ({p['length']:,} 字符)\n"
                        
                        return report
                        
                    except Exception as e:
                        return f"❌ 创建失败: {str(e)}"
                
                create_collection_btn.click(
                    fn=create_collection_from_extracted,
                    outputs=[create_output]
                )
            
            with gr.Tab("💾 导出摘要"):
                gr.Markdown("### 导出集合摘要为文本文件")
                
                export_path = gr.Textbox(
                    label="集合文件路径",
                    placeholder="data/collections/all_papers.json",
                    value="data/collections/all_papers.json"
                )
                export_btn = gr.Button("💾 导出摘要", variant="primary")
                export_output = gr.Textbox(label="导出结果", lines=15)
                
                def export_summary(path):
                    try:
                        from utils.collection_ui import export_collection_summary
                        result, _ = export_collection_summary(path)
                        return result
                    except Exception as e:
                        return f"❌ 导出失败: {str(e)}"
                
                export_btn.click(
                    fn=export_summary,
                    inputs=[export_path],
                    outputs=[export_output]
                )
        
        # ==================== Tab 3: 论文处理 ====================
        with gr.Tab("📖 论文处理"):
            gr.Markdown("## 论文清洗与分析")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 1️⃣ 清洗论文")
                    gr.Markdown("删除引用、参考文献等无关内容")
                    clean_btn = gr.Button("🧹 清洗论文", variant="primary", size="lg")
                    clean_output = gr.Textbox(label="清洗结果", lines=20)
                    
                    clean_btn.click(
                        fn=clean_papers,
                        outputs=clean_output
                    )
                
                with gr.Column():
                    gr.Markdown("### 2️⃣ 分析论文（DeepSeek）")
                    gr.Markdown("""
                    使用DeepSeek分析论文核心内容和算法实现逻辑
                    
                    **分析内容：**
                    - 📋 论文核心内容（研究问题、创新点）
                    - 🔬 核心算法实现逻辑（算法原理、关键步骤）
                    - ✨ 技术亮点和贡献
                    
                    **输出格式：** Markdown
                    """)
                    analyze_btn = gr.Button("🔬 分析论文", variant="primary", size="lg")
                    analyze_output = gr.Textbox(label="分析结果", lines=20)
                    
                    analyze_btn.click(
                        fn=analyze_papers,
                        outputs=analyze_output
                    )
            
            # 查看分析结果
            with gr.Row():
                gr.Markdown("### 3️⃣ 查看分析结果")
            
            with gr.Row():
                with gr.Column(scale=2):
                    view_paper_key = gr.Textbox(
                        label="论文键名",
                        placeholder="paper_1",
                        value="paper_1"
                    )
                with gr.Column(scale=1):
                    view_analysis_btn = gr.Button("👁️ 查看分析", variant="secondary")
            
            analysis_viewer = gr.Markdown(label="分析内容")
            
            view_analysis_btn.click(
                fn=view_analysis,
                inputs=[view_paper_key],
                outputs=[analysis_viewer]
            )
        
        # ==================== Tab 4: 想法生成 ====================
        with gr.Tab("💡 想法生成"):
            gr.Markdown("## 创新想法生成（DeepSeek）")
            gr.Markdown("""
            基于论文分析结果，生成创新性强的研究想法
            
            **输入格式**：
            ```
            【Paper_1】论文名：分析内容...
            【Paper_2】论文名：分析内容...
            【Paper_3】论文名：分析内容...
            ```
            
            **生成内容**：
            - 多个创新想法（详细描述）
            - 创新性评分（0-100分）
            - 按评分从高到低排序
            """)
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 1️⃣ 生成创新想法")
                    generate_btn = gr.Button("💡 生成想法", variant="primary", size="lg")
                    ideas_output = gr.Textbox(label="生成结果", lines=20)
                    
                    generate_btn.click(
                        fn=generate_ideas,
                        outputs=ideas_output
                    )
                
                with gr.Column():
                    gr.Markdown("### 2️⃣ 查看完整想法")
                    view_ideas_btn = gr.Button("👁️ 查看想法", variant="secondary", size="lg")
                    ideas_viewer = gr.Markdown(label="想法内容")
                    
                    view_ideas_btn.click(
                        fn=view_ideas,
                        outputs=ideas_viewer
                    )
        
        # ==================== Tab 5: 代码生成 ====================
        with gr.Tab("💻 代码生成"):
            gr.Markdown("## 代码实现生成")
            gr.Markdown("### 5️⃣ 生成Python代码")
            
            code_btn = gr.Button("💻 生成代码", variant="primary", size="lg")
            code_output = gr.Code(label="生成的代码", language="python", lines=20)
            
            code_btn.click(fn=generate_code, outputs=code_output)
        
        # ==================== Tab 6: 完整流程 ====================
        with gr.Tab("🚀 完整流程"):
            gr.Markdown("## 一键运行完整流程")
            gr.Markdown("""
            ### 流程说明
            1. 提取PDF文档
            2. 清洗论文内容
            3. 分析论文（翻译、推导）
            4. 生成创新想法
            5. 筛选最优想法
            6. 详细化想法
            7. 生成代码实现
            
            ⚠️ **注意**: 完整流程可能需要较长时间（取决于论文数量和API速度）
            """)
            
            full_btn = gr.Button("🚀 运行完整流程", variant="primary", size="lg")
            full_output = gr.Textbox(label="执行结果", lines=20)
            
            full_btn.click(fn=run_full_pipeline, outputs=full_output)
        
        # ==================== Footer ====================
        gr.Markdown("""
        ---
        ### 📚 使用说明
        1. 先在"配置"页面设置API密钥
        2. 在"PDF提取"页面上传或指定PDF
        3. 按顺序执行各个步骤，或直接运行"完整流程"
        4. 结果会自动保存在 `data/` 目录下
        
        ### 💡 提示
        - PDF URL必须是公开可访问的链接
        - MinerU提供更高精度，但需要PDF URL
        - PyPDF2可处理本地文件，但精度较低
        
        ### 📖 文档
        - [使用指南](README.md)
        - [MinerU指南](docs/MINERU_GUIDE.md)
        - [API配置](docs/API_KEY_CONFIG.md)
        """)
    
    return app


# ==================== 启动 ====================

def main():
    """启动Web UI"""
    app = create_ui()
    
    print("""
╔════════════════════════════════════════════════════════════╗
║              AgentColab Web UI 启动中...                   ║
╚════════════════════════════════════════════════════════════╝

界面将在浏览器中自动打开
如未自动打开，请访问: http://localhost:7860

按 Ctrl+C 停止服务器
""")
    
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )


if __name__ == "__main__":
    main()

