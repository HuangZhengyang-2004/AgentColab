"""
论文集合管理功能 - 用于Web UI
"""

import json
from pathlib import Path
from typing import Tuple, Dict


def load_collection_info(collection_path: str) -> str:
    """加载并显示集合信息"""
    try:
        if not collection_path or not Path(collection_path).exists():
            return "❌ 请先提取PDF或选择有效的集合文件"
        
        with open(collection_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        papers = data.get('papers', {})
        metadata = data.get('metadata', {})
        
        info = f"📚 论文集合信息\n"
        info += f"=" * 60 + "\n\n"
        info += f"📊 统计:\n"
        info += f"  • 总论文数: {len(papers)}\n"
        info += f"  • 总字符数: {sum(p['content_length'] for p in papers.values()):,}\n"
        info += f"  • 创建时间: {metadata.get('created_at', 'N/A')[:19]}\n\n"
        
        info += f"📄 论文列表:\n"
        for key in sorted(papers.keys(), key=lambda x: int(x.split('_')[1])):
            paper = papers[key]
            name = paper['name'][:50] + "..." if len(paper['name']) > 50 else paper['name']
            info += f"  {key}: {name}\n"
            info += f"         ({paper['content_length']:,} 字符)\n"
        
        return info
        
    except Exception as e:
        return f"❌ 加载失败: {str(e)}"


def view_paper_content(collection_path: str, paper_key: str) -> str:
    """查看特定论文内容"""
    try:
        if not collection_path or not Path(collection_path).exists():
            return "❌ 请先提取PDF或选择有效的集合文件"
        
        with open(collection_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        papers = data.get('papers', {})
        
        if paper_key not in papers:
            available = ', '.join(sorted(papers.keys()))
            return f"❌ 论文 '{paper_key}' 不存在\n可用的键: {available}"
        
        paper = papers[paper_key]
        
        output = f"📄 {paper_key}\n"
        output += f"=" * 60 + "\n\n"
        output += f"名称: {paper['name']}\n"
        output += f"长度: {paper['content_length']:,} 字符\n"
        output += f"添加时间: {paper.get('added_at', 'N/A')[:19]}\n\n"
        output += f"内容预览 (前1000字符):\n"
        output += "-" * 60 + "\n"
        output += paper['content'][:1000]
        output += "\n\n..." if len(paper['content']) > 1000 else ""
        
        return output
        
    except Exception as e:
        return f"❌ 错误: {str(e)}"


def export_collection_summary(collection_path: str) -> Tuple[str, str]:
    """导出集合摘要为文本文件"""
    try:
        if not collection_path or not Path(collection_path).exists():
            return "❌ 请先提取PDF或选择有效的集合文件", ""
        
        with open(collection_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        papers = data.get('papers', {})
        metadata = data.get('metadata', {})
        
        # 生成摘要文本
        summary = f"论文集合摘要\n"
        summary += f"{'=' * 60}\n\n"
        summary += f"创建时间: {metadata.get('created_at', 'N/A')}\n"
        summary += f"总论文数: {len(papers)}\n"
        summary += f"总字符数: {sum(p['content_length'] for p in papers.values()):,}\n\n"
        
        summary += f"论文详情:\n"
        summary += f"{'-' * 60}\n\n"
        
        for key in sorted(papers.keys(), key=lambda x: int(x.split('_')[1])):
            paper = papers[key]
            summary += f"{key}:\n"
            summary += f"  名称: {paper['name']}\n"
            summary += f"  长度: {paper['content_length']:,} 字符\n"
            summary += f"  添加: {paper.get('added_at', 'N/A')[:19]}\n"
            summary += f"  预览: {paper['content'][:200]}...\n\n"
        
        # 保存摘要
        output_path = collection_path.replace('.json', '_summary.txt')
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(summary)
        
        return f"✓ 摘要已导出到:\n{output_path}\n\n{summary}", output_path
        
    except Exception as e:
        return f"❌ 错误: {str(e)}", ""


def merge_collections(collection_paths: str) -> Tuple[str, str]:
    """合并多个集合"""
    try:
        paths = [p.strip() for p in collection_paths.strip().split('\n') if p.strip()]
        
        if len(paths) < 2:
            return "❌ 请输入至少2个集合路径（每行一个）", ""
        
        from utils.paper_collection import PaperCollection
        
        # 创建新集合
        merged = PaperCollection()
        
        total_loaded = 0
        for path in paths:
            if not Path(path).exists():
                return f"❌ 文件不存在: {path}", ""
            
            temp = PaperCollection.load_from_json(path)
            merged.add_papers_batch(temp.get_all_contents())
            total_loaded += len(temp)
        
        # 保存合并结果
        output_path = "data/collections/merged_collection.json"
        merged.save_to_json(output_path)
        
        report = f"✓ 合并成功！\n\n"
        report += f"合并了 {len(paths)} 个集合\n"
        report += f"总论文数: {total_loaded} → {len(merged)}\n"
        report += f"保存位置: {output_path}\n"
        
        return report, output_path
        
    except Exception as e:
        return f"❌ 错误: {str(e)}", ""

