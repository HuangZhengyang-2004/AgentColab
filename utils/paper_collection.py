"""
论文集合管理工具
用于管理多篇论文的解析结果
"""

import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime


class PaperCollection:
    """论文集合类，管理多篇论文的解析内容"""
    
    def __init__(self):
        """初始化论文集合"""
        self.papers = {}  # 格式: {"paper_1": {"name": "...", "content": "..."}}
        self.metadata = {
            "created_at": datetime.now().isoformat(),
            "total_papers": 0,
            "last_updated": datetime.now().isoformat()
        }
    
    def add_paper(self, paper_name: str, content: str, index: Optional[int] = None) -> str:
        """
        添加一篇论文
        
        Args:
            paper_name: 论文名称
            content: 论文解析内容
            index: 论文索引（可选，不指定则自动递增）
        
        Returns:
            paper_key: 论文的键名（如 "paper_1"）
        """
        if index is None:
            index = len(self.papers) + 1
        
        paper_key = f"paper_{index}"
        
        self.papers[paper_key] = {
            "name": paper_name,
            "content": content,
            "added_at": datetime.now().isoformat(),
            "content_length": len(content)
        }
        
        self.metadata["total_papers"] = len(self.papers)
        self.metadata["last_updated"] = datetime.now().isoformat()
        
        return paper_key
    
    def add_papers_batch(self, papers: Dict[str, str]) -> Dict[str, str]:
        """
        批量添加论文
        
        Args:
            papers: 字典 {paper_name: content}
        
        Returns:
            映射字典 {paper_name: paper_key}
        """
        mapping = {}
        for i, (name, content) in enumerate(papers.items(), start=len(self.papers) + 1):
            paper_key = self.add_paper(name, content, index=i)
            mapping[name] = paper_key
        
        return mapping
    
    def get_paper(self, paper_key: str) -> Optional[Dict]:
        """
        获取单篇论文
        
        Args:
            paper_key: 论文键名（如 "paper_1"）
        
        Returns:
            论文信息字典，包含 name 和 content
        """
        return self.papers.get(paper_key)
    
    def get_paper_content(self, paper_key: str) -> Optional[str]:
        """
        获取论文内容
        
        Args:
            paper_key: 论文键名（如 "paper_1"）
        
        Returns:
            论文内容文本
        """
        paper = self.papers.get(paper_key)
        return paper["content"] if paper else None
    
    def get_paper_by_name(self, paper_name: str) -> Optional[Dict]:
        """
        根据论文名称查找论文
        
        Args:
            paper_name: 论文名称
        
        Returns:
            包含 paper_key 和论文信息的字典
        """
        for key, paper in self.papers.items():
            if paper["name"] == paper_name:
                return {"paper_key": key, **paper}
        return None
    
    def list_papers(self) -> List[Dict]:
        """
        列出所有论文
        
        Returns:
            论文列表，每项包含 paper_key, name, content_length
        """
        result = []
        for key in sorted(self.papers.keys(), key=lambda x: int(x.split('_')[1])):
            paper = self.papers[key]
            result.append({
                "paper_key": key,
                "name": paper["name"],
                "content_length": paper["content_length"],
                "added_at": paper.get("added_at", "")
            })
        return result
    
    def get_all_contents(self) -> Dict[str, str]:
        """
        获取所有论文的内容
        
        Returns:
            字典 {paper_key: content}
        """
        return {key: paper["content"] for key, paper in self.papers.items()}
    
    def get_summary(self) -> Dict:
        """
        获取集合摘要信息
        
        Returns:
            摘要字典
        """
        total_chars = sum(p["content_length"] for p in self.papers.values())
        return {
            **self.metadata,
            "total_characters": total_chars,
            "papers": [
                {"key": k, "name": p["name"], "length": p["content_length"]}
                for k, p in sorted(self.papers.items(), 
                                  key=lambda x: int(x[0].split('_')[1]))
            ]
        }
    
    def save_to_json(self, filepath: str):
        """
        保存为JSON文件
        
        Args:
            filepath: 保存路径
        """
        data = {
            "metadata": self.metadata,
            "papers": self.papers
        }
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"✓ 已保存到: {filepath}")
    
    def save_to_pickle(self, filepath: str):
        """
        保存为Pickle文件（更快，但不可读）
        
        Args:
            filepath: 保存路径
        """
        data = {
            "metadata": self.metadata,
            "papers": self.papers
        }
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"✓ 已保存到: {filepath}")
    
    @classmethod
    def load_from_json(cls, filepath: str) -> 'PaperCollection':
        """
        从JSON文件加载
        
        Args:
            filepath: 文件路径
        
        Returns:
            PaperCollection实例
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        collection = cls()
        collection.metadata = data.get("metadata", collection.metadata)
        collection.papers = data.get("papers", {})
        
        print(f"✓ 已加载 {len(collection.papers)} 篇论文")
        return collection
    
    @classmethod
    def load_from_pickle(cls, filepath: str) -> 'PaperCollection':
        """
        从Pickle文件加载
        
        Args:
            filepath: 文件路径
        
        Returns:
            PaperCollection实例
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        collection = cls()
        collection.metadata = data.get("metadata", collection.metadata)
        collection.papers = data.get("papers", {})
        
        print(f"✓ 已加载 {len(collection.papers)} 篇论文")
        return collection
    
    @classmethod
    def from_extracted_dir(cls, extracted_dir: str = "data/extracted") -> 'PaperCollection':
        """
        从extracted目录加载所有已提取的论文
        
        Args:
            extracted_dir: extracted目录路径
        
        Returns:
            PaperCollection实例
        """
        from pathlib import Path
        
        collection = cls()
        extracted_path = Path(extracted_dir)
        
        # 查找所有_extracted.txt文件
        txt_files = list(extracted_path.glob("*_extracted.txt"))
        
        for i, txt_file in enumerate(sorted(txt_files), start=1):
            paper_name = txt_file.stem.replace("_extracted", "")
            
            # 读取内容
            with open(txt_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            collection.add_paper(paper_name, content, index=i)
            print(f"✓ 加载: {paper_name}")
        
        print(f"\n✓ 共加载 {len(collection.papers)} 篇论文")
        return collection
    
    def __len__(self):
        """返回论文数量"""
        return len(self.papers)
    
    def __getitem__(self, paper_key: str):
        """支持字典式访问"""
        return self.papers.get(paper_key)
    
    def __repr__(self):
        """字符串表示"""
        return f"PaperCollection(papers={len(self.papers)})"
    
    def __str__(self):
        """打印信息"""
        lines = [
            f"论文集合 (共 {len(self.papers)} 篇)",
            "=" * 50
        ]
        for key in sorted(self.papers.keys(), key=lambda x: int(x.split('_')[1])):
            paper = self.papers[key]
            lines.append(f"{key}: {paper['name']} ({paper['content_length']} 字符)")
        return "\n".join(lines)


def create_collection_from_extraction(papers_dict: Dict[str, str]) -> PaperCollection:
    """
    便捷函数：从提取结果创建集合
    
    Args:
        papers_dict: 提取结果 {paper_name: content}
    
    Returns:
        PaperCollection实例
    
    Example:
        >>> from agents import PDFExtractorAgent
        >>> agent = PDFExtractorAgent()
        >>> results = agent.run()  # 返回 {name: content}
        >>> collection = create_collection_from_extraction(results)
        >>> collection.save_to_json("data/collections/my_papers.json")
    """
    collection = PaperCollection()
    collection.add_papers_batch(papers_dict)
    return collection

