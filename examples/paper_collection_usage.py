"""
论文集合使用示例
演示如何使用PaperCollection管理多篇论文
"""

from utils.paper_collection import PaperCollection, create_collection_from_extraction


def example_1_manual_creation():
    """示例1：手动创建论文集合"""
    print("=" * 60)
    print("示例1：手动创建论文集合")
    print("=" * 60)
    
    # 创建集合
    collection = PaperCollection()
    
    # 添加论文
    collection.add_paper(
        paper_name="Deep Learning Survey",
        content="This is the content of the first paper..."
    )
    
    collection.add_paper(
        paper_name="Transformer Architecture",
        content="This is the content of the second paper..."
    )
    
    collection.add_paper(
        paper_name="Attention Mechanism",
        content="This is the content of the third paper..."
    )
    
    # 查看集合
    print(collection)
    print()
    
    # 访问特定论文
    paper1 = collection.get_paper("paper_1")
    print(f"Paper 1: {paper1['name']}")
    print(f"Content: {paper1['content'][:50]}...")
    print()


def example_2_batch_creation():
    """示例2：批量创建（从提取结果）"""
    print("=" * 60)
    print("示例2：批量创建论文集合")
    print("=" * 60)
    
    # 模拟PDF提取结果
    extraction_results = {
        "Paper A on Neural Networks": "Neural networks are...",
        "Paper B on Computer Vision": "Computer vision techniques...",
        "Paper C on NLP": "Natural language processing..."
    }
    
    # 创建集合
    collection = create_collection_from_extraction(extraction_results)
    
    # 列出所有论文
    papers = collection.list_papers()
    for paper in papers:
        print(f"{paper['paper_key']}: {paper['name']} ({paper['content_length']} 字符)")
    print()


def example_3_from_extracted_directory():
    """示例3：从data/extracted目录加载"""
    print("=" * 60)
    print("示例3：从extracted目录加载")
    print("=" * 60)
    
    # 从已提取的论文创建集合
    collection = PaperCollection.from_extracted_dir("data/extracted")
    
    # 查看摘要
    summary = collection.get_summary()
    print(f"\n总论文数: {summary['total_papers']}")
    print(f"总字符数: {summary['total_characters']}")
    print(f"创建时间: {summary['created_at']}")
    print()


def example_4_save_and_load():
    """示例4：保存和加载"""
    print("=" * 60)
    print("示例4：保存和加载集合")
    print("=" * 60)
    
    # 创建集合
    collection = PaperCollection()
    collection.add_paper("Paper 1", "Content 1...")
    collection.add_paper("Paper 2", "Content 2...")
    
    # 保存为JSON
    collection.save_to_json("data/collections/test_collection.json")
    
    # 保存为Pickle（更快，但不可读）
    collection.save_to_pickle("data/collections/test_collection.pkl")
    
    # 加载
    loaded_collection = PaperCollection.load_from_json(
        "data/collections/test_collection.json"
    )
    print(f"加载的集合: {loaded_collection}")
    print()


def example_5_integration_with_agent():
    """示例5：与PDFExtractorAgent集成"""
    print("=" * 60)
    print("示例5：与PDFExtractorAgent集成")
    print("=" * 60)
    
    # 实际使用场景
    print("""
# 完整工作流示例：

from agents import PDFExtractorAgent
from utils.paper_collection import create_collection_from_extraction

# 1. 提取PDF
agent = PDFExtractorAgent(use_mineru=True)
results = agent.run()  # 返回 {paper_name: content}

# 2. 创建集合
collection = create_collection_from_extraction(results)

# 3. 保存集合
collection.save_to_json("data/collections/my_papers.json")

# 4. 后续使用
collection = PaperCollection.load_from_json("data/collections/my_papers.json")

# 5. 访问论文
for paper_key in ["paper_1", "paper_2", "paper_3"]:
    paper = collection.get_paper(paper_key)
    print(f"{paper_key}: {paper['name']}")
    
# 6. 获取所有内容（用于后续分析）
all_contents = collection.get_all_contents()
# 返回: {"paper_1": "content...", "paper_2": "content...", ...}

# 7. 按名称查找
paper = collection.get_paper_by_name("某篇论文的名字")
if paper:
    print(f"找到论文: {paper['paper_key']}")
    """)


def example_6_access_patterns():
    """示例6：各种访问方式"""
    print("=" * 60)
    print("示例6：不同的访问方式")
    print("=" * 60)
    
    collection = PaperCollection()
    collection.add_paper("Paper Alpha", "Content Alpha...")
    collection.add_paper("Paper Beta", "Content Beta...")
    collection.add_paper("Paper Gamma", "Content Gamma...")
    
    # 方式1：通过键访问
    paper = collection["paper_1"]
    print(f"方式1 - 字典访问: {paper['name']}")
    
    # 方式2：获取论文对象
    paper = collection.get_paper("paper_2")
    print(f"方式2 - get_paper: {paper['name']}")
    
    # 方式3：只获取内容
    content = collection.get_paper_content("paper_3")
    print(f"方式3 - get_paper_content: {content[:30]}...")
    
    # 方式4：按名称查找
    paper = collection.get_paper_by_name("Paper Beta")
    print(f"方式4 - get_paper_by_name: {paper['paper_key']}")
    
    # 方式5：获取所有内容
    all_contents = collection.get_all_contents()
    print(f"方式5 - get_all_contents: {len(all_contents)} 篇论文")
    
    # 方式6：列出所有论文
    papers = collection.list_papers()
    print(f"方式6 - list_papers:")
    for p in papers:
        print(f"  {p['paper_key']}: {p['name']}")
    print()


if __name__ == "__main__":
    # 运行所有示例
    example_1_manual_creation()
    example_2_batch_creation()
    example_3_from_extracted_directory()
    example_4_save_and_load()
    example_5_integration_with_agent()
    example_6_access_patterns()
    
    print("=" * 60)
    print("✓ 所有示例运行完成！")
    print("=" * 60)

