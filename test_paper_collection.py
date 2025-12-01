"""
简化的论文集合使用示例（不依赖其他模块）
"""

import sys
sys.path.insert(0, '/home/hzy/Tom/Agent_Colab')

from utils.paper_collection import PaperCollection, create_collection_from_extraction


print("=" * 60)
print("论文集合功能演示")
print("=" * 60)
print()

# 示例1：创建集合并添加论文
print("1. 创建集合并添加论文")
print("-" * 60)
collection = PaperCollection()

collection.add_paper(
    paper_name="Deep Learning Survey",
    content="This is a comprehensive survey on deep learning methods..."
)

collection.add_paper(
    paper_name="Transformer Architecture",
    content="The Transformer model revolutionized NLP..."
)

collection.add_paper(
    paper_name="Attention Is All You Need",
    content="We propose a new simple network architecture..."
)

print(collection)
print()

# 示例2：访问论文
print("2. 访问特定论文")
print("-" * 60)
paper1 = collection.get_paper("paper_1")
print(f"Paper Key: paper_1")
print(f"Name: {paper1['name']}")
print(f"Content: {paper1['content'][:50]}...")
print(f"Length: {paper1['content_length']} 字符")
print()

# 示例3：列出所有论文
print("3. 列出所有论文")
print("-" * 60)
papers = collection.list_papers()
for paper in papers:
    print(f"  {paper['paper_key']}: {paper['name']} ({paper['content_length']} 字符)")
print()

# 示例4：保存和加载
print("4. 保存到JSON")
print("-" * 60)
collection.save_to_json("data/collections/demo_papers.json")
print()

# 示例5：从extracted目录加载
print("5. 从extracted目录加载实际论文")
print("-" * 60)
try:
    real_collection = PaperCollection.from_extracted_dir("data/extracted")
    print(f"\n实际论文集合:")
    summary = real_collection.get_summary()
    print(f"  总论文数: {summary['total_papers']}")
    print(f"  总字符数: {summary['total_characters']:,}")
    print(f"\n论文列表:")
    for p in summary['papers']:
        print(f"  {p['key']}: {p['name'][:50]}... ({p['length']:,} 字符)")
except Exception as e:
    print(f"  注意: {e}")
print()

#示例6：获取所有内容
print("6. 获取所有内容（用于后续处理）")
print("-" * 60)
all_contents = collection.get_all_contents()
print(f"获取到 {len(all_contents)} 篇论文的内容:")
for key, content in all_contents.items():
    print(f"  {key}: {len(content)} 字符")
print()

print("=" * 60)
print("✓ 演示完成！")
print("=" * 60)

