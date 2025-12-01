"""
测试论文清洗功能
"""

import sys
sys.path.insert(0, '/home/hzy/Tom/Agent_Colab')

from agents.paper_cleaner_agent import PaperCleanerAgent
from utils.paper_collection import PaperCollection


def test_clean_sample_text():
    """测试清洗示例文本"""
    print("=" * 60)
    print("测试1: 清洗示例文本")
    print("=" * 60)
    
    sample_text = """
    # A Novel Algorithm for Signal Processing
    
    ## Abstract
    This paper presents a novel algorithm for signal processing...
    
    ## 1. Introduction
    Signal processing is an important field [1]. Previous work by Smith et al. (2020) 
    showed that this approach works well [2-5].
    
    ## 2. Methodology
    Our method uses the following approach...
    The formula is: x = y + z
    
    ## 3. Results
    We achieved 95% accuracy...
    
    ## 4. Conclusion
    This paper demonstrated...
    
    ## References
    [1] Smith, J. (2020). "Signal Processing Methods"
    [2] Jones, A. (2019). "Advanced Techniques"
    [3] Brown, B. et al. (2018). "Novel Approaches"
    
    ## Acknowledgments
    We thank our colleagues for helpful discussions.
    This work was supported by NSF Grant 12345.
    
    ## Appendix A: Supplementary Material
    Additional experimental results...
    """
    
    agent = PaperCleanerAgent()
    cleaned = agent.clean_paper(sample_text)
    
    print("\n清洗后的文本:")
    print("-" * 60)
    print(cleaned)
    print("-" * 60)
    print(f"\n原始长度: {len(sample_text)}")
    print(f"清洗后长度: {len(cleaned)}")
    print(f"删除率: {(1 - len(cleaned)/len(sample_text))*100:.1f}%")
    

def test_clean_actual_papers():
    """测试清洗实际论文"""
    print("\n" + "=" * 60)
    print("测试2: 清洗实际论文")
    print("=" * 60)
    
    # 从集合加载论文
    collection = PaperCollection.load_from_json("data/collections/all_papers.json")
    papers = collection.get_all_contents()
    
    print(f"\n加载了 {len(papers)} 篇论文")
    
    # 清洗第一篇论文
    if papers:
        paper_key = list(papers.keys())[0]
        paper_content = papers[paper_key]
        
        print(f"\n清洗 {paper_key}:")
        print(f"原始长度: {len(paper_content):,} 字符")
        
        agent = PaperCleanerAgent()
        cleaned_content = agent.clean_paper(paper_content)
        
        print(f"清洗后长度: {len(cleaned_content):,} 字符")
        print(f"删除率: {(1 - len(cleaned_content)/len(paper_content))*100:.1f}%")
        
        # 显示清洗后的预览
        print("\n清洗后内容预览（前500字符）:")
        print("-" * 60)
        print(cleaned_content[:500])
        print("...")


def test_clean_all_papers():
    """测试清洗所有论文并保存"""
    print("\n" + "=" * 60)
    print("测试3: 清洗所有论文并保存")
    print("=" * 60)
    
    agent = PaperCleanerAgent()
    
    # 运行清洗（会自动从集合加载）
    cleaned_papers = agent.run()
    
    print(f"\n✓ 清洗完成！共处理 {len(cleaned_papers)} 篇论文")
    
    # 显示统计
    print("\n清洗统计:")
    print("-" * 60)
    
    # 加载原始集合对比
    original_collection = PaperCollection.load_from_json("data/collections/all_papers.json")
    original_papers = original_collection.get_all_contents()
    
    for paper_key in cleaned_papers.keys():
        original_len = len(original_papers.get(paper_key, ""))
        cleaned_len = len(cleaned_papers[paper_key])
        removal_rate = (1 - cleaned_len/original_len)*100 if original_len > 0 else 0
        
        print(f"{paper_key}:")
        print(f"  原始: {original_len:,} 字符")
        print(f"  清洗: {cleaned_len:,} 字符")
        print(f"  删除: {removal_rate:.1f}%")
    
    # 验证保存的文件
    print("\n保存的文件:")
    print("-" * 60)
    
    from pathlib import Path
    cleaned_dir = Path("data/cleaned")
    if cleaned_dir.exists():
        cleaned_files = list(cleaned_dir.glob("*_cleaned.txt"))
        for f in cleaned_files:
            print(f"  ✓ {f.name} ({f.stat().st_size:,} 字节)")
    
    # 验证集合文件
    collection_file = Path("data/collections/all_papers_cleaned.json")
    if collection_file.exists():
        print(f"\n  ✓ 集合文件: {collection_file.name}")
        
        # 加载并验证
        cleaned_collection = PaperCollection.load_from_json(str(collection_file))
        print(f"    包含 {len(cleaned_collection)} 篇清洗后的论文")


def test_pattern_matching():
    """测试各种参考文献格式的匹配"""
    print("\n" + "=" * 60)
    print("测试4: 参考文献格式匹配")
    print("=" * 60)
    
    test_cases = [
        "## References\n[1] Smith...",
        "## REFERENCES\n[1] Smith...",
        "7. References\n[1] Smith...",
        "VII. REFERENCES\n[1] Smith...",
        "\nReferences\n\n[1] Smith...",
        "## 参考文献\n[1] 张三...",
    ]
    
    agent = PaperCleanerAgent()
    
    for i, test_text in enumerate(test_cases, 1):
        result = agent._find_section_start(test_text, agent.section_keywords['references'])
        found = "✓ 找到" if result is not None else "✗ 未找到"
        print(f"测试 {i}: {found}")
        print(f"  文本: {test_text[:30].strip()}...")


if __name__ == "__main__":
    print("╔════════════════════════════════════════════════════════════╗")
    print("║          论文清洗功能测试                                  ║")
    print("╚════════════════════════════════════════════════════════════╝")
    
    # 运行所有测试
    test_clean_sample_text()
    test_pattern_matching()
    
    # 测试实际论文
    try:
        test_clean_actual_papers()
        test_clean_all_papers()
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("✓ 测试完成！")
    print("=" * 60)

