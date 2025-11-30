"""
MinerU API测试脚本
用于测试MinerU PDF解析功能
"""

import os
from utils.mineru_client import get_mineru_client
from agents.pdf_extractor_agent import PDFExtractorAgent


def test_mineru_api():
    """测试MinerU API连接和基本功能"""
    print("="*60)
    print("MinerU API 测试")
    print("="*60)
    
    # 检查API密钥
    api_key = os.getenv('MINERU_API_KEY')
    if not api_key:
        print("❌ 未设置 MINERU_API_KEY 环境变量")
        print("\n请设置API密钥后再试:")
        print("  export MINERU_API_KEY='your_api_key_here'")
        return
    
    print(f"✓ API密钥已设置: {api_key[:8]}...")
    
    try:
        # 创建客户端
        print("\n1. 初始化MinerU客户端...")
        client = get_mineru_client(api_key)
        print("   ✓ 客户端创建成功")
        
        # 测试PDF URL（使用MinerU官方示例）
        test_url = "https://cdn-mineru.openxlab.org.cn/demo/example.pdf"
        print(f"\n2. 测试PDF解析...")
        print(f"   URL: {test_url}")
        
        # 创建任务
        print("\n   创建解析任务...")
        result = client.create_task(
            file_url=test_url,
            model_version="vlm",
            data_id="test_example"
        )
        
        task_id = result["data"]["task_id"]
        print(f"   ✓ 任务创建成功")
        print(f"   Task ID: {task_id}")
        
        # 等待完成
        print("\n   等待任务完成...")
        status = client.wait_for_task(task_id, max_wait_time=300, poll_interval=3)
        
        if status.state == "done":
            print(f"   ✓ 解析完成！")
            print(f"   结果URL: {status.full_zip_url}")
            
            # 下载结果
            print("\n3. 下载解析结果...")
            save_dir = "data/extracted/test_mineru"
            files = client.download_result(status.full_zip_url, save_dir)
            
            print(f"   ✓ 下载完成")
            print(f"   Markdown: {files.get('markdown')}")
            print(f"   JSON: {files.get('json')}")
            
            # 显示部分内容
            if files.get('markdown'):
                with open(files['markdown'], 'r', encoding='utf-8') as f:
                    content = f.read()
                print(f"\n4. 提取的内容预览:")
                print("   " + "-"*56)
                print("   " + content[:500].replace('\n', '\n   '))
                if len(content) > 500:
                    print("   ...")
                print("   " + "-"*56)
                print(f"   总长度: {len(content)} 字符")
            
            print("\n" + "="*60)
            print("✅ MinerU API 测试成功！")
            print("="*60)
            
        else:
            print(f"   ❌ 解析失败: {status.err_msg}")
    
    except Exception as e:
        print(f"\n❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()


def test_pdf_extractor_agent():
    """测试PDF提取Agent"""
    print("\n" + "="*60)
    print("PDF提取Agent 测试")
    print("="*60)
    
    api_key = os.getenv('MINERU_API_KEY')
    if not api_key:
        print("❌ 未设置 MINERU_API_KEY，跳过测试")
        return
    
    try:
        # 创建Agent
        print("\n1. 创建PDF提取Agent...")
        agent = PDFExtractorAgent(use_mineru=True)
        print("   ✓ Agent创建成功")
        
        # 测试URL提取
        test_url = "https://cdn-mineru.openxlab.org.cn/demo/example.pdf"
        print(f"\n2. 使用Agent提取PDF...")
        print(f"   URL: {test_url}")
        
        content = agent.extract_from_url(
            pdf_url=test_url,
            pdf_name="test_example",
            model_version="vlm"
        )
        
        print(f"\n   ✓ 提取成功！")
        print(f"   内容长度: {len(content)} 字符")
        print(f"\n   内容预览:")
        print("   " + "-"*56)
        print("   " + content[:300].replace('\n', '\n   '))
        if len(content) > 300:
            print("   ...")
        print("   " + "-"*56)
        
        print("\n" + "="*60)
        print("✅ PDF提取Agent 测试成功！")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()


def main():
    """主函数"""
    print("""
╔════════════════════════════════════════════════════════════╗
║              MinerU PDF解析 测试程序                       ║
╚════════════════════════════════════════════════════════════╝

本程序将测试MinerU API的各项功能。

测试内容:
1. MinerU API 基本功能
2. PDF提取Agent集成

""")
    
    try:
        # 测试1: MinerU API
        test_mineru_api()
        
        # 测试2: PDF提取Agent
        print("\n\n")
        test_pdf_extractor_agent()
        
    except KeyboardInterrupt:
        print("\n\n用户中断测试")
    except Exception as e:
        print(f"\n\n测试程序异常: {str(e)}")


if __name__ == "__main__":
    main()

