#!/usr/bin/env python3
"""
MinerU 简单测试脚本
快速验证MinerU API是否可用
"""

import os
import sys

def check_api_key():
    """检查API密钥"""
    print("="*60)
    print("1️⃣  检查MinerU API密钥")
    print("="*60)
    
    # 检查环境变量
    env_key = os.getenv('MINERU_API_KEY')
    
    if env_key:
        masked = env_key[:10] + "..." + env_key[-4:] if len(env_key) > 14 else "***"
        print(f"✓ 环境变量已设置: {masked}")
        return env_key
    
    # 检查配置文件
    try:
        from utils.config_loader import config_loader
        config = config_loader.get_api_config('mineru')
        config_key = config.get('api_key', '')
        
        if config_key:
            masked = config_key[:10] + "..." + config_key[-4:] if len(config_key) > 14 else "***"
            print(f"✓ 配置文件已设置: {masked}")
            return config_key
    except Exception as e:
        print(f"⚠ 读取配置文件失败: {e}")
    
    print("❌ 未找到API密钥")
    print("\n请设置API密钥:")
    print("  方式1: export MINERU_API_KEY='your_key'")
    print("  方式2: 在 config.yaml 中配置")
    print("\n获取密钥: https://mineru.net/")
    return None


def test_mineru_api(api_key):
    """测试MinerU API"""
    print("\n" + "="*60)
    print("2️⃣  测试MinerU API连接")
    print("="*60)
    
    try:
        from utils.mineru_client import MinerUClient
        
        print("创建MinerU客户端...")
        client = MinerUClient(api_key)
        print("✓ 客户端创建成功")
        
        # 使用官方示例PDF测试
        test_url = "https://cdn-mineru.openxlab.org.cn/demo/example.pdf"
        print(f"\n测试PDF: {test_url}")
        
        print("\n创建解析任务...")
        result = client.create_task(
            file_url=test_url,
            model_version="vlm",
            data_id="test_example"
        )
        
        task_id = result["data"]["task_id"]
        print(f"✓ 任务创建成功")
        print(f"  Task ID: {task_id}")
        
        print("\n等待任务完成（这可能需要几分钟）...")
        print("  [提示] 你可以按 Ctrl+C 中断等待")
        
        try:
            status = client.wait_for_task(task_id, max_wait_time=300, poll_interval=3)
            
            if status.state == "done":
                print("\n✅ 解析完成！")
                print(f"  结果URL: {status.full_zip_url}")
                
                # 下载并查看结果
                print("\n下载解析结果...")
                save_dir = "data/extracted/test_mineru"
                files = client.download_result(status.full_zip_url, save_dir)
                
                if files.get('markdown'):
                    with open(files['markdown'], 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    print("\n" + "="*60)
                    print("3️⃣  提取的内容预览")
                    print("="*60)
                    preview = content[:500]
                    print(preview)
                    if len(content) > 500:
                        print("...")
                    
                    print(f"\n总长度: {len(content)} 字符")
                    print(f"保存位置: {files['markdown']}")
                    
                    print("\n" + "="*60)
                    print("🎉 MinerU测试成功！")
                    print("="*60)
                    return True
            else:
                print(f"\n❌ 解析失败: {status.err_msg}")
                return False
                
        except KeyboardInterrupt:
            print("\n\n⚠️  用户中断等待")
            print(f"任务 {task_id} 仍在处理中")
            print("你可以稍后使用以下命令查询结果：")
            print(f"  client.get_task_status('{task_id}')")
            return False
            
    except Exception as e:
        print(f"\n❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_pdf_extractor_agent(api_key):
    """测试PDF提取Agent"""
    print("\n" + "="*60)
    print("4️⃣  测试PDF提取Agent")
    print("="*60)
    
    try:
        from agents.pdf_extractor_agent import PDFExtractorAgent
        
        print("创建PDF提取Agent...")
        agent = PDFExtractorAgent(use_mineru=True)
        print("✓ Agent创建成功")
        
        test_url = "https://cdn-mineru.openxlab.org.cn/demo/example.pdf"
        print(f"\n提取PDF: {test_url}")
        
        content = agent.extract_from_url(
            pdf_url=test_url,
            pdf_name="agent_test",
            model_version="vlm"
        )
        
        print("\n✓ Agent提取成功！")
        print(f"  内容长度: {len(content)} 字符")
        print(f"  保存位置: data/extracted/agent_test_extracted.txt")
        
        print("\n" + "="*60)
        print("✅ PDF提取Agent测试成功！")
        print("="*60)
        return True
        
    except Exception as e:
        print(f"\n❌ Agent测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    print("""
╔════════════════════════════════════════════════════════════╗
║          MinerU PDF解析 快速测试                           ║
╚════════════════════════════════════════════════════════════╝

本测试将验证:
  1. API密钥配置
  2. MinerU API连接
  3. PDF解析功能
  4. Agent集成

测试使用MinerU官方示例PDF，不消耗你的额度。
""")
    
    # 检查API密钥
    api_key = check_api_key()
    
    if not api_key:
        print("\n请先设置API密钥后再运行测试")
        sys.exit(1)
    
    # 询问是否继续
    print("\n是否开始测试？ (y/n): ", end="")
    try:
        choice = input().strip().lower()
        if choice != 'y':
            print("测试已取消")
            sys.exit(0)
    except:
        # 如果是非交互环境，自动继续
        print("y")
    
    # 测试MinerU API
    api_success = test_mineru_api(api_key)
    
    if not api_success:
        print("\n⚠️  MinerU API测试未完成")
        print("请检查:")
        print("  1. API密钥是否正确")
        print("  2. 网络连接是否正常")
        print("  3. 查看上面的错误信息")
        sys.exit(1)
    
    # 测试Agent
    agent_success = test_pdf_extractor_agent(api_key)
    
    # 总结
    print("\n" + "="*60)
    print("📊 测试总结")
    print("="*60)
    print(f"  MinerU API:     {'✅ 通过' if api_success else '❌ 失败'}")
    print(f"  PDF提取Agent:   {'✅ 通过' if agent_success else '❌ 失败'}")
    
    if api_success and agent_success:
        print("\n🎉 所有测试通过！MinerU模块可以正常使用了。")
        print("\n下一步:")
        print("  1. 将你的PDF上传到云存储获取URL")
        print("  2. 使用 agent.extract_from_url() 提取PDF")
        print("  3. 或使用 ./run.sh full 运行完整流程")
    else:
        print("\n⚠️  部分测试失败，请检查配置")
    
    print("="*60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n用户中断测试")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n测试程序异常: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

