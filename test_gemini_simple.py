#!/usr/bin/env python3
"""
简单的 Gemini API 测试
"""

import os
import google.generativeai as genai

def main():
    print("=" * 60)
    print("🔍 Gemini API 测试")
    print("=" * 60)
    
    # 1. 获取 API Key
    api_key = os.getenv('GOOGLE_API_KEY')
    
    if not api_key:
        print("❌ 未设置 GOOGLE_API_KEY 环境变量")
        print("\n请先设置:")
        print("export GOOGLE_API_KEY='your-api-key-here'")
        print("\n或者直接在这里输入你的 API Key 进行测试:")
        api_key = input("API Key: ").strip()
        
        if not api_key:
            print("❌ 未提供 API Key，退出测试")
            return
    
    print(f"✅ API Key: {api_key[:10]}...{api_key[-4:]}")
    print()
    
    # 2. 配置 API
    genai.configure(api_key=api_key)
    
    # 3. 列出可用模型
    print("📋 可用的 Gemini 模型:")
    print("-" * 60)
    
    available_models = []
    try:
        for model in genai.list_models():
            if 'generateContent' in model.supported_generation_methods:
                available_models.append(model.name)
                print(f"✅ {model.name}")
        print()
    except Exception as e:
        print(f"❌ 列出模型失败: {str(e)}")
        return
    
    if not available_models:
        print("❌ 没有找到可用的模型")
        return
    
    # 4. 测试配置中的模型
    print("🧪 测试配置中的模型:")
    print("-" * 60)
    
    test_models = [
        "gemini-1.5-flash",  # config.yaml 中的配置
        "models/gemini-1.5-flash",
        "gemini-1.5-flash-latest",
        "models/gemini-1.5-flash-latest",
    ]
    
    successful_model = None
    
    for model_name in test_models:
        try:
            print(f"\n🔍 测试: {model_name}")
            model = genai.GenerativeModel(model_name)
            
            # 发送简单请求
            response = model.generate_content("请用一句话说明什么是人工智能。")
            
            if response.text:
                print(f"   ✅ 成功！")
                print(f"   📥 响应: {response.text}")
                successful_model = model_name
                break  # 找到可用的就停止
            else:
                print(f"   ⚠️  无响应内容")
                
        except Exception as e:
            error_msg = str(e)
            if "404" in error_msg:
                print(f"   ❌ 404 - 模型不存在")
            elif "429" in error_msg:
                print(f"   ⚠️  429 - 配额超限")
                if "quota" in error_msg.lower():
                    print(f"   💡 建议: 等待配额恢复或使用其他模型")
            else:
                print(f"   ❌ 错误: {error_msg[:150]}")
    
    # 5. 总结
    print("\n" + "=" * 60)
    print("📊 测试结果")
    print("=" * 60)
    
    if successful_model:
        print(f"✅ 找到可用的模型: {successful_model}")
        print("\n💡 推荐配置 (config.yaml):")
        print(f"   model: \"{successful_model}\"")
        print("\n🎉 Gemini API 配置正确！")
    else:
        print("❌ 所有测试的模型都失败了")
        print("\n💡 建议:")
        print("   1. 检查 API Key 是否有效")
        print("   2. 检查是否有可用配额")
        print("   3. 尝试使用 DeepSeek 作为备选:")
        print("      api_provider: 'deepseek'")
        print("      model: 'deepseek-chat'")
    
    print("=" * 60)


if __name__ == "__main__":
    main()

