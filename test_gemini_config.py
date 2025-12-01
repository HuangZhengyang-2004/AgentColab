#!/usr/bin/env python3
"""
测试 Gemini API 配置
"""

import os
import sys
import google.generativeai as genai

def test_gemini_models():
    """测试不同的 Gemini 模型名称"""
    
    # 获取 API Key - 优先从环境变量，其次从 config.yaml
    api_key = os.getenv('GOOGLE_API_KEY')
    
    if not api_key:
        # 尝试从 config_loader 读取
        try:
            sys.path.insert(0, os.path.dirname(__file__))
            from utils.config_loader import config_loader
            api_config = config_loader.get_api_config('gemini')
            api_key = api_config.get('api_key', '')
        except Exception as e:
            print(f"⚠️  从 config.yaml 读取失败: {str(e)}")
    
    if not api_key:
        print("❌ 未设置 GOOGLE_API_KEY")
        print("\n请设置环境变量:")
        print("export GOOGLE_API_KEY='your-api-key'")
        print("\n或在 config.yaml 中配置 api_keys.google_api_key")
        return
    
    print("=" * 60)
    print("🔍 测试 Gemini API 配置")
    print("=" * 60)
    print(f"✅ API Key 已设置: {api_key[:10]}...")
    print()
    
    # 配置 API
    genai.configure(api_key=api_key)
    
    # 1. 列出所有可用模型
    print("📋 步骤1: 列出所有可用的 Gemini 模型")
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
    
    # 2. 测试不同的模型名称格式
    print("📋 步骤2: 测试不同的模型名称格式")
    print("-" * 60)
    
    test_models = [
        "gemini-1.5-flash",
        "models/gemini-1.5-flash",
        "gemini-1.5-flash-latest",
        "models/gemini-1.5-flash-latest",
        "gemini-1.5-pro",
        "models/gemini-1.5-pro",
        "gemini-2.0-flash-exp",
        "models/gemini-2.0-flash-exp",
    ]
    
    successful_models = []
    
    for model_name in test_models:
        try:
            print(f"\n🧪 测试: {model_name}")
            model = genai.GenerativeModel(model_name)
            response = model.generate_content("请用一句话说明什么是AI。")
            
            if response.text:
                print(f"   ✅ 成功！")
                print(f"   📥 响应: {response.text[:100]}...")
                successful_models.append(model_name)
            else:
                print(f"   ⚠️  无响应内容")
                
        except Exception as e:
            error_msg = str(e)
            if "404" in error_msg:
                print(f"   ❌ 404错误 - 模型不存在")
            elif "429" in error_msg:
                print(f"   ⚠️  429错误 - 配额超限")
            elif "RECITATION" in error_msg or "finish_reason" in error_msg:
                print(f"   ⚠️  RECITATION错误 - 内容被过滤")
            else:
                print(f"   ❌ 错误: {error_msg[:100]}")
    
    # 3. 总结
    print("\n" + "=" * 60)
    print("📊 测试总结")
    print("=" * 60)
    
    if successful_models:
        print(f"✅ 成功的模型 ({len(successful_models)}):")
        for model in successful_models:
            print(f"   • {model}")
        
        print("\n💡 推荐配置 (config.yaml):")
        print(f"   model: \"{successful_models[0]}\"")
    else:
        print("❌ 没有成功的模型")
        print("\n可能的原因:")
        print("   1. API Key 无效")
        print("   2. 配额已用完")
        print("   3. 网络连接问题")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    test_gemini_models()

