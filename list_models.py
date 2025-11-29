"""
列出所有可用的 Gemini 模型
"""

import os
import google.generativeai as genai

# 配置 API
api_key = os.getenv('GOOGLE_API_KEY')
if not api_key:
    print("错误: 请设置环境变量 GOOGLE_API_KEY")
    exit(1)

genai.configure(api_key=api_key)

print("正在获取可用模型列表...\n")
print("=" * 70)

for model in genai.list_models():
    print(f"模型名称: {model.name}")
    print(f"显示名称: {model.display_name}")
    print(f"支持的方法: {', '.join(model.supported_generation_methods)}")
    print("-" * 70)

