"""
Google Gemini API 调用示例
这个脚本展示了如何使用 Google Gemini API 进行文本生成和对话
"""

import os
import google.generativeai as genai


def setup_gemini_api():
    """
    配置 Gemini API
    需要在环境变量中设置 GOOGLE_API_KEY
    """
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        raise ValueError("请设置环境变量 GOOGLE_API_KEY")
    
    genai.configure(api_key=api_key)
    return api_key


def simple_text_generation():
    """
    示例1: 简单的文本生成
    """
    print("=" * 50)
    print("示例1: 简单文本生成")
    print("=" * 50)
    
    # 初始化模型
    model = genai.GenerativeModel('gemini-2.5-flash')
    
    # 生成文本
    prompt = "请用简短的话解释什么是人工智能"
    response = model.generate_content(prompt)
    
    print(f"提问: {prompt}")
    print(f"回答: {response.text}\n")


def streaming_generation():
    """
    示例2: 流式生成（实时输出）
    """
    print("=" * 50)
    print("示例2: 流式生成")
    print("=" * 50)
    
    model = genai.GenerativeModel('gemini-2.5-flash')
    
    prompt = "写一首关于春天的短诗"
    print(f"提问: {prompt}")
    print("回答: ", end="")
    
    response = model.generate_content(prompt, stream=True)
    for chunk in response:
        print(chunk.text, end="", flush=True)
    print("\n")


def chat_conversation():
    """
    示例3: 多轮对话
    """
    print("=" * 50)
    print("示例3: 多轮对话")
    print("=" * 50)
    
    model = genai.GenerativeModel('gemini-2.5-flash')
    chat = model.start_chat(history=[])
    
    # 第一轮对话
    message1 = "你好，我想学习 Python 编程"
    response1 = chat.send_message(message1)
    print(f"用户: {message1}")
    print(f"AI: {response1.text}\n")
    
    # 第二轮对话（基于上下文）
    message2 = "我应该从哪里开始？"
    response2 = chat.send_message(message2)
    print(f"用户: {message2}")
    print(f"AI: {response2.text}\n")
    
    # 显示完整对话历史
    print("完整对话历史:")
    for i, msg in enumerate(chat.history):
        role = "用户" if msg.role == "user" else "AI"
        print(f"{i+1}. {role}: {msg.parts[0].text[:50]}...")


def generation_with_config():
    """
    示例4: 使用配置参数控制生成
    """
    print("=" * 50)
    print("示例4: 配置参数控制")
    print("=" * 50)
    
    # 配置生成参数
    generation_config = {
        "temperature": 0.9,  # 控制随机性 (0.0-1.0)
        "top_p": 1,
        "top_k": 1,
        "max_output_tokens": 200,  # 最大输出长度
    }
    
    # 安全设置
    safety_settings = [
        {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
        {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
    ]
    
    model = genai.GenerativeModel(
        'gemini-1.5-flash',
        generation_config=generation_config,
        safety_settings=safety_settings
    )
    
    prompt = "创造性地描述一个未来城市"
    response = model.generate_content(prompt)
    
    print(f"提问: {prompt}")
    print(f"回答: {response.text}\n")


def list_available_models():
    """
    示例5: 列出可用的模型
    """
    print("=" * 50)
    print("示例5: 可用模型列表")
    print("=" * 50)
    
    print("可用的 Gemini 模型:")
    for model in genai.list_models():
        if 'generateContent' in model.supported_generation_methods:
            print(f"- {model.name}")
            print(f"  描述: {model.description}")
            print(f"  输入Token限制: {model.input_token_limit}")
            print(f"  输出Token限制: {model.output_token_limit}")
            print()


def main():
    """
    主函数
    """
    try:
        # 配置 API
        setup_gemini_api()
        print("✓ API 配置成功\n")
        
        # 运行各个示例
        simple_text_generation()
        streaming_generation()
        chat_conversation()
        generation_with_config()
        list_available_models()
        
        print("=" * 50)
        print("所有示例执行完成！")
        print("=" * 50)
        
    except Exception as e:
        print(f"错误: {str(e)}")
        print("\n提示:")
        print("1. 确保已安装依赖: pip install -r requirements.txt")
        print("2. 确保已设置环境变量: export GOOGLE_API_KEY='your-api-key'")
        print("3. 获取 API Key: https://makersuite.google.com/app/apikey")


if __name__ == "__main__":
    main()

