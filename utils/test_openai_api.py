import os
from openai import OpenAI

def test_openai_api():
    api_key = 'your-api-key'
    api_base = 'https://openai.arnotho.com/v1'
    client = OpenAI(api_key=api_key, base_url=api_base)
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": "你好，请简单介绍一下自己"}
            ],
            max_tokens=100
        )
        
        print("✅ API 测试成功")
        print(f"回复: {response.choices[0].message.content}")
        print(f"使用 tokens: {response.usage.total_tokens}")
        
    except Exception as e:
        print(f"❌ API 测试失败: {e}")

if __name__ == "__main__":

    test_openai_api()