from transformers import AutoConfig, AutoTokenizer

# 替换为你实际的下载路径
local_model_path = "/your/path/to/embodied-eval-main/embodied_eval/data/Embodied-VLM-8B-RFT-0307"

try:
    # 1. 测试加载配置文件
    config = AutoConfig.from_pretrained(local_model_path, trust_remote_code=True)
    print("✅ 配置文件 (Config) 加载成功！")

    # 2. 测试加载分词器
    tokenizer = AutoTokenizer.from_pretrained(local_model_path, trust_remote_code=True)
    print("✅ 分词器 (Tokenizer) 加载成功！")
    
    print("🎉 基础验证通过，模型文件结构基本完整！")

except Exception as e:
    print("❌ 加载失败，模型文件可能不完整。错误信息如下：")
    print(e)