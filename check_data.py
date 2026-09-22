from datasets import load_dataset
from transformers import AutoTokenizer

print(">>> [1/3] 测试数据拉取...")
# 若国内网络受阻，取消下行注释
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

dataset = load_dataset("simplescaling/s1K_tokenized", split="train")
print(f"✅ 数据拉取成功！共 {len(dataset)} 条样本。")
print(f"数据集包含的字段: {list(dataset.features.keys())}")

print("\n>>> [2/3] 查看原始 'text' 字段前 200 字...")
sample_text = dataset[0]["text"]
print("--------------------------------------------------")
print(sample_text[:200] + "\n... (中间思考过程省略) ...")
print("--------------------------------------------------")

print("\n>>> [3/3] 测试 Tokenizer 对 'text' 进行分词编码...")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
tokenized_sample = tokenizer(sample_text)
input_ids = tokenized_sample["input_ids"]

print(f"✅ 分词成功！该样本被分为了 {len(input_ids)} 个 Token。")
print(f"解码前 30 个 Token 验证:\n{tokenizer.decode(input_ids[:30])}")
print("\n🎉 第一步数据通路验证完全通过！")
