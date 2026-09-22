import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "./qwen0.5b"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.float16, device_map="cuda"
)

prompt = "<|im_start|>system\nYou are a helpful assistant. Think step by step.<|im_end|>\n<|im_start|>user\nWhich is bigger, 9.9 or 9.11?<|im_end|>\n<|im_start|>assistant\n<thought>\n"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

# 1. 模拟生成第一段思考
out = model.generate(**inputs, max_new_tokens=30, do_sample=False)
raw_thought = tokenizer.decode(out[0])

# 2. Budget Forcing 介入：若思考过短，强行在尾部追加 "Wait"
forced_thought = raw_thought.rstrip() + " Wait,"

# 3. 强迫模型从 Wait 处再次反思
new_inputs = tokenizer(forced_thought, return_tensors="pt").to("cuda")
final_out = model.generate(**new_inputs, max_new_tokens=60, do_sample=False)

print("=== 强行注入 Wait 后的思考演化 ===")
print(tokenizer.decode(final_out[0]))
print("\n✅ Budget Forcing 逻辑跑通！")
