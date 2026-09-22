import os

# 1. 强制在 Python 最顶层杀死失效镜像，重置为官方源
os.environ["HF_ENDPOINT"] = "https://huggingface.co"
os.environ["WANDB_DISABLED"] = "true"

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


def run_dry_run():
    print(">>> 正在加载 0.5B 测试模型到 5070 Mobile 显存 (bfloat16)...")
    model_id = "./qwen0.5b"

    # 尝试优先读取本地已有的缓存文件，彻底绕过网络请求
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            local_files_only=True,
        )
        print(">>> 成功从本地缓存极速加载模型！")
    except Exception:
        print(">>> 本地缓存不全，正在从官方源安全下载...")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="cuda"
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(">>> 加载数据集并提取前 4 条进行极简训练...")
    ds = load_dataset("simplescaling/s1K_tokenized", split="train").select(range(4))

    training_args = SFTConfig(
        output_dir="./test_ckpt",
        max_steps=2,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=1e-5,
        bf16=True,
        fp16=False,
        max_seq_length=512,
        dataset_text_field="text",
        logging_steps=1,
        save_strategy="no",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        processing_class=tokenizer,
    )

    print(">>> 开始执行本地训练冒烟测试...")
    trainer.train()
    print(
        "\n🎉🎉🎉 恭喜！本地 5070 Mobile 训练全流程（前向/反传/优化器步进）完美跑通！"
    )


if __name__ == "__main__":
    run_dry_run()
