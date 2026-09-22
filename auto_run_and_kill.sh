#!/bin/bash
set -e

# ================= 1. 配置区域 =================
# 通知密钥统一放在 config/.env（FEISHU_WEBHOOK / FEISHU_SECRET），
# 由 scripts/run_eval.sh 自动加载，无需在此硬编码。

# ================= 2. 启动评测（跳过 SFT）=================
# 直接拉取 simplescaling/s1-32B，跑原版 Budget Forcing（强塞 "Wait"）推理机制。
# 统一入口见 scripts/run_eval.sh（内置 watchdog + 飞书通知）。
echo ">>> [$(date)] 评测开始，跳过 SFT 训练..."
start_time=$(date +%s)

bash scripts/run_eval.sh --config config/config.yaml

end_time=$(date +%s)
cost_time=$(( (end_time - start_time) / 60 ))
echo ">>> [$(date)] 评测完成！耗时: ${cost_time} 分钟，结果见 outputs/ 目录。"

# ================= 3. 手机推送通知 (由 s1_eval.notify 统一发送) =================
# 评测的开始/进度/完成/报错/超时 都会自动推送飞书，无需在此重复发送。
# 如需手动补发一条：
python send_feishu.py "评测结束，耗时 ${cost_time} 分钟" --event complete || true

# ================= 4. 触发自动关机/销毁 (按平台二选一) =================
echo ">>> 正在执行自动停机，终止扣费..."

# 【情况 A：如果你在 AutoDL】
# AutoDL 官方自带关机脚本，执行后实例立即变为“已关机”，停止 GPU 扣费
if [ -f "/usr/bin/autodl-shutdown" ]; then
    /usr/bin/autodl-shutdown
fi

# 【情况 B：如果你在 Vast.ai】
# Vast.ai 可以直接通过其 CLI 销毁容器实例（开机时需配置好 VAST_CONTAINER_ID）
# vastai destroy instance $CONTAINER_ID

# 【兜底方案】强行关闭虚拟机系统
shutdown -h now