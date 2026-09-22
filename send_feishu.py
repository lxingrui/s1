import base64
import hashlib
import hmac
import json
import os
import sys
import time
import urllib.request


def _get_credentials():
    """从环境变量 / config/.env 读取 webhook 与 secret（不再硬编码密钥）。"""
    if not os.environ.get("FEISHU_WEBHOOK"):
        for env_path in ("config/.env", ".env"):
            if os.path.exists(env_path):
                with open(env_path, encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            key, _, value = line.partition("=")
                            os.environ.setdefault(
                                key.strip(), value.strip().strip('"').strip("'")
                            )
                break
    return os.environ.get("FEISHU_WEBHOOK", ""), os.environ.get("FEISHU_SECRET", "")


def gen_sign(secret, timestamp):
    """飞书官方标准签名算法 (HMAC-SHA256)"""
    string_to_sign = f"{timestamp}\n{secret}"
    hmac_code = hmac.new(
        string_to_sign.encode("utf-8"), digestmod=hashlib.sha256
    ).digest()
    sign = base64.b64encode(hmac_code).decode("utf-8")
    return sign


def send_feishu_notification(title, message):
    webhook, secret = _get_credentials()
    if not webhook:
        print("❌ 未配置 FEISHU_WEBHOOK（请在 config/.env 中设置）")
        return
    current_time = int(time.time())
    sign = gen_sign(secret, current_time)

    # 包含签名、时间戳与消息卡片的完整报文
    payload = {
        "timestamp": str(current_time),
        "sign": sign,
        "msg_type": "interactive",
        "card": {
            "config": {"wide_screen_mode": True},
            "header": {
                "title": {"tag": "plain_text", "content": f"🔔 {title}"},
                "template": "blue",
            },
            "elements": [
                {
                    "tag": "div",
                    "text": {
                        "tag": "lark_md",
                        "content": (
                            f"**实验项目**: s1-32B 全量复现\n**运行状态**:"
                            f" {message}\n**通知方式**: 密钥签名鉴权"
                        ),
                    },
                }
            ],
        },
    }

    try:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            webhook, data=data, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req) as resp:
            res = json.loads(resp.read().decode("utf-8"))
            if res.get("StatusCode") == 0 or res.get("code") == 0:
                print("✅ 飞书通知推送成功（签名验证通过）！")
            else:
                print(f"❌ 推送失败，飞书返回: {res}")
    except Exception as e:
        print(f"❌ 网络推送异常: {e}")


if __name__ == "__main__":
    status_text = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "训练成功完成！权重已同步至 HuggingFace，云端实例正在关机。"
    )
    send_feishu_notification("s1 训练任务状态通知", status_text)
