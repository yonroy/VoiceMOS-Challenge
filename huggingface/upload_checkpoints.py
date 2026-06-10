"""
Đẩy checkpoint VoiceMOS 2026 lên Hugging Face Models.

Cách dùng (sau khi đã có tài khoản HF + token write):
    pip install -U huggingface_hub
    huggingface-cli login            # dán token (write) — hoặc set HF_TOKEN env var
    python huggingface/upload_checkpoints.py

Trước khi chạy: sửa HF_USER cho đúng username của bạn.
Checkpoint lớn (1.27GB) → upload dùng git-lfs ngầm, cần mạng khỏe; có resume nếu đứt.
"""
import os
from huggingface_hub import HfApi, create_repo

# ── SỬA Ở ĐÂY ────────────────────────────────────────────────────────────────
HF_USER   = "tranminhtoan140601"                     # << username HF
REPO_ID   = f"{HF_USER}/voicemos2026-track2-emotion"  # repo Models chứa checkpoint
PRIVATE   = False                                     # True nếu muốn để riêng tư trước
PROJECT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # gốc dự án

# (đường dẫn local → tên file trên HF). Bỏ dòng nào không muốn đẩy.
FILES = {
    os.path.join(PROJECT, "cache", "ft_emotion_full_20epoch.pt"): "ft_emotion_full_20epoch.pt",  # exp08 cảm xúc (TỐT NHẤT)
    os.path.join(PROJECT, "cache", "ft_qmos_utmos.pt"):           "ft_qmos_utmos.pt",            # exp13 QMOS
    os.path.join(PROJECT, "ft_joint_full.pt"):                    "ft_joint_full.pt",            # exp11 fusion 2-backbone
}
# ──────────────────────────────────────────────────────────────────────────────


def main():
    assert HF_USER != "YOUR_HF_USERNAME", "❌ Sửa HF_USER thành username HF của bạn trước đã."
    api = HfApi()
    print(f"→ Tạo/đảm bảo repo: {REPO_ID} (private={PRIVATE})")
    create_repo(REPO_ID, repo_type="model", private=PRIVATE, exist_ok=True)

    # Model card
    card = os.path.join(os.path.dirname(__file__), "model_card_README.md")
    if os.path.exists(card):
        api.upload_file(path_or_fileobj=card, path_in_repo="README.md",
                        repo_id=REPO_ID, repo_type="model")
        print("✅ Đã đẩy README (model card)")

    for local, remote in FILES.items():
        if not os.path.exists(local):
            print(f"⚠️ BỎ QUA (không thấy): {local}")
            continue
        sz = os.path.getsize(local) / 1e6
        print(f"→ Upload {remote} ({sz:.0f} MB) ... (lớn → có thể lâu)")
        api.upload_file(path_or_fileobj=local, path_in_repo=remote,
                        repo_id=REPO_ID, repo_type="model")
        print(f"✅ Xong {remote}")

    print(f"\n🎉 Hoàn tất → https://huggingface.co/{REPO_ID}")


if __name__ == "__main__":
    main()
