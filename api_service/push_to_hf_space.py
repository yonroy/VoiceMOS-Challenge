#!/usr/bin/env python3
"""
Push thư mục api_service/ lên Hugging Face Space (SDK = Docker).

Cách dùng (PowerShell):
    $env:HF_TOKEN = "hf_xxx"          # token WRITE (Settings → Access Tokens)
    python push_to_hf_space.py        # tạo Space mặc định + upload
    python push_to_hf_space.py --repo yonroy/voicemos2026-api --private

Yêu cầu: pip install huggingface_hub
Lưu ý: KHÔNG hardcode token vào file. Nếu HF repo checkpoint Track 2 để PRIVATE,
       thêm Secret HF_TOKEN cho Space (--set-token-secret) để pull được lúc chạy.
"""
import argparse
import os
import sys

from huggingface_hub import HfApi, create_repo

HERE = os.path.dirname(os.path.abspath(__file__))

# Không đẩy những thứ này lên Space (rác / nặng / không cần để build)
IGNORE = [
    "__pycache__/*", "*/__pycache__/*", "*.pyc",
    "models/*", ".venv/*", ".git/*", "*.pt", "*.zip",
    "push_to_hf_space.py",          # bản thân script này
    "tests/*",                       # smoke test, không cần trên Space
]


def main():
    ap = argparse.ArgumentParser(description="Deploy VoiceMOS 3-track API lên HF Space (Docker).")
    ap.add_argument("--repo", default="yonroy/voicemos2026-api",
                    help="Space repo id dạng <user>/<space-name>")
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN"),
                    help="HF token WRITE (mặc định lấy từ env HF_TOKEN)")
    ap.add_argument("--private", action="store_true", help="Tạo Space ở chế độ private")
    ap.add_argument("--hardware", default=None,
                    help="Nâng phần cứng Space, vd: cpu-upgrade, t4-small, t4-medium (mặc định: free cpu-basic)")
    ap.add_argument("--set-token-secret", action="store_true",
                    help="Đặt Secret HF_TOKEN cho Space (cần nếu repo checkpoint là private)")
    ap.add_argument("--message", default="Deploy VoiceMOS 2026 3-track MOS API (FastAPI · Docker Space)")
    args = ap.parse_args()

    if not args.token:
        sys.exit("❌ Thiếu token. Đặt $env:HF_TOKEN = 'hf_...' hoặc truyền --token.")

    api = HfApi(token=args.token)

    print(f"→ Tạo/đảm bảo Space: {args.repo} (sdk=docker, private={args.private})")
    create_repo(args.repo, repo_type="space", space_sdk="docker",
                private=args.private, exist_ok=True, token=args.token)

    if args.set_token_secret:
        print("→ Đặt Secret HF_TOKEN cho Space (để pull checkpoint private)")
        api.add_space_secret(repo_id=args.repo, key="HF_TOKEN", value=args.token)

    if args.hardware:
        print(f"→ Nâng hardware: {args.hardware}")
        api.request_space_hardware(repo_id=args.repo, hardware=args.hardware)

    print(f"→ Upload thư mục {HERE} …")
    api.upload_folder(
        folder_path=HERE,
        repo_id=args.repo,
        repo_type="space",
        ignore_patterns=IGNORE,
        commit_message=args.message,
    )

    url = f"https://huggingface.co/spaces/{args.repo}"
    app_url = f"https://{args.repo.replace('/', '-').lower()}.hf.space"
    print("\n✅ Đã push. Space sẽ tự build (xem tab 'Logs').")
    print(f"   Trang Space : {url}")
    print(f"   API base    : {app_url}")
    print(f"   Swagger UI  : {app_url}/docs")


if __name__ == "__main__":
    main()
