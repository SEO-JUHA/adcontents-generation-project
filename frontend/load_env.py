# load_env.py (프로젝트 루트)
from __future__ import annotations
import os
from dotenv import load_dotenv, find_dotenv

# .env를 "프로젝트 루트 기준"으로 확실히 찾도록
# usecwd=True: 실행 위치가 어딘지와 무관하게, 현재 파일 기준 상위에서 탐색
def ensure_env_loaded(override: bool = False) -> None:
    path = find_dotenv(filename=".env", usecwd=True)
    if path:
        load_dotenv(path, override=override)
    # 필수 키 점검(필요시 경고만)
    if not os.getenv("OPENAI_API_KEY"):
        print("[load_env] WARN: OPENAI_API_KEY not set (still running without GPT).")
