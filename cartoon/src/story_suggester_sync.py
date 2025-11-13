from __future__ import annotations
import os, json, base64, mimetypes
from pathlib import Path
from typing import List, Optional
from openai import OpenAI
import sys, pathlib

from frontend.load_env import ensure_env_loaded
ensure_env_loaded()

from openai import OpenAI
_client = OpenAI()

def _require_openai_key():
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "[story_suggester] OPENAI_API_KEY not set. "
            "Load via .env (python-dotenv) or export before launch.\n"
            f"CWD={pathlib.Path().resolve()} EXE={sys.executable}"
        )
    return key

def _to_data_url(path: str) -> str:
    p = Path(path)
    mime = mimetypes.guess_type(p.name)[0] or "image/png"
    b64 = base64.b64encode(p.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{b64}"

def _coerce_4(items: List[str]) -> List[str]:
    # 정확히 4개로 맞춤(부족시 빈칸, 초과시 잘라냄)
    xs = list(items[:4])
    while len(xs) < 4:
        xs.append("")
    return xs

def suggest_story(
    *,
    brand_intro: str,
    core_message: str,
    panel_image_paths: List[str],
    language: str = "ko",           # "ko"|"en"
    max_chars: int = 42,            # 패널별 글자 제한(캡션 바에 예쁘게)
    model: Optional[str] = None,    # None=기본(환경설정)
) -> List[str]:

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("[story_suggester] OPENAI_API_KEY not set; return empty captions.")
        return ["", "", "", ""]

    client = OpenAI(api_key=api_key)
    model = model or os.getenv("CARTOON_STORY_MODEL", "gpt-4.1-mini")
     
    # 이미지들을 data URL로 첨부
    imgs = []
    for p in (panel_image_paths or [])[:4]:
        try:
            imgs.append({"type": "image_url", "image_url": {"url": _to_data_url(p)}})
        except Exception as e:
            print(f"[story_suggester] skip image {p}: {e}")

    # 시스템/유저 메시지(출력 포맷은 JSON 고정)
    sys = (
        "You write ultra concise ad copy for Instagram 4-panel posts. "
        "Given brand info, core message and 4 panel images, produce an engaging 4-sentence story: "
        "one short sentence per panel, in order. Keep it suitable for overlay captions; no hashtags, no emojis, "
        "no quotes, no numbering. Return pure JSON: {\"lines\": [\"...\",\"...\",\"...\",\"...\"]}."
    )
    locale = "Korean" if language.lower().startswith("ko") else "English"

    user_text = (
        f"Brand intro: {brand_intro}\n"
        f"Core message: {core_message}\n\n"
        f"Write {locale} copy.\n"
        f"Hard limits per sentence: ≤ {max_chars} characters.\n"
        f"Style: friendly, brand-aligned, high-contrast caption, no hashtags/emojis."
    )

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": sys},
                {"role": "user", "content": [{"type": "text", "text": user_text}] + imgs},
            ],
        )
        content = resp.choices[0].message.content or ""
        # 모델이 종종 마크다운을 붙이는 걸 대비해 JSON만 뽑기
        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end != -1 and end > start:
            js = json.loads(content[start:end+1])
            lines = js.get("lines") or []
        else:
            # JSON 실패시 줄바꿈 기반 폴백
            lines = [ln.strip("-• ").strip() for ln in content.splitlines() if ln.strip()][:4]
        lines = [ln[:max_chars] for ln in lines]
        return _coerce_4(lines)
    except Exception as e:
        print("[story_suggester] OpenAI call failed:", e)
        return ["", "", "", ""]


# ---- FastAPI Router ----
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional

router = APIRouter(tags=["story-suggester"])

@router.get("/ping")
async def ping(): return {"ok": True}

class SuggestRequest(BaseModel):
    brand_intro: str = ""
    core_message: str = ""
    panel_image_paths: List[str] = []
    language: str = "ko"
    max_chars: int = 42
    model: Optional[str] = None

class SuggestResponse(BaseModel):
    lines: List[str]

@router.post("/suggest", response_model=SuggestResponse)
async def api_suggest(req: SuggestRequest):
    try:
        lines = suggest_story(
            brand_intro=req.brand_intro,
            core_message=req.core_message,
            panel_image_paths=req.panel_image_paths,
            language=req.language,
            max_chars=req.max_chars,
            model=req.model
        )
        return SuggestResponse(lines=lines)
    except SystemExit as e: raise HTTPException(status_code=400, detail=str(e))
    except Exception as e: raise HTTPException(status_code=500, detail=f"suggest_story failed: {e}")
