from __future__ import annotations
from typing import Tuple, List, Optional, Dict, Any
import os, sys
import json
from pathlib import Path
from collections import deque
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[2]
sp = str(ROOT)
if sp not in sys.path:
    sys.path.insert(0, sp)

try:
    from load_env import ensure_env_loaded
    ensure_env_loaded()
except Exception:
    try:
        from dotenv import load_dotenv, find_dotenv
        load_dotenv(find_dotenv(filename=".env", usecwd=True))
    except Exception:
        pass


# ========== GPT toggle & helpers ==========
def _gpt_enabled() -> bool:
    return (
        os.getenv("CARTOON_USE_GPT", "").strip() not in ("", "0", "false", "False")
        and bool(os.getenv("OPENAI_API_KEY"))
    )

def _default_system_prompt() -> str:
    """
    최신 시스템 프롬프트 (person_food 역할 포함)
    """
    return (
        "You are a prompt normalizer for a cafe comic generator.\n"
        "Input will be JSON with keys: brand_intro, category_hint, optional story_beat, optional panel_role.\n"
        "Return a JSON object with keys 'positive' and 'negative'.\n\n"
        "Rules:\n"
        "- Always stay in the cafe domain (exterior/storefront, interior, hero food/coffee, or atmospheric cafe).\n"
        "- Positive = background/scene/mood/composition & brand color harmony. English, <60 tokens.\n"
        "- Negative = hard constraints: no readable text/typography/slogans/watermarks/logo imprints; "
        "avoid clutter/busy layout; avoid low quality/compression artifacts.\n\n"
        "Role-specific constraints:\n"
        "- exterior: cafe facade/storefront/signage; architectural composition; no people.\n"
        "- interior: seating/counter/materials/lighting; cozy ambience; no people.\n"
        "- food: hero shot of a signature menu (coffee/latte/pastry). No hands/people.\n"
        "- person_food: one person holding the featured food/drink; waist-up framing; clean background; "
        "focus on the food in hand; allow a single person only; no crowd; no text.\n"
        "- mood: atmospheric cafe scene consistent with brand colors; no people.\n"
    )

def _load_system_prompt() -> str:
    if os.getenv("CARTOON_USE_SYSTEM_PROMPT_FILE", "").strip() in ("1", "true", "True"):
        sys_path = Path(__file__).resolve().parents[1] / "prompts" / "system_prompt.txt"
        try:
            return sys_path.read_text(encoding="utf-8")
        except Exception:
            pass
    return _default_system_prompt()

_client = None
def _client_singleton():
    global _client
    if _client is None:
        _client = OpenAI()
    return _client

def _call_gpt_for_json(messages, *, model=None):
    cli = _client_singleton()
    model = model or os.getenv("CARTOON_GPT_MODEL", "gpt-4.1-mini")
    temp = float(os.getenv("CARTOON_GPT_TEMP", "0.2"))
    try:
        r = cli.chat.completions.create(
            model=model,
            temperature=temp,
            messages=messages
        )
        text = (r.choices[0].message.content or "").strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0]
        return json.loads(text)
    except Exception as e:
        print(f"[prompt_builder_sync] GPT call failed: {e}")
        return None

# ========== Role inference (panel) ==========
_KO_EN_KEYS: Dict[str, List[str]] = {
    "exterior": ["exterior", "facade", "storefront", "signage", "외관", "간판", "입구"],
    "food": ["food", "menu", "dish", "coffee", "latte", "dessert", "음식", "메뉴", "커피", "라떼", "디저트"],
    "interior": ["interior", "seating", "counter", "barista", "매장", "인테리어", "좌석", "테이블", "바"],
    "mood": ["mood", "concept", "vibe", "atmosphere", "무드", "컨셉", "분위기"],
    # 사람+음식 역할 키워드
    "person_food": ["person", "people", "holding", "hand", "hands", "사람", "들고", "손", "인물", "손에", "들다"],
}

def _env_true(name: str, default: bool = True) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return str(v).strip().lower() not in ("0", "false", "no", "off", "")

# person_food 강제 정책
_FORCE_PF = _env_true("CARTOON_FORCE_PERSON_FOOD", True)
_FORCE_PF_POS = os.getenv("CARTOON_FORCE_PF_POS", "last").strip().lower()   # last | any
_FORCE_PF_RELAX = _env_true("CARTOON_FORCE_PF_RELAX", False)

# 최근 4개 역할 추적(한 작업 묶음에서 4회 연속 호출 가정)
_RECENT_ROLES = deque(maxlen=4)

def _contains_food_hint(text: str) -> bool:
    t = (text or "").lower()
    return any(k in t for k in _KO_EN_KEYS["food"])

def _coerce_person_food_if_needed(inferred_role: str, story_beat: str) -> str:
    """
    최근 4개 창 내에 person_food가 없다면 정책에 따라 person_food로 강제 전환.
    - 'last' : 4번째 호출에서 없으면 마지막 컷을 person_food로 전환(RELAX=1이면 힌트 없어도 강제).
    - 'any'  : 창에 아직 없고, 이 비트가 food/인물 힌트면 곧바로 person_food 부여.
    """
    role = inferred_role

    if not _FORCE_PF:
        _RECENT_ROLES.append(role)
        return role

    has_pf = any(r == "person_food" for r in _RECENT_ROLES)
    window_len = len(_RECENT_ROLES)

    if has_pf:
        _RECENT_ROLES.append(role)
        return role

    if _FORCE_PF_POS == "any":
        if role != "person_food" and (_FORCE_PF_RELAX or _contains_food_hint(story_beat) or inferred_role == "food"):
            role = "person_food"
        _RECENT_ROLES.append(role)
        return role

    # 기본: last
    if window_len == 3 and role != "person_food":
        if _FORCE_PF_RELAX or _contains_food_hint(story_beat) or inferred_role == "food":
            role = "person_food"
    _RECENT_ROLES.append(role)
    return role

def _infer_panel_role(story_beat: str) -> str:
    beat = (story_beat or "").lower()
    for role, keys in _KO_EN_KEYS.items():
        if any(k in beat for k in keys):
            return _coerce_person_food_if_needed(role, story_beat)
    # 기본 mood → 강제 정책도 적용
    return _coerce_person_food_if_needed("mood", story_beat)

# ========== Public API ==========
def build_prompts(brand_intro: str, category_hint: str = "cafe") -> Tuple[str, str]:
    """
    입력: 브랜드 소개(국/영문)
    출력: (positive, negative)
    - CARTOON_USE_GPT=1 && OPENAI_API_KEY 있을 때만 GPT 사용.
    - 실패/미설정 시 규칙 기반 폴백.
    """
    brand_intro = (brand_intro or "").strip().replace("\n", " ")
    if _gpt_enabled():
        sys = _load_system_prompt()
        user = json.dumps({
            "brand_intro": brand_intro,
            "category_hint": category_hint,
        }, ensure_ascii=False)
        data = _call_gpt_for_json(
            [{"role": "system", "content": sys}, {"role": "user", "content": user}]
        )
        if isinstance(data, dict) and "positive" in data and "negative" in data:
            pos = str(data["positive"]).strip()
            neg = str(data["negative"]).strip()
            if pos and neg:
                return (pos, neg)

    # 규칙 기반 폴백 (사람 전역 금지는 하지 않음)
    base = [
        f"{category_hint} background",
        "professional design, high quality",
        "cohesive color grading, consistent style",
        f"reflect brand: {brand_intro}" if brand_intro else "",
    ]
    positive = ", ".join([t for t in base if t])

    negative = ", ".join([
        "text, typography, slogans, watermark, logo imprint",
        "cluttered, busy layout",
        "low quality, compression artifacts",
    ])
    return positive, negative

def build_prompts_panel(brand_intro: str, story_beat: str, *_, **__) -> Tuple[str, str]:
    brand_intro = (brand_intro or "").strip().replace("\n", " ")
    story_beat = (story_beat or "").strip().replace("\n", " ")
    role = _infer_panel_role(story_beat)

    if _gpt_enabled():
        sys = _load_system_prompt()
        user = json.dumps({
            "brand_intro": brand_intro,
            "story_beat": story_beat,
            "panel_role": role,
            "category_hint": "instagram comic panel"
        }, ensure_ascii=False)
        data = _call_gpt_for_json(
            [{"role": "system", "content": sys}, {"role": "user", "content": user}]
        )
        if isinstance(data, dict) and "positive" in data and "negative" in data:
            pos = str(data["positive"]).strip()
            neg = str(data["negative"]).strip()
            if pos and neg:
                return (pos, neg)

    # ---- 규칙 기반 폴백(역할별) ----
    if role == "exterior":
        pos = ", ".join([
            "coffee shop exterior, cafe facade, storefront",
            "photorealistic, cinematic lighting, blue hour, wet asphalt reflections",
            "volumetric light, shallow depth of field, physically based rendering cues",
            f"reflect brand: {brand_intro}" if brand_intro else "",
        ])
        neg = ", ".join([
            "readable text, typography, slogans, watermark, logo imprint, signage text, brand name, letters",
            "cluttered background, messy scene",
            "low quality, compression artifacts",
        ])
    elif role == "food":
        pos = ", ".join([
            "hero shot of signature coffee or pastry",
            "studio lighting, appetizing, soft shadows, shallow depth of field",
            "clean backdrop, minimal props, product-centric framing",
            "cohesive grading, consistent style",
            f"reflect brand: {brand_intro}" if brand_intro else "",
        ])
        neg = ", ".join([
            "people, faces, hands",
            "text, typography, slogan, watermark, logo imprint",
            "dirty table, cluttered background",
            "low quality, compression artifacts",
        ])
    elif role == "interior":
        pos = ", ".join([
            "cafe interior with seating/counter, warm wood and stone textures",
            "cozy ambience, balanced composition, practical lighting",
            "subtle brand colors in materials/furniture",
            "cohesive grading, consistent style",
            f"reflect brand: {brand_intro}" if brand_intro else "",
        ])
        neg = ", ".join([
            "people, faces, portrait",
            "text, typography, slogan, watermark, logo imprint",
            "overcrowded, clutter",
            "low quality, compression artifacts",
        ])
    elif role == "person_food":
        pos = ", ".join([
            "one person holding the featured food or drink, waist-up framing",
            "focus on the food in hand, friendly natural pose, clean background",
            "clear human figure visible, hands visible, mid-shot portrait composition",
            "soft lighting, cohesive grading, consistent illustration style",
            f"reflect brand: {brand_intro}" if brand_intro else "",
        ])
        neg = ", ".join([
            "multiple people, crowd, group photo",
            "strong readable text, typography, slogans, watermark, logo imprint",
            "busy background, clutter, messy props",
            "low quality, compression artifacts",
        ])
    else:  # mood
        pos = ", ".join([
            "atmospheric cafe scene that matches the brand concept",
            "minimal clutter, soft light, calm mood",
            "cohesive color grading, consistent style",
            f"reflect brand: {brand_intro}" if brand_intro else "",
        ])
        neg = ", ".join([
            "person, people, faces, portrait",
            "strong readable text, typography, slogan, watermark, logo imprint",
            "noisy, gritty, busy layout",
            "low quality, compression artifacts",
        ])

    if os.getenv("CARTOON_DEBUG_PROMPTS", "0").strip().lower() not in ("0", "false", "no", "off"):
        print(f"[panel] role={role}")

    return pos, neg

# ========== Captions ==========
def suggest_captions(
    *,
    brand_intro: str,
    core_message: Optional[str] = None,
    template: str = "cafe_bi",
    language: str = "ko"
) -> List[str]:
    """
    4컷 캡션(짧은 문장) 추천.
    - GPT 사용 시: 역할(외관/음식/내부/사람+음식)에 맞춰 4문장 반환.
    - 실패/미설정 시: 안전한 기본 문구로 폴백.
    """
    brand_intro = (brand_intro or "").strip()
    core_message = (core_message or brand_intro).strip()

    beats = (
        [
            "Exterior/signage with brand identity mood",
            "Hero food/menu item",
            "Interior/experience resembling our store with BI colors",
            "A person holding the hero food/drink"
        ] if template.lower() == "cafe_bi" else
        ["Panel 1", "Panel 2", "Panel 3", "Panel 4"]
    )

    if _gpt_enabled():
        try:
            sys = (
                "You write concise, on-brand social captions. "
                "Return a JSON object with key 'captions' as an array of 4 short lines. "
                "Each line <= 55 Korean chars (or <= 90 English chars), no hashtags, no emojis."
            )
            user = json.dumps({
                "brand_intro": brand_intro,
                "core_message": core_message,
                "template": template,
                "language": language,
                "beats": beats
            }, ensure_ascii=False)
            data = _call_gpt_for_json(
                [{"role": "system", "content": sys}, {"role": "user", "content": user}],
                model=os.getenv("CARTOON_GPT_MODEL", "gpt-4.1-mini"),
            )
            if isinstance(data, dict) and isinstance(data.get("captions"), list):
                caps = [str(x).strip() for x in data["captions"]][:4]
                caps = [c for c in caps if c]
                if len(caps) == 4:
                    return caps
        except Exception as e:
            print("[prompt_builder_sync] suggest_captions GPT failed:", e)

    # 폴백
    if template.lower() == "cafe_bi":
        return [
            f"{core_message} — 간판에서 시작되는 우리의 무드",
            "대표 메뉴 한 입의 설렘",
            "머물고 싶은 내부, 편안한 온도",
            "손에 담은 시그니처, 오늘의 포토 스팟",
        ]
    else:
        return [
            f"{core_message} — 첫 장면",
            "두 번째 장면",
            "세 번째 장면",
            "마지막 장면",
        ]
        

try:
    from fastapi import APIRouter, HTTPException
    from pydantic import BaseModel

    class BuildReq(BaseModel):
        brand_intro: str
        category_hint: str = "cafe"

    class PanelReq(BaseModel):
        brand_intro: str
        story_beat: str

    class CaptionsReq(BaseModel):
        brand_intro: str
        core_message: Optional[str] = None
        template: str = "cafe_bi"
        language: str = "ko"

    router = APIRouter(prefix="/prompts", tags=["prompts"])

    @router.post("/build")
    def api_build(req: BuildReq):
        try:
            pos, neg = build_prompts(req.brand_intro, req.category_hint)
            return {"positive": pos, "negative": neg}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"build_prompts failed: {e}")

    @router.post("/panel")
    def api_panel(req: PanelReq):
        try:
            pos, neg = build_prompts_panel(req.brand_intro, req.story_beat)
            return {"positive": pos, "negative": neg}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"build_prompts_panel failed: {e}")

    @router.post("/captions")
    def api_captions(req: CaptionsReq):
        try:
            caps = suggest_captions(
                brand_intro=req.brand_intro,
                core_message=req.core_message,
                template=req.template,
                language=req.language,
            )
            return {"captions": caps}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"suggest_captions failed: {e}")

    try:
        __all__
    except NameError:
        __all__ = ["build_prompts", "build_prompts_panel", "suggest_captions"]

    if "router" not in __all__:
        __all__.append("router")

except Exception:
    pass        


__all__ = ["build_prompts", "build_prompts_panel", "suggest_captions"]