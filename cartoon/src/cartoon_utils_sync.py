from __future__ import annotations
import contextlib
from typing import Optional, Tuple

import streamlit as st


# Navigation
def goto_poster_step(step_idx: int):
    """
    특정 단계(step_idx)로 이동하도록 '의도'만 기록.
    """
    st.session_state["_nav_to"] = ("poster_step", int(step_idx))


def go_to_home():
    """
    홈으로 이동하도록 '의도'만 기록.
    """
    st.session_state["_nav_to"] = ("home", None)


def _set_query_param(key: str, value: Optional[str]):
    """
    Streamlit 최신 API(st.query_params)가 있을 때만 안전하게 설정.
    """
    with contextlib.suppress(Exception):
        if value is None:
            # pop 동작 미지원 환경 대비: 빈 값으로 둔다
            qp = st.query_params
            if key in qp:
                qp.pop(key, None)
            st.query_params = qp
        else:
            st.query_params[key] = value  # type: ignore[attr-defined]


def run_pending_nav():
    nav: Optional[Tuple[str, Optional[int]]] = st.session_state.pop("_nav_to", None)  # type: ignore[assignment]
    if not nav:
        return

    kind, arg = nav
    if kind == "home":
        # 프로젝트 내 메인 앱 파일로 이동 (필요시 경로 조정)
        st.switch_page("app.py")
        return

    if kind == "poster_step":
        step = int(arg or 0)
        st.session_state["poster_step"] = step
        _set_query_param("step", str(step))
        # cartoon용 페이지 경로 (필요 시 맞는 파일명으로 교체)
        st.switch_page("pages/poster.py")
        return



# Flash message (toast-style)
def flash(message: str, level: str = "info"):
    """
    - level: 'info' | 'warning' | 'success' | 'error'
    """
    st.session_state["flash"] = {"message": message, "level": level}


def consume_flash():
    """
    flash()로 기록된 메시지를 읽어 적절한 스트림릿 함수로 출력하고 제거.
    """
    data = st.session_state.pop("flash", None)
    if not data:
        return
    level = (data.get("level") or "info").lower()
    msg = data.get("message", "")

    # level→출력 함수 매핑
    fn = {
        "info": st.info,
        "warning": st.warning,
        "success": st.success,
        "error": st.error,
    }.get(level, st.info)

    fn(msg)


# ---- FastAPI Router ----
from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional, Dict, Any

router = APIRouter(tags=["cartoon-utils"])

@router.get("/ping")
async def ping(): return {"ok": True}

_FLASH_STORE: Dict[str, Dict[str, Any]] = {}

class NavPosterStepRequest(BaseModel):
    step_idx: int

@router.post("/nav/poster-step")
async def api_goto_poster_step(req: NavPosterStepRequest):
    return {"navigate_to": "poster_step", "step": int(req.step_idx)}

@router.post("/nav/home")
async def api_go_home():
    return {"navigate_to": "home"}

class FlashRequest(BaseModel):
    message: str
    level: str = "info"
    client_id: Optional[str] = "default"

@router.post("/flash")
async def api_flash(req: FlashRequest):
    cid = (req.client_id or "default").strip() or "default"
    _FLASH_STORE[cid] = {"message": req.message, "level": req.level}
    try: flash(req.message, req.level)
    except Exception: pass
    return {"ok": True}

class ConsumeFlashRequest(BaseModel):
    client_id: Optional[str] = "default"

class ConsumeFlashResponse(BaseModel):
    message: Optional[str] = None
    level: Optional[str] = None

@router.post("/flash/consume", response_model=ConsumeFlashResponse)
async def api_consume_flash(req: ConsumeFlashRequest):
    cid = (req.client_id or "default").strip() or "default"
    data = _FLASH_STORE.pop(cid, None)
    try: consume_flash()
    except Exception: pass
    if not data: return ConsumeFlashResponse()
    return ConsumeFlashResponse(message=str(data.get("message") or ""), level=str(data.get("level") or "info"))
