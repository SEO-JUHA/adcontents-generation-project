# pages/logo_pages/04_generate.py
# TITLE: 🔮 Step 4/4. Generate — 결과 확인 (대표 선택/ZIP/콜라주 제거, 저장/다운로드만)
from __future__ import annotations

import os, io, base64, time, json
from typing import List, Optional, Dict, Any

import streamlit as st
from PIL import Image, ImageOps

TITLE = "🔮 Step 4/4. Generate — 결과 확인"

# =============================
# ★ 계정 전환 가드 & 세션/캐시 리셋 (추가)
# =============================
LOGO_STATE_KEYS = [
    # 공통 진행 상태
    "logo_step",
    # 생성 결과/메타
    "gen_images_b64", "last_job_id", "last_job_data", "used_prompt",
    # 앞단계 산출물(혹시 남아있으면 전 사용자 자취가 보임)
    "sketch_png_b64","sketch_final_png_b64","sketch_result_b64",
    "sketch_canvas_b64","sketch_bytes_b64","sketch_rgba_b64",
    "mask_text_png_b64","text_preview_png_b64","text_export_png_b64",
    "text_info","text_info_json","canny_b64","canny_edges_b64",
    # 브리프 관련
    "brief_payload","brief_id","palette","ref_img_b64",
    "gpt_prompt_seed","prompt_bundle",
]

def _read_current_owner() -> tuple[Optional[int], Optional[str]]:
    """앱이 로그인 후 세션에 넣어둔 사용자 정보(user_id/username)를 읽는다."""
    uid = st.session_state.get("user_id")
    uname = st.session_state.get("username")
    # 보조: auth_user dict 지원
    au = st.session_state.get("auth_user")
    if uid is None and isinstance(au, dict):
        uid = au.get("user_id") or au.get("id")
    if (not uname) and isinstance(au, dict):
        uname = au.get("username") or au.get("name")
    return (uid if uid is not None else None, uname if uname else None)

def _reset_logo_state_all():
    for k in LOGO_STATE_KEYS:
        st.session_state.pop(k, None)
    # 사용자 A의 cache_data가 B에게 보이는 것을 방지
    try:
        st.cache_data.clear()
    except Exception:
        pass
    st.session_state["cache_bust"] = str(int(time.time() * 1000))

def _ensure_session_owner_guard():
    """현재 로그인 사용자가 바뀌었거나(계정 전환) 로그아웃/로그인으로 상태가 달라졌으면 초기화."""
    cur_owner = _read_current_owner()          # (uid, uname) — 둘 다 None이면 '비로그인'
    prev_owner = st.session_state.get("_logo_session_owner")  # 직전 소유자 스냅샷
    if cur_owner != prev_owner:
        _reset_logo_state_all()
        st.session_state["_logo_session_owner"] = cur_owner

# =============================
# 환경 변수 (백엔드 경로/엔드포인트)
# =============================
BACKEND_BASE       = os.environ.get("LOGO_BACKEND_URL", "http://127.0.0.1:8000")
JOB_ENDPOINT       = os.environ.get("LOGO_JOB_ENDPOINT", "/logo/generate/{job_id}")
GENERATE_ENDPOINT  = os.environ.get("LOGO_GENERATE_ENDPOINT", "/logo/generate")
SELECT_ENDPOINT    = os.environ.get("LOGO_SELECT_ENDPOINT", "/logo/selection")

# ▼ Logos 저장/조회/삭제 엔드포인트
LOGOS_UPLOAD_ENDPOINT = os.environ.get("LOGO_UPLOAD_ENDPOINT", "/logos/upload")
LOGOS_LIST_ENDPOINT   = os.environ.get("LOGO_LIST_ENDPOINT",   "/logos")
LOGOS_DELETE_ENDPOINT = os.environ.get("LOGO_DELETE_ENDPOINT", "/logos/{logo_id}")
LOGO_UPLOAD_FIELD     = os.environ.get("LOGO_UPLOAD_FIELD", "file")  # 기본: file (FastAPI UploadFile)

NAV_MODE = os.environ.get("LOGO_NAV_MODE", "router").lower().strip()  # router | pages

JOB_URL    = lambda job_id: f"{BACKEND_BASE.rstrip('/')}{JOB_ENDPOINT.format(job_id=job_id)}"
GEN_URL    = f"{BACKEND_BASE.rstrip('/')}{GENERATE_ENDPOINT}"
SELECT_URL = f"{BACKEND_BASE.rstrip('/')}{SELECT_ENDPOINT}"

# Logos URL
LOGOS_UPLOAD_URL = f"{BACKEND_BASE.rstrip('/')}{LOGOS_UPLOAD_ENDPOINT}"
LOGOS_LIST_URL   = f"{BACKEND_BASE.rstrip('/')}{LOGOS_LIST_ENDPOINT}"
LOGOS_DELETE_URL = lambda logo_id: f"{BACKEND_BASE.rstrip('/')}{LOGOS_DELETE_ENDPOINT.format(logo_id=logo_id)}"

# =============================
# 순수 유틸
# =============================
def _clean_b64(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.strip()
    if s.startswith("data:image"):
        s = s.split(",", 1)[1]
    return "".join(s.split())

def _b64_to_img(b64png: str) -> Optional[Image.Image]:
    try:
        s = _clean_b64(b64png)
        raw = base64.b64decode(s, validate=False)
        im = Image.open(io.BytesIO(raw))
        return ImageOps.exif_transpose(im.convert("RGBA"))
    except Exception:
        return None

def _to_data_url(b64png: str) -> str:
    s = (b64png or "").strip()
    return s if s.startswith("data:image") else f"data:image/png;base64,{s}"

def _img_to_bytes(img: Image.Image, fmt: str="PNG") -> bytes:
    buf = io.BytesIO(); img.save(buf, format=fmt); return buf.getvalue()

def _post_json(url: str, payload: dict, timeout: int = 30):
    import requests
    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()

def _get_json(url: str, timeout: int = 15, params: Optional[dict]=None):
    import requests
    r = requests.get(url, timeout=timeout, params=params)
    r.raise_for_status()
    return r.json()

# -----------------------------
# 생성/폴링 (메타 포함)
# -----------------------------
def _start_generate(
    brief_id: int,
    sketch_b64: Optional[str],
    mask_b64: Optional[str],
    *,
    num_images: int = 4,
    seed: Optional[int] = None,
    extra: Optional[Dict[str, Any]] = None,
    use_llm_prompt: bool = True,
    return_debug: bool = True,
    force_mode: Optional[str] = None,
    text_info: Optional[Dict[str, Any]] = None,
    prompt_overrides: Optional[Dict[str, Any]] = None,
    gpt_prompt_seed: Optional[str] = None,
    gpt_messages: Optional[List[Dict[str, str]]] = None,
) -> Optional[str]:
    mode = force_mode or (
        "dual" if (sketch_b64 and mask_b64)
        else ("scribble" if sketch_b64 else ("canny" if mask_b64 else "canny"))
    )

    payload: Dict[str, Any] = {
        "brief_id": int(brief_id),
        "sketch_png_b64": sketch_b64,
        "text_mask_png_b64": mask_b64,
        "num_images": int(num_images),
        "seed": None if (seed in (0, "", None)) else int(seed),
        "preprocess_mode": mode,
        "use_llm_prompt": bool(use_llm_prompt),
        "return_debug": bool(return_debug),
        "text_info": text_info,
        "llm_inputs": {
            "prompt_overrides": prompt_overrides or {},
            "gpt_prompt_seed": gpt_prompt_seed or "",
            "gpt_messages": gpt_messages or [],
        },
    }

    if extra:
        extra.pop("prompt", None)
        if extra.get("positive_prompt") or extra.get("negative_prompt"):
            payload["use_llm_prompt"] = False
        payload.update(extra)

    try:
        data = _post_json(GEN_URL, payload, timeout=60)
        return data.get("job_id")
    except Exception as e:
        st.error(f"재생성 시작 실패: {e}")
        return None

def _poll_job(job_id: str, timeout_sec: int = 300, interval_sec: float = 2.0) -> Optional[Dict[str, Any]]:
    t0 = time.time()
    with st.spinner("이미지 생성 중…"):
        while True:
            try:
                data = _get_json(JOB_URL(job_id), timeout=15)
                status = data.get("status", "pending")
                if status == "done":
                    return data
                if status == "error":
                    st.error(f"생성 오류: {data.get('error')}"); return None
            except Exception as e:
                st.warning(f"폴링 에러: {e}")
            if time.time() - t0 > timeout_sec:
                st.warning("생성 대기 시간이 초과되었습니다.")
                return None
            time.sleep(interval_sec)

def _goto_step(step: int):
    if NAV_MODE == "router":
        try:
            st.query_params["step"] = str(step)
        except Exception:
            st.experimental_set_query_params(step=str(step))
        st.rerun()
    else:
        target = {
            1: "pages/logo_pages/01_sketch.py",
            2: "pages/logo_pages/02_text.py",
            3: "pages/logo_pages/03_brief.py",
            4: "pages/logo_pages/04_generate.py",
        }.get(step, "pages/logo_pages/01_sketch.py")
        try:
            st.switch_page(target);  return
        except Exception:
            try:
                st.query_params["step"] = str(step)
            except Exception:
                st.experimental_set_query_params(step=str(step))
            st.rerun()

# =============================
# ▼ Logos 저장/갤러리 유틸 (로그인 필수)
# =============================
def _login_identity() -> Optional[dict]:
    """세션에서 user_id 또는 username을 찾아 반환. auth_user 호환."""
    user_id = st.session_state.get("user_id")
    username = st.session_state.get("username")
    if user_id is not None:
        return {"user_id": str(user_id)}
    if username:
        return {"username": str(username)}
    au = st.session_state.get("auth_user")
    if isinstance(au, dict):
        uid = au.get("user_id") or au.get("id")
        uname = au.get("username") or au.get("name")
        if uid is not None:
            st.session_state["user_id"] = uid
            return {"user_id": str(uid)}
        if uname:
            st.session_state["username"] = uname
            return {"username": str(uname)}
    return None

def _upload_logo_bytes(image_bytes: bytes, filename: str = "selected.png") -> Optional[dict]:
    ident = _login_identity()
    if not ident:
        st.error("로그인 정보가 없습니다. (user_id 또는 username 필요)")
        return None
    import requests
    files = {LOGO_UPLOAD_FIELD: (filename, io.BytesIO(image_bytes), "image/png")}
    try:
        r = requests.post(LOGOS_UPLOAD_URL, files=files, data=ident, timeout=30)
        if r.status_code >= 400:
            st.error(f"저장 실패 [{r.status_code}]: {r.text}")
            return None
        return r.json()
    except Exception as e:
        st.error(f"저장 실패(네트워크): {e}")
        return None

def _fetch_my_logos() -> List[dict]:
    ident = _login_identity()
    if not ident:
        return []
    params = {"page": 1, "size": 50, **ident}
    try:
        data = _get_json(LOGOS_LIST_URL, timeout=15, params=params)
        if isinstance(data, dict) and "items" in data:
            return data["items"]
        if isinstance(data, list):
            return data
        return []
    except Exception as e:
        st.error(f"목록 조회 실패: {e}")
        return []

def _delete_logo(logo_id: int) -> bool:
    ident = _login_identity()
    if not ident:
        st.error("로그인 정보가 없습니다. (삭제 불가)")
        return False
    import requests
    try:
        r = requests.delete(LOGOS_DELETE_URL(logo_id), params=ident, timeout=15)
        r.raise_for_status()
        return True
    except Exception as e:
        st.error(f"삭제 실패: {e}")
        return False

# =============================
# UI
# =============================
def render():
    try:
        st.set_page_config(page_title=TITLE, page_icon="🔮", layout="wide")
    except Exception:
        pass

    # ★★★ 계정 전환/로그아웃/로그인 변화 감지 → 상태 초기화
    _ensure_session_owner_guard()

    st.session_state["logo_step"] = 4

    # 이전 사용자 잔상이 있었다면 위 가드에서 이미 비워짐
    if not st.session_state.get("gen_images_b64") and not st.session_state.get("last_job_id"):
        _goto_step(3 if st.session_state.get("brief_id") else 1)
        return

    st.progress(1.0, text="Step 4/4 — Generate")
    st.title(TITLE)
    st.caption("브리프에서 요청한 생성 작업 결과입니다. 4장을 검토하고 저장하세요.")

    imgs_b64: Optional[List[str]] = st.session_state.get("gen_images_b64")
    job_id: Optional[str]         = st.session_state.get("last_job_id")
    brief_id: Optional[int]       = st.session_state.get("brief_id")

    # Step1/2 산출물 회수 (있을 때만)
    sketch_b64 = None
    for k in ("sketch_png_b64","sketch_final_png_b64","sketch_canvas_b64","sketch_result_b64","sketch_rgba_b64","sketch_bytes_b64"):
        if st.session_state.get(k):
            sketch_b64 = st.session_state.get(k); break
    mask_b64 = st.session_state.get("mask_text_png_b64")

    # LLM 컨텍스트
    text_info        = st.session_state.get("text_info")
    prompt_overrides = st.session_state.get("prompt_bundle")
    gpt_prompt_seed  = st.session_state.get("gpt_prompt_seed")
    gpt_messages     = None

    # 결과 없으면 폴링
    if (not imgs_b64) and job_id:
        st.info("백엔드에서 생성 결과를 가져오는 중…")
        job_json = _poll_job(job_id=job_id, timeout_sec=300, interval_sec=2.0)
        if job_json:
            imgs_b64 = job_json.get("images_b64") or []
            st.session_state["gen_images_b64"] = imgs_b64
            st.session_state["last_job_data"]  = job_json

    if imgs_b64 and job_id and not st.session_state.get("last_job_data"):
        try:
            st.session_state["last_job_data"] = _get_json(JOB_URL(job_id), timeout=10)
        except Exception:
            pass

    if not imgs_b64:
        st.warning("표시할 이미지가 없습니다. 브리프로 이동합니다.")
        _goto_step(3); return

    # ===== 결과 그리드 =====
    N = len(imgs_b64)
    st.subheader(f"결과 미리보기 · {N}장")

    is_logged_in = _login_identity() is not None
    if not is_logged_in:
        st.info("🔐 저장 기능을 사용하려면 로그인(user_id 또는 username)이 필요합니다.")

    grid_cols = 2 if N <= 4 else 3
    rows = (N + grid_cols - 1) // grid_cols
    idx = 0
    for _ in range(rows):
        cols = st.columns(grid_cols)
        for c in cols:
            if idx >= N: break
            with c:
                im = _b64_to_img(imgs_b64[idx])
                if im is None:
                    data_url = _to_data_url(imgs_b64[idx])
                    st.image(data_url, use_container_width=True, caption=f"#{idx+1}")
                    dl_bytes = base64.b64decode(data_url.split(",", 1)[1])
                else:
                    st.image(im, use_container_width=True, caption=f"#{idx+1}")
                    dl_bytes = _img_to_bytes(im)

                colb1, colb2 = st.columns(2)
                with colb1:
                    st.download_button("⬇️ 다운로드", data=dl_bytes, file_name=f"logo_{idx+1:02d}.png",
                                       mime="image/png", use_container_width=True)
                with colb2:
                    st.button(
                        "💾 저장",
                        key=f"save_{idx}",
                        use_container_width=True,
                        disabled=not is_logged_in,
                        on_click=lambda b=dl_bytes, i=idx: _upload_logo_bytes(b, filename=f"logo_{i+1:02d}.png") if is_logged_in else None
                    )
            idx += 1

    # ===== 생성 메타 =====
    job_meta = st.session_state.get("last_job_data") or {}
    used_prompt = job_meta.get("used_prompt", {})
    debug = job_meta.get("debug", {})
    with st.expander("🔧 생성에 사용된 정보 (클릭하여 펼치기)", expanded=False):
        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown("**프롬프트**")
            st.caption("Positive"); st.code(used_prompt.get("positive", ""), language="text")
            st.caption("Negative"); st.code(used_prompt.get("negative", ""), language="text")
            st.caption("LLM 모델"); st.write(used_prompt.get("model", "-"))
            st.markdown("---")
            st.markdown("**모드 & 가이던스/파라미터**")
            st.write(f"Mode: `{debug.get('mode','-')}`")
            controls = debug.get("controls", {})
            if controls: st.code(json.dumps(controls, ensure_ascii=False, indent=2), language="json")
            else: st.write("—")
            if "text_info" in job_meta:
                st.markdown("---")
                st.markdown("**text_info (2단계 가이드 요약)**")
                st.code(json.dumps(job_meta.get("text_info"), ensure_ascii=False, indent=2), language="json")
            if "llm_inputs" in job_meta:
                st.markdown("**llm_inputs (3단계 LLM 힌트)**")
                st.code(json.dumps(job_meta.get("llm_inputs"), ensure_ascii=False, indent=2), language="json")
        with c2:
            st.markdown("**참고한 조건 프리뷰**")
            pcolA, pcolB = st.columns(2)
            with pcolA:
                st.caption("Canny Preview")
                b64 = debug.get("canny_preview_b64")
                if b64: st.image(_to_data_url(b64), use_container_width=True)
                else: st.write("—")
            with pcolB:
                st.caption("Scribble Preview")
                b64 = debug.get("scribble_preview_b64")
                if b64: st.image(_to_data_url(b64), use_container_width=True)
                else: st.write("—")

    # ===== 재생성(3가지 컨트롤만 노출) =====
    st.markdown("---")
    st.subheader("다시 생성하기")
    st.info(f"스케치 전달 여부: {'✅' if bool(sketch_b64) else '❌'}  /  텍스트마스크: {'✅' if bool(mask_b64) else '❌'}")

    regen_cols = st.columns(4)
    with regen_cols[0]:
        regen_n = st.number_input("개수", 1, 8, 4, 1)
    with regen_cols[1]:
        regen_seed = st.number_input("Seed(빈칸=랜덤)", value=0, step=1)
    regen_seed_val = None if regen_seed == 0 else int(regen_seed)
    with regen_cols[2]:
        text_lock = st.slider("텍스트 고정력 (Canny)", 0.0, 2.0, 0.9, 0.05)
    with regen_cols[3]:
        symbol_lock = st.slider("심볼 고정력 (Scribble)", 0.0, 2.0, 0.45, 0.05)

    guidance = st.slider("프롬프트 충성도 (Guidance)", 1.0, 12.0, 6.5, 0.5)

    def _do_regen():
        if not brief_id:
            st.error("brief_id가 없어 재생성을 시작할 수 없습니다."); return

        extra = {
            "preprocess_mode": "dual" if (sketch_b64 and mask_b64) else ("scribble" if sketch_b64 else "canny"),
            "return_debug": True,
            "use_llm_prompt": True,

            # 3가지 핵심 컨트롤만 전달
            "canny_cn_scale":    float(text_lock),
            "scribble_cn_scale": float(symbol_lock),
            "guidance_scale":    float(guidance),
        }

        job = _start_generate(
            int(brief_id),
            sketch_b64,
            mask_b64,
            num_images=int(regen_n),
            seed=regen_seed_val,
            extra=extra,
            text_info=text_info,
            prompt_overrides=prompt_overrides,
            gpt_prompt_seed=gpt_prompt_seed,
            gpt_messages=None,
        )
        if not job: return
        st.session_state["last_job_id"] = job

        job_json = _poll_job(job, timeout_sec=300, interval_sec=2.0)
        if job_json and job_json.get("images_b64"):
            st.session_state["gen_images_b64"] = job_json["images_b64"]
            st.session_state["last_job_data"]  = job_json
            st.success("재생성 완료! 화면을 업데이트합니다.")
            st.rerun()
        else:
            st.warning("재생성 결과가 없습니다.")

    st.button("🔁 다시 생성", on_click=_do_regen, use_container_width=True)

    # ===== 내가 저장한 로고 (로그인 필수) =====
    st.markdown("---")
    st.subheader("📁 내가 저장한 로고")
    is_logged_in = _login_identity() is not None
    if not is_logged_in:
        st.info("🔐 로그인 후에 내가 저장한 로고 목록을 볼 수 있습니다.")
    else:
        gallery = _fetch_my_logos()
        if not gallery:
            st.write("아직 저장된 로고가 없습니다.")
        else:
            rows = (len(gallery) + 3) // 4
            for r in range(rows):
                cols = st.columns(4)
                for c in range(4):
                    idx = r*4 + c
                    if idx >= len(gallery): break
                    item = gallery[idx]
                    with cols[c]:
                        img_path = item.get("image_url") or item.get("url")
                        if img_path:
                            full_url = img_path if img_path.startswith("http") else f"{BACKEND_BASE.rstrip('/')}{img_path}"
                            st.image(full_url, caption=f"#{item.get('id','?')} • {item.get('created_at','')}", use_container_width=True)
                        else:
                            st.write("(이미지 경로 없음)")
                        if st.button(f"삭제 #{item.get('id','?')}", key=f"del_{item.get('id','?')}"):
                            if _delete_logo(int(item["id"])):
                                st.toast("삭제 완료", icon="🗑️")
                                st.rerun()

    # # 네비
    # st.markdown("---")
    # n1, n2 = st.columns(2)
    # with n1:
    #     if st.button("◀ 브리프로 돌아가기", use_container_width=True):
    #         _goto_step(3)
    # with n2:
    #     if st.button("① 스케치로 돌아가기", use_container_width=True):
    #         _goto_step(1)

if __name__ == "__main__":
    render()
