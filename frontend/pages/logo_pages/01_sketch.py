# pages/logo_pages/01_sketch.py
# ✏️ Step 1/4. Sketch

from __future__ import annotations
import io, base64, hashlib, uuid, time   # ← time 추가
from datetime import datetime
from typing import Optional, Tuple

import numpy as np
from PIL import Image, ImageOps, ImageDraw
import streamlit as st

# ===== 계정 전환 가드 & 세션/캐시 리셋 유틸 =====
LOGO_STATE_KEYS = [
    "generated_images", "used_prompt", "job_id", "brief_id",
    "sketch_png_b64","sketch_final_png_b64","sketch_result_b64",
    "sketch_canvas_b64","sketch_bytes_b64","sketch_rgba_b64",
    "text_mask_png_b64","text_info","llm_inputs",
    "logo_step",
]

def _read_current_owner() -> tuple[Optional[int], Optional[str]]:
    uid = st.session_state.get("user_id")
    uname = st.session_state.get("username")
    return uid, uname

def _reset_logo_state_all():
    for k in LOGO_STATE_KEYS:
        if k in st.session_state:
            del st.session_state[k]
    try:
        st.cache_data.clear()
    except Exception:
        pass
    st.session_state["cache_bust"] = str(int(time.time() * 1000))

def _ensure_session_owner_guard():
    cur_owner = _read_current_owner()
    prev_owner = st.session_state.get("_logo_session_owner")
    if cur_owner and cur_owner != prev_owner and (cur_owner[0] is not None or cur_owner[1]):
        _reset_logo_state_all()
        st.session_state["_logo_session_owner"] = cur_owner

# ===== 캔버스 모듈 확인 =====
try:
    from streamlit_drawable_canvas import st_canvas  # type: ignore
    _HAS_CANVAS = True
except Exception:
    _HAS_CANVAS = False

TITLE = "✏️ Step 1/4. Sketch"
CANVAS_W = CANVAS_H = 1024  # 고정

# ================= Utils =================
def pil_to_data_url(im: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    im.save(buf, format=fmt)
    return f"data:image/{fmt.lower()};base64,{base64.b64encode(buf.getvalue()).decode()}"

def pil_to_b64_png(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def pil_to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def b64_to_pil(b64png: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64png))).convert("RGBA")

def file_to_pil(uploaded) -> Image.Image:
    img = Image.open(uploaded)
    img = ImageOps.exif_transpose(img)
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGBA")
    return img

def np_to_pil(arr: np.ndarray) -> Image.Image:
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        return Image.fromarray(arr, mode="L").convert("RGBA")
    if arr.ndim == 3 and arr.shape[-1] == 4:
        return Image.fromarray(arr, mode="RGBA")
    return Image.fromarray(arr, mode="RGB").convert("RGBA")

def overlay_grid(img: Image.Image, step: int = 128, alpha: int = 80) -> Image.Image:
    im = img.copy()
    draw = ImageDraw.Draw(im, "RGBA")
    w, h = im.size
    draw.rectangle([(0, 0), (w - 1, h - 1)], outline=(0, 0, 0, alpha))
    for x in range(step, w, step):
        draw.line([(x, 0), (x, h)], fill=(0, 0, 0, alpha // 2), width=1)
    for y in range(step, h, step):
        draw.line([(0, y), (w, y)], fill=(0, 0, 0, alpha // 2), width=1)
    draw.line([(w // 2, 0), (w // 2, h)], fill=(0, 0, 0, alpha), width=2)
    draw.line([(0, h // 2), (w, h // 2)], fill=(0, 0, 0, alpha), width=2)
    return im

def _rgba_or_hex_to_hex(v) -> str:
    if isinstance(v, str) and v.startswith("#"):
        return v
    if isinstance(v, tuple) and len(v) in (3, 4):
        r, g, b = v[:3]
        return f"#{r:02x}{g:02x}{b:02x}"
    return "#000000"

def fit_to_1024(img: Image.Image, mode: str = "Stretch", letterbox_color: Tuple[int,int,int,int] = (255,255,255,255)) -> Image.Image:
    img = img.convert("RGBA")
    w, h = img.size

    if mode == "Stretch":
        return img.resize((CANVAS_W, CANVAS_H), Image.LANCZOS)

    scale_contain = min(CANVAS_W / w, CANVAS_H / h)
    scale_cover   = max(CANVAS_W / w, CANVAS_H / h)

    if mode == "Contain":
        nw, nh = int(w * scale_contain), int(h * scale_contain)
        resized = img.resize((nw, nh), Image.LANCZOS)
        canvas = Image.new("RGBA", (CANVAS_W, CANVAS_H), letterbox_color)
        ox, oy = (CANVAS_W - nw) // 2, (CANVAS_H - nh) // 2
        canvas.alpha_composite(resized, (ox, oy))
        return canvas

    if mode == "Cover":
        nw, nh = int(w * scale_cover), int(h * scale_cover)
        resized = img.resize((nw, nh), Image.LANCZOS)
        left = (nw - CANVAS_W) // 2
        top  = (nh - CANVAS_H) // 2
        return resized.crop((left, top, left + CANVAS_W, CANVAS_H + top))

    return img.resize((CANVAS_W, CANVAS_H), Image.LANCZOS)

# ── 내부 라우터 ─────────────────────────────
def _go_text_page():
    st.session_state["logo_step"] = 1
    try:
        st.query_params.update({"step": "1"})
    except Exception:
        st.experimental_set_query_params(step="1")
    st.rerun()

# ===== 세션 저장 =====
def _save_sketch_to_session(img: Image.Image):
    b64 = pil_to_b64_png(img)
    for k in ("sketch_png_b64","sketch_final_png_b64","sketch_result_b64","sketch_canvas_b64","sketch_bytes_b64","sketch_rgba_b64"):
        st.session_state[k] = b64
    st.session_state["sketch_W"], st.session_state["sketch_H"] = (CANVAS_W, CANVAS_H)

def _clear_sketch_from_session():  # [CHANGED] 건너뛰기 시 스케치 완전 제거
    for k in ("sketch_png_b64","sketch_final_png_b64","sketch_result_b64","sketch_canvas_b64","sketch_bytes_b64","sketch_rgba_b64","sketch_W","sketch_H"):
        st.session_state.pop(k, None)

# =============== Page ===============
def render():
    try:
        st.set_page_config(page_title=TITLE, page_icon="✏️", layout="wide")
    except Exception:
        pass

    # ★★★ 로그인 사용자 변경 가드 (가장 먼저 실행) ★★★
    _ensure_session_owner_guard()

    st.progress(25, text="Step 1/4 — Sketch")
    st.title("✏️ Step 1. Sketch")
    st.caption("이 단계에서는 만들고 싶은 로고 이미지를 스케치 합니다. 다음 단계(2/4 Text)에서는 텍스트를 배치합니다.")

    with st.expander("🧭 사용 방법 ", expanded=True):
        st.markdown(
            "**로고의 심볼 HINT 이미지**를  제작하는 페이지 입니다  \n"
            "**원하는 HINT 입력 방식**을 선택하고 이미지를 업로드 하거나, 캔버스 위에 직접 그려보세요!  \n"
            "캔버스에 그리거나, 업로드 이미지를 배경으로 두고 **즉시 미리보기**를 확인합니다.  \n"
            "HINT를 참고하여 로고를 제작해 드릴게요!"
        )

    st.info("출력 해상도: **1024 × 1024 px 고정** ")
    st.divider()

    left, right = st.columns([1.35, 1], gap="large")

    # ────────── 우측: 도구/옵션 패널 ──────────
    with right:
        st.subheader("🎛️ 도구 & 옵션")
        drawing_mode = st.selectbox(
            "드로잉 모드",
            ["freedraw", "line", "rect", "circle", "transform", None],
            index=0,
            format_func=lambda x: "그리기 없음" if x is None else x,
            key="draw_mode",
        )
        stroke_w = st.slider("선 두께", 1, 80, 6, key="stroke_w")

        stroke_color_label = st.radio("펜/지우개 선택", ("검정", "흰색"), index=0, horizontal=True, key="stroke_color_choice")
        if stroke_color_label == "검정":
            st.session_state["stroke_c"] = "#000000"
        else:
            st.session_state["stroke_c"] = "#FFFFFF"

        st.markdown("#### 업로드 맞춤 방식")
        fit_mode = st.selectbox(
            "이미지 업로드 시 1024×1024에 맞추는 방법",
            ["Stretch(늘림)", "Contain(여백 추가)", "Cover(가운데 크롭)"],
            index=0, key="fit_mode",
        )

        letterbox_color = st.session_state.get("letterbox_color", "#FFFFFF")
        if "Contain" in st.session_state.get("fit_mode", "Stretch"):
            letterbox_color = st.color_picker("Contain 여백 색상", letterbox_color, key="letterbox_color")

        st.markdown("#### 빈 캔버스 옵션 (직접 그리기/건너뛰기)")
        bg_color_blank = st.color_picker("빈 캔버스 배경색", "#ffffff", key="bg_blank")

        st.markdown("---")
        guide_grid = st.toggle("격자 가이드(미리보기 전용)", value=True, key="guide_grid")

    # ────────── 좌측: 입력/캔버스 ──────────
    result_img: Optional[Image.Image] = None
    with left:
        st.subheader("🧩 입력 방식")
        mode = st.radio(
            "원하는 방식을 고르세요",
            options=["직접 그리기", "이미지 업로드", "건너뛰기(빈 배경)"],
            horizontal=True, key="sketch_mode",
        )

        st.markdown("### 🎨 스케치 영역")
        canvas = None

        if mode == "직접 그리기":
            if not _HAS_CANVAS:
                st.error("`streamlit-drawable-canvas` 미설치: `pip install streamlit-drawable-canvas`")
            else:
                bg_hex = _rgba_or_hex_to_hex(st.session_state.get("bg_blank", "#ffffff"))
                canvas = st_canvas(
                    fill_color="#0000ff55",
                    stroke_width=st.session_state["stroke_w"],
                    stroke_color=st.session_state["stroke_c"],
                    background_color=bg_hex,
                    height=CANVAS_H, width=CANVAS_W,
                    drawing_mode=st.session_state["draw_mode"],
                    update_streamlit=True,
                    key="canvas_draw_fixed",
                )
                if canvas is not None and canvas.image_data is not None:
                    result_img = np_to_pil(canvas.image_data)

        elif mode == "이미지 업로드":
            up = st.file_uploader("이미지 업로드 (PNG/JPG)", type=["png", "jpg", "jpeg"], key="upload_main")
            reset = st.button("↩️ 캔버스 리셋", use_container_width=True)

            if up is None:
                st.info("이미지를 업로드하면 1024×1024로 맞춘 배경 위에서 바로 그릴 수 있어요.")
            else:
                raw = file_to_pil(up)
                fit_choice = "Stretch" if "Stretch" in st.session_state.get("fit_mode", "Stretch") \
                    else ("Contain" if "Contain" in st.session_state.get("fit_mode", "Stretch") else "Cover")

                if fit_choice == "Contain":
                    hexv = st.session_state.get("letterbox_color", "#FFFFFF")
                    lb_rgba = (int(hexv[1:3], 16), int(hexv[3:5], 16), int(hexv[5:7], 16), 255)
                else:
                    lb_rgba = (255, 255, 255, 255)

                fitted = fit_to_1024(raw, mode=fit_choice, letterbox_color=lb_rgba)

                if not _HAS_CANVAS:
                    st.warning("캔버스 모듈이 없어 업로드 이미지만 저장할 수 있습니다. (`pip install streamlit-drawable-canvas`)")
                    st.image(fitted, caption="1024×1024로 맞춘 업로드 이미지", use_container_width=True)
                    result_img = fitted
                else:
                    initial = {
                        "objects": [{
                            "type": "image",
                            "left": 0, "top": 0,
                            "width": CANVAS_W, "height": CANVAS_H,
                            "scaleX": 1, "scaleY": 1,
                            "opacity": 1,
                            "src": pil_to_data_url(fitted, fmt="PNG"),
                            "selectable": False, "evented": False,
                        }]
                    }
                    key_seed_src = hashlib.md5(fitted.tobytes()).hexdigest()[:8]
                    key_seed_mode = hashlib.md5(fit_choice.encode()).hexdigest()[:4]
                    key_seed_reset = uuid.uuid4().hex[:6] if reset else "noreset"
                    canvas_key = f"canvas_upload_{key_seed_src}_{key_seed_mode}_{key_seed_reset}"

                    canvas = st_canvas(
                        fill_color="#0000ff55",
                        stroke_width=st.session_state["stroke_w"],
                        stroke_color=st.session_state["stroke_c"],
                        background_color=None,
                        height=CANVAS_H, width=CANVAS_W,
                        drawing_mode=st.session_state["draw_mode"],
                        initial_drawing=initial,
                        update_streamlit=True,
                        key=canvas_key,
                    )

                    result_img = fitted
                    if canvas is not None and canvas.image_data is not None:
                        result_img = np_to_pil(canvas.image_data)

        else:
            # [CHANGED] 건너뛰기: 스케치 이미지를 생성/저장하지 않음
            result_img = None
            _clear_sketch_from_session()
            st.info("스케치를 건너뛰고 진행합니다. 스케치 이미지는 전달되지 않습니다.")

    st.markdown("---")
    c1, c2, c3 = st.columns(3)

    with c1:
        if result_img is not None:
            st.download_button(
                "⬇️ PNG 다운로드",
                data=pil_to_png_bytes(result_img),
                file_name=f"sketch_{datetime.now():%Y%m%d_%H%M%S}.png",
                mime="image/png",
                use_container_width=True,
            )
        else:
            st.download_button("⬇️ PNG 다운로드", b"", disabled=True, use_container_width=True)

    with c2:
        # [UNCHANGED] 결과 이미지가 있을 때만 세션에 저장
        if result_img is not None:
            new_b64 = pil_to_b64_png(result_img)
            if st.session_state.get("sketch_png_b64") != new_b64:
                for k in ("sketch_png_b64","sketch_final_png_b64","sketch_result_b64",
                          "sketch_canvas_b64","sketch_bytes_b64","sketch_rgba_b64"):
                    st.session_state[k] = new_b64
                st.session_state["sketch_W"], st.session_state["sketch_H"] = (CANVAS_W, CANVAS_H)
            st.caption("자동 저장됨 (세션)")
        else:
            st.caption("")

    with c3:
        # [CHANGED] 건너뛰기일 땐 스케치 없이 다음 단계로 진행 허용
        if st.button("➡️ 다음 (2/4) Text / Masking", type="primary", use_container_width=True):
            if st.session_state.get("sketch_mode") == "건너뛰기(빈 배경)":
                _go_text_page()
            elif st.session_state.get("sketch_png_b64"):
                _go_text_page()
            else:
                st.warning("먼저 스케치를 만들고 미리보기를 확인해 주세요.")

# ===== entry =====
if __name__ == "__main__":
    render()
