# pages/logo_pages/02_text.py
# TITLE: ✏️ Step 2/4. Text — 스케치 위에 텍스트 마스크 만들기 (1024 고정 · pad=0 · 윤곽선 항상 표시 · 가이드 원 토글만)

from __future__ import annotations

import io, os, math, base64, glob, json, time
from typing import List, Optional, Tuple

import streamlit as st
from PIL import Image, ImageDraw, ImageFont, ImageOps

# === 계정 전환 가드 & 세션/캐시 리셋 유틸 (추가) ===
LOGO_STATE_KEYS = [
    # 공통 생성/브리프/잡
    "generated_images", "used_prompt", "job_id", "brief_id", "logo_step",
    "llm_inputs",
    # 스케치/드로잉
    "sketch_png_b64","sketch_final_png_b64","sketch_result_b64",
    "sketch_canvas_b64","sketch_bytes_b64","sketch_rgba_b64",
    "canny_b64","canny_edges_b64",
    # 텍스트 단계
    "text_preview_png_b64","text_export_png_b64","mask_text_png_b64",
    "text_info","text_info_json",
    # 참조 이미지 등
    "ref_img_b64","mask_final_png_b64",
    # 스킵 플래그
    "text_skipped",
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
    prev_owner = st.session_state.get("_logo_session_owner")  # (uid, uname)
    if cur_owner and cur_owner != prev_owner and (cur_owner[0] is not None or cur_owner[1]):
        _reset_logo_state_all()
        st.session_state["_logo_session_owner"] = cur_owner

# === Streamlit × drawable-canvas 호환 패치 (image_to_url 없을 때 보강) ===
def _make_image_to_url_patch():
    try:
        import numpy as np  # type: ignore
    except Exception:
        np = None  # noqa: F841
    try:
        from streamlit.runtime.media_file_manager import media_file_manager
    except Exception:
        media_file_manager = None

    def _to_pil(img, channels="RGB"):
        if isinstance(img, Image.Image):
            im = img
        else:
            try:
                import numpy as np  # type: ignore
                if isinstance(img, np.ndarray):
                    if img.dtype != np.uint8:
                        img = img.astype("uint8")
                    if img.ndim == 2:
                        im = Image.fromarray(img, mode="L")
                    else:
                        im = Image.fromarray(img)
                elif isinstance(img, (bytes, bytearray)):
                    im = Image.open(io.BytesIO(img))
                else:
                    return None, str(img)
            except Exception:
                return None, str(img)
        if channels in ("RGB", "RGBA", "L") and im.mode != channels:
            im = im.convert(channels)
        return ImageOps.exif_transpose(im), None

    def image_to_url(image, width, clamp=False, channels="RGB", output_format="PNG", image_id=None):
        pil, maybe_url = _to_pil(image, channels)
        if maybe_url is not None:
            return maybe_url
        if isinstance(width, (int, float)) and width and pil.width != int(width):
            ratio = float(width) / float(pil.width)
            pil = pil.resize((int(width), max(1, int(pil.height * ratio))), Image.LANCZOS)
        fmt = (output_format or "PNG").upper()
        buf = io.BytesIO(); pil.save(buf, format=fmt); data = buf.getvalue()
        mime = f"image/{fmt.lower()}"
        if hasattr(st, "runtime") and media_file_manager is not None:
            return media_file_manager.add(data, mimetype=mime, file_extension=fmt.lower())
        b64 = base64.b64encode(data).decode("ascii")
        return f"data:{mime};base64,{b64}"

    try:
        import streamlit.elements.image as st_image
        if not hasattr(st_image, "image_to_url"):
            st_image.image_to_url = image_to_url
    except Exception:
        pass
_make_image_to_url_patch()
# === 패치 끝 ===

TITLE = "✏️ Step 2/4. Text — 스케치 위에 텍스트 마스크 만들기"

# ============================
# Constants
# ============================
CANVAS = 1024
PAD_FIXED = 0
MIN_R, MAX_R = 1, 2048

# ============================
# Font utils
# ============================
FONT_DIRS = [
    "/usr/share/fonts",
    "/usr/local/share/fonts",
    os.path.expanduser("~/.local/share/fonts"),
]
FONT_EXTS = (".ttf", ".otf", ".ttc")

@st.cache_resource(show_spinner=False)
def list_system_fonts() -> List[str]:
    paths: List[str] = []
    for d in FONT_DIRS:
        if not os.path.isdir(d):
            continue
        for ext in FONT_EXTS:
            paths.extend(glob.glob(os.path.join(d, "**", f"*{ext}"), recursive=True))
    uniq = sorted(list({p for p in paths if os.path.exists(p)}))
    return uniq

def nice_font_label(path: str) -> str:
    base = os.path.basename(path)
    label = os.path.splitext(base)[0]
    label = label.replace("NanumGothic", "Nanum Gothic").replace("NanumMyeongjo", "Nanum Myeongjo")
    label = label.replace("NotoSansCJK", "Noto Sans CJK").replace("NotoSerifCJK", "Noto Serif CJK")
    return label

@st.cache_resource(show_spinner=False)
def _load_font_cached(path: Optional[str], size: int, index: int = 0):
    if path and os.path.exists(path):
        try:
            return ImageFont.truetype(path, size, index=index)
        except Exception:
            return None
    return None

def load_font_any(path: Optional[str], size: int, index: int = 0, fallback: bool = True):
    f = _load_font_cached(path, size, index=index)
    if f is not None:
        return f
    if fallback:
        for p in [
            "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
            "/usr/share/fonts/truetype/unfonts-core/UnDotum.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]:
            f2 = _load_font_cached(p, size)
            if f2 is not None:
                return f2
        return ImageFont.load_default()
    return ImageFont.load_default()

# ============================
# Helpers
# ============================
def _parse_hex_to_rgba(hex_color: str, alpha: int):
    h = hex_color.lstrip("#")
    if len(h) == 3:
        h = "".join([c*2 for c in h])
    return (int(h[0:2],16), int(h[2:4],16), int(h[4:6],16), int(alpha))

def _draw_stroke_text(draw, xy, text, base_font, fill_rgba, stroke_width_px):
    draw.text(xy, text, font=base_font, fill=fill_rgba, stroke_width=stroke_width_px, stroke_fill=fill_rgba)

def _hr_params(s: int, stroke_w: int, scale: int):
    margin = max(round(s * 0.16), stroke_w * 2 + 4)
    inner = max(1, s - 2*margin)
    font_size = int(inner * 0.78)
    BOX_HR = s * scale
    MARGIN_HR = margin * scale
    FONT_SIZE_HR = font_size * scale
    STROKE_HR = stroke_w * scale
    ROT_EXTRA_HR = math.ceil(0.21 * BOX_HR)
    SAFE_PAD_HR = ROT_EXTRA_HR + 2*max(STROKE_HR, 2) + 6
    return BOX_HR, MARGIN_HR, FONT_SIZE_HR, STROKE_HR, SAFE_PAD_HR

def _angles_for_text(n, r, theta_center, s, track, direction=+1):
    if n <= 0: return []
    if n == 1: return [theta_center]
    delta = (s + track) / max(1, r)
    delta *= direction
    return [theta_center + (i - (n-1)/2)*delta for i in range(n)]

def _make_glyph_square_HR(
    ch: str,
    box_hr: int,
    base_font_hr,
    margin_hr: int,
    safe_pad_hr: int,
    stroke_hr: int,
    text_color=(0, 0, 0, 255),
):
    # 임시 드로잉 컨텍스트(텍스트 bbox 계산용)
    dmy = Image.new("RGBA", (4, 4), (0, 0, 0, 0))
    d = ImageDraw.Draw(dmy)

    pad = int(safe_pad_hr)
    box_padded = int(box_hr + pad * 2)

    g = Image.new("RGBA", (box_padded, box_padded), (0, 0, 0, 0))
    dg = ImageDraw.Draw(g)

    # 텍스트 실제 bbox
    bbox = d.textbbox((0, 0), ch, font=base_font_hr, anchor="lt", stroke_width=int(stroke_hr))
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

    inner = max(1, int(box_hr) - 2 * int(margin_hr))

    # 텍스트를 정사각 내부 중앙 정렬
    tx = pad + int(margin_hr) + (inner - tw) / 2 - bbox[0]
    ty = pad + int(margin_hr) + (inner - th) / 2 - bbox[1]

    _draw_stroke_text(dg, (tx, ty), ch, base_font_hr, text_color, int(stroke_hr))

    top_y = ty
    base_y = min(ty + th, box_padded - 2)

    return g, top_y, base_y, box_padded

def _paste_rotated_tile(canvas, tile_hr, theta, pivot_y_hr, out_center, radius, scale, extra_rot_pi=False):
    cxp, cyp = out_center
    bw_hr, bh_hr = tile_hr.size
    center_hr = (bw_hr/2.0, float(pivot_y_hr))
    deg = -math.degrees(theta) + (180.0 if extra_rot_pi else 0.0)
    rot_hr = tile_hr.rotate(deg, resample=Image.BICUBIC, expand=False, center=center_hr)
    rot = rot_hr.resize((int(round(bw_hr/scale)), int(round(bh_hr/scale))), Image.LANCZOS)
    px = cxp + radius * math.sin(theta)
    py = cyp - radius * math.cos(theta)
    ox = int(round(px - (center_hr[0]/scale)))
    oy = int(round(py - (center_hr[1]/scale)))
    canvas.alpha_composite(rot, (ox,oy))

def _draw_straight_text(canvas, text, font, color_rgba, x, y, angle_deg, stroke_px, anchor_mode: str = "center"):
    tile = Image.new("RGBA", canvas.size, (0,0,0,0))
    d = ImageDraw.Draw(tile)
    if anchor_mode == "center":
        bbox = d.textbbox((0,0), text, font=font, anchor="lt", stroke_width=stroke_px)
        tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
        x0 = x - tw/2 - bbox[0]
        y0 = y - th/2 - bbox[1]
    else:
        x0, y0 = x, y
    _draw_stroke_text(d, (x0,y0), text, font, color_rgba, stroke_px)
    rot = tile.rotate(-angle_deg, resample=Image.BICUBIC, expand=False, center=(x,y))
    canvas.alpha_composite(rot)

# ============================
# 세션 이미지 로드
# ============================
def _b64_to_rgba(b64str: Optional[str]) -> Optional[Image.Image]:
    if not b64str:
        return None
    try:
        im = Image.open(io.BytesIO(base64.b64decode(b64str)))
        if im.mode != "RGBA":
            im = im.convert("RGBA")
        return ImageOps.exif_transpose(im)
    except Exception:
        return None

def _find_sketch_image_from_session() -> Optional[Image.Image]:
    for k in ["sketch_png_b64", "sketch_rgba_b64", "sketch_image_b64", "sketch_canvas_b64", "sketch_result_b64"]:
        if st.session_state.get(k):
            im = _b64_to_rgba(st.session_state.get(k))
            if im is not None:
                return im
    return None

def _find_ref_or_masks_from_session() -> Tuple[Optional[Image.Image], Optional[Image.Image]]:
    ref = _b64_to_rgba(st.session_state.get("ref_img_b64")) if st.session_state.get("ref_img_b64") else None
    mask = _b64_to_rgba(st.session_state.get("mask_text_png_b64")) if st.session_state.get("mask_text_png_b64") else None
    return ref, mask

def _letterbox_to_canvas(img: Image.Image, target_wh: Tuple[int,int]) -> Image.Image:
    Wp, Hp = target_wh
    bg = Image.new("RGBA", (Wp, Hp), (0,0,0,0))
    tmp = img.copy()
    tmp.thumbnail((Wp, Hp))
    px = (Wp - tmp.width)//2
    py = (Hp - tmp.height)//2
    bg.alpha_composite(tmp, (px, py))
    return bg

# ============================
# Nav helper
# ============================
def _set_query_params_safe(**kwargs):
    try:
        st.query_params.update({k: str(v) for k, v in kwargs.items()})
        return
    except Exception:
        pass
    try:
        st.experimental_set_query_params(**kwargs)
    except Exception:
        pass

def _goto_step(step: int):
    st.session_state["logo_step"] = int(step)
    _set_query_params_safe(step=int(step))
    st.session_state["_nav_trigger"] = os.urandom(4).hex()
    st.rerun()

def _go_next_brief():
    _goto_step(2)

def _skip_sketch(set_empty_bg: bool = True):
    for k in [
        "sketch_png_b64", "sketch_rgba_b64", "sketch_image_b64",
        "sketch_canvas_b64", "sketch_result_b64",
        "canny_b64", "canny_edges_b64",
    ]:
        st.session_state.pop(k, None)
    st.session_state["sketch_skipped"] = True
    if set_empty_bg:
        st.session_state["_text_bg_choice"] = "빈 배경"

def _clear_text_from_session():
    """텍스트 산출물 및 사용 플래그/값 초기화"""
    for k in [
        "text_preview_png_b64","text_export_png_b64","mask_text_png_b64",
        "text_info","text_info_json"
    ]:
        st.session_state.pop(k, None)
    st.session_state["use_arc_top"] = False
    st.session_state["use_arc_bottom"] = False
    st.session_state["use_straight"] = False
    st.session_state["top_text_ui"] = ""
    st.session_state["bottom_text_ui"] = ""
    st.session_state["straight_text_ui"] = ""
    st.session_state["text_skipped"] = True

def _skip_and_next_brief():
    _skip_sketch(set_empty_bg=True)
    _clear_text_from_session()
    _goto_step(2)

# ============================
# Shared state
# ============================
def _ensure_defaults():  
    ss = st.session_state
    ss.setdefault("W", CANVAS); ss.setdefault("H", CANVAS)
    ss["W"] = CANVAS; ss["H"] = CANVAS
    ss.setdefault("pad", PAD_FIXED); ss["pad"] = PAD_FIXED

    ss.setdefault("cx", CANVAS//2)
    ss.setdefault("cy", CANVAS//2)
    ss.setdefault("r", 220)

    ss.setdefault("straight_x", CANVAS//2)
    ss.setdefault("straight_y", int(CANVAS*0.875))

    ss.setdefault("outline_alpha", 200)
    ss.setdefault("outline_width", 2)

    ss.setdefault("text_hex", "#282828")
    ss.setdefault("text_alpha", 255)

    ss.setdefault("_text_bg_choice", "자동(스케치→드로잉마스크→텍스트마스크→브리프)")
    ss.setdefault("_text_bg_uploaded", None)

    # ✔ 기본 텍스트는 빈 문자열
    ss.setdefault("top_text_ui", "")
    ss.setdefault("bottom_text_ui", "")
    ss.setdefault("straight_text_ui", "")

    # ✔ 기본 사용여부는 비활성
    ss.setdefault("use_arc_top", False)
    ss.setdefault("use_arc_bottom", False)
    ss.setdefault("use_straight", False)

    ss.setdefault("s_top_ui", 88); ss.setdefault("track_top_ui", 0)
    ss.setdefault("theta_top_ui", 0); ss.setdefault("ro_top_ui", 0)

    ss.setdefault("s_bot_ui", 88); ss.setdefault("track_bot_ui", 0)
    ss.setdefault("theta_bot_ui", 180)
    ss.setdefault("ro_bot_ui", -int(round(ss["s_bot_ui"]*0.25)))

    ss.setdefault("straight_size_ui", 72)
    ss.setdefault("straight_angle_ui", 0)

    ss.setdefault("stroke_w_top_ui", 3)
    ss.setdefault("stroke_w_bot_ui", 3)
    ss.setdefault("stroke_w_straight_ui", 3)

    ss.setdefault("regular_path_top", None)
    ss.setdefault("regular_path_bottom", None)
    ss.setdefault("regular_path_straight", None)

    ss.setdefault("text_skipped", False)
    
    ss.setdefault("preview_overlay_guide", False)

# ============================
# Main render
# ============================
def render():
    try:
        st.set_page_config(page_title=TITLE, page_icon="✏️", layout="wide")
    except Exception:
        pass

    # ★★★ 로그인 사용자 변경 가드 (가장 먼저 실행) ★★★
    _ensure_session_owner_guard()

    st.progress(50, text="Step 2/4 — Text")
    st.title(TITLE)

    with st.expander("📘 사용 방법", expanded=True):
        st.markdown(
            """
1. 로고의 **텍스트 HINT** 이미지를 제작하는 페이지입니다.  
2. **텍스트 배치**(원형 위/아래/직선)를 선택하고 문구를 입력하세요.  
3. **글꼴/간격/두께/위치**를 조절해 개성 있는 로고 텍스트를 만들 수 있습니다.  
4. 만든 **HINT**를 참고하여 다음 단계에서 로고를 제작합니다.
            """
        )

    st.info("① 우측 탭(원형 위/원형 아래/직선)에서 설정 → ② 좌측 미리보기 확인 → ③ 자동 저장 후 단계 이동")

    _ensure_defaults()
    _ = _find_sketch_image_from_session()
    _ = _find_ref_or_masks_from_session()

    left, right = st.columns([1.15, 1])

    # ------ 우측 탭 ------
    with right:
        tabs = st.tabs(["원형 위", "원형 아래", "직선"])
        sys_fonts = list_system_fonts()
        sys_labels = [nice_font_label(p) for p in sys_fonts] if sys_fonts else []

        # 원형 위
        with tabs[0]:
            st.checkbox("원형 텍스트(윗쪽) 사용", key="use_arc_top")  # 기본 False
            st.text_input("텍스트", key="top_text_ui", placeholder="")

            st.markdown("#### 이동")
            c1, c2, c3 = st.columns(3)
            cy_top = c1.number_input("원 중심 Y", 0, CANVAS, value=int(st.session_state["cy"]), step=1, key="cy_top")
            cx_top = c2.number_input("원 중심 X", 0, CANVAS, value=int(st.session_state["cx"]), step=1, key="cx_top")
            r_top  = c3.number_input("반지름 r", MIN_R, MAX_R, value=int(st.session_state["r"]), step=1, key="r_top")

            st.markdown("#### 폰트 선택")
            idx_top = st.selectbox(
                "윗쪽 폰트",
                options=(range(len(sys_fonts)) if sys_fonts else [0]),
                format_func=(lambda k: sys_labels[k] if sys_fonts else "(시스템 폰트가 없습니다 — 기본 폰트 사용)"),
                key="font_sel_top",
            )
            st.session_state["regular_path_top"] = (sys_fonts[idx_top] if sys_fonts else None)

            st.markdown("#### 두께 조절 (stroke 고정)")
            st.session_state["stroke_w_top_ui"] = st.slider(
                "스트로크(px)", 0, 12, int(st.session_state.get("stroke_w_top_ui", 3)),
                key="stroke_w_top"
            )

            st.markdown("#### 텍스트 배치(윗쪽)")
            c_t1, c_t2 = st.columns(2)
            st.session_state["s_top_ui"] = int(c_t1.number_input("글자 크기 변경 ", 8, 512, int(st.session_state["s_top_ui"]), key="s_top_val"))
            st.session_state["track_top_ui"] = int(c_t2.number_input("윗쪽 간격(px)", -40, 200, int(st.session_state["track_top_ui"]), key="track_top_val"))
            st.session_state["theta_top_ui"] = int(st.slider("회전 (deg / 0=12시)", -180, 180, int(st.session_state["theta_top_ui"]), key="theta_top_val"))
            st.session_state["ro_top_ui"] = int(st.slider("텍스트 높이 조절", -256, 256, int(st.session_state["ro_top_ui"]), key="ro_top_val"))

        # 원형 아래
        with tabs[1]:
            st.checkbox("원형 텍스트(아래쪽) 사용", key="use_arc_bottom")  # 기본 False
            st.text_input("텍스트", key="bottom_text_ui", placeholder="")

            st.markdown("#### 이동")
            c1, c2, c3 = st.columns(3)
            cy_bot = c1.number_input("원 중심 Y", 0, CANVAS, value=int(st.session_state["cy"]), step=1, key="cy_bot")
            cx_bot = c2.number_input("원 중심 X", 0, CANVAS, value=int(st.session_state["cx"]), step=1, key="cx_bot")
            r_bot  = c3.number_input("반지름 r", MIN_R, MAX_R, value=int(st.session_state["r"]), step=1, key="r_bot")

            st.markdown("#### 폰트 선택")
            idx_bot = st.selectbox(
                "아래쪽 폰트",
                options=(range(len(sys_fonts)) if sys_fonts else [0]),
                format_func=(lambda k: sys_labels[k] if sys_fonts else "(시스템 폰트가 없습니다 — 기본 폰트 사용)"),
                key="font_sel_bot",
            )
            st.session_state["regular_path_bottom"] = (sys_fonts[idx_bot] if sys_fonts else None)

            st.markdown("#### 두께 조절 (stroke 고정)")
            st.session_state["stroke_w_bot_ui"] = st.slider(
                "스트로크(px)", 0, 12, int(st.session_state.get("stroke_w_bot_ui", 3)),
                key="stroke_w_bot"
            )

            st.markdown("#### 텍스트 배치(아래쪽)")
            c_b1, c_b2 = st.columns(2)
            st.session_state["s_bot_ui"] = int(c_b1.number_input("글자 크기 변경", 8, 512, int(st.session_state["s_bot_ui"]), key="s_bot_val"))
            st.session_state["track_bot_ui"] = int(c_b2.number_input("아래쪽 간격(px)", -40, 200, int(st.session_state["track_bot_ui"]), key="track_bot_val"))
            st.session_state["theta_bot_ui"] = int(st.slider("회전 (deg / 180=6시)", 0, 360, int(st.session_state["theta_bot_ui"]), key="theta_bot_val"))
            default_bot_offset = -int(round(int(st.session_state["s_bot_ui"]) * 0.25))
            st.session_state["ro_bot_ui"] = int(st.slider("텍스트 높이 조절", -256, 256, int(st.session_state.get("ro_bot_ui", default_bot_offset)), key="ro_bot_val"))

        # 직선
        with tabs[2]:
            st.checkbox("직선 텍스트 사용", key="use_straight")  # 기본 False
            st.text_input("텍스트", key="straight_text_ui", placeholder="")

            st.markdown("#### 이동 (좌/우/상/하)")
            c1, c2 = st.columns(2)
            st.session_state["straight_x"] = int(c1.number_input("X", 0, CANVAS, value=int(st.session_state["straight_x"]), step=1, key="straight_x_val"))
            st.session_state["straight_y"] = int(c2.number_input("Y", 0, CANVAS, value=int(st.session_state["straight_y"]), step=1, key="straight_y_val"))

            st.markdown("#### 폰트 선택")
            idx_str = st.selectbox(
                "직선 폰트",
                options=(range(len(sys_fonts)) if sys_fonts else [0]),
                format_func=(lambda k: sys_labels[k] if sys_fonts else "(시스템 폰트가 없습니다 — 기본 폰트 사용)"),
                key="font_sel_st",
            )
            st.session_state["regular_path_straight"] = (sys_fonts[idx_str] if sys_fonts else None)

            st.markdown("#### 두께 조절 (stroke 고정)")
            st.session_state["stroke_w_straight_ui"] = st.slider(
                "스트로크(px)", 0, 12, int(st.session_state.get("stroke_w_straight_ui", 3)),
                key="stroke_w_st"
            )

            st.markdown("#### 글자 크기/회전")
            c_s1, c_s2 = st.columns(2)
            st.session_state["straight_size_ui"] = int(c_s1.number_input("글자 크기(px)", 6, 512, int(st.session_state["straight_size_ui"]), key="straight_size_val"))
            st.session_state["straight_angle_ui"] = int(c_s2.slider("회전 각도(deg)", -180, 180, int(st.session_state["straight_angle_ui"]), key="straight_angle_val"))

        # ★★★ 탭 UI → 세션 단일 소스 동기화 (한 번만) ★★★
        def _sync_circle_from_widgets_once():
            ss = st.session_state

            def _changed(group):
                keys = [f"cx_{group}", f"cy_{group}", f"r_{group}"]
                changed = False
                for k in keys:
                    prevk = f"_prev_{k}"
                    if k in ss and ss.get(prevk) != ss.get(k):
                        changed = True
                    # prev 값 갱신(없으면 초기화)
                    if k in ss:
                        ss[prevk] = ss[k]
                return changed

            top_changed = _changed("top")
            bot_changed = _changed("bot")

            # 최근 변경 우선순위: 이번 렌더에서 바뀐 쪽 -> 직전 기록(_last_circle_src)
            src = None
            if top_changed and not bot_changed:
                src = "top"
            elif bot_changed and not top_changed:
                src = "bot"
            elif top_changed and bot_changed:
                # 동시에 바뀌면 직전 기록 반대로 토글 (간단 처리)
                src = "top" if ss.get("_last_circle_src") != "top" else "bot"
            else:
                # 이번에 변화 없으면 직전 선택 유지
                src = ss.get("_last_circle_src", "bot")  # 기존 동작과 비슷하게 '아래' 기본

            if src == "top" and all(k in ss for k in ("cx_top","cy_top","r_top")):
                ss["cx"], ss["cy"], ss["r"] = int(ss["cx_top"]), int(ss["cy_top"]), int(ss["r_top"])
            elif src == "bot" and all(k in ss for k in ("cx_bot","cy_bot","r_bot")):
                ss["cx"], ss["cy"], ss["r"] = int(ss["cx_bot"]), int(ss["cy_bot"]), int(ss["r_bot"])

            ss["_last_circle_src"] = src


        _sync_circle_from_widgets_once()

    # ------ 좌측: 미리보기 ------
    with left:
        st.markdown("### 미리보기")

        W, H = CANVAS, CANVAS
        pad = PAD_FIXED
        Wp, Hp = W + 2*pad, H + 2*pad

        # ★★★ 렌더 직전: 클램프 & 로컬 갱신 ★★★
        cx = int(st.session_state["cx"])
        cy = int(st.session_state["cy"])
        r  = int(st.session_state["r"])
        max_r_allowed = max(MIN_R, min(cx, cy, W - cx, H - cy))
        if r > max_r_allowed:
            st.session_state["r"] = max_r_allowed
            r = max_r_allowed

        # (선택) 현재값 표시
        # st.caption(f"cx={cx}, cy={cy}, r={r} (max={max_r_allowed})")

        def _select_background_preview() -> Tuple[Optional[Image.Image], str]:
            choice = st.session_state.get("_text_bg_choice", "자동(스케치→드로잉마스크→텍스트마스크→브리프)")
            label = "(없음)"
            if choice == "스케치 이미지":
                auto_s = _find_sketch_image_from_session()
                if auto_s is not None:
                    return _letterbox_to_canvas(auto_s, (Wp, Hp)), "스케치(sketch_png_b64)"
                return None, label
            if choice == "드로잉 결과 마스크":
                im = _b64_to_rgba(st.session_state.get("mask_final_png_b64"))
                if im is not None:
                    return _letterbox_to_canvas(im, (Wp, Hp)), "드로잉(mask_final_png_b64)"
                return None, label
            if choice == "현재 텍스트 마스크":
                if st.session_state.get("mask_text_png_b64"):
                    im = _b64_to_rgba(st.session_state.get("mask_text_png_b64"))
                    if im is not None:
                        return _letterbox_to_canvas(im, (Wp, Hp)), "텍스트(mask_text_png_b64)"
                return None, label
            if choice == "브리프 참고 이미지":
                ref, _ = _find_ref_or_masks_from_session()
                if ref is not None:
                    return _letterbox_to_canvas(ref, (Wp, Hp)), "브리프(ref_img_b64)"
                return None, label
            auto_s = _find_sketch_image_from_session()
            for cand, lab in [
                (auto_s, "스케치(sketch_png_b64)"),
                (_b64_to_rgba(st.session_state.get("mask_final_png_b64")), "드로잉(mask_final_png_b64)"),
                (_b64_to_rgba(st.session_state.get("mask_text_png_b64")), "텍스트(mask_text_png_b64)"),
                (_find_ref_or_masks_from_session()[0], "브리프(ref_img_b64)"),
            ]:
                if cand is not None:
                    return _letterbox_to_canvas(cand, (Wp, Hp)), lab
            return None, label

        bg_layer, bg_label = _select_background_preview()
        st.caption(f"배경 소스: {bg_label}")
        st.checkbox("가이드 원 겹쳐보기 (다운로드 미포함)", key="preview_overlay_guide")

    # ========= 렌더 =========
    text_hex = st.session_state.get("text_hex", "#282828")
    text_alpha = int(st.session_state.get("text_alpha", 255))

    # 렌더에서 사용할 최종 좌표/반지름(위에서 클램프된 값 사용)
    pad_val = st.session_state["pad"]
    cxp, cyp = cx + pad_val, cy + pad_val
    Wp, Hp = CANVAS + 2*pad_val, CANVAS + 2*pad_val

    out = Image.new("RGBA", (Wp, Hp), (0, 0, 0, 0))

    stroke_w_top = int(st.session_state.get("stroke_w_top_ui", 3))
    stroke_w_bot = int(st.session_state.get("stroke_w_bot_ui", 3))
    stroke_w_st  = int(st.session_state.get("stroke_w_straight_ui", 3))
    scale = 3  # 내부 고정

    s_top = int(st.session_state.get("s_top_ui", 88))
    track_top = int(st.session_state.get("track_top_ui", 0))
    theta0_top_deg = int(st.session_state.get("theta_top_ui", 0))

    s_bot = int(st.session_state.get("s_bot_ui", 88))
    track_bot = int(st.session_state.get("track_bot_ui", 0))
    theta0_bot_deg = int(st.session_state.get("theta_bot_ui", 180))

    radial_offset_top = int(st.session_state.get("ro_top_ui", 0))
    radial_offset_bot = int(st.session_state.get("ro_bot_ui", -int(round(s_bot*0.25))))

    straight_size = int(st.session_state.get("straight_size_ui", 72))
    straight_angle = int(st.session_state.get("straight_angle_ui", 0))

    regular_top = st.session_state.get("regular_path_top", None)
    regular_bot = st.session_state.get("regular_path_bottom", None)
    regular_str = st.session_state.get("regular_path_straight", None)

    BOX_HR_T, MARGIN_HR_T, FONT_SIZE_HR_T, STROKE_HR_T, SAFE_PAD_HR_T = _hr_params(int(s_top), int(stroke_w_top), int(scale))
    BOX_HR_B, MARGIN_HR_B, FONT_SIZE_HR_B, STROKE_HR_B, SAFE_PAD_HR_B = _hr_params(int(s_bot), int(stroke_w_bot), int(scale))
    TEXT_COLOR = _parse_hex_to_rgba(text_hex, text_alpha)

    font_top_hr = load_font_any(regular_top, FONT_SIZE_HR_T, fallback=True)
    font_bot_hr = load_font_any(regular_bot, FONT_SIZE_HR_B, fallback=True)

    top_text_val = st.session_state.get("top_text_ui", "")
    bottom_text_val = st.session_state.get("bottom_text_ui", "")
    straight_text = st.session_state.get("straight_text_ui", "") if st.session_state.get("use_straight", False) else ""

    r_top_eff = max(1, r + radial_offset_top)
    r_bot_eff = max(1, r + radial_offset_bot)

    if st.session_state.get("use_arc_top", False) and top_text_val:
        theta0_top = math.radians(theta0_top_deg)
        ths_top = _angles_for_text(len(top_text_val), r_top_eff, theta0_top, s_top, track_top, direction=+1)
        for ch, th in zip(top_text_val, ths_top):
            tile_hr, top_y_hr, base_y_hr, _ = _make_glyph_square_HR(
                ch, BOX_HR_T, font_top_hr, MARGIN_HR_T, SAFE_PAD_HR_T, STROKE_HR_T, text_color=TEXT_COLOR
            )
            pivot_mid_hr = int(round((top_y_hr + base_y_hr) / 2))
            _paste_rotated_tile(out, tile_hr, th, pivot_mid_hr, (cxp,cyp), r_top_eff, int(scale), extra_rot_pi=False)

    if st.session_state.get("use_arc_bottom", False) and bottom_text_val:
        theta0_bot = math.radians(theta0_bot_deg)
        ths_bot = _angles_for_text(len(bottom_text_val), r_bot_eff, theta0_bot, s_bot, track_bot, direction=-1)
        for ch, th in zip(bottom_text_val, ths_bot):
            tile_hr, top_y_hr, base_y_hr, _ = _make_glyph_square_HR(
                ch, BOX_HR_B, font_bot_hr, MARGIN_HR_B, SAFE_PAD_HR_B, STROKE_HR_B, text_color=TEXT_COLOR
            )
            pivot_mid_hr = int(round((top_y_hr + base_y_hr) / 2))
            _paste_rotated_tile(out, tile_hr, th, pivot_mid_hr, (cxp,cyp), r_bot_eff, int(scale), extra_rot_pi=True)

    if st.session_state.get("use_straight", False) and (straight_text or ""):
        straight_font = load_font_any(regular_str, int(straight_size), fallback=True)
        _draw_straight_text(
            out, straight_text, straight_font, TEXT_COLOR,
            x=st.session_state["straight_x"] + pad_val,
            y=st.session_state["straight_y"] + pad_val,
            angle_deg=straight_angle,
            stroke_px=int(stroke_w_st),
            anchor_mode="center"
        )

    # 윤곽 박스 + 미리보기 합성
    preview_base = Image.new("RGBA", (Wp, Hp), (0,0,0,0))
    dprev = ImageDraw.Draw(preview_base)
    dprev.rectangle((pad_val, pad_val, pad_val+CANVAS-1, pad_val+CANVAS-1), outline=(0,0,0,200), width=2)

    def _select_bg_for_export():
        auto_s = _find_sketch_image_from_session()
        for cand in [auto_s,
                     _b64_to_rgba(st.session_state.get("mask_final_png_b64")),
                     _b64_to_rgba(st.session_state.get("mask_text_png_b64")),
                     _find_ref_or_masks_from_session()[0]]:
            if cand is not None:
                return _letterbox_to_canvas(cand, (Wp, Hp))
        return None

    bg_layer2 = _select_bg_for_export()
    preview_img = Image.new("RGBA", (Wp, Hp), (0, 0, 0, 0))
    if bg_layer2 is not None:
        preview_img.alpha_composite(bg_layer2, (0, 0))
    preview_img.alpha_composite(preview_base, (0, 0))
    preview_img.alpha_composite(out, (0, 0))

    if st.session_state.get("preview_overlay_guide", False):
        guide = Image.new("RGBA", (Wp, Hp), (0,0,0,0))
        dg = ImageDraw.Draw(guide)
        gx0, gy0 = cxp - r, cyp - r
        gx1, gy1 = cxp + r, cyp + r
        dg.ellipse((gx0, gy0, gx1, gy1), outline=(0, 200, 255, 180), width=2)
        preview_img.alpha_composite(guide, (0,0))

    # 좌측 프리뷰 표시
    show_w = int(st.session_state.get("preview_w", 600))
    ratio = show_w / float(Wp if Wp > 0 else 1)
    show_h = int(round((Hp if Hp > 0 else 1) * ratio))
    preview_resized = preview_img.resize((show_w, show_h), Image.LANCZOS)
    with left:
        st.image(preview_resized, caption="미리보기", use_container_width=False)

    # ===== 세션 저장(자동) =====
    alpha = out.split()[-1]  # 텍스트 알파 채널
    is_alpha_empty = (alpha.getbbox() is None)
    user_forced_skip = bool(st.session_state.get("text_skipped", False))

    if is_alpha_empty or user_forced_skip:
        # 텍스트 없음 → 저장하지 않고 정리
        for k in ["text_preview_png_b64","text_export_png_b64","mask_text_png_b64","text_info","text_info_json"]:
            st.session_state.pop(k, None)
        if user_forced_skip:
            st.info("텍스트 단계를 스킵했습니다. 텍스트 마스크는 생성/저장되지 않습니다.")
        else:
            st.info("입력된 텍스트가 없어서 텍스트 마스크는 생성/저장하지 않았습니다.")
    else:
        # 산출물 저장
        Wp, Hp = preview_img.size
        crop_box = (pad_val, pad_val, pad_val + CANVAS, pad_val + CANVAS)

        mask_rgba_full = Image.new("RGBA", (Wp, Hp), (255,255,255,0))
        mask_rgba_full.putalpha(alpha)

        export_rgba_full = Image.new("RGBA", (Wp, Hp), (0,0,0,0))
        export_rgba_full.alpha_composite(out, (0,0))
        export_rgba = export_rgba_full.crop(crop_box)
        mask_rgba = mask_rgba_full.crop(crop_box)

        buf_prev = io.BytesIO(); preview_img.save(buf_prev, format="PNG")
        st.session_state["text_preview_png_b64"] = base64.b64encode(buf_prev.getvalue()).decode("ascii")

        buf_export = io.BytesIO(); export_rgba.save(buf_export, format="PNG")
        st.session_state["text_export_png_b64"] = base64.b64encode(buf_export.getvalue()).decode("ascii")

        buf_mask = io.BytesIO(); mask_rgba.save(buf_mask, format="PNG")
        st.session_state["mask_text_png_b64"] = base64.b64encode(buf_mask.getvalue()).decode("ascii")

        text_info = {
            "canvas": {"w": CANVAS, "h": CANVAS, "pad": PAD_FIXED},
            "modes": {
                "use_arc_top": bool(st.session_state.get("use_arc_top", False)),
                "use_arc_bottom": bool(st.session_state.get("use_arc_bottom", False)),
                "use_straight": bool(st.session_state.get("use_straight", False)),
            },
            "circle": {
                "cx": int(cx),
                "cy": int(cy),
                "r":  int(r),
                "top": {
                    "text": st.session_state.get("top_text_ui",""),
                    "s": int(s_top), "track": int(track_top),
                    "theta_deg": int(st.session_state["theta_top_ui"]),
                    "radial_offset": int(st.session_state["ro_top_ui"]),
                },
                "bottom": {
                    "text": st.session_state.get("bottom_text_ui",""),
                    "s": int(s_bot), "track": int(track_bot),
                    "theta_deg": int(st.session_state["theta_bot_ui"]),
                    "radial_offset": int(st.session_state["ro_bot_ui"]),
                },
            },
            "straight": {
                "text": st.session_state.get("straight_text_ui","") if st.session_state.get("use_straight", False) else "",
                "x": int(st.session_state.get("straight_x", CANVAS//2)),
                "y": int(st.session_state.get("straight_y", int(CANVAS*0.875))),
                "angle_deg": int(st.session_state.get("straight_angle_ui", 0)),
                "size_px": int(st.session_state.get("straight_size_ui", 72)),
                "anchor": "center",
            },
            "style": {
                "text_hex": st.session_state.get("text_hex", "#282828"),
                "text_alpha": int(st.session_state.get("text_alpha", 255)),
                "supersample": 3,
                "stroke_w_top": int(st.session_state.get("stroke_w_top_ui", 3)),
                "stroke_w_bottom": int(st.session_state.get("stroke_w_bot_ui", 3)),
                "stroke_w_straight": int(st.session_state.get("stroke_w_straight_ui", 3)),
                "font_regular_top": st.session_state.get("regular_path_top", None),
                "font_regular_bottom": st.session_state.get("regular_path_bottom", None),
                "font_regular_straight": st.session_state.get("regular_path_straight", None),
            },
            "assets": {"mask_text_png_b64": st.session_state.get("mask_text_png_b64", None)}
        }
        st.session_state["text_info"] = text_info
        st.session_state["text_info_json"] = json.dumps(text_info, ensure_ascii=False)

        st.success("세션 저장 완료: text_export_png_b64, mask_text_png_b64, text_preview_png_b64, text_info, text_info_json")

    st.markdown("---")
    c1, c2 = st.columns([1,1])
    with c1:
        st.button("다음(브리프) ▶", type="primary", use_container_width=True, on_click=_go_next_brief)
    with c2:
        st.button("스킵 후 브리프 ▶", help="지금까지 입력된 텍스트를 모두 버리고 브리프로 이동합니다.", use_container_width=True, on_click=_skip_and_next_brief)

if __name__ == "__main__":
    render()
