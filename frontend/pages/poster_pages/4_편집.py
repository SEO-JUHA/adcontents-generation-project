import os
import io
import glob
import base64
import pathlib
from typing import List, Tuple, Optional

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

from streamlit.elements import image as _st_image_mod
_old_image_to_url = getattr(_st_image_mod, "image_to_url", None)

def _image_to_url_compat(img, *args, **kwargs) -> str:
    # 이미 URL/data-url이면 그대로
    if isinstance(img, str) and img.startswith(("http://", "https://", "data:")):
        return img

    # numpy → PIL
    if isinstance(img, np.ndarray):
        try:
            img = Image.fromarray(img)
        except Exception:
            return str(type(img))

    # 파일 경로면 data-url로
    if isinstance(img, str) and os.path.exists(img):
        try:
            with open(img, "rb") as f:
                data = f.read()
            b64 = base64.b64encode(data).decode("ascii")
            ext = os.path.splitext(img)[1].lower()
            if ext in [".jpg", ".jpeg"]:
                mime = "image/jpeg"
            elif ext == ".webp":
                mime = "image/webp"
            else:
                mime = "image/png"
            return f"data:{mime};base64,{b64}"
        except Exception:
            return img

    if isinstance(img, Image.Image):
        buf = io.BytesIO()
        try:
            img.convert("RGB").save(buf, format="PNG")
        except Exception:
            buf = io.BytesIO()
            img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return "data:image/png;base64," + b64

    return str(img)

_st_image_mod.image_to_url = _image_to_url_compat

from streamlit_drawable_canvas import st_canvas

try:
    import cv2
except Exception:
    cv2 = None

# ↓↓↓ 이 아래에 추가 ↓↓↓
import numpy as _np
from PIL import Image as _PILImage

def _rgb_to_bgr(arr):
    # numpy 3채널이면 cv2 없이도 채널 뒤집기로 처리
    if isinstance(arr, _np.ndarray) and arr.ndim == 3 and arr.shape[2] == 3:
        return arr[..., ::-1]  # RGB -> BGR
    if isinstance(arr, _PILImage.Image):
        return _rgb_to_bgr(_np.array(arr.convert("RGB"), dtype=_np.uint8))
    if cv2 is not None and hasattr(cv2, "cvtColor"):
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    raise AttributeError("RGB->BGR 변환 불가")

def _bgr_to_rgb(arr):
    if isinstance(arr, _np.ndarray) and arr.ndim == 3 and arr.shape[2] == 3:
        return arr[..., ::-1]  # BGR -> RGB
    if isinstance(arr, _PILImage.Image):
        return _bgr_to_rgb(_np.array(arr.convert("RGB"), dtype=_np.uint8))
    if cv2 is not None and hasattr(cv2, "cvtColor"):
        return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    raise AttributeError("BGR->RGB 변환 불가")


try:
    from poster.src.upscaler import upscale_if_needed
except Exception:
    upscale_if_needed = None


ROOT = pathlib.Path(__file__).resolve().parents[3]
OUTPUT_DIR = ROOT / "poster" / "outputs" / "edited"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Font utils  (시스템 폰트 탐색/로드)
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

def _rerun():
    try:
        st.rerun()
    except Exception:
        try:
            st.experimental_rerun()
        except Exception:
            pass


# === 상태/히스토리 & 합성 유틸 ===
def _init_state():
    st.session_state.setdefault("edited_image", None)  # 지우개 적용 누적본
    st.session_state.setdefault("layers", [])          # 텍스트/이미지 레이어
    st.session_state.setdefault("hist", [])            # Undo
    st.session_state.setdefault("redo", [])            # Redo

def _snapshot_layers():
    return [dict(L) for L in st.session_state.get("layers", [])]

def _push_history():
    st.session_state["hist"].append(_snapshot_layers())
    st.session_state["redo"].clear()

def _undo_layers():
    if st.session_state["hist"]:
        st.session_state["redo"].append(_snapshot_layers())
        st.session_state["layers"] = st.session_state["hist"].pop()
        st.toast("Undo 완료")

def _redo_layers():
    if st.session_state["redo"]:
        st.session_state["hist"].append(_snapshot_layers())
        st.session_state["layers"] = st.session_state["redo"].pop()
        st.toast("Redo 완료")

def _stroke_text(draw: ImageDraw.ImageDraw, xy: Tuple[int,int], text: str, font: ImageFont.FreeTypeFont,
                 fill=(255,255,255,255), stroke_width=2, stroke_fill=(0,0,0,255), align_center=True):
    x, y = xy
    if stroke_width > 0:
        for dx in range(-stroke_width, stroke_width+1):
            for dy in range(-stroke_width, stroke_width+1):
                if dx*dx + dy*dy <= stroke_width*stroke_width:
                    draw.text((x+dx, y+dy), text, font=font, fill=stroke_fill, anchor="mm" if align_center else None)
    draw.text((x, y), text, font=font, fill=fill, anchor="mm" if align_center else None)

def _compose_layers_over(img_rgba: Image.Image) -> Image.Image:
    """누적본(또는 기본 이미지) 위에 현재 세션 레이어를 합성"""
    out = img_rgba.copy()
    for L in st.session_state.get("layers", []):
        if L.get("type") == "text":
            draw = ImageDraw.Draw(out)
            _stroke_text(draw, (L["x"], L["y"]), L["text"], L["font"],
                         fill=L["fill"], stroke_width=L["stroke_w"], stroke_fill=L["stroke_fill"],
                         align_center=L.get("align_center", True))
        elif L.get("type") == "image":
            layer = L["image"].copy()
            op = L.get("opacity", 1.0)
            if op < 1.0:
                a = layer.split()[-1].point(lambda t: int(t*op))
                layer.putalpha(a)
            out.alpha_composite(layer, dest=(int(L["x"]), int(L["y"])))
    return out

def _add_layer(layer: dict):
    _push_history()
    st.session_state["layers"].append(layer)

def _move_layer(idx: int, direction: int):
    L = st.session_state["layers"]
    j = max(0, min(len(L)-1, idx+direction))
    if j != idx:
        _push_history(); L[idx], L[j] = L[j], L[idx]

def _delete_layer(idx: int):
    L = st.session_state["layers"]
    if 0 <= idx < len(L):
        _push_history(); L.pop(idx)


# === 이미지/마스크 유틸 ===
def _last_pick_xy(canvas_json, fallback_xy, base_w, base_h, canvas_w, canvas_h):
    bx, by = fallback_xy
    try:
        if canvas_json and canvas_json.get("objects"):
            obj = canvas_json["objects"][-1]
            px = int(obj.get("left", canvas_w//2))
            py = int(obj.get("top",  canvas_h//2))
            sx = base_w / max(1, canvas_w)
            sy = base_h / max(1, canvas_h)
            bx, by = int(px * sx), int(py * sy)
    except Exception:
        pass
    return bx, by

def _load_image(path: str) -> Image.Image:
    im = Image.open(path)
    if im.mode != "RGBA":
        im = im.convert("RGBA")
    return im

def _fit_image_to_box(img: Image.Image, box_wh: Tuple[int,int], keep_aspect=True, cover=True) -> Image.Image:
    W, H = box_wh
    img = img.convert("RGBA")
    if keep_aspect:
        rw, rh = img.width, img.height
        scale = max(W / rw, H / rh) if cover else min(W / rw, H / rh)
        new = img.resize((int(rw*scale), int(rh*scale)), Image.LANCZOS)
        if cover:
            left = max(0, (new.width - W) // 2); top = max(0, (new.height - H) // 2)
            new  = new.crop((left, top, left+W, top+H))
        else:
            canvas = Image.new("RGBA", (W, H), (0,0,0,0))
            canvas.alpha_composite(new, dest=((W-new.width)//2, (H-new.height)//2))
            new = canvas
        return new
    return img.resize((W, H), Image.LANCZOS)

def _extract_slot_boxes(mask_path: Optional[str], fallback_layout: Optional[str], min_area_ratio=0.005) -> List[Tuple[int,int,int,int]]:
    boxes: List[Tuple[int,int,int,int]] = []
    if cv2 is None: return boxes
    def _boxes_from_alpha(img_rgba: Image.Image):
        a = np.array(img_rgba.split()[-1]); a_bin = (a > 0).astype(np.uint8)*255
        contours, _ = cv2.findContours(a_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out = []; min_area = img_rgba.width*img_rgba.height*min_area_ratio
        for c in contours:
            x,y,w,h = cv2.boundingRect(c)
            if w*h >= min_area: out.append((x,y,x+w,y+h))
        return out
    try:
        if mask_path and os.path.exists(mask_path):
            boxes = _boxes_from_alpha(_load_image(mask_path))
        elif fallback_layout and os.path.exists(fallback_layout):
            ly = _load_image(fallback_layout); arr = np.array(ly)
            white = (arr[...,0:3] > 245).all(axis=-1); alpha0 = arr[...,3] < 5
            cand = (white | alpha0).astype(np.uint8)*255
            contours, _ = cv2.findContours(cand, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            min_area = ly.width*ly.height*min_area_ratio; boxes = []
            for c in contours:
                x,y,w,h = cv2.boundingRect(c)
                if w*h >= min_area: boxes.append((x,y,x+w,y+h))
    except Exception: boxes = []
    boxes.sort(key=lambda b: (b[1], b[0])); return boxes

# def _inpaint_rgba(base_rgba: Image.Image, mask_gray: Image.Image, method: str = "telea", radius: int = 7) -> Image.Image:
#     if cv2 is None:
#         st.warning("OpenCV 미설치: 'opencv-python-headless' 필요")
#         return base_rgba
#     import cv2 as _cv2
#     bgr = _cv2.cvtColor(np.array(base_rgba)[..., :3], _cv2.COLOR_RGB2BGR)
#     mask = np.array(mask_gray.convert("L"))
#     algo = _cv2.INPAINT_TELEA if method == "telea" else _cv2.INPAINT_NS
#     out = _cv2.inpaint(bgr, (mask > 0).astype(np.uint8) * 255, radius, algo)
#     rgb = _cv2.cvtColor(out, _cv2.COLOR_BGR2RGB)
#     out_rgba = np.array(base_rgba); out_rgba[..., :3] = rgb
#     return Image.fromarray(out_rgba, mode="RGBA")

def _inpaint_rgba(base_rgba, mask_gray, method: str = "telea", radius: int = 7):
    """
    base_rgba: PIL.Image | np.ndarray  (RGB/RGBA/그레이 모두 허용, 최종 RGBA로 처리)
    mask_gray: PIL.Image | np.ndarray  (H×W 단일채널, 0/255 or bool도 허용)
    """
    try:
        import cv2 as _cv2
    except Exception as e:
        st.warning("OpenCV 미설치 또는 임포트 실패: opencv-python-headless 필요")
        return base_rgba

    # ---- 1) base_rgba → RGBA uint8 보장 ----
    if isinstance(base_rgba, Image.Image):
        if base_rgba.mode != "RGBA":
            base_rgba = base_rgba.convert("RGBA")
        img_rgba = np.array(base_rgba, dtype=np.uint8)
    else:
        img_rgba = np.asarray(base_rgba)
        if img_rgba.ndim == 2:  # (H,W) → RGB → RGBA
            img_rgba = np.stack([img_rgba]*3, axis=-1)
        if img_rgba.shape[-1] == 3:  # RGB → RGBA
            alpha = np.full(img_rgba.shape[:2] + (1,), 255, dtype=np.uint8)
            img_rgba = np.concatenate([img_rgba, alpha], axis=-1)
        if img_rgba.dtype != np.uint8:
            img_rgba = img_rgba.astype(np.uint8)

    H, W, C = img_rgba.shape
    if C != 4:
        raise ValueError(f"expect RGBA, got shape={img_rgba.shape}")

    # ---- 2) mask → (H,W) uint8 0/255 보장 ----
    if isinstance(mask_gray, Image.Image):
        mask_np = np.array(mask_gray.convert("L"), dtype=np.uint8)
    else:
        mask_np = np.asarray(mask_gray)
        if mask_np.ndim == 3:  # 3채널 마스크가 오면 그레이로
            if mask_np.shape[2] == 3:
                r, g, b = mask_np[...,0], mask_np[...,1], mask_np[...,2]
                mask_np = (0.299*r + 0.587*g + 0.114*b).astype(_np.uint8)
            else:
                mask_np = mask_np[..., 0]
        if mask_np.dtype != np.uint8:
            mask_np = mask_np.astype(np.uint8)

    if mask_np.shape[:2] != (H, W):
        raise ValueError(f"mask shape {mask_np.shape} != image shape {(H, W)}")
    if mask_np.max() <= 1:
        mask_np = (mask_np > 0).astype(np.uint8) * 255

    # ---- 3) RGB만 인페인트 후 RGBA에 덮기 ----
    rgb = img_rgba[..., :3]
    bgr = _rgb_to_bgr(rgb)

    algo = _cv2.INPAINT_TELEA if method.lower() == "telea" else _cv2.INPAINT_NS
    try:
        bgr_fixed = _cv2.inpaint(bgr, mask_np, float(radius), algo)
    except Exception as e:
        raise RuntimeError(f"cv2.inpaint 실패: {e}")

    rgb_fixed = _bgr_to_rgb(bgr_fixed)
    out = img_rgba.copy()
    out[..., :3] = rgb_fixed
    return Image.fromarray(out, mode="RGBA")



# ===디자인 지문 & 초기화 유틸 ===
def _fp_stat(p: Optional[str]) -> str:
    try:
        if p and os.path.exists(p):
            return f"{p}:{int(os.path.getmtime(p))}"
        return str(p)
    except Exception:
        return str(p)

def _design_fingerprint(chosen_path: Optional[str],
                        layout_path: Optional[str],
                        slots_mask_path: Optional[str],
                        chosen_version: Optional[int]) -> str:
    # 선택 파일의 변경(mtime) + 레이아웃/마스크 + 선택 카운터를 함께 묶어서 지문 생성
    return "|".join([
        _fp_stat(chosen_path),
        _fp_stat(layout_path),
        _fp_stat(slots_mask_path),
        str(chosen_version if chosen_version is not None else 0),
    ])

def _reset_edit_state():
    st.session_state["edited_image"] = None
    st.session_state["layers"] = []
    st.session_state["hist"] = []
    st.session_state["redo"] = []


# === UI ===
def render():
    st.set_page_config(page_title="4) 후처리 · 편집", layout="wide")
    st.markdown(
        """
        <style>
        .block-container{padding-top:0.6rem; padding-bottom:0.6rem;}
        [data-testid="stImage"] img{ max-width: none !important; width: auto !important; height: auto !important; }
        </style>
        """, unsafe_allow_html=True
    )

    st.title("4) 후처리 · 편집")
    _init_state()

    chosen_path = st.session_state.get("chosen_path")
    if not chosen_path or not os.path.exists(chosen_path):
        st.warning("3) 페이지에서 이미지를 선택한 뒤 넘어와주세요.")
        st.stop()

    layout_path = st.session_state.get("selected_layout_path")
    slots_mask_path = st.session_state.get("slots_mask_path")
    chosen_version = st.session_state.get("chosen_version", 0)

    # 선택/레이아웃/마스크/선택버전 지문이 바뀌면 편집 상태 초기화
    cur_fp = _design_fingerprint(chosen_path, layout_path, slots_mask_path, chosen_version)
    if st.session_state.get("last_design_fp") != cur_fp:
        _reset_edit_state()
        st.session_state["last_design_fp"] = cur_fp
        st.toast("선택한 디자인이 변경되어 편집 상태를 초기화했어요.", icon="✨")

    base0 = _load_image(chosen_path)

    # 상단 툴바
    with st.container(border=True):
        c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
        with c1:
            tool = st.segmented_control("도구", options=["지우개", "텍스트", "사진"], default="지우개", key="tool_top")
        with c2:
            zoom = st.slider("화면 배율(%)", 40, 120, 70, 5)
        with c3:
            col_cap = st.number_input("표시 최대폭(px)", 360, 1200, 560, 10)
        with c4:
            st.markdown("**히스토리**")
            b1, b2 = st.columns(2)
            with b1: st.button("뒤로 가기", on_click=_undo_layers, key="undo_top")
            with b2: st.button("앞으로 가기", on_click=_redo_layers, key="redo_top")

    # 표시 크기 (좌우 동일 폭)
    aspect = base0.width / max(1, base0.height)
    disp_w = st.slider("표시 폭(px) – 좌우 동일", 320, 1200, 560, 10,
                       help="드로잉 캔버스와 우측 미리보기 폭을 동일하게 맞춥니다. (기본값 560)")
    target_w = int(min(disp_w, col_cap))
    target_h = int(round(target_w / aspect))

    current_base = (st.session_state.get("edited_image") or base0)  # 누적본

    # 텍스트/사진 옵션(상단)
    text_val, size = "가게 최고의 시그니처 메뉴!", 48
    color, stroke_color, stroke_w, align_center = "#FFFFFF", "#000000", 3, True
    font = None; x_text = current_base.width//2; y_text = current_base.height//2

    keep_aspect, cover, opacity, snap = True, True, 1.0, True
    target_idx = 0
    place_img, x_img, y_img = None, 0, 0
    w_img = min(512, current_base.width); h_img = min(512, current_base.height)

    # 텍스트 도구 UI
    if tool == "텍스트":
        t1, t2, t3, t4, t5 = st.columns([1.6, 0.9, 0.2, 0.2, 1.2])
        with t1: text_val = st.text_input("텍스트", text_val, key="txt_top")
        with t2: size = st.slider("크기", 12, 160, size, key="size_top")
        with t3: color = st.color_picker("색상", color, key="color_top")
        with t4: stroke_color = st.color_picker("외곽선", stroke_color, key="scolor_top")
        with t5: stroke_w = st.slider("외곽선 두께", 0, 12, stroke_w, key="sw_top")

        t6, t7, t8 = st.columns([1.1, 1.1, 2.0])
        with t6:
            align_center = st.toggle("중앙 기준", align_center, key="ac_top")
            x_text = st.number_input("X", 0, current_base.width, x_text, key="x_top")
            y_text = st.number_input("Y", 0, current_base.height, y_text, key="y_top")

        with t7:
            fonts = list_system_fonts()
            labels = [nice_font_label(p) for p in fonts]
            sel_idx = st.selectbox(
                "시스템 폰트",
                options=list(range(len(labels))) if labels else [0],
                format_func=(lambda i: labels[i] if labels else "없음"),
            )
            selected_font_path = fonts[sel_idx] if fonts else None
            font_index = st.number_input("TTC 서브폰트 인덱스", min_value=0, max_value=16, value=0, step=1,
                                         help="*.ttc인 경우 0부터 시도")

        with t8:
            upl_font = st.file_uploader("폰트 업로드 (.ttf/.otf/.ttc)", type=["ttf","otf","ttc"], key="font_top")
            uploaded_path = None
            if upl_font is not None:
                tmp_dir = pathlib.Path(st.session_state.get("_font_tmp_dir", OUTPUT_DIR / "_tmp_fonts"))
                tmp_dir.mkdir(parents=True, exist_ok=True)
                tmp_path = tmp_dir / upl_font.name
                with open(tmp_path, "wb") as f:
                    f.write(upl_font.getbuffer())
                uploaded_path = str(tmp_path)
                st.caption(f"업로드 폰트 사용: {upl_font.name}")

            effective_font_path = uploaded_path or selected_font_path
            try:
                font = load_font_any(effective_font_path, size=int(size), index=int(font_index), fallback=True)
            except Exception:
                font = ImageFont.load_default()

    # 사진 도구 UI
    if tool == "사진":
        p1, p2, p3, p4 = st.columns([1.1, 1.1, 1.2, 1.6])
        with p1:
            keep_aspect = st.toggle("비율 유지", True, key="ka_top")
            cover = st.toggle("슬롯 채우기", True, key="cov_top")
        with p2:
            opacity = st.slider("불투명도", 0.0, 1.0, 1.0, 0.05, key="op_top")
            snap = st.toggle("슬롯 스냅", True, key="sn_top")
        with p3:
            boxes = _extract_slot_boxes(slots_mask_path, layout_path)
            st.caption(f"감지 슬롯: {len(boxes)}")
            target_idx = st.number_input("슬롯 인덱스", 0, max(0, len(boxes)-1), 0, key="si_top")
        with p4:
            upl = st.file_uploader("이미지 업로드", type=["png","jpg","jpeg","webp"], key="upl_top")
            scale_pct = st.slider("크기 비율(%)", 10, 200, 100, 5, key="scale_pct",
                          help="슬롯 스냅이 켜진 경우 슬롯 크기를 기준으로 최종 배치 크기를 조절합니다.")
            if upl is not None:
                place_img = Image.open(io.BytesIO(upl.read())).convert("RGBA")
                x_img = st.number_input("X", 0, current_base.width, 0, key="x_top_img")
                y_img = st.number_input("Y", 0, current_base.height, 0, key="y_top_img")
                w_img = st.number_input("너비", 1, current_base.width, min(512, current_base.width), key="w_top_img")
                h_img = st.number_input("높이", 1, current_base.height, min(512, current_base.height), key="h_top_img")
            else:
                place_img = None
                x_img, y_img = 0, 0
                w_img = min(512, current_base.width)
                h_img = min(512, current_base.height)

    # 본문: 좌(드로잉/좌표) · 우(미리보기)
    left, right = st.columns([1.05, 1.05])

    # 좌측: 드로잉 / 좌표
    with left:
        st.subheader("드로잉 / 좌표 픽업")

        # 기본 배경은 현재 누적본
        bg_left_base = current_base

        # 사진 도구에서 업로드/사이즈/스냅/좌표를 좌측 배경에도 즉시 반영
        if tool == "사진" and place_img is not None:
            if snap and 'boxes' in locals() and boxes:
                x0, y0, x1, y1 = boxes[int(target_idx)]
                W, H = (x1 - x0, y1 - y0)
                layer_tmp = _fit_image_to_box(place_img, (W, H), keep_aspect=keep_aspect, cover=cover)
                px_for_left, py_for_left = x0, y0
            else:
                layer_tmp = _fit_image_to_box(place_img, (int(w_img), int(h_img)),
                                            keep_aspect=keep_aspect, cover=cover)
                px_for_left, py_for_left = int(x_img), int(y_img)

            if opacity < 1.0:
                a = layer_tmp.split()[-1].point(lambda t: int(t * opacity))
                layer_tmp.putalpha(a)

            # 원본 해상도에서 합성 후 표시 크기로 리사이즈
            overlay_preview_for_left = bg_left_base.copy()
            overlay_preview_for_left.alpha_composite(layer_tmp, dest=(px_for_left, py_for_left))
            bg_L_pil = overlay_preview_for_left.resize((target_w, target_h), Image.LANCZOS).convert("RGB")
        else:
            # 지우개/텍스트 또는 사진 업로드 전
            bg_L_pil = bg_left_base.resize((target_w, target_h), Image.LANCZOS).convert("RGB")

        if tool == "지우개":
            canvas_L = st_canvas(
                background_image=bg_L_pil,
                background_color="#00000000",
                fill_color="#00000000",
                stroke_color="#0000FFFF",
                stroke_width=st.session_state.get("brush_val", 24),
                update_streamlit=True,
                height=target_h,
                width=target_w,
                drawing_mode="freedraw",
                display_toolbar=True,
                key="draw_left_erase",
            )
            st.slider("브러시 두께", 3, 80, 24, 1, key="brush_val")
        else:
            canvas_L = st_canvas(
                background_image=bg_L_pil,
                background_color="#00000000",
                fill_color="#00000000",
                stroke_color="#FF0040",
                stroke_width=2,
                update_streamlit=True,
                height=target_h,
                width=target_w,
                drawing_mode="point",
                display_toolbar=False,
                key="draw_left_pick",
            )

    # 우측: 라이브 결과 미리보기 + 커밋
    with right:
        st.subheader("미리보기 (후처리/편집 반영)")
        preview_base = _compose_layers_over(current_base)

        if tool == "지우개":
            if canvas_L and canvas_L.image_data is not None:
                mask_small = Image.fromarray((canvas_L.image_data[..., 3] > 0).astype(np.uint8) * 255)
                mask = mask_small.resize((current_base.width, current_base.height), Image.NEAREST)
                tmp = _inpaint_rgba(current_base, mask, method="telea", radius=7)
                preview_img = _compose_layers_over(tmp)
            else:
                preview_img = preview_base

            st.image(preview_img.resize((target_w, target_h), Image.LANCZOS),
                     caption="지우개 미리보기", width=target_w)

            if st.button("지우개 적용하기", type="primary"):
                if canvas_L and canvas_L.image_data is not None:
                    mask_small = Image.fromarray((canvas_L.image_data[..., 3] > 0).astype(np.uint8) * 255)
                    mask = mask_small.resize((current_base.width, current_base.height), Image.NEAREST)
                    out = _inpaint_rgba(current_base, mask, method="telea", radius=7)
                    st.session_state["edited_image"] = out
                    st.success("지우개 적용 완료")
                else:
                    st.info("마스크가 없습니다. 좌측에서 칠해 주세요.")

        elif tool == "텍스트":
            # 좌측 클릭으로 좌표 픽업(없으면 입력값)
            bx, by = x_text, y_text
            if canvas_L.json_data and canvas_L.json_data.get("objects"):
                obj = canvas_L.json_data["objects"][-1]
                px = int(obj.get("left", target_w//2)); py = int(obj.get("top", target_h//2))
                scale = current_base.width / target_w
                bx, by = int(px * scale), int(py * scale)

            tmp = preview_base.copy()
            draw = ImageDraw.Draw(tmp)

            def _hex(h: str):
                h=h.lstrip('#')
                if len(h)==6: r,g,b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16); return (r,g,b,255)
                if len(h)==8: r,g,b,a = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16), int(h[6:8],16); return (r,g,b,a)
                return (255,255,255,255)

            _stroke_text(draw, (bx, by), text_val, font,
                         fill=_hex(color), stroke_width=int(stroke_w),
                         stroke_fill=_hex(stroke_color), align_center=bool(align_center))

            st.image(tmp.resize((target_w, target_h), Image.LANCZOS),
                     caption="텍스트 미리보기", width=target_w)

            if st.button("텍스트 레이어 추가", type="primary"):
                _add_layer({
                    "type": "text",
                    "text": text_val,
                    "x": int(bx),
                    "y": int(by),
                    "font": font,
                    "fill": _hex(color),
                    "stroke_w": int(stroke_w),
                    "stroke_fill": _hex(stroke_color),
                    "align_center": bool(align_center),
                })
                st.success("텍스트 레이어가 추가되었습니다.")

        else:  # 사진
            tmp = preview_base.copy()
            bx, by = x_img, y_img
            if canvas_L.json_data and canvas_L.json_data.get("objects"):
                obj = canvas_L.json_data["objects"][-1]
                px = int(obj.get("left", target_w//2)); py = int(obj.get("top", target_h//2))
                scale = current_base.width / target_w
                bx, by = int(px * scale), int(py * scale)

            if place_img is not None:
                boxes = _extract_slot_boxes(slots_mask_path, layout_path) if snap else []
                if snap and boxes:
                    x0,y0,x1,y1 = boxes[int(target_idx)]
                    W,H = (x1-x0, y1-y0)
                    layer_img = _fit_image_to_box(place_img, (W,H), keep_aspect=keep_aspect, cover=cover)
                    px, py = x0, y0
                else:
                    layer_img = _fit_image_to_box(place_img, (int(w_img), int(h_img)), keep_aspect=keep_aspect, cover=cover)
                    px, py = int(bx), int(by)

                try:
                    if int(scale_pct) != 100:
                        new_w = max(1, int(layer_img.width  * int(scale_pct) / 100))
                        new_h = max(1, int(layer_img.height * int(scale_pct) / 100))
                        layer_img = layer_img.resize((new_w, new_h), Image.LANCZOS)
                except Exception:
                    pass

                if opacity < 1.0:
                    a = layer_img.split()[-1].point(lambda t: int(t*opacity)); layer_img.putalpha(a)
                tmp.alpha_composite(layer_img, dest=(px, py))

                st.image(tmp.resize((target_w, target_h), Image.LANCZOS),
                         caption="사진 미리보기", width=target_w)

                if st.button("사진 레이어 추가", type="primary"):
                    _add_layer({
                        "type": "image",
                        "image": layer_img,
                        "x": int(px),
                        "y": int(py),
                        "opacity": float(opacity),
                    })
                    st.success("사진 레이어가 추가되었습니다.")
            else:
                st.info("오른쪽 상단에서 사진을 업로드하면 미리보기가 표시됩니다.")

    # 레이어 / 내보내기
    st.divider(); st.subheader("레이어 / 내보내기")
    layers = st.session_state.get("layers", [])
    if not layers:
        st.caption("레이어가 없습니다. 텍스트/사진 레이어를 추가하세요.")
    else:
        for i, L in enumerate(reversed(layers)):
            idx = len(layers) - 1 - i
            col1, col2, col3, col4, col5 = st.columns([0.7, 0.6, 0.6, 2.4, 1.2])
            with col1: st.code(L.get("type"), language=None)
            with col2:
                if st.button("▲", key=f"up_{idx}"): _move_layer(idx, +1)
            with col3:
                if st.button("▼", key=f"dn_{idx}"): _move_layer(idx, -1)
            with col4:
                txt = L["text"] if L.get("type")=="text" else f"img {L['image'].size}"
                st.write(f"({idx}) x={L.get('x')}, y={L.get('y')} · {txt}")
            with col5:
                if st.button("삭제", key=f"del_{idx}"):
                    _delete_layer(idx)
                    _rerun()

    # 저장
    st.divider(); st.subheader("편집본 저장")
    composed_now = _compose_layers_over(st.session_state.get("edited_image") or base0)
    fn_default = pathlib.Path(chosen_path).with_suffix("")
    out_name = st.text_input("파일명", f"{fn_default.name}_edited.png")
    cA, cB = st.columns(2)
    with cA:
        if st.button("PNG로 저장", type="primary"):
            out_path = OUTPUT_DIR / out_name; composed_now.save(out_path, format="PNG")
            st.success(f"저장됨: {out_path}")
    with cB:
        bio = io.BytesIO(); composed_now.save(bio, format="PNG")
        st.download_button("PNG 다운로드", data=bio.getvalue(), file_name=out_name, mime="image/png")

    # 최종 업스케일
    st.divider(); st.subheader("최종 업스케일 (프리셋)")
    grid_cfg = st.session_state.get("grid_cfg", {})
    fw, fh = grid_cfg.get("final_resolution", [2000, 2828])
    presets = {
        "YAML 최종": (fw, fh),
        "인스타 4:5 1080×1350": (1080, 1350),
        "인스타 4:5 1440×1792": (1440, 1792),
        "A4 300dpi 2480×3508": (2480, 3508),
    }
    chosen = []
    pc1, pc2 = st.columns(2)
    for j, (name, wh) in enumerate(presets.items()):
        with (pc1 if j % 2 == 0 else pc2):
            if st.checkbox(f"{name} {wh[0]}×{wh[1]}", key=f"preset_{j}"):
                chosen.append((name, wh))

    if st.button("선택 프리셋 업스케일", type="primary"):
        if not chosen:
            st.warning("프리셋을 1개 이상 선택하세요.")
        else:
            base_to_export = composed_now
            for name, (W, H) in chosen:
                if upscale_if_needed is None:
                    out = base_to_export.resize((W, H), Image.LANCZOS)
                else:
                    out = upscale_if_needed(base_to_export, (W, H))
                fname = f"{pathlib.Path(chosen_path).stem}_{W}x{H}.png"
                out_path = OUTPUT_DIR / fname
                out.save(out_path)
                st.success(f"[{name}] 저장 완료: {out_path}")
                with open(out_path, "rb") as f:
                    st.download_button(f"다운로드: {fname}", f.read(), file_name=fname)

    st.caption("Tip: 좌측에서 바로 그려보며 우측 미리보기를 확인하세요. ‘지우개 적용하기’ 또는 ‘레이어 추가’로 결과를 커밋합니다.")


if __name__ == "__main__":
    render()