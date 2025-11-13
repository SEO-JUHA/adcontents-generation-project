from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union, Dict, Any
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageChops
import textwrap
import os
import re

try:
    from backend.database import SessionLocal, FourcutsImage
except Exception as e:
    raise RuntimeError(f"[caption_overlay_sync] backend.database import failed: {e}")

# Font helpers
_FONT_HINTS = [
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansKR-Regular.otf",
    "/usr/share/fonts/truetype/noto/NotoSansKR-Regular.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    "/usr/share/fonts/truetype/nanum/NanumSquare.ttf",
    "/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf",
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]

def _find_font(explicit_path: Optional[str] = None) -> Optional[str]:
    if explicit_path and Path(explicit_path).exists():
        return explicit_path
    for p in _FONT_HINTS:
        if Path(p).exists():
            return p
    return None

def _load_font(font_path: Optional[str], size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    p = _find_font(font_path)
    try:
        if p:
            return ImageFont.truetype(p, size=size)
    except Exception:
        pass
    return ImageFont.load_default()

def _wrap_by_char_limit(text: str, limit: int = 25) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    out, line = [], ""
    def flush():
        nonlocal line
        if line:
            out.append(line)
            line = ""
    for tok in text.split():
        if len(tok) > limit:
            flush()
            for i in range(0, len(tok), limit):
                out.append(tok[i:i+limit])
        else:
            if not line:
                line = tok
            elif len(line) + 1 + len(tok) <= limit:
                line += " " + tok
            else:
                flush()
                line = tok
    flush()

    final = []
    for ln in out:
        if len(ln) <= limit:
            final.append(ln)
        else:
            for i in range(0, len(ln), limit):
                final.append(ln[i:i+limit])
    return final

def _wrap_text_for_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    wrapped = textwrap.wrap(text, width=40, break_long_words=False, replace_whitespace=False)
    lines: List[str] = []
    for para in ("\n".join(wrapped)).splitlines():
        if not para:
            lines.append("")
            continue
        buf = ""
        for ch in para:
            test = (buf + ch) if buf else ch
            bbox = draw.textbbox((0, 0), test, font=font)
            if bbox[2] - bbox[0] <= max_width:
                buf = test
            else:
                if buf:
                    lines.append(buf)
                    buf = ch
                else:
                    lines.append(ch)
                    buf = ""
        if buf:
            lines.append(buf)
    return lines

def _measure_lines(draw: ImageDraw.ImageDraw, lines: List[str], font: ImageFont.ImageFont, line_spacing: int) -> Tuple[int, int]:
    w = 0
    h = 0
    for i, ln in enumerate(lines):
        bbox = draw.textbbox((0, 0), ln, font=font, stroke_width=0)
        lw = bbox[2] - bbox[0]
        lh = bbox[3] - bbox[1]
        w = max(w, lw)
        h += lh + (line_spacing if i < len(lines) - 1 else 0)
    return w, h

def _next_version_path(dst_dir: Path, stem: str) -> Path:
    dst_dir.mkdir(parents=True, exist_ok=True)
    v = 1
    while True:
        p = dst_dir / f"{stem}_v{v:03d}.jpg"
        if not p.exists():
            return p
        v += 1

def _find_bases_dir(p: Path) -> Optional[Path]:
    # overlays → bases 형제 폴더를 우선 탐색
    for parent in [p.parent, p.parent.parent, p.parent.parent.parent]:
        if parent and parent.exists():
            cand = parent / "bases"
            if cand.exists() and cand.is_dir():
                return cand
    return None

def _resolve_base_path(any_path: Union[str, os.PathLike]) -> Path:
    p = Path(any_path).resolve()
    if "_base" in p.stem or "bases" in p.parts:
        return p

    bases_dir = _find_bases_dir(p)
    # 1) panel 번호 기반 탐색
    m = re.search(r"panel_(\d+)", p.stem)
    if bases_dir and m:
        idx = int(m.group(1))
        cand = bases_dir / f"panel_{idx:02d}_base.jpg"
        if cand.exists():
            return cand

    # 2) stem 치환 기반 탐색
    stem_base = re.sub(r"_caption.*$", "_base", p.stem)
    if bases_dir:
        cand = bases_dir / f"{stem_base}.jpg"
        if cand.exists():
            return cand

    # 3) 같은 폴더도 시도
    cand2 = p.parent / f"{stem_base}.jpg"
    if cand2.exists():
        return cand2

    print(f"[overlay_panel] WARN: base image not found for {p}; using given path")
    return p


# Core APIs
def overlay_panel(
    base_path: Union[str, Image.Image],
    text: str,
    out_dir: Union[str, os.PathLike],
    *,
    bar_mode: str = "tight",
    corner_radius: int = 10,
    bottom_margin: int = 20,
    max_chars_per_line: int = 25,
    stroke_width: int = 3,
    stroke_fill: Tuple[int,int,int] = (0,0,0),
    text_fill: Tuple[int,int,int] = (255,255,255),
    tight_pad_x: int = 28,
    tight_pad_y: int = 12,
    force_base: bool = True,   # 기본적으로 항상 베이스 강제
) -> str:
    # 경로/이미지 인식
    if isinstance(base_path, Image.Image):
        src_img = base_path
        src_path = None
    else:
        src_path = Path(base_path)
        if force_base:
            src_path = _resolve_base_path(src_path)
        src_img = Image.open(src_path).convert("RGB")

    # 렌더
    over = draw_caption_bar(
        src_img, text,
        bar_mode=bar_mode,
        corner_radius=corner_radius,
        bottom_margin=bottom_margin,
        max_chars_per_line=max_chars_per_line,
        stroke_width=stroke_width,
        stroke_fill=stroke_fill,
        text_fill=text_fill,
        tight_pad_x=tight_pad_x,
        tight_pad_y=tight_pad_y,
    )

    # 파일명은 베이스 기준으로 생성
    base_name = "panel" if src_path is None else src_path.stem
    stem = base_name.replace("_base", "_caption")
    dst_dir = Path(out_dir)
    out_path = _next_version_path(dst_dir, stem)
    over.save(out_path, quality=95)
    return str(out_path)

def draw_caption_bar(
    image: Image.Image,
    text: str,
    *,
    bar_height_ratio: float = 0.18,
    bar_alpha: int = 220,
    bar_color: Tuple[int, int, int] = (0, 0, 0),
    padding: int = 28,
    corner_radius: int = 0,
    font_path: Optional[str] = None,
    font_size: Optional[int] = None,
    line_spacing: Optional[int] = None,
    stroke_width: int = 2,
    stroke_fill: Tuple[int, int, int] = (0, 0, 0),
    text_fill: Tuple[int, int, int] = (255, 255, 255),
    align: str = "center",
    bar_mode: str = "tight",
    bottom_margin: int = 25,
    tight_pad_x: int = 28,
    tight_pad_y: int = 12,
    tight_min_width_ratio: float = 0.35,
    tight_max_width_ratio: float = 0.92,
    text_shadow: bool = True,
    shadow_offset: Tuple[int, int] = (0, 2),
    shadow_fill_rgba: Tuple[int, int, int, int] = (0, 0, 0, 200),
    max_chars_per_line: int = 25,
) -> Image.Image:
    """
    이미지 하단에 캡션 바를 그리고 텍스트를 자동 크기/줄바꿈으로 배치.
    반환: 새로운 PIL.Image (원본 보존)
    """
    im = image.convert("RGB")
    W, H = im.size
    canvas = im.copy()
    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    tmp_draw = ImageDraw.Draw(Image.new("RGB", (10, 10)))

    # 폰트 크기 탐색
    base_bar_h = max(60, int(H * max(0.08, min(bar_height_ratio, 0.45))))
    avail_w_full = W - padding * 2
    avail_h_full = base_bar_h - padding * 2
    if avail_w_full <= 10 or avail_h_full <= 10:
        return canvas

    base_size = font_size or max(16, int(avail_h_full * 0.28))
    candidates = [int(base_size * f) for f in (1.05, 1.0, 0.92, 0.85, 0.78, 0.7, 0.64, 0.58)]
    if font_size is not None:
        candidates = [font_size]

    best_font = None
    best_lines: List[str] = []
    best_size = base_size

    raw_lines = _wrap_by_char_limit(text, limit=max_chars_per_line)
    max_width_tight = int(W * tight_max_width_ratio) - 2 * tight_pad_x

    for sz in candidates:
        font = _load_font(font_path, sz)
        ls = line_spacing if line_spacing is not None else max(4, sz // 6)
        lines_px: List[str] = []
        
        for ln in raw_lines:
            buf = ""
            for ch in ln:
                test = buf + ch
                bbox = tmp_draw.textbbox((0, 0), test, font=font)
                limit_w = avail_w_full if (bar_mode == "full") else max_width_tight
                if bbox[2] - bbox[0] <= limit_w:
                    buf = test
                else:
                    if buf:
                        lines_px.append(buf)
                        buf = ch
                    else:
                        lines_px.append(ch)
                        buf = ""
            if buf:
                lines_px.append(buf)

        _, total_h = _measure_lines(tmp_draw, lines_px, font, ls)

        if total_h <= (avail_h_full if bar_mode == "full" else H) and lines_px:
            best_font = font
            best_lines = lines_px
            best_size = sz
            break

    if best_font is None:
        best_size = max(12, min(base_size, int(avail_h_full * 0.35)))
        best_font = _load_font(font_path, best_size)
        ls = line_spacing if line_spacing is not None else max(4, best_size // 6)
        best_lines = raw_lines
    else:
        ls = line_spacing if line_spacing is not None else max(4, best_size // 6)

    # 실제 렌더링
    draw = ImageDraw.Draw(overlay)

    if bar_mode == "full":
        bar_top = H - base_bar_h
        rect = (0, bar_top, W, H)
        if corner_radius > 0:
            # 아래쪽 모서리만 둥글게
            mask = Image.new("L", (W, H), 0)
            mdraw = ImageDraw.Draw(mask)
            mdraw.rounded_rectangle(rect, radius=corner_radius, fill=255)
            bar_layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            bdraw = ImageDraw.Draw(bar_layer)
            bdraw.rectangle(rect, fill=(bar_color[0], bar_color[1], bar_color[2], int(bar_alpha)))
            overlay = Image.composite(bar_layer, overlay, mask)
        else:
            draw.rectangle(rect, fill=(bar_color[0], bar_color[1], bar_color[2], int(bar_alpha)))

        # 텍스트 배치
        _, total_h = _measure_lines(tmp_draw, best_lines, best_font, ls)
        y = bar_top + (base_bar_h - total_h) // 2

        for ln in best_lines:
            bbox = tmp_draw.textbbox((0, 0), ln, font=best_font, stroke_width=stroke_width)
            lw = bbox[2] - bbox[0]
            if align == "left":
                x = padding
            elif align == "right":
                x = W - padding - lw
            else:
                x = (W - lw) // 2

            if text_shadow:
                draw.text((x + shadow_offset[0], y + shadow_offset[1]), ln, font=best_font, fill=shadow_fill_rgba)
            draw.text((x, y), ln, font=best_font, fill=text_fill, stroke_width=stroke_width, stroke_fill=stroke_fill)
            y += (bbox[3] - bbox[1]) + ls

    else:  # tight
        # 텍스트 블록 크기
        text_w, text_h = _measure_lines(tmp_draw, best_lines, best_font, ls)
        bw = text_w + 2 * tight_pad_x
        bh = text_h + 2 * tight_pad_y

        min_w = int(W * max(0.0, min(tight_min_width_ratio, 1.0)))
        max_w = int(W * max(0.0, min(tight_max_width_ratio, 1.0)))
        bw = max(min_w, min(bw, max_w))

        x0 = (W - bw) // 2
        y0 = H - bottom_margin - bh
        x1 = x0 + bw
        y1 = y0 + bh

        draw.rounded_rectangle([x0, y0, x1, y1], radius=max(8, corner_radius),
                               fill=(bar_color[0], bar_color[1], bar_color[2], int(bar_alpha)))

        # 텍스트 중앙 배치
        y = y0 + (bh - text_h) // 2 - 12
        for ln in best_lines:
            bbox = tmp_draw.textbbox((0, 0), ln, font=best_font, stroke_width=stroke_width)
            lw = bbox[2] - bbox[0]
            if align == "left":
                x = x0 + tight_pad_x
            elif align == "right":
                x = x1 - tight_pad_x - lw
            else:
                x = x0 + (bw - lw) // 2

            if text_shadow:
                draw.text((x + shadow_offset[0], y + shadow_offset[1]), ln, font=best_font, fill=shadow_fill_rgba)
            draw.text((x, y), ln, font=best_font, fill=text_fill, stroke_width=stroke_width, stroke_fill=stroke_fill)
            y += (bbox[3] - bbox[1]) + ls

    out = Image.alpha_composite(canvas.convert("RGBA"), overlay)
    return out.convert("RGB")

# 2x2 grid (원형 유지)
def _open_if_path(x: Union[str, Image.Image]) -> Image.Image:
    return Image.open(x).convert("RGB") if isinstance(x, (str, os.PathLike)) else x.convert("RGB")

def compose_2x2(
    images: Sequence[Union[str, Image.Image]],
    out_path: Optional[str] = None,
    *,
    final_side: int = 2160,
    pad_px: int = 16,
    background: Tuple[int, int, int] = (0, 0, 0),
    save_quality: int = 95,
) -> Image.Image:
    if not images:
        raise ValueError("images must contain at least 1 image")
    ims = [_open_if_path(i) for i in images]
    if len(ims) < 4:
        last = ims[-1]
        while len(ims) < 4:
            ims.append(last.copy())

    S = int(final_side)
    P = int(pad_px)
    panel_w = (S - P * 3) // 2
    panel_h = (S - P * 3) // 2

    def fit(im: Image.Image, w: int, h: int) -> Image.Image:
        iw, ih = im.size
        scale = min(w / iw, h / ih)
        nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
        resized = im.resize((nw, nh), Image.LANCZOS)
        out = Image.new("RGB", (w, h), background)
        ox = (w - nw) // 2
        oy = (h - nh) // 2
        out.paste(resized, (ox, oy))
        return out

    tiles = [fit(im, panel_w, panel_h) for im in ims[:4]]
    grid = Image.new("RGB", (S, S), background)
    positions = [
        (P, P),
        (P * 2 + panel_w, P),
        (P, P * 2 + panel_h),
        (P * 2 + panel_w, P * 2 + panel_h),
    ]
    for t, pos in zip(tiles, positions):
        grid.paste(t, pos)

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        grid.save(out_path, quality=save_quality)
    return grid

__all__ = ["draw_caption_bar", "compose_2x2", "overlay_panel"]

# FastAPI Router
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

router = APIRouter(tags=["caption-overlay"])

@router.get("/ping")
def ping():
    return {"ok": True}

# --- DB helper ---
def _db_save_image(path: str, user_id: Optional[int] = None) -> None:
    if not path:
        return
    db = SessionLocal()
    try:
        db.add(FourcutsImage(user_id=user_id, image_path=str(path)))
        db.commit()
    except Exception as e:
        db.rollback()
        print(f"[WARN] DB save failed: {e}")
    finally:
        db.close()

# overlay
class OverlayRequest(BaseModel):
    base_path: str
    text: str
    out_dir: str
    bar_mode: str = "tight"
    corner_radius: int = 10
    bottom_margin: int = 18
    max_chars_per_line: int = 25
    stroke_width: int = 3
    stroke_fill: Tuple[int,int,int] = (0,0,0)
    text_fill: Tuple[int,int,int] = (255,255,255)
    tight_pad_x: int = 28
    tight_pad_y: int = 12
    force_base: bool = True

class OverlayResponse(BaseModel):
    out_path: str

@router.post("/overlay", response_model=OverlayResponse)
def api_overlay(req: OverlayRequest, request: Request):
    try:
        out = overlay_panel(
            base_path=req.base_path, text=req.text, out_dir=req.out_dir, bar_mode=req.bar_mode,
            corner_radius=req.corner_radius, bottom_margin=req.bottom_margin, max_chars_per_line=req.max_chars_per_line,
            stroke_width=req.stroke_width, stroke_fill=req.stroke_fill, text_fill=req.text_fill,
            tight_pad_x=req.tight_pad_x, tight_pad_y=req.tight_pad_y, force_base=req.force_base
        )

        user_id: Optional[int] = None
        try:
            raw = request.headers.get("X-User-Id")
            user_id = int(raw) if raw is not None else None
        except Exception:
            user_id = None
        _db_save_image(out, user_id=user_id)
        return OverlayResponse(out_path=out)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"overlay_panel failed: {e}")

# compose-2x2
class ComposeRequest(BaseModel):
    images: List[str]
    out_path: Optional[str] = None
    final_side: int = 2160
    pad_px: int = 16
    background: Tuple[int,int,int] = (0,0,0)
    save_quality: int = 95

class ComposeResponse(BaseModel):
    out_path: Optional[str] = None

@router.post("/compose-2x2", response_model=ComposeResponse)
def api_compose(req: ComposeRequest, request: Request):
    try:
        im = compose_2x2(
            req.images,
            out_path=req.out_path,
            final_side=req.final_side,
            pad_px=req.pad_px,
            background=req.background,
            save_quality=req.save_quality
        )
        outp: Optional[str] = req.out_path
        if not outp:
            tmp = Path("/tmp") / ("comic_2x2_" + Path(req.images[0]).stem + ".jpg")
            tmp.parent.mkdir(parents=True, exist_ok=True)
            im.save(tmp, quality=req.save_quality)
            outp = str(tmp)

        # 저장 기록
        user_id: Optional[int] = None
        try:
            raw = request.headers.get("X-User-Id")
            user_id = int(raw) if raw is not None else None
        except Exception:
            user_id = None
        _db_save_image(outp, user_id=user_id)

        return ComposeResponse(out_path=outp)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"compose_2x2 failed: {e}")

# draw-bar
class DrawBarRequest(BaseModel):
    image_path: str
    text: str
    out_path: str
    bar_height_ratio: float = 0.18
    bar_alpha: int = 220
    bar_color: Tuple[int,int,int] = (0,0,0)
    padding: int = 28
    corner_radius: int = 0
    font_path: Optional[str] = None
    font_size: Optional[int] = None
    line_spacing: Optional[int] = None
    stroke_width: int = 2
    stroke_fill: Tuple[int,int,int] = (0,0,0)
    text_fill: Tuple[int,int,int] = (255,255,255)
    align: str = "center"
    bar_mode: str = "tight"
    bottom_margin: int = 25
    tight_pad_x: int = 28
    tight_pad_y: int = 12
    tight_min_width_ratio: float = 0.35
    tight_max_width_ratio: float = 0.92
    text_shadow: bool = True
    shadow_offset: Tuple[int,int] = (0,2)
    shadow_fill_rgba: Tuple[int,int,int,int] = (0,0,0,200)
    max_chars_per_line: int = 25

class DrawBarResponse(BaseModel):
    out_path: str

@router.post("/draw-bar", response_model=DrawBarResponse)
def api_draw_bar(req: DrawBarRequest, request: Request):
    try:
        with Image.open(req.image_path) as im:
            im = im.convert("RGB")
            over = draw_caption_bar(
                image=im, text=req.text,
                bar_height_ratio=req.bar_height_ratio, bar_alpha=req.bar_alpha, bar_color=req.bar_color,
                padding=req.padding, corner_radius=req.corner_radius, font_path=req.font_path, font_size=req.font_size,
                line_spacing=req.line_spacing, stroke_width=req.stroke_width, stroke_fill=req.stroke_fill, text_fill=req.text_fill,
                align=req.align, bar_mode=req.bar_mode, bottom_margin=req.bottom_margin, tight_pad_x=req.tight_pad_x, tight_pad_y=req.tight_pad_y,
                tight_min_width_ratio=req.tight_min_width_ratio, tight_max_width_ratio=req.tight_max_width_ratio, text_shadow=req.text_shadow,
                shadow_offset=req.shadow_offset, shadow_fill_rgba=req.shadow_fill_rgba, max_chars_per_line=req.max_chars_per_line,
            )
        outp = Path(req.out_path)
        outp.parent.mkdir(parents=True, exist_ok=True)
        over.save(outp, quality=95)

        user_id: Optional[int] = None
        try:
            raw = request.headers.get("X-User-Id")
            user_id = int(raw) if raw is not None else None
        except Exception:
            user_id = None
        _db_save_image(str(outp), user_id=user_id)

        return DrawBarResponse(out_path=str(outp))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"draw_caption_bar failed: {e}")