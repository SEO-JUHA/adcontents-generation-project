from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import json
import numpy as np
from PIL import Image, ImageOps, ImageFilter, ImageStat, ImageChops, ImageDraw

def _try_imports(engine: str | None):
    if engine == "paddle":
        try:
            from paddleocr import PaddleOCR  # type: ignore
            return ("paddle", PaddleOCR)
        except Exception as e:
            print("[WARN] PaddleOCR not available:", e)
    if engine == "tesseract":
        try:
            import pytesseract  # noqa: F401
            return ("tesseract", None)
        except Exception as e:
            print("[WARN] pytesseract not available:", e)
    return (None, None)

def _normalize_lang_for_paddle(lang: str | None) -> str:
    if not lang:
        return "en"
    s = lang.strip().lower()
    if s in ("eng","english"): return "en"
    if "kor" in s or "ko" in s or "korean" in s or "+" in s: return "korean"
    if s in ("en","korean","japan","latin","arabic","cyrillic","devanagari","ch","chinese","chinese_cht","ta","te","ka"):
        return "ch" if s == "chinese" else s
    return "en"

def load_mask_any(slots_path: str | Path, size: Tuple[int,int]) -> Image.Image:
    W,H = size
    src = Image.open(slots_path)
    if src.mode == "RGBA":
        a = src.split()[-1].resize((W,H), Image.LANCZOS)
        mn, mx = a.getextrema()
        if mn == 255 and mx == 255:
            gray = ImageOps.grayscale(src).resize((W,H), Image.LANCZOS)
            med = ImageStat.Stat(gray).median[0]
            if med > 200:
                mask = gray.point(lambda v: 255 if v < 40 else 0)
            else:
                mask = gray.point(lambda v: 255 if v > 215 else 0)
            return mask.filter(ImageFilter.MaxFilter(3)).filter(ImageFilter.MinFilter(3))
        return a.point(lambda v: 255 if v>0 else 0)
    else:
        gray = ImageOps.grayscale(src).resize((W,H), Image.LANCZOS)
        med = ImageStat.Stat(gray).median[0]
        if med > 200:
            mask = gray.point(lambda v: 255 if v < 40 else 0)
        else:
            mask = gray.point(lambda v: 255 if v > 215 else 0)
        return mask.filter(ImageFilter.MaxFilter(3)).filter(ImageFilter.MinFilter(3))

def ocr_boxes(img_path: str, slot_mask_L: Image.Image,
              engine_kind: str | None, engine_cls, min_conf=0.5, lang="en") -> List[List[float]]:
    boxes: List[List[float]] = []
    full_pil = Image.open(img_path).convert("RGB")
    W,H = full_pil.size

    if engine_kind == "paddle":
        lang = _normalize_lang_for_paddle(lang)
        PaddleOCR = engine_cls
        try:
            try:
                ocr = PaddleOCR(use_textline_orientation=True, lang=lang)
            except TypeError:
                ocr = PaddleOCR(use_angle_cls=True, lang=lang)
        except Exception as e:
            print("[WARN] PaddleOCR init failed:", e)
            return boxes
        np_img = np.array(ImageOps.exif_transpose(full_pil))
        res = ocr.ocr(np_img, cls=True)
        for line in res:
            for det in line:
                pts, info = det
                try:
                    conf = float(info[1]) if isinstance(info, (list,tuple)) else float(info.get("score", 1.0))
                except Exception:
                    conf = 1.0
                if conf < min_conf: continue
                xs = [int(p[0]) for p in pts]; ys = [int(p[1]) for p in pts]
                x1,y1,x2,y2 = max(0,min(xs)), max(0,min(ys)), min(W,max(xs)), min(H,max(ys))
                cx, cy = (x1+x2)//2, (y1+y2)//2
                if slot_mask_L.getpixel((cx,cy))>0:
                    boxes.append([x1,y1,x2,y2,float(conf)])
    elif engine_kind == "tesseract":
        import pytesseract  # type: ignore
        try:
            data = pytesseract.image_to_data(full_pil, lang=(lang or "eng"), output_type=pytesseract.Output.DICT)
        except TypeError:
            data = pytesseract.image_to_data(full_pil, output_type=pytesseract.Output.DICT)
        n = len(data.get("text", []))
        for i in range(n):
            try:
                conf = float(data["conf"][i])
            except Exception:
                conf = -1.0
            if conf < (min_conf*100): continue
            x,y,w,h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
            if w*h==0: continue
            cx, cy = x + w//2, y + h//2
            if 0<=cx<W and 0<=cy<H and slot_mask_L.getpixel((cx,cy))>0:
                boxes.append([x, y, x+w, y+h, float(conf/100.0)])
    else:
        print("[INFO] No OCR engine selected; returning empty list.")
    return boxes

def rects_to_mask(rects: List[List[float]], size: Tuple[int,int], pad=14) -> Image.Image:
    W,H = size
    a = Image.new("L", (W,H), 0)
    d = ImageDraw.Draw(a)
    for x1,y1,x2,y2,_ in rects:
        x1p,y1p = max(0, int(x1)-pad), max(0, int(y1)-pad)
        x2p,y2p = min(W, int(x2)+pad), min(H, int(y2)+pad)
        d.rectangle([x1p,y1p,x2p,y2p], fill=255)
    return a

def union_mask(slot_L: Image.Image, boxes_L: Image.Image) -> Image.Image:
    return ImageChops.lighter(slot_L, boxes_L).point(lambda v: 255 if v>0 else 0)

def _to_cv(img: Image.Image):
    import cv2
    return cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)

def _from_cv(bgr) -> Image.Image:
    import cv2
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

def fast_inpaint(img: Image.Image, mask_L: Image.Image, method="telea", radius=5,
                 tone=True, alpha=0.22) -> Image.Image:
    import cv2
    W,H = img.size
    bgr = _to_cv(img)
    m_cv = np.array(mask_L, dtype=np.uint8)
    flag = cv2.INPAINT_TELEA if method=="telea" else cv2.INPAINT_NS
    result = cv2.inpaint(bgr, (m_cv>0).astype(np.uint8)*255, int(radius), flag)
    out = _from_cv(result)
    if tone:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(21,21))
        dil = cv2.dilate(m_cv, kernel, iterations=1)
        ring = (dil>0) & (m_cv==0)
        ring_rgb = np.array(img)[ring]
        if ring_rgb.size>0:
            mean = ring_rgb.mean(axis=0).astype(np.uint8)
            tint = Image.new("RGB", (W,H), tuple(int(x) for x in mean))
            out = Image.blend(out.convert("RGB"), tint, alpha=float(alpha))
    return out

def erase_text_regions(
    image_path: str,
    slots_mask_path: str | Path | None,
    out_dir: str | Path,
    *,
    engine: str = "paddle",
    lang: str = "korean",
    min_conf: float = 0.5,
    boxes_pad: int = 14,
    mask_mode: str = "union",  # union | boxes | slots
    method: str = "telea",     # telea | ns
    radius: int = 5,
    tone: bool = True,
    alpha: float = 0.22,
    write_debug: bool = True,
) -> str:

    img = Image.open(image_path).convert("RGB")
    W,H = img.size
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    slot_L = None
    if slots_mask_path and Path(slots_mask_path).exists():
        slot_L = load_mask_any(str(slots_mask_path), (W,H))

    kind, engine_cls = _try_imports(engine if engine!="none" else None)
    rects = ocr_boxes(image_path, (slot_L or Image.new("L",(W,H),255)), kind, engine_cls, min_conf=min_conf, lang=lang)
    boxes_L = rects_to_mask(rects, (W,H), pad=boxes_pad) if rects else Image.new("L",(W,H),0)

    if mask_mode == "boxes":
        final_L = boxes_L
    elif mask_mode == "slots":
        final_L = slot_L or Image.new("L",(W,H),0)
    else:
        final_L = union_mask((slot_L or Image.new("L",(W,H),0)), boxes_L)

    if write_debug:
        (out_dir/"ocr_boxes.json").write_text(json.dumps({"boxes": rects}, indent=2), encoding="utf-8")
        dbg = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        dbg.putalpha(final_L)
        dbg.save(out_dir / "slots_mask_ocr.png")

    out_img = fast_inpaint(img, final_L, method=method, radius=radius, tone=tone, alpha=alpha)
    out_path = out_dir / (Path(image_path).stem + "_erase.png")
    out_img.save(out_path)
    return str(out_path)