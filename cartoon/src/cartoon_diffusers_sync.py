from __future__ import annotations
import json, os, random, pathlib, shutil
from datetime import datetime
from typing import Dict, Any, List, Optional, Mapping, Tuple
from dotenv import load_dotenv; load_dotenv()

from PIL import Image, ImageOps
import numpy as np

try:
    from .comfy_api_sync import post_prompt, wait_and_collect, stage_to_input
    from .prompt_builder_sync import build_prompts, build_prompts_panel
    try:
        from .upscaler_sync import upscale_if_needed
    except Exception:
        upscale_if_needed = None
    from .workflow_ops_sync import (
        load_prompt, set_prompts, set_layout_image, set_logo_image,
        set_control_strength, set_controlnet_model, set_ipadapter_weight,
        set_base_resolution, set_sampler
    )
    from .post_ocr_erase_sync import erase_text_regions
    from .caption_overlay_sync import draw_caption_bar, compose_2x2
    from .story_suggester_sync import suggest_story
except Exception:
    # 폴백(구 네임스페이스)
    from cartoon.src.comfy_api_sync import post_prompt, wait_and_collect, stage_to_input
    try:
        from cartoon.src.prompt_builder_sync import build_prompts, build_prompts_panel
    except Exception:
        from cartoon.src.prompt_builder_sync import build_prompts
        def build_prompts_panel(brand_intro: str, story_beat: str, *_, **__):
            pos, neg = build_prompts(f"{brand_intro} | {story_beat}", category_hint="instagram comic panel")
            return pos, neg
    try:
        from cartoon.src.upscaler_sync import upscale_if_needed
    except Exception:
        upscale_if_needed = None
    from cartoon.src.workflow_ops_sync import (
        load_prompt, set_prompts, set_layout_image, set_logo_image,
        set_control_strength, set_controlnet_model, set_ipadapter_weight,
        set_base_resolution, set_sampler
    )
    from cartoon.src.post_ocr_erase_sync import erase_text_regions
    from cartoon.src.caption_overlay_sync import draw_caption_bar, compose_2x2
    try:
        from cartoon.src.story_suggester_sync import suggest_story
    except Exception:
        suggest_story = None

from frontend.load_env import ensure_env_loaded; ensure_env_loaded()

# 업스케일 폴백
if "upscale_if_needed" not in globals() or upscale_if_needed is None:
    def upscale_if_needed(path: str, final_wh: tuple[int, int]) -> str:
        return path

# ---- Paths / Constants ----
_THIS = pathlib.Path(__file__).resolve()
CARTOON_ROOT = _THIS.parents[1]
PROJECT_ROOT = CARTOON_ROOT.parent

DATA_DIR = CARTOON_ROOT / "data"
LAYOUTS_INDEX = DATA_DIR / "layouts" / "index.json"
WORKFLOW_PATH = CARTOON_ROOT / "workflows" / "base_prompt.json"

_GRID_CANDIDATES = [CARTOON_ROOT / "configs" / "experiment.grid.yaml"]

def _pick_grid_path() -> pathlib.Path:
    for p in _GRID_CANDIDATES:
        if p.exists():
            return p
    return _GRID_CANDIDATES[0]

GRID_CFG_PATH = _pick_grid_path()

DEFAULT_RECIPE: Dict[str, Any] = dict(
    sampler_name="dpmpp_2m",
    scheduler="karras",
    steps=22,
    cfg=7.0,
    ip_w=0.6,
)

# ---- Helpers ----
def _env_true(name: str, default: bool = True) -> bool:
    v = os.getenv(name)
    if v is None: return default
    return str(v).strip().lower() not in ("0","false","no","off","")

def _style_ref_path() -> Optional[str]:
    p = (os.getenv("CARTOON_STYLE_REF") or "").strip()
    if not p: return None
    try:
        rp = _resolve_path(p)
        return rp if pathlib.Path(rp).exists() else None
    except Exception:
        return None

def _mid(vals: List[float]) -> float:
    if not vals: return 0.8
    if len(vals) == 1: return float(vals[0])
    lo, hi = float(vals[0]), float(vals[1])
    return (lo + hi) / 2.0

def _first(arr, default=None):
    try: return (arr or [default])[0]
    except Exception: return default

def _recipe_from_grid(gc: Dict[str, Any]) -> Dict[str, Any]:
    return dict(
        sampler_name=_first(gc.get("sampler_name"), DEFAULT_RECIPE["sampler_name"]),
        scheduler=_first(gc.get("scheduler"), DEFAULT_RECIPE["scheduler"]),
        steps=int(_first(gc.get("steps"), DEFAULT_RECIPE["steps"])),
        cfg=float(_first(gc.get("cfg"), DEFAULT_RECIPE["cfg"])),
        ip_w=float(_first(gc.get("ipadapter_weight"), DEFAULT_RECIPE["ip_w"])),
    )

def _apply_env_overrides(recipe: Dict[str, Any], ctrl_strength: float) -> tuple[Dict[str, Any], float, str]:
    mode = (os.getenv("CARTOON_REF_MODE") or "strong").strip().lower()
    ipw_env = os.getenv("CARTOON_IP_W")
    ctrl_env = os.getenv("CARTOON_CTRL_STRENGTH")

    r = dict(recipe)
    if mode == "soft":
        r["ip_w"] = max(0.6, float(r.get("ip_w", 1.0)))
        ctrl_strength = min(0.70, ctrl_strength)
    elif mode == "strong":
        r["ip_w"] = max(1.35, float(r.get("ip_w", 1.0)))
        ctrl_strength = max(0.90, ctrl_strength)
    else:  # medium
        r["ip_w"] = max(1.0, float(r.get("ip_w", 1.0)))
        ctrl_strength = max(0.80, ctrl_strength)

    if ipw_env:
        try: r["ip_w"] = float(ipw_env)
        except Exception: pass
    if ctrl_env:
        try: ctrl_strength = float(ctrl_env)
        except Exception: pass
    return r, ctrl_strength, mode

def _control_from_grid(gc: Dict[str, Any]) -> float:
    cs = gc.get("control_strength_defaults", [0.8])
    if isinstance(cs, (list, tuple)): return _mid(list(map(float, cs)))
    return float(cs) if isinstance(cs, (int, float)) else 0.8

def load_grid_cfg() -> Dict[str, Any]:
    p = GRID_CFG_PATH
    if p.exists():
        try:
            import yaml
            cfg = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
            return cfg
        except Exception:
            pass
    return {
        "designs_per_layout": 3,
        "base_resolution": [1080, 1080],
        "final_resolution": [2160, 2160],
        "control_strength_defaults": [0.70, 0.90],
        "controlnet_name": "controlnet-union-sdxl-1.0/diffusion_pytorch_model.safetensors",
        "auto_erase": {
            "enabled": True, "engine": "paddle", "langs": None, "lang": "korean",
            "min_conf": 0.5, "boxes_pad": 14, "mask_mode": "union",
            "method": "telea", "radius": 5, "tone": True, "alpha": 0.22, "write_debug": True,
        },
    }

def _resolve_path(path_like: str) -> str:
    p = pathlib.Path(path_like)
    if p.is_absolute(): return str(p)
    cand = (PROJECT_ROOT / p).resolve()
    if cand.exists(): return str(cand)
    cand2 = (CARTOON_ROOT / p).resolve()
    if cand2.exists(): return str(cand2)
    parts = p.parts
    if parts and parts[0] == "cartoon":
        rest = pathlib.Path(*parts[1:]) if len(parts) > 1 else pathlib.Path(".")
        cand3 = (CARTOON_ROOT / rest).resolve()
        if cand3.exists(): return str(cand3)
    poster_root = PROJECT_ROOT / "poster"
    cand4 = (poster_root / p).resolve()
    if cand4.exists(): return str(cand4)
    if parts and parts[0] == "poster":
        rest = pathlib.Path(*parts[1:]) if len(parts) > 1 else pathlib.Path(".")
        cand5 = (poster_root / rest).resolve()
        if cand5.exists(): return str(cand5)
    cand6 = (pathlib.Path.cwd() / p).resolve()
    if cand6.exists(): return str(cand6)
    return str(cand2)

def load_layouts() -> List[Dict[str, Any]]:
    if not LAYOUTS_INDEX.exists():
        raise SystemExit(f"Missing layouts index: {LAYOUTS_INDEX}")
    try:
        catalog = json.loads(LAYOUTS_INDEX.read_text(encoding="utf-8"))
    except Exception as e:
        raise SystemExit(f"Invalid JSON at {LAYOUTS_INDEX}: {e}")
    layouts = catalog.get("layouts", [])
    if not layouts:
        raise SystemExit("No layouts found in index.json")
    return layouts

def find_layout(layouts: List[Dict[str, Any]], layout_id: str) -> Dict[str, Any]:
    for it in layouts:
        if it.get("id") == layout_id:
            return it
    raise SystemExit(f"Layout id not found: {layout_id}")

def _output_dir() -> pathlib.Path:
    return CARTOON_ROOT / "outputs" / datetime.now().strftime("%Y%m%d_%H%M%S")

def _guess_slots_mask_from_layout(layout_path: str) -> Optional[str]:
    p = pathlib.Path(layout_path)
    candidates = [
        p.with_name(p.stem + "_alpha.png"),
        p.with_name(p.stem.replace("_line", "_alpha") + ".png") if "_line" in p.stem else None,
        p.with_name("slots_mask.png"),
        p.with_name("slots.png"),
    ]
    for c in candidates:
        if c and c.exists():
            return str(c)
    return None

def _auto_erase_config(grid_cfg: Dict[str, Any]) -> Dict[str, Any]:
    defaults = {
        "enabled": True, "engine": "paddle", "langs": None, "lang": "korean",
        "min_conf": 0.5, "boxes_pad": 14, "mask_mode": "union",
        "method": "telea", "radius": 5, "tone": True, "alpha": 0.22, "write_debug": True,
    }
    cfg = (grid_cfg or {}).get("auto_erase", {})
    merged = {**defaults, **(cfg if isinstance(cfg, dict) else {})}
    env_toggle = os.getenv("CARTOON_AUTO_ERASE") or os.getenv("POSTER_AUTO_ERASE")
    if env_toggle is not None:
        merged["enabled"] = (str(env_toggle).strip() not in ("0","false","False"))

    langs = merged.get("langs")
    if not langs or (isinstance(langs, list) and len(langs) == 0):
        raw = str(merged.get("lang", "korean"))
        if "," in raw: langs = [s.strip() for s in raw.split(",") if s.strip()]
        elif "+" in raw: langs = [s.strip() for s in raw.split("+") if s.strip()]
        else: langs = [raw]
    merged["langs"] = langs
    return merged

def _count_mask_pixels(mask_png_path: pathlib.Path) -> int:
    try:
        with Image.open(mask_png_path) as m:
            if m.mode != "L": m = ImageOps.grayscale(m)
            arr = np.array(m); return int((arr > 0).sum())
    except Exception:
        return -1

def _erase_multilang_best(image_path: str, slots_mask_guess: Optional[str],
                          out_dir: pathlib.Path, erase_cfg: Dict[str, Any]) -> str:
    langs = erase_cfg.get("langs") or [erase_cfg.get("lang", "korean")]
    langs = [str(x).strip() for x in langs if str(x).strip()]
    if len(langs) == 1:
        return erase_text_regions(
            image_path=image_path, slots_mask_path=slots_mask_guess, out_dir=out_dir,
            engine=erase_cfg.get("engine", "paddle"), lang=langs[0],
            min_conf=float(erase_cfg.get("min_conf", 0.5)), boxes_pad=int(erase_cfg.get("boxes_pad", 14)),
            mask_mode=str(erase_cfg.get("mask_mode", "union")), method=str(erase_cfg.get("method", "telea")),
            radius=int(erase_cfg.get("radius", 5)), tone=bool(erase_cfg.get("tone", True)),
            alpha=float(erase_cfg.get("alpha", 0.22)), write_debug=bool(erase_cfg.get("write_debug", True)),
        )
    trials = []
    for lg in langs:
        sub = out_dir / f"ae_{lg}"
        sub.mkdir(parents=True, exist_ok=True)
        try:
            out_p = erase_text_regions(
                image_path=image_path, slots_mask_path=slots_mask_guess, out_dir=sub,
                engine=erase_cfg.get("engine", "paddle"), lang=lg,
                min_conf=float(erase_cfg.get("min_conf", 0.5)), boxes_pad=int(erase_cfg.get("boxes_pad", 14)),
                mask_mode=str(erase_cfg.get("mask_mode", "union")), method=str(erase_cfg.get("method", "telea")),
                radius=int(erase_cfg.get("radius", 5)), tone=bool(erase_cfg.get("tone", True)),
                alpha=float(erase_cfg.get("alpha", 0.22)), write_debug=bool(erase_cfg.get("write_debug", True)),
            )
            boxes_json = sub / "ocr_boxes.json"
            n_boxes = 0
            if boxes_json.exists():
                try:
                    data = json.loads(boxes_json.read_text(encoding="utf-8"))
                    n_boxes = len(data.get("boxes", []))
                except Exception:
                    n_boxes = 0
            mask_png = sub / "slots_mask_ocr.png"
            mask_pixels = _count_mask_pixels(mask_png) if mask_png.exists() else -1
            trials.append((lg, out_p, n_boxes, mask_pixels))
        except Exception as e:
            print(f"[WARN] auto_erase(lang={lg}) failed: {e}")
    if not trials:
        return image_path
    trials.sort(key=lambda t: (t[2], t[3]), reverse=True)
    best_lang, best_path, _, _ = trials[0]
    final_name = pathlib.Path(image_path).with_suffix("").name + "_erase.png"
    final_path = out_dir / final_name
    try:
        shutil.copy2(best_path, final_path)
        print(f"[INFO] auto_erase selected lang={best_lang} -> {final_path}")
        return str(final_path)
    except Exception:
        return best_path

def _clean_panel_word_if_any(img_path: str, out_dir: pathlib.Path) -> str:
    if not _env_true("CARTOON_CLEAR_PANEL_TEXT", True):
        return img_path
    engine = os.getenv("CARTOON_OCR_ENGINE", "tesseract")
    kw = os.getenv("CARTOON_CLEAR_KEYWORDS", "panel,penel")
    keywords = [k.strip() for k in kw.split(",") if k.strip()]
    try:
        bottom_ratio = float(os.getenv("CARTOON_CLEAR_BOTTOM_RATIO", "0.6"))
    except Exception:
        bottom_ratio = 0.6
    try:
        return erase_text_regions(
            image_path=img_path, slots_mask_path=None, out_dir=out_dir,
            engine=engine, lang="eng", min_conf=0.3, boxes_pad=6,
            mask_mode="boxes", method="telea", radius=3, tone=True, alpha=0.0, write_debug=False,
            keywords=keywords, bottom_ratio=bottom_ratio
        )
    except TypeError:
        print("[clean_panel_word] erase_text_regions without keywords support; skip cleaning.")
        return img_path
    except Exception as e:
        print("[clean_panel_word] skip due to:", e)
        return img_path

# ---- Face-safe presets (person_food only) ----
_PF_POS_FACE = (
    "natural skin texture, detailed eyes, clean facial features, "
    "photographic portrait, sharp focus, soft lighting"
)
_PF_NEG_FACE = (
    "blurry face, disfigured face, deformed, extra eyes, bad anatomy, "
    "lowres, oversaturated skin, jpeg artifacts, text, watermark"
)

def _is_person_food(beat: str) -> bool:
    return "person_food" in (beat or "").lower()

def _tune_recipe_for_person(recipe: Dict[str, Any]) -> Dict[str, Any]:
    tuned = dict(recipe)
    tuned["steps"] = max(28, int(recipe.get("steps", 22)))
    tuned["cfg"]   = min(6.2, float(recipe.get("cfg", 7.0)))
    tuned["sampler_name"] = recipe.get("sampler_name", "dpmpp_2m")
    tuned["scheduler"]    = recipe.get("scheduler", "karras")
    return tuned

def _pf_ipw_cap(default_ipw: float) -> float:
    try: cap = float(os.getenv("CARTOON_PF_IPW_MAX_FACE", "0.62"))
    except Exception: cap = 0.62
    return min(default_ipw, cap)

def _pf_ctrl_cap(default_ctrl: float) -> float:
    try: cap = float(os.getenv("CARTOON_PF_CTRL_MAX_FACE", "0.78"))
    except Exception: cap = 0.78
    return min(default_ctrl, cap)

def _maybe_face_restore(path: str) -> str:
    if os.getenv("CARTOON_FACE_RESTORE", "0").lower() in ("0","false","no",""):
        return path
    try:
        from gfpgan import GFPGANer  # Optional
        restorer = GFPGANer(model_path=None)
        im = Image.open(path).convert("RGB")
        np_img = np.array(im)[:, :, ::-1]
        _, _, restored = restorer.enhance(np_img, has_aligned=False, only_center_face=False, paste_back=True)
        outp = pathlib.Path(path).with_suffix("").as_posix() + "_face.jpg"
        Image.fromarray(restored[:, :, ::-1]).save(outp, quality=95)
        return outp
    except Exception:
        return path

# ---- render core ----
def _render_single_panel(
    *, panel_positive: str, panel_negative: str, layout_path_abs: str,
    ref_image_abs: Optional[str], logo_fallback_abs: Optional[str],
    grid_cfg: Dict[str, Any], recipe: Dict[str, Any], ctrl_strength: float,
    seed: Optional[int], upscale: bool, out_dir: pathlib.Path,
    override_ip_w: Optional[float] = None,
    override_ctrl: Optional[float] = None,
    disable_auto_erase: bool = False,
    override_steps: Optional[int] = None,
    override_cfg: Optional[float] = None,
) -> str:
    prompt = load_prompt(str(WORKFLOW_PATH))
    set_controlnet_model(
        prompt,
        grid_cfg.get("controlnet_name", "controlnet-union-sdxl-1.0/diffusion_pytorch_model.safetensors")
    )
    base_w, base_h = grid_cfg.get("base_resolution", [1080 , 1080])
    set_base_resolution(prompt, base_w, base_h)
    set_layout_image(prompt, stage_to_input(layout_path_abs))
    set_prompts(prompt, positive=panel_positive, negative=panel_negative)

    ctrl = float(override_ctrl if override_ctrl is not None else ctrl_strength)
    ipw  = float(override_ip_w if override_ip_w is not None else recipe["ip_w"])
    set_control_strength(prompt, ctrl)
    set_ipadapter_weight(prompt, ipw)

    ref_abs = ref_image_abs or logo_fallback_abs
    if ref_abs:
        set_logo_image(prompt, stage_to_input(ref_abs))

    set_sampler(
        prompt,
        seed=int(seed) if seed is not None else random.randint(1, 2**31 - 1),
        steps=int(override_steps if override_steps is not None else recipe["steps"]),
        cfg=float(override_cfg if override_cfg is not None else recipe["cfg"]),
        sampler_name=str(recipe["sampler_name"]),
        scheduler=str(recipe["scheduler"]),
        denoise=1.0,
    )
    pid = post_prompt(prompt)
    outs = wait_and_collect(pid)

    final_w, final_h = grid_cfg.get("final_resolution", [2160, 2160])
    finals = [upscale_if_needed(p, (final_w, final_h)) for p in outs] if upscale else outs

    erase_cfg = _auto_erase_config(grid_cfg)
    if bool(erase_cfg.get("enabled", True)) and not disable_auto_erase:
        slots_mask_guess = _guess_slots_mask_from_layout(layout_path_abs)
        post_paths = []
        for pth in finals:
            try:
                ep = _erase_multilang_best(pth, slots_mask_guess, out_dir, erase_cfg)
                post_paths.append(ep)
            except Exception as e:
                print(f"[WARN] auto_erase failed for {pth}: {e}")
                post_paths.append(pth)
        finals = post_paths

    return finals[0]

def _defer_overlay() -> bool:
    return (os.getenv("CARTOON_DEFER_OVERLAY", "1").lower() not in ("0","false","no"))

# ---- 4컷: 스토리 비트 입력 ----
def generate_4cut_comic(
    *, logo_path: str, brand_intro: str, layout_id: str, story_beats: List[str],
    captions: Optional[List[str]] = None, seed: Optional[int] = None,
    upscale: bool = False, make_grid: bool = False, grid_side: int = 2160, grid_pad_px: int = 16,
) -> Dict[str, Any]:

    if len(story_beats) != 4:
        raise SystemExit("story_beats must be length 4")
    if not WORKFLOW_PATH.exists():
        raise SystemExit("Missing workflow API prompt: base_prompt.json")

    defer_overlay = _defer_overlay()
    auto_story_on = (os.getenv("CARTOON_AUTO_STORY", "1").lower() not in ("0","false","no"))

    grid_cfg = load_grid_cfg()
    lock_recipe = bool(grid_cfg.get("lock_recipe")) or bool(os.getenv("CARTOON_LOCK_RECIPE") or os.getenv("POSTER_LOCK_RECIPE"))
    if lock_recipe:
        recipe = _recipe_from_grid(grid_cfg)
        ctrl_strength = _control_from_grid(grid_cfg)
    else:
        recipe = dict(**DEFAULT_RECIPE)
        ctrl_strength = _control_from_grid(grid_cfg)
    recipe, ctrl_strength, ref_mode = _apply_env_overrides(recipe, ctrl_strength)

    layouts = load_layouts()
    layout_path_abs = _resolve_path(find_layout(layouts, layout_id)["path"])
    if not pathlib.Path(layout_path_abs).exists():
        raise SystemExit(f"Layout image not found: {layout_path_abs}")
    logo_abs = _resolve_path(logo_path) if logo_path else None
    if logo_abs and (not pathlib.Path(logo_abs).exists()):
        logo_abs = None

    out_dir = _output_dir()
    bases_dir = out_dir / "bases"
    overlays_dir = out_dir / "overlays"
    out_dir.mkdir(parents=True, exist_ok=True)

    lock_style  = _env_true("CARTOON_STYLE_LOCK", True)
    style_ref   = _style_ref_path()
    style_anchor: Optional[str] = None

    panels_raw: List[str] = []
    for i, beat in enumerate(story_beats, start=1):
        pos, neg = build_prompts_panel(brand_intro, beat)
        if _is_person_food(beat):
            pos = f"{pos}, {_PF_POS_FACE}"
            neg = f"{neg}, {_PF_NEG_FACE}"

        ref_for_this = style_ref or (style_anchor if (lock_style and style_anchor) else None)

        is_person_food = _is_person_food(beat)
        override_ip_w = None
        override_ctrl = None
        disable_auto_erase = False
        recipe_to_use = recipe

        if is_person_food:
            pf_policy = (os.getenv("CARTOON_PF_REF_POLICY") or "none").strip().lower()  # none|content|style
            if pf_policy == "none":
                ref_for_this = None
            elif pf_policy == "content":
                pf_ref = os.getenv("CARTOON_PERSON_FOOD_REF", "").strip()
                ref_for_this = _resolve_path(pf_ref) if pf_ref else ref_for_this

            override_ctrl = _pf_ctrl_cap(float(ctrl_strength))
            override_ip_w = _pf_ipw_cap(float(recipe["ip_w"]))
            disable_auto_erase = (os.getenv("CARTOON_PF_NO_ERASE", "1").lower() not in ("0","false","no"))
            recipe_to_use = _tune_recipe_for_person(recipe)

        panel = _render_single_panel(
            panel_positive=pos, panel_negative=neg, layout_path_abs=layout_path_abs,
            ref_image_abs=ref_for_this, logo_fallback_abs=logo_abs, grid_cfg=grid_cfg, recipe=recipe_to_use,
            ctrl_strength=ctrl_strength, seed=(seed + (i - 1)) if seed is not None else None,
            upscale=upscale, out_dir=out_dir,
            override_ip_w=override_ip_w, override_ctrl=override_ctrl,
            disable_auto_erase=disable_auto_erase,
            override_steps=(recipe_to_use["steps"] if is_person_food else None),
            override_cfg=(recipe_to_use["cfg"] if is_person_food else None),
        )
        panels_raw.append(panel)

        if (i == 1) and lock_style and (style_ref is None):
            style_anchor = panel

    # 베이스 저장 + 'panel' 문구 클린 + (옵션) Face Restore
    base_panels: List[str] = []
    for i, p in enumerate(panels_raw, start=1):
        cleaned_path = _clean_panel_word_if_any(p, out_dir)
        # 사람 컷이면 선택적 Face Restore
        beat = story_beats[i-1] if i-1 < len(story_beats) else ""
        final_src = _maybe_face_restore(cleaned_path) if _is_person_food(beat) else cleaned_path
        im = Image.open(final_src).convert("RGB")
        dst = bases_dir / f"panel_{i:02d}_base.jpg"
        dst.parent.mkdir(parents=True, exist_ok=True)
        im.save(dst, quality=95)
        base_panels.append(str(dst))

    # (옵션) 즉시 오버레이
    final_panels: List[str] = []
    grid_base = None
    grid_overlay = None

    if not defer_overlay:
        texts = captions
        if (not texts) and auto_story_on and (suggest_story is not None):
            try:
                model = os.getenv("CARTOON_STORY_MODEL", "gpt-4.1-mini")
                lang  = os.getenv("CARTOON_STORY_LANG", "ko")
                maxc  = int(os.getenv("CARTOON_STORY_MAXCHARS", "42"))
                core_msg = story_beats[0] if story_beats else (brand_intro or "")
                lines = suggest_story(
                    brand_intro=brand_intro, core_message=core_msg,
                    panel_image_paths=base_panels, language=lang, max_chars=maxc, model=model,
                )
                if any(lines): texts = lines
            except Exception as e:
                print("[story] GPT captions failed, fallback to beats:", e)
        texts = texts if (texts and len(texts) == 4) else story_beats

        cap_mode = os.getenv("CARTOON_CAPTION_MODE", "tight").strip().lower()
        for i, (b, cap) in enumerate(zip(base_panels, texts), start=1):
            im = Image.open(b).convert("RGB")
            try:
                over = draw_caption_bar(
                    im, cap,
                    bar_mode=cap_mode,
                    corner_radius=18 if cap_mode == "tight" else 0,
                    bottom_margin=1000 if cap_mode == "tight" else 0,
                    max_chars_per_line=int(os.getenv("CARTOON_CAPTION_MAXCHARS", "25")),
                    stroke_width=int(os.getenv("CARTOON_CAPTION_STROKE", "3")),
                    stroke_fill=(0, 0, 0), text_fill=(255, 255, 255),
                )
            except TypeError:
                over = draw_caption_bar(im, cap)
            out_p = overlays_dir / f"panel_{i:02d}_caption_v001.jpg"
            out_p.parent.mkdir(parents=True, exist_ok=True)
            over.save(out_p, quality=95)
            final_panels.append(str(out_p))

        if make_grid:
            gp = overlays_dir / "comic_2x2_v001.jpg"
            compose_2x2(final_panels, str(gp), final_side=int(grid_side), pad_px=int(grid_pad_px))
            grid_overlay = str(gp)

    if make_grid and not grid_overlay:
        gp = bases_dir / "comic_2x2_base.jpg"
        compose_2x2(base_panels, str(gp), final_side=int(grid_side), pad_px=int(grid_pad_px))
        grid_base = str(gp)

    manifest = {
        "mode": "4cut_comic",
        "brand_intro": brand_intro,
        "story_beats": story_beats,
        "captions_initial": captions,
        "layout_id": layout_id,
        "layout_path": layout_path_abs,
        "seed_base": seed,
        "upscale": bool(upscale),
        "deferred_overlay": bool(defer_overlay),
        "ref_mode": ref_mode,
        "style_lock": bool(lock_style),
        "style_ref": style_ref,
        "style_anchor": style_anchor,
        "panel_bases": base_panels,
        "panel_overlays": final_panels,
        "grid_base": grid_base,
        "grid_overlay": grid_overlay,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    try:
        (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass

    return {
        "panel_bases": base_panels,
        "panel_images": final_panels or base_panels,
        "grid_image": grid_overlay or grid_base,
        "manifest_path": str(out_dir / "manifest.json"),
    }

# ---- 4컷: 자산 기반 ----
def _nth(seq, i):
    try: return (seq or [None])[i]
    except Exception: return None

def _choose_first_exist(paths: List[Optional[str]]) -> Optional[str]:
    for p in paths:
        if not p: continue
        rp = _resolve_path(p)
        if pathlib.Path(rp).exists(): return rp
    return None

def panel_plan_from_core_message(core_message: str, assets: Mapping[str, Any]) -> Dict[str, Any]:
    def _as_list(x):
        if x is None: return []
        return x if isinstance(x, (list, tuple)) else [x]
    menus = [a for a in (assets.get("menus") or []) if a]
    store = [a for a in (assets.get("store") or []) if a]
    interior_imgs = [a for a in (assets.get("interior") or []) if a]
    logo  = assets.get("logo")

    beats = [
        "exterior | 간판이 보이는 카페 외관/입구",
        "food | 대표 메뉴(업로드한 음식 이미지) 클로즈업",
        "interior | 좌석/카운터가 보이는 내부 공간감",
        "person_food | 사람이 대표 메뉴를 손에 들고 있는 장면",
    ]
    ref_img = [
        _choose_first_exist([_first(store), logo, _first(menus)]),
        _choose_first_exist([_first(menus), _nth(menus, 1), logo, _first(store)]),
        _choose_first_exist([_first(interior_imgs), _nth(store, 1), _first(store)]),
        _choose_first_exist([_first(menus), _nth(menus, 1), logo]),
    ]
    return {"story_beats": beats, "panel_refs": ref_img, "logo": logo or None, "layout_id": assets.get("layout_id")}

def generate_4cut_from_assets(
    *, core_message: str, images: Mapping[str, Any], captions: Optional[List[str]] = None,
    seed: Optional[int] = None, upscale: bool = False, make_grid: bool = False,
    grid_side: int = 2160, grid_pad_px: int = 16,
) -> Dict[str, Any]:

    if not WORKFLOW_PATH.exists():
        raise SystemExit("Missing workflow API prompt: base_prompt.json")

    defer_overlay = _defer_overlay()

    plan = panel_plan_from_core_message(core_message, images)
    beats: List[str] = plan["story_beats"]
    panel_refs: List[Optional[str]] = plan["panel_refs"]
    logo_abs = _resolve_path(plan.get("logo")) if plan.get("logo") else None

    layouts = load_layouts()
    layout_id = plan.get("layout_id") or (images.get("layout_id") if images else None)
    if not layout_id:
        raise SystemExit("layout_id not provided in images/plan")
    layout_path_abs = _resolve_path(find_layout(layouts, layout_id)["path"])
    if not pathlib.Path(layout_path_abs).exists():
        raise SystemExit(f"Layout image not found: {layout_path_abs}")
    if logo_abs and (not pathlib.Path(logo_abs).exists()):
        logo_abs = None

    grid_cfg = load_grid_cfg()
    lock_recipe = bool(grid_cfg.get("lock_recipe")) or bool(os.getenv("CARTOON_LOCK_RECIPE") or os.getenv("POSTER_LOCK_RECIPE"))
    if lock_recipe:
        recipe = _recipe_from_grid(grid_cfg); ctrl_strength = _control_from_grid(grid_cfg)
    else:
        recipe = dict(**DEFAULT_RECIPE); ctrl_strength = _control_from_grid(grid_cfg)
    recipe, ctrl_strength, ref_mode = _apply_env_overrides(recipe, ctrl_strength)

    out_dir = _output_dir()
    bases_dir = out_dir / "bases"
    overlays_dir = out_dir / "overlays"
    out_dir.mkdir(parents=True, exist_ok=True)

    lock_style  = _env_true("CARTOON_STYLE_LOCK", True)
    style_ref   = _style_ref_path()
    style_anchor: Optional[str] = None

    panel_paths: List[str] = []
    for i, beat in enumerate(beats, start=1):
        pos, neg = build_prompts_panel(core_message, beat)
        if _is_person_food(beat):
            pos = f"{pos}, {_PF_POS_FACE}"
            neg = f"{neg}, {_PF_NEG_FACE}"

        content_ref = panel_refs[i - 1] if i - 1 < len(panel_refs) else None
        ref_for_this = style_ref or (style_anchor if (lock_style and style_anchor) else content_ref)

        is_person_food = _is_person_food(beat)
        override_ip_w = None
        override_ctrl = None
        disable_auto_erase = False
        recipe_to_use = recipe

        if is_person_food:
            pf_policy = (os.getenv("CARTOON_PF_REF_POLICY") or "none").strip().lower()
            if pf_policy == "none":
                ref_for_this = None
            elif pf_policy == "content":
                ref_for_this = content_ref or ref_for_this
                pf_ref = os.getenv("CARTOON_PERSON_FOOD_REF", "").strip()
                if pf_ref: ref_for_this = _resolve_path(pf_ref)

            override_ctrl = _pf_ctrl_cap(float(ctrl_strength))
            override_ip_w = _pf_ipw_cap(float(recipe["ip_w"]))
            disable_auto_erase = (os.getenv("CARTOON_PF_NO_ERASE", "1").lower() not in ("0","false","no"))
            recipe_to_use = _tune_recipe_for_person(recipe)

        panel = _render_single_panel(
            panel_positive=pos, panel_negative=neg, layout_path_abs=layout_path_abs,
            ref_image_abs=ref_for_this, logo_fallback_abs=logo_abs, grid_cfg=grid_cfg, recipe=recipe_to_use,
            ctrl_strength=ctrl_strength, seed=(seed + (i - 1)) if seed is not None else None,
            upscale=upscale, out_dir=out_dir,
            override_ip_w=override_ip_w, override_ctrl=override_ctrl, disable_auto_erase=disable_auto_erase,
            override_steps=(recipe_to_use["steps"] if is_person_food else None),
            override_cfg=(recipe_to_use["cfg"] if is_person_food else None),
        )
        panel_paths.append(panel)

        if (i == 1) and lock_style and (style_ref is None):
            style_anchor = panel

    base_panels: List[str] = []
    for i, p in enumerate(panel_paths, start=1):
        cleaned_path = _clean_panel_word_if_any(p, out_dir)
        beat = beats[i-1] if i-1 < len(beats) else ""
        final_src = _maybe_face_restore(cleaned_path) if _is_person_food(beat) else cleaned_path
        im = Image.open(final_src).convert("RGB")
        dst = bases_dir / f"panel_{i:02d}_base.jpg"
        dst.parent.mkdir(parents=True, exist_ok=True)
        im.save(dst, quality=95)
        base_panels.append(str(dst))

    final_panels: List[str] = []
    grid_base = None
    grid_overlay = None

    if not defer_overlay:
        texts = captions
        if (not texts) and (os.getenv("CARTOON_AUTO_STORY", "1").lower() not in ("0","false","no")) and (suggest_story is not None):
            try:
                model = os.getenv("CARTOON_STORY_MODEL", "gpt-4.1-mini")
                lang  = os.getenv("CARTOON_STORY_LANG", "ko")
                maxc  = int(os.getenv("CARTOON_STORY_MAXCHARS", "42"))
                lines = suggest_story(
                    brand_intro=core_message, core_message=core_message,
                    panel_image_paths=base_panels, language=lang, max_chars=maxc, model=model,
                )
                if any(lines): texts = lines
            except Exception as e:
                print("[story] GPT captions failed, fallback to beats:", e)
        texts = texts if (texts and len(texts) == 4) else beats

        cap_mode = os.getenv("CARTOON_CAPTION_MODE", "tight").strip().lower()
        for i, (b, cap) in enumerate(zip(base_panels, texts), start=1):
            im = Image.open(b).convert("RGB")
            try:
                over = draw_caption_bar(
                    im, cap, bar_mode=cap_mode,
                    corner_radius=18 if cap_mode == "tight" else 0,
                    bottom_margin=1000 if cap_mode == "tight" else 0,
                    max_chars_per_line=int(os.getenv("CARTOON_CAPTION_MAXCHARS", "25")),
                    stroke_width=int(os.getenv("CARTOON_CAPTION_STROKE", "3")),
                    stroke_fill=(0,0,0), text_fill=(255,255,255),
                )
            except TypeError:
                over = draw_caption_bar(im, cap)
            out_p = overlays_dir / f"panel_{i:02d}_caption_v001.jpg"
            out_p.parent.mkdir(parents=True, exist_ok=True)
            over.save(out_p, quality=95)
            final_panels.append(str(out_p))

        if make_grid:
            gp = overlays_dir / "comic_2x2_v001.jpg"
            compose_2x2(final_panels, str(gp), final_side=int(grid_side), pad_px=int(grid_pad_px))
            grid_overlay = str(gp)

    if make_grid and not grid_overlay:
        gp = bases_dir / "comic_2x2_base.jpg"
        compose_2x2(base_panels, str(gp), final_side=int(grid_side), pad_px=int(grid_pad_px))
        grid_base = str(gp)

    manifest = {
        "mode": "4cut_from_assets", "core_message": core_message, "images": images, "beats": beats,
        "captions_initial": captions, "layout_path": layout_path_abs, "seed_base": seed, "upscale": bool(upscale),
        "deferred_overlay": bool(defer_overlay), "ref_mode": ref_mode, "style_lock": _env_true("CARTOON_STYLE_LOCK", True),
        "style_ref": _style_ref_path(), "style_anchor": style_anchor,
        "panel_bases": base_panels, "panel_overlays": final_panels,
        "grid_base": grid_base, "grid_overlay": grid_overlay,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    try:
        (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass

    return {
        "panel_bases": base_panels,
        "panel_images": final_panels or base_panels,
        "grid_image": grid_overlay or grid_base,
        "manifest_path": str(out_dir / "manifest.json"),
    }

__all__ = ["generate_4cut_comic", "panel_plan_from_core_message", "generate_4cut_from_assets"]

# ---- FastAPI Router ----
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from typing import Any, Dict, List, Mapping, Optional

router = APIRouter(tags=["cartoon-diffusers"])

@router.get("/ping")
async def ping(): return {"ok": True}

class Generate4CutComicRequest(BaseModel):
    logo_path: str
    brand_intro: str
    layout_id: str
    story_beats: List[str]
    captions: Optional[List[str]] = None
    seed: Optional[int] = None
    upscale: bool = False
    make_grid: bool = False
    grid_side: int = 2160
    grid_pad_px: int = 16

class GenerateResponse(BaseModel):
    panel_bases: List[str]
    panel_images: List[str]
    grid_image: Optional[str] = None
    manifest_path: Optional[str] = None

async def _save_images(request: Request, paths: list[str]) -> None:
    try:
        pool = request.app.state.db_pool
    except Exception:
        return
    user_id = request.headers.get("X-User-Id") or None
    try:
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                for p in (paths or []):
                    try:
                        await cur.execute("INSERT INTO `4cuts_image` (user_id, image_path) VALUES (%s, %s)", (user_id, p))
                    except Exception:
                        pass
    except Exception:
        pass

@router.post("/generate-4cut-comic", response_model=GenerateResponse)
async def api_generate_4cut_comic(req: Generate4CutComicRequest, request: Request):
    try:
        if len(req.story_beats) != 4: raise HTTPException(400, "story_beats must be length 4")
        res = generate_4cut_comic(
            logo_path=req.logo_path, brand_intro=req.brand_intro, layout_id=req.layout_id,
            story_beats=req.story_beats, captions=req.captions, seed=req.seed,
            upscale=req.upscale, make_grid=req.make_grid, grid_side=req.grid_side, grid_pad_px=req.grid_pad_px,
        )
        paths = []
        try:
            paths.extend(res.get("panel_bases") or [])
            paths.extend(res.get("panel_images") or [])
            gi = res.get("grid_image")
            if gi: paths.append(gi)
        except Exception: pass
        await _save_images(request, [p for p in paths if p])
        return GenerateResponse(**res)
    except SystemExit as e: raise HTTPException(status_code=400, detail=str(e))
    except Exception as e: raise HTTPException(status_code=500, detail=f"generate_4cut_comic failed: {e}")

class PanelPlanRequest(BaseModel):
    core_message: str
    assets: Mapping[str, Any]

class PanelPlanResponse(BaseModel):
    story_beats: List[str]
    panel_refs: List[Optional[str]]
    logo: Optional[str]
    layout_id: Optional[str]

@router.post("/panel-plan", response_model=PanelPlanResponse)
async def api_panel_plan(req: PanelPlanRequest):
    try:
        plan = panel_plan_from_core_message(req.core_message, req.assets)
        return PanelPlanResponse(
            story_beats=plan.get("story_beats", []),
            panel_refs=plan.get("panel_refs", []),
            logo=plan.get("logo"),
            layout_id=plan.get("layout_id"),
        )
    except SystemExit as e: raise HTTPException(status_code=400, detail=str(e))
    except Exception as e: raise HTTPException(status_code=500, detail=f"panel_plan_from_core_message failed: {e}")

class GenerateFromAssetsRequest(BaseModel):
    core_message: str
    images: Mapping[str, Any]
    captions: Optional[List[str]] = None
    seed: Optional[int] = None
    upscale: bool = False
    make_grid: bool = False
    grid_side: int = 2160
    grid_pad_px: int = 16

@router.post("/generate-4cut-from-assets", response_model=GenerateResponse)
async def api_generate_4cut_from_assets(req: GenerateFromAssetsRequest, request: Request):
    try:
        res = generate_4cut_from_assets(
            core_message=req.core_message, images=req.images, captions=req.captions, seed=req.seed,
            upscale=req.upscale, make_grid=req.make_grid, grid_side=req.grid_side, grid_pad_px=req.grid_pad_px,
        )
        paths = []
        try:
            paths.extend(res.get("panel_bases") or [])
            paths.extend(res.get("panel_images") or [])
            gi = res.get("grid_image")
            if gi: paths.append(gi)
        except Exception: pass
        await _save_images(request, [p for p in paths if p])
        return GenerateResponse(**res)
    except SystemExit as e: raise HTTPException(status_code=400, detail=str(e))
    except Exception as e: raise HTTPException(status_code=500, detail=f"generate_4cut_from_assets failed: {e}")