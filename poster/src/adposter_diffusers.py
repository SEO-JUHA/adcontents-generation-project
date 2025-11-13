import json
import os
import random
import pathlib
import shutil
from datetime import datetime
from typing import Dict, Any, List, Optional

from PIL import Image, ImageOps
import numpy as np

try:
    from .comfy_api import post_prompt, wait_and_collect, stage_to_input
    from .prompt_builder import build_prompts
    from .upscaler import upscale_if_needed
    from .workflow_ops import (
        load_prompt, set_prompts, set_layout_image, set_logo_image,
        set_control_strength, set_controlnet_model, set_ipadapter_weight,
        set_base_resolution, set_sampler
    )
    # ▼ 자동 후처리: OCR 기반 인페인트
    from .post_ocr_erase import erase_text_regions
    # 이미지 재랭킹
    try:
        from .gpt_ranker import rank_images
    except Exception:
        from poster.src.gpt_ranker import rank_images
        
except Exception:
    # Streamlit/CLI 실행 위치에 따른 임포트 실패 대비
    from poster.src.comfy_api import post_prompt, wait_and_collect, stage_to_input
    from poster.src.prompt_builder import build_prompts
    from poster.src.upscaler import upscale_if_needed
    from poster.src.workflow_ops import (
        load_prompt, set_prompts, set_layout_image, set_logo_image,
        set_control_strength, set_controlnet_model, set_ipadapter_weight,
        set_base_resolution, set_sampler
    )
    # ▼ 자동 후처리: OCR 기반 인페인트
    from poster.src.post_ocr_erase import erase_text_regions
    # ▼ (선택) 이미지 재랭킹
    try:
        from poster.src.gpt_ranker import rank_images
    except Exception:
        rank_images = None  # 완전 미사용 폴백

_THIS = pathlib.Path(__file__).resolve()
POSTER_ROOT = _THIS.parents[1]
PROJECT_ROOT = POSTER_ROOT.parent

DATA_DIR = POSTER_ROOT / "data"
LAYOUTS_INDEX = DATA_DIR / "layouts" / "index.json"
WORKFLOW_PATH = POSTER_ROOT / "workflows" / "base_prompt.json"

_GRID_CANDIDATES = [
    POSTER_ROOT / "configs" / "experiment.grid.yaml",
]


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
    ip_w=1.0,
)


def _mid(vals: List[float]) -> float:
    if not vals:
        return 0.8
    if len(vals) == 1:
        return float(vals[0])
    lo, hi = float(vals[0]), float(vals[1])
    return (lo + hi) / 2.0


def _first(arr, default=None):
    try:
        return (arr or [default])[0]
    except Exception:
        return default


def _recipe_from_grid(gc: Dict[str, Any]) -> Dict[str, Any]:
    return dict(
        sampler_name=_first(gc.get("sampler_name"), DEFAULT_RECIPE["sampler_name"]),
        scheduler=_first(gc.get("scheduler"), DEFAULT_RECIPE["scheduler"]),
        steps=int(_first(gc.get("steps"), DEFAULT_RECIPE["steps"])),
        cfg=float(_first(gc.get("cfg"), DEFAULT_RECIPE["cfg"])),
        ip_w=float(_first(gc.get("ipadapter_weight"), DEFAULT_RECIPE["ip_w"])),
    )


def _control_from_grid(gc: Dict[str, Any]) -> float:
    cs = gc.get("control_strength_defaults", [0.8])
    if isinstance(cs, (list, tuple)):
        return _mid(list(map(float, cs)))
    return float(cs) if isinstance(cs, (int, float)) else 0.8


def load_grid_cfg() -> Dict[str, Any]:
    """
    운영 설정(experiment.grid.yaml)을 로드합니다.
    파일이 없거나 파싱 실패 시 안전한 기본값을 리턴합니다.
    """
    p = GRID_CFG_PATH
    if p.exists():
        try:
            import yaml
            cfg = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
            return cfg
        except Exception:
            pass
    # 기본값 (auto_erase + rerank 포함)
    return {
        "designs_per_layout": 3,
        "base_resolution": [1000, 1416],
        "final_resolution": [2000, 2828],
        "control_strength_defaults": [0.70, 0.90],
        "controlnet_name": "controlnet-union-sdxl-1.0/diffusion_pytorch_model.safetensors",
        "auto_erase": {
            "enabled": True,
            "engine": "paddle",
            # 다국어는 langs 배열 권장 (없으면 lang 단일값)
            "langs": None,
            "lang": "korean",
            "min_conf": 0.5,
            "boxes_pad": 14,
            "mask_mode": "union",  # union | boxes | slots
            "method": "telea",     # telea | ns
            "radius": 5,
            "tone": True,
            "alpha": 0.22,
            "write_debug": True,
        },
        "rerank": {
            "enabled": False,
            "method": "gpt",        # "gpt" | "local"
            "top_k": 3,
            "model": "gpt-4o-mini",
            "max_images": 12
        }
    }


def _resolve_path(path_like: str) -> str:
    p = pathlib.Path(path_like)
    if p.is_absolute():
        return str(p)
    cand = (PROJECT_ROOT / p).resolve()
    if cand.exists():
        return str(cand)
    parts = p.parts
    if parts and parts[0] == "poster":
        rest = pathlib.Path(*parts[1:]) if len(parts) > 1 else pathlib.Path(".")
        cand2 = (POSTER_ROOT / rest).resolve()
        if cand2.exists():
            return str(cand2)
    cand3 = (POSTER_ROOT / p).resolve()
    if cand3.exists():
        return str(cand3)
    cand4 = (pathlib.Path.cwd() / p).resolve()
    if cand4.exists():
        return str(cand4)
    return str(cand3)


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
    # 호출 시점마다 안전하게 새 폴더
    return POSTER_ROOT / "outputs" / datetime.now().strftime("%Y%m%d_%H%M%S")


# 레이아웃 이미지 경로에서 슬롯 마스크 자동 추정
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
    """grid 설정에서 auto_erase 섹션을 읽고 기본값과 병합"""
    defaults = {
        "enabled": True,
        "engine": "paddle",
        "langs": None,   # 다국어 배열 권장
        "lang": "korean",
        "min_conf": 0.5,
        "boxes_pad": 14,
        "mask_mode": "union",
        "method": "telea",
        "radius": 5,
        "tone": True,
        "alpha": 0.22,
        "write_debug": True,
    }
    cfg = (grid_cfg or {}).get("auto_erase", {})
    if not isinstance(cfg, dict):
        cfg = {}
    merged = {**defaults, **cfg}
    # 환경변수로 전체 on/off 가능 (POSTER_AUTO_ERASE=0/1)
    env_toggle = os.getenv("POSTER_AUTO_ERASE")
    if env_toggle is not None:
        merged["enabled"] = (str(env_toggle).strip() not in ("0", "false", "False"))

    langs = merged.get("langs")
    if not langs or (isinstance(langs, list) and len(langs) == 0):
        raw = str(merged.get("lang", "korean"))
        if "," in raw:
            langs = [s.strip() for s in raw.split(",") if s.strip()]
        elif "+" in raw:
            langs = [s.strip() for s in raw.split("+") if s.strip()]
        else:
            langs = [raw]
    merged["langs"] = langs

    return merged



# 다국어 OCR 인페인트 보조 유틸
def _count_mask_pixels(mask_png_path: pathlib.Path) -> int:
    try:
        with Image.open(mask_png_path) as m:
            if m.mode != "L":
                m = ImageOps.grayscale(m)
            arr = np.array(m)
            return int((arr > 0).sum())
    except Exception:
        return -1


def _erase_multilang_best(
    image_path: str,
    slots_mask_guess: Optional[str],
    out_dir: pathlib.Path,
    erase_cfg: Dict[str, Any],
) -> str:

    langs = erase_cfg.get("langs") or [erase_cfg.get("lang", "korean")]
    langs = [str(x).strip() for x in langs if str(x).strip()]
    if len(langs) == 1:
        return erase_text_regions(
            image_path=image_path,
            slots_mask_path=slots_mask_guess,
            out_dir=out_dir,
            engine=erase_cfg.get("engine", "paddle"),
            lang=langs[0],
            min_conf=float(erase_cfg.get("min_conf", 0.5)),
            boxes_pad=int(erase_cfg.get("boxes_pad", 14)),
            mask_mode=str(erase_cfg.get("mask_mode", "union")),
            method=str(erase_cfg.get("method", "telea")),
            radius=int(erase_cfg.get("radius", 5)),
            tone=bool(erase_cfg.get("tone", True)),
            alpha=float(erase_cfg.get("alpha", 0.22)),
            write_debug=bool(erase_cfg.get("write_debug", True)),
        )

    trials = []
    for lg in langs:
        sub = out_dir / f"ae_{lg}"
        sub.mkdir(parents=True, exist_ok=True)
        try:
            out_p = erase_text_regions(
                image_path=image_path,
                slots_mask_path=slots_mask_guess,
                out_dir=sub,
                engine=erase_cfg.get("engine", "paddle"),
                lang=lg,
                min_conf=float(erase_cfg.get("min_conf", 0.5)),
                boxes_pad=int(erase_cfg.get("boxes_pad", 14)),
                mask_mode=str(erase_cfg.get("mask_mode", "union")),
                method=str(erase_cfg.get("method", "telea")),
                radius=int(erase_cfg.get("radius", 5)),
                tone=bool(erase_cfg.get("tone", True)),
                alpha=float(erase_cfg.get("alpha", 0.22)),
                write_debug=bool(erase_cfg.get("write_debug", True)),
            )
            # 점수 산정: (1) OCR 박스 수, (2) 마스크 픽셀 합
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
        # 실패 시 원본 유지
        return image_path

    # 우선순위: n_boxes DESC → mask_pixels DESC
    trials.sort(key=lambda t: (t[2], t[3]), reverse=True)
    best_lang, best_path, _, _ = trials[0]

    # 최종 산출물 경로를 out_dir 루트로 통일
    final_name = pathlib.Path(image_path).with_suffix("").name + "_erase.png"
    final_path = out_dir / final_name
    try:
        shutil.copy2(best_path, final_path)
        print(f"[INFO] auto_erase selected lang={best_lang} -> {final_path}")
        return str(final_path)
    except Exception:
        return best_path


# 생성 실행기
def _run_single_config_multi_designs(
    logo_path: str,
    brand_intro: str,
    layout_path: str,
    grid_cfg: Dict[str, Any],
    recipe: Dict[str, Any],
    control_strength: float,
    num_designs: int,
    base_seed: Optional[int],
    upscale: bool,
    out_dir: Optional[pathlib.Path] = None,
) -> List[str]:

    out_dir = out_dir if isinstance(out_dir, pathlib.Path) else _output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    base_w, base_h = grid_cfg.get("base_resolution", [1000, 1416])
    final_w, final_h = grid_cfg.get("final_resolution", [2000, 2828])
    controlnet_name = grid_cfg.get(
        "controlnet_name",
        "controlnet-union-sdxl-1.0/diffusion_pytorch_model.safetensors"
    )

    pos, neg = build_prompts(brand_intro, category_hint="cafe")

    logo_name = stage_to_input(logo_path)
    layout_name = stage_to_input(layout_path)

    if base_seed is None:
        seeds = [random.randint(1, 2**31 - 1) for _ in range(max(1, int(num_designs)))]
    else:
        start = int(base_seed)
        seeds = [start + i for i in range(max(1, int(num_designs)))]

    results: List[str] = []

    # 자동 후처리 설정 로드
    erase_cfg = _auto_erase_config(grid_cfg)
    auto_erase_enabled = bool(erase_cfg.get("enabled", True))
    # 슬롯 마스크 경로 자동 추정
    slots_mask_guess = _guess_slots_mask_from_layout(layout_path) if auto_erase_enabled else None

    for i, seed in enumerate(seeds, start=1):
        prompt = load_prompt(str(WORKFLOW_PATH))
        set_controlnet_model(prompt, controlnet_name)
        set_prompts(prompt, positive=pos, negative=neg)
        set_layout_image(prompt, layout_name)
        set_logo_image(prompt, logo_name)
        set_base_resolution(prompt, base_w, base_h)
        set_control_strength(prompt, strength=float(control_strength))
        set_ipadapter_weight(prompt, weight=float(recipe["ip_w"]))
        set_sampler(
            prompt,
            seed=int(seed),
            steps=int(recipe["steps"]),
            cfg=float(recipe["cfg"]),
            sampler_name=str(recipe["sampler_name"]),
            scheduler=str(recipe["scheduler"]),
            denoise=1.0,
        )
        prompt_id = post_prompt(prompt)
        image_paths = wait_and_collect(prompt_id)

        if upscale:
            finals = [upscale_if_needed(p, (final_w, final_h)) for p in image_paths]
        else:
            finals = image_paths

        # 자동 OCR 인페인트 (활성화 시)
        if auto_erase_enabled:
            post_paths: List[str] = []
            for pth in finals:
                try:
                    ep = _erase_multilang_best(
                        image_path=pth,
                        slots_mask_guess=slots_mask_guess,
                        out_dir=out_dir,
                        erase_cfg=erase_cfg,
                    )
                    post_paths.append(ep)
                except Exception as e:
                    print(f"[WARN] auto_erase failed for {pth}: {e}")
                    post_paths.append(pth)  # 실패 시 원본 유지
            finals = post_paths

        results.extend(finals)

    # 후보 재정렬/상위 K 선정
    try:
        rr = (grid_cfg or {}).get("rerank", {})
        if rr and bool(rr.get("enabled", False)):
            method = str(rr.get("method", "gpt"))
            k = int(rr.get("top_k", 3))
            model = str(rr.get("model", "gpt-4o-mini"))
            max_images = int(rr.get("max_images", 12))
            cand = results[:max_images]
            if 'rank_images' in globals() and callable(rank_images):
                ranked = rank_images(cand, brand_intro, method=method, top_k=k, model=model)
                results = ranked
                print(f"[INFO] rerank({method}) -> top{len(results)}")
            else:
                print("[WARN] rank_images not available; skip rerank.")
    except Exception as e:
        print(f"[WARN] rerank skipped: {e}")

    return results


def generate_designs(
    logo_path: str,
    brand_intro: str,
    layout_id: str,
    sampler_name: str = DEFAULT_RECIPE["sampler_name"],
    scheduler: str = DEFAULT_RECIPE["scheduler"],
    steps: int = DEFAULT_RECIPE["steps"],
    cfg: float = DEFAULT_RECIPE["cfg"],
    ip_weight: float = DEFAULT_RECIPE["ip_w"],
    control_strength: Optional[float] = None,
    num_designs: int = 3,
    seed: Optional[int] = None,
    upscale: bool = False,
) -> List[str]:

    if not WORKFLOW_PATH.exists():
        raise SystemExit(
            f"Missing workflow API prompt: {WORKFLOW_PATH}\n"
            f"- ComfyUI UI에서 현재 그래프를 'Save (API)'로 저장해 주세요."
        )

    grid_cfg = load_grid_cfg()
    lock_recipe = bool(grid_cfg.get("lock_recipe")) or bool(os.getenv("POSTER_LOCK_RECIPE"))

    # 레시피/강도 확정
    if lock_recipe:
        recipe = _recipe_from_grid(grid_cfg)
        ctrl_strength = _control_from_grid(grid_cfg)
        num_designs = int(grid_cfg.get("designs_per_layout", num_designs))
        print("[LOCK] Using recipe from experiment.grid.yaml (overrides ignored)")
    else:
        recipe = dict(
            sampler_name=sampler_name,
            scheduler=scheduler,
            steps=int(steps),
            cfg=float(cfg),
            ip_w=float(ip_weight),
        )
        ctrl_strength = float(control_strength) if control_strength is not None \
            else _mid(grid_cfg.get("control_strength_defaults", [0.70, 0.90]))

    # 레이아웃 id → 경로
    layouts = load_layouts()
    target = find_layout(layouts, layout_id)
    layout_path_abs = _resolve_path(target["path"])

    logo_path_abs = _resolve_path(logo_path)
    if not pathlib.Path(logo_path_abs).exists():
        raise SystemExit(f"Logo not found: {logo_path_abs}")
    if not pathlib.Path(layout_path_abs).exists():
        raise SystemExit(f"Layout image not found: {layout_path_abs}")

    out_dir = _output_dir()
    results = _run_single_config_multi_designs(
        logo_path=logo_path_abs,
        brand_intro=brand_intro,
        layout_path=layout_path_abs,
        grid_cfg=grid_cfg,
        recipe=recipe,
        control_strength=ctrl_strength,
        num_designs=int(num_designs),
        base_seed=seed,
        upscale=upscale,
        out_dir=out_dir,
    )

    manifest = {
        "grid_cfg": grid_cfg,
        "recipe": recipe,
        "control_strength": ctrl_strength,
        "logo": logo_path_abs,
        "layout": layout_path_abs,
        "num_designs": int(num_designs),
        "seed_base": seed,
        "upscale": bool(upscale),
        "results": results,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }

    # 자동 후처리/재랭킹 설정 기록
    ae = _auto_erase_config(grid_cfg)
    manifest["auto_erase"] = {k: ae[k] for k in ae.keys() if k != "enabled"} | {"enabled": bool(ae.get("enabled", True))}
    manifest["rerank"] = (grid_cfg or {}).get("rerank", {})

    try:
        (out_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        print(f"[INFO] Manifest saved: {out_dir/'manifest.json'}")
    except Exception as e:
        print(f"[WARN] Failed to save manifest: {e}")

    return results


def run_single_config_multi_designs(
    *,
    logo_path: str,
    brand_intro: str,
    layout_path: str,
    grid_cfg: Dict[str, Any],
    recipe: Dict[str, Any],
    control_strength: float,
    num_designs: int = 3,
    base_seed: Optional[int] = None,
    upscale: bool = False,
) -> List[str]:

    if not WORKFLOW_PATH.exists():
        raise SystemExit(
            f"Missing workflow API prompt: {WORKFLOW_PATH}\n"
            f"- ComfyUI UI에서 현재 그래프를 'Save (API)'로 저장해 주세요."
        )

    # 경로 보정
    logo_path_abs = _resolve_path(logo_path)
    layout_path_abs = _resolve_path(layout_path)

    if not pathlib.Path(logo_path_abs).exists():
        raise SystemExit(f"Logo not found: {logo_path_abs}")
    if not pathlib.Path(layout_path_abs).exists():
        raise SystemExit(f"Layout image not found: {layout_path_abs}")
    
    if not recipe:
        recipe = _recipe_from_grid(grid_cfg or load_grid_cfg())
    else:
        recipe = dict(
            sampler_name=recipe.get("sampler_name", DEFAULT_RECIPE["sampler_name"]),
            scheduler=recipe.get("scheduler", DEFAULT_RECIPE["scheduler"]),
            steps=int(recipe.get("steps", DEFAULT_RECIPE["steps"])),
            cfg=float(recipe.get("cfg", DEFAULT_RECIPE["cfg"])),
            ip_w=float(recipe.get("ip_w", DEFAULT_RECIPE["ip_w"])),
        )

    # 컨트롤 강도 기본치
    if control_strength is None:
        ctrl_strength = _control_from_grid(grid_cfg or load_grid_cfg())
    else:
        ctrl_strength = float(control_strength)

    # 출력 폴더 생성
    out_dir = _output_dir()

    # 생성 실행
    results = _run_single_config_multi_designs(
        logo_path=logo_path_abs,
        brand_intro=brand_intro,
        layout_path=layout_path_abs,
        grid_cfg=(grid_cfg or load_grid_cfg()),
        recipe=recipe,
        control_strength=ctrl_strength,
        num_designs=int(num_designs),
        base_seed=base_seed,
        upscale=bool(upscale),
        out_dir=out_dir,
    )

    # 매니페스트 저장
    manifest = {
        "grid_cfg": (grid_cfg or load_grid_cfg()),
        "recipe": recipe,
        "control_strength": ctrl_strength,
        "logo": logo_path_abs,
        "layout": layout_path_abs,
        "num_designs": int(num_designs),
        "seed_base": base_seed,
        "upscale": bool(upscale),
        "results": results,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    ae = _auto_erase_config(grid_cfg or {})
    manifest["auto_erase"] = {k: ae[k] for k in ae.keys() if k != "enabled"} | {"enabled": bool(ae.get("enabled", True))}
    manifest["rerank"] = (grid_cfg or {}).get("rerank", {})

    try:
        (out_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        print(f"[INFO] Manifest saved: {out_dir/'manifest.json'}")
    except Exception as e:
        print(f"[WARN] Failed to save manifest: {e}")

    return results


__all__ = [
    "load_grid_cfg",
    "run_single_config_multi_designs",
    "generate_designs",
]