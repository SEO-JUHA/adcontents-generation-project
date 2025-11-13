from PIL import Image
from typing import Tuple
import pathlib

def upscale_if_needed(path: str, final_wh: Tuple[int, int]) -> str:
    p = pathlib.Path(path)
    if not p.exists():
        try:
            import yaml
            here = pathlib.Path(__file__).resolve()
            cfg_path = here.parents[1] / "configs" / "routing.yaml"
            cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
            base = cfg.get("save_dir_hint")
            if base:
                candidate = pathlib.Path(base) / path
                if candidate.exists():
                    p = candidate
        except Exception:
            return str(p)

    with Image.open(p) as img:
        if img.mode not in ("RGB", "RGBA"):
            img = img.convert("RGB")
        if (img.width, img.height) == final_wh:
            return str(p)
        out = img.resize(final_wh, Image.LANCZOS)
        out_path = p.with_name(p.stem + f"_{final_wh[0]}x{final_wh[1]}" + p.suffix)
        out.save(out_path)
        return str(out_path)