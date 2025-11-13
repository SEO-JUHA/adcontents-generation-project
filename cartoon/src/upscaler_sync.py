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

# ---- FastAPI Router ----
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field, model_validator
from typing import Tuple

router = APIRouter(tags=["upscaler"])

@router.get("/ping")
async def ping(): return {"ok": True}

class IfNeededRequest(BaseModel):
    path: str
    final_wh: Tuple[int, int]

    @model_validator(mode="before")
    def _coerce_final_wh(cls, data):
        v = data.get("final_wh")
        if isinstance(v, (list, tuple)) and len(v) == 2:
            data["final_wh"] = (int(v[0]), int(v[1]))
            return data
        if isinstance(v, str) and "x" in v:
            w,h = v.lower().split("x",1)
            data["final_wh"] = (int(w), int(h))
            return data
        return data

class IfNeededResponse(BaseModel):
    out_path: str

async def _save_image(request: Request, path: str) -> None:
    try:
        pool = request.app.state.db_pool
    except Exception:
        return
    user_id = request.headers.get("X-User-Id") or None
    try:
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "INSERT INTO `4cuts_image` (user_id, image_path) VALUES (%s, %s)",
                    (user_id, path)
                )
    except Exception:
        pass

@router.post("/if-needed", response_model=IfNeededResponse)
async def api_upscale_if_needed(req: IfNeededRequest, request: Request):
    try:
        out = upscale_if_needed(req.path, req.final_wh)
        await _save_image(request, out)
        return IfNeededResponse(out_path=out)
    except FileNotFoundError as e: raise HTTPException(status_code=404, detail=str(e))
    except Exception as e: raise HTTPException(status_code=500, detail=f"upscale_if_needed failed: {e}")
