import os
import json
import yaml
import time
import shutil
import urllib.request
import urllib.error
from typing import Dict, Any, List, Optional
from pathlib import Path
from PIL import Image

# Routing / URL helpers
def _load_routing() -> Dict[str, Any]:
    try:
        here = Path(__file__).resolve()
        cfg_path = here.parents[1] / "configs" / "routing.yaml"
        with open(cfg_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}

def _url(path: str) -> str:
    base = os.environ.get("COMFY_URL") or _load_routing().get("comfy_url") or "http://127.0.0.1:8188"
    return f"{base}{path}"

def system_stats() -> Dict[str, Any]:
    url = _url("/system_stats")
    with urllib.request.urlopen(url) as resp:
        return json.loads(resp.read().decode("utf-8"))

# File staging
def _guess_input_dir() -> Optional[str]:
    env = os.environ.get("COMFY_INPUT_DIR")
    if env:
        return env
    cfg = _load_routing()
    if cfg.get("input_dir"):
        return str(cfg["input_dir"])
    here = Path(__file__).resolve()
    guess_cartoon = here.parents[1] / "ComfyUI" / "input"
    return str(guess_cartoon)

def stage_to_input(local_path: str) -> str:
    src = Path(local_path).resolve()
    if not src.exists():
        raise FileNotFoundError(f"[stage] source not found: {src}")

    input_dir = _guess_input_dir()
    if not input_dir:
        print(f"[stage] no input_dir configured, skip copy; return basename={src.name}")
        return src.name

    input_dir_p = Path(input_dir)
    input_dir_p.mkdir(parents=True, exist_ok=True)

    CARTOON_ROOT = Path(__file__).resolve().parents[1]
    try:
        rel_from_cartoon = src.relative_to(CARTOON_ROOT)  # e.g., data/...
        if rel_from_cartoon.parts and rel_from_cartoon.parts[0] == "data":
            dst = (input_dir_p / rel_from_cartoon).resolve()
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                try:
                    shutil.copy2(str(src), str(dst))
                    print(f"[stage] copied -> {dst}")
                except Exception as e:
                    print(f"[stage][WARN] copy failed: {src} -> {dst} ({e}); fallback to basename")
                    dst2 = input_dir_p / src.name
                    try:
                        if str(src) != str(dst2):
                            shutil.copy2(str(src), str(dst2))
                            print(f"[stage] copied(basename) -> {dst2}")
                    except Exception as e2:
                        print(f"[stage][WARN] basename copy failed: {src} -> {dst2} ({e2}); return basename anyway")
                    return src.name
            else:
                print(f"[stage] exists -> {dst} (skip copy)")
            return str(rel_from_cartoon).replace("\\", "/")
    except Exception:
        pass

    dst = input_dir_p / src.name
    try:
        if str(src) != str(dst):
            shutil.copy2(str(src), str(dst))
            print(f"[stage] copied -> {dst}")
        else:
            print(f"[stage] source is already in input/: {dst}")
    except Exception as e:
        print(f"[stage][WARN] copy failed {src} -> {dst}: {e}; return basename anyway")
    return src.name

# Job polling
def get_history(prompt_id: str) -> Dict[str, Any]:
    url = _url(f"/history/{prompt_id}")
    with urllib.request.urlopen(url) as resp:
        return json.loads(resp.read().decode("utf-8"))

def _is_file_stable(path: str, checks: int = 4, interval: float = 0.25) -> bool:
    last_sz = -1
    for _ in range(checks):
        try:
            sz = os.path.getsize(path)
            if sz <= 0:
                time.sleep(interval)
                continue
        except OSError:
            return False

        if sz == last_sz:
            try:
                with Image.open(path) as im:
                    im.verify()
            except Exception:
                return False
            return True

        last_sz = sz
        time.sleep(interval)
    return False

def post_prompt(prompt_json: Dict[str, Any]) -> str:
    def _sanitize_for_json(obj):
        if isinstance(obj, set):
            return [_sanitize_for_json(v) for v in obj]
        if isinstance(obj, tuple):
            return [_sanitize_for_json(v) for v in obj]
        if isinstance(obj, dict):
            cleaned = {}
            for k, v in obj.items():
                if isinstance(k, str) and k.startswith("_"):
                    continue
                cleaned[k] = _sanitize_for_json(v)
            return cleaned
        if isinstance(obj, list):
            return [_sanitize_for_json(v) for v in obj]
        return obj

    prompt_clean = _sanitize_for_json(prompt_json)
    payload = {"prompt": prompt_clean, "client_id": "cartoon-cli"}
    data = json.dumps(payload).encode("utf-8")
    url = _url("/prompt")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as resp:
            res = json.loads(resp.read().decode("utf-8"))
        pid = res.get("prompt_id")
        if not pid:
            raise RuntimeError(f"Invalid response from /prompt: {res}")
        return pid
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="ignore")
        raise SystemExit(
            f"[ComfyUI /prompt 에러] {e}\n"
            f"- 요청 URL: {url}\n"
            f"- 응답 바디: {body}\n"
            f"- 'workflows/base_prompt.json'이 'Save (API)' 파일인지, 그리고 모델/가중치/입력 이미지가 서버 경로에 존재하는지 확인하세요."
        )
    except Exception as e:
        raise SystemExit(
            f"[ComfyUI /prompt 연결 실패] {e}\n"
            f"- 요청 URL: {url}\n"
            f"- COMFY_URL 또는 configs/routing.yaml(comfy_url)을 확인하세요."
        )

def wait_and_collect(prompt_id: str) -> List[str]:
    cfg = _load_routing()
    timeout = int(cfg.get("timeout", 600))
    poll = float(cfg.get("poll_interval", 0.5))
    save_dir_hint = cfg.get("save_dir_hint", "ComfyUI")

    here = Path(__file__).resolve()
    base_path = (here.parents[1] / save_dir_hint).resolve() if not os.path.isabs(save_dir_hint) else Path(save_dir_hint).resolve()
    base = str(base_path)

    output_dir = None
    try:
        candidate = os.path.join(base, "output")
        if os.path.isdir(candidate):
            output_dir = candidate
    except Exception:
        output_dir = None

    def snapshot_outputs(root: str) -> set:
        if not root or not os.path.isdir(root):
            return set()
        acc = set()
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                if fn.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                    acc.add(os.path.join(dirpath, fn))
        return acc

    fs_baseline = snapshot_outputs(output_dir) if output_dir else set()
    t0 = time.time()
    last_hist = None

    while True:
        time.sleep(poll)
        try:
            hist = get_history(prompt_id)
            last_hist = hist
            node_outs = hist.get(prompt_id, {}).get("outputs", {})
            candidate_paths: List[str] = []
            for _nid, out_dict in node_outs.items():
                if isinstance(out_dict, dict):
                    for key in ("images", "gifs", "videos"):
                        items = out_dict.get(key, [])
                        if isinstance(items, list):
                            for art in items:
                                if isinstance(art, dict) and "filename" in art:
                                    fn = art["filename"]
                                    sub = (art.get("subfolder") or "").strip()
                                    if os.path.isabs(fn):
                                        abs_path = fn
                                    else:
                                        abs_path = os.path.join(base, "output", sub, fn) if sub else os.path.join(base, "output", fn)
                                    candidate_paths.append(abs_path)
            stable = [p for p in candidate_paths if _is_file_stable(p)]
            if stable:
                return stable
        except Exception:
            pass

        if output_dir:
            now_snap = snapshot_outputs(output_dir)
            def safe_mtime(p: str) -> float:
                try:
                    return os.path.getmtime(p)
                except Exception:
                    return 0.0
            created = sorted(now_snap - fs_baseline, key=safe_mtime)
            stable = [p for p in created if _is_file_stable(p)]
            if stable:
                return stable

        if time.time() - t0 > timeout:
            raise TimeoutError(
                f"ComfyUI history timeout: {prompt_id}\n"
                f"- waited={int(time.time() - t0)}s, poll={poll}s, timeout={timeout}s\n"
                f"- output_dir={output_dir}\n"
                f"- last_history_keys={list(last_hist.keys()) if isinstance(last_hist, dict) else type(last_hist)}"
            )

# FastAPI Router
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Any, Dict, Optional

router = APIRouter(tags=["comfy-api"])

@router.get("/ping")
def ping():
    return {"ok": True}

@router.get("/routing")
def routing():
    try:
        return _load_routing()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"_load_routing failed: {e}")

@router.get("/stats")
def stats():
    try:
        return system_stats()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"system_stats failed: {e}")

class StageRequest(BaseModel):
    local_path: str

@router.post("/stage")
def stage(req: StageRequest):
    try:
        return {"basename": stage_to_input(req.local_path)}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"stage_to_input failed: {e}")

class PromptRequest(BaseModel):
    prompt_json: Optional[Dict[str, Any]] = None
    prompt: Optional[Dict[str, Any]] = None

@router.post("/prompt")
def prompt(req: PromptRequest):
    try:
        payload = req.prompt_json or req.prompt
        if payload is None:
            raise HTTPException(status_code=400, detail="prompt_json (or prompt) is required")
        pid = post_prompt(payload)
        return {"result": pid}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"post_prompt failed: {e}")

@router.get("/history/{prompt_id}")
def history(prompt_id: str):
    try:
        return get_history(prompt_id)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"get_history failed: {e}")

class WaitRequest(BaseModel):
    prompt_id: str

@router.post("/wait")
def wait(req: WaitRequest):
    try:
        return {"outputs": wait_and_collect(req.prompt_id)}
    except TimeoutError as e:
        raise HTTPException(status_code=504, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"wait_and_collect failed: {e}")