# backend/main.py
# ============================================================
# SDXL + ControlNet(dual: Canny + Scribble) 기반 로고 생성 백엔드
# + 디버그: OpenAI 헬스체크 & 프롬프트 생성 확인
# + 로고 저장/목록/삭제 (로그인 필수: user_id 또는 username)
# + 레거시 정적 경로(/static/data/...)는 "파일시스템 링크" 대신
#   "미들웨어 리라이트"로 안전하게 호환 (무한 data/ 중첩 방지)
# ============================================================

from __future__ import annotations

# ---------- 표준 라이브러리 ----------
import os, io, re, json, base64, threading, uuid, time, gc, logging
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path

# ---------- 서드파티 ----------
import numpy as np
from PIL import Image, ImageOps
import cv2
import torch

from fastapi import APIRouter, FastAPI, HTTPException, File, UploadFile, Form, Body, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from dotenv import load_dotenv
from openai import OpenAI

from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    StableDiffusionXLControlNetPipeline,
    EulerAncestralDiscreteScheduler,
    # MultiControlNetModel,
)


from huggingface_hub import snapshot_download

# ★ 레거시 /static/data/* → /static/* 리라이트용
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import RedirectResponse

# DB 객체: User, Logo, GeneratedPost, UsersData, FourcutsImage 는 database.py에 정의
from backend.database import (
    SessionLocal, Base, engine,
    User, Logo, GeneratedPost, UsersData, FourcutsImage
)

# --- ControlNet 구간 보정 (end > start를 항상 보장) ---
EPS = float(os.getenv("CN_EPS", "0.001"))

def _fix_cn_windows(scale_arg, start_arg, end_arg, eps: float = EPS):
    # 0~1로 클램프
    s = [float(max(0.0, min(1.0, x))) for x in start_arg]
    e = [float(max(0.0, min(1.0, x))) for x in end_arg]

    for i in range(len(s)):
        # scale이 0이면: 영향은 없지만 검증을 통과하는 '초소형 창'을 부여
        if scale_arg[i] <= 0.0:
            s[i] = 0.0
            e[i] = max(eps, min(1.0 - eps, eps))  # 보통 eps
            continue

        # 정상 활성화 케이스: end > start 보장
        if e[i] <= s[i]:
            e[i] = min(1.0 - eps, s[i] + eps)

        # end가 1.0이면 내부 인덱스 이슈 → 1.0 - eps
        if e[i] >= 1.0:
            e[i] = 1.0 - eps
            if e[i] <= s[i]:
                s[i] = max(0.0, e[i] - eps)

        # start가 1.0 이상인 비정상 → 1.0 - 2*eps 로 낮춤
        if s[i] >= 1.0:
            s[i] = 1.0 - (2 * eps)
            e[i] = 1.0 - eps
            if e[i] <= s[i]:
                s[i] = max(0.0, e[i] - eps)

    return s, e



# ============================================================
# 로깅
# ============================================================
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
log = logging.getLogger("backend")

# 공통 설정 (.env 강제 로드 + OpenAI 초기화)
DOTENV_PATH = (Path(__file__).resolve().parent.parent / ".env")
load_dotenv(dotenv_path=DOTENV_PATH, override=False)

for p in (Path("/app/.env"), Path("/app/deploy/.env"), Path(".env")):
    if p.exists():
        load_dotenv(p, override=False)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
if not OPENAI_API_KEY:
    raise RuntimeError(
        f"OPENAI_API_KEY가 비었습니다. 확인 경로: {DOTENV_PATH} "
        "(또는 환경변수/시크릿로 주입 필요)"
    )

def _mask(s: str, left: int = 6, right: int = 4) -> str:
    if not s: return ""
    if len(s) <= left + right:
        return "*" * len(s)
    return s[:left] + "*" * (len(s) - left - right) + s[-4:]

print(f"[env] OPENAI_API_KEY: {_mask(OPENAI_API_KEY)}", flush=True)

OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL") or None
OPENAI_ORG_ID   = os.getenv("OPENAI_ORG_ID") or None

openai_client = OpenAI(
    api_key=OPENAI_API_KEY,
    **({"base_url": OPENAI_BASE_URL} if OPENAI_BASE_URL else {}),
    **({"organization": OPENAI_ORG_ID} if OPENAI_ORG_ID else {}),
)

# 프롬프트 생성에 사용할 LLM
PROMPT_LLM_MODEL = os.getenv("PROMPT_LLM_MODEL", "gpt-5")
USE_LLM_PROMPT   = os.getenv("USE_LLM_PROMPT", "true").lower() in ("1", "true", "yes")

app = FastAPI(title="Logo Backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # 데모용: 운영에서는 화이트리스트 권장
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# 레거시 /static/data/* → /static/* 리라이트 미들웨어
# ============================================================
class LegacyStaticRewrite(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        path = request.url.path
        if path.startswith("/static/data/"):
            new_path = "/static/" + path[len("/static/data/"):]
            return RedirectResponse(new_path, status_code=307)
        return await call_next(request)

app.add_middleware(LegacyStaticRewrite)

# === storage 경로를 프로젝트 루트 기준으로 고정 ===
THIS_DIR = Path(__file__).resolve().parent        # backend/
PROJECT_ROOT = THIS_DIR.parent                    # 루트

DEFAULT_STORAGE = PROJECT_ROOT / "storage"
STORAGE_DIR = Path(os.getenv("STORAGE_DIR", str(DEFAULT_STORAGE))).resolve()

LOGO_ROOT = STORAGE_DIR / "logos"
LOGO_ROOT.mkdir(parents=True, exist_ok=True)

# /static/* -> STORAGE_DIR/*
app.mount("/static", StaticFiles(directory=str(STORAGE_DIR)), name="static")

log.info("[storage] STORAGE_DIR=%s exists=%s", STORAGE_DIR, STORAGE_DIR.exists())
log.info("[storage] LOGO_ROOT=%s  exists=%s", LOGO_ROOT, LOGO_ROOT.exists())

# ============================================================
# 로고 생성 파트 (SDXL + ControlNet Dual)
# ============================================================
CHECKPOINT_DIR = os.getenv("CHECKPOINT_DIR", "/home/uv-env/checkpoints").rstrip("/")
os.environ.setdefault("HF_HOME", os.path.join(CHECKPOINT_DIR, ".cache"))
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

SDXL_MODEL_ID      = os.getenv("SDXL_MODEL_ID", "stabilityai/stable-diffusion-xl-base-1.0")
VAE_MODEL_ID       = os.getenv("VAE_MODEL_ID",  "madebyollin/sdxl-vae-fp16-fix")
CANNY_CN_ID        = os.getenv("CANNY_CN_ID",   "xinsir/controlnet-canny-sdxl-1.0")
SCRIBBLE_CN_ID     = os.getenv("SCRIBBLE_CN_ID","xinsir/controlnet-scribble-sdxl-1.0")

SDXL_LOCAL_DIR     = os.getenv("SDXL_LOCAL_DIR",  "sdxl/base-1.0")
VAE_LOCAL_DIR      = os.getenv("VAE_LOCAL_DIR",   "vae/sdxl-vae-fp16-fix")
CANNY_LOCAL_DIR    = os.getenv("CANNY_LOCAL_DIR", "controlnet/canny-sdxl-1.0")
SCRIBBLE_LOCAL_DIR = os.getenv("SCRIBBLE_LOCAL_DIR","controlnet/scribble-sdxl-1.0")

OUT_SIZE         = 1024
DEFAULT_STEPS    = int(os.getenv("GEN_STEPS", "32"))
DEFAULT_GUIDE    = float(os.getenv("GEN_GUIDE", "6.5"))

DEFAULT_CN_SCALE = float(os.getenv("GEN_CN_SCALE", "0.8"))
DEFAULT_CN_START = float(os.getenv("GEN_CN_START", "0.0"))
DEFAULT_CN_END   = float(os.getenv("GEN_CN_END",   "0.9"))

DEFAULT_CN_SCALE_CANNY    = float(os.getenv("GEN_CN_SCALE_CANNY",    "0.9"))
DEFAULT_CN_START_CANNY    = float(os.getenv("GEN_CN_START_CANNY",    "0.0"))
DEFAULT_CN_END_CANNY      = float(os.getenv("GEN_CN_END_CANNY",      "0.9"))

DEFAULT_CN_SCALE_SCRIBBLE = float(os.getenv("GEN_CN_SCALE_SCRIBBLE", "0.6"))
DEFAULT_CN_START_SCRIBBLE = float(os.getenv("GEN_CN_START_SCRIBBLE", "0.0"))
DEFAULT_CN_END_SCRIBBLE   = float(os.getenv("GEN_CN_END_SCRIBBLE",   "0.9"))

CANNY_LOW        = int(os.getenv("CANNY_LOW", "50"))
CANNY_HIGH       = int(os.getenv("CANNY_HIGH", "150"))
CANNY_DILATE     = os.getenv("CANNY_DILATE", "true").lower() in ("1","true","yes")
CANNY_KSIZE      = int(os.getenv("CANNY_KSIZE", "2"))

SCRIBBLE_THRESH  = int(os.getenv("SCRIBBLE_THRESH", "150"))
SCRIBBLE_INVERT  = os.getenv("SCRIBBLE_INVERT", "false").lower() in ("1","true","yes")

CN_ACTIVE_THRESH = float(os.getenv("CN_ACTIVE_THRESH", "0.2"))
USE_CPU_OFFLOAD  = os.getenv("USE_CPU_OFFLOAD", "false").lower() in ("1","true","yes")

def _ensure_model(repo_id: str, subdir: str) -> str:
    target_dir = os.path.join(CHECKPOINT_DIR, subdir)
    model_index = os.path.join(target_dir, "model_index.json")
    if os.path.isdir(target_dir) and os.path.exists(model_index):
        return target_dir
    os.makedirs(target_dir, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        local_dir=target_dir,
        local_dir_use_symlinks=False,
        resume_download=True,
        ignore_patterns=["*.msgpack"],
    )
    return target_dir

def _clean_b64(s: Optional[str]) -> Optional[str]:
    if not s: return None
    s = s.strip()
    if s.startswith("data:image"):
        s = s.split(",", 1)[1]
    return "".join(s.split())

def _b64_to_pil_rgba(b64png: Optional[str]) -> Optional[Image.Image]:
    s = _clean_b64(b64png)
    if not s: return None
    try:
        raw = base64.b64decode(s, validate=False)
        im = Image.open(io.BytesIO(raw))
        return ImageOps.exif_transpose(im.convert("RGBA"))
    except Exception:
        return None

def _pil_to_b64_png(img: Image.Image) -> str:
    buf = io.BytesIO(); img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")

def _to_data_url(b64: str) -> str:
    return f"data:image/png;base64,{b64}"

def _resize_square(pil: Image.Image, size: int = 1024) -> Image.Image:
    w, h = pil.size
    if w <= h:
        nw, nh = size, max(size, int(h * (size / w)))
    else:
        nh, nw = size, max(size, int(w * (size / h)))
    pil = pil.resize((nw, nh), Image.LANCZOS)
    x0 = max(0, (pil.width - size)//2)
    y0 = max(0, (pil.height - size)//2)
    return pil.crop((x0, y0, x0+size, y0+size))

def _make_canny_image(source: Image.Image, low: int, high: int, dilate: bool, ksize: int) -> Image.Image:
    gray = np.array(source.convert("L"))
    gray = cv2.GaussianBlur(gray, (3,3), 0)
    edges = cv2.Canny(gray, low, high)
    if dilate:
        kernel = np.ones((ksize, ksize), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
    edges_rgb = np.stack([edges, edges, edges], axis=-1)
    return Image.fromarray(edges_rgb)

def _make_scribble_image(source: Image.Image, thresh: int, invert: bool) -> Image.Image:
    img = np.array(source.convert("L"))
    _, bw = cv2.threshold(img, int(thresh), 255, cv2.THRESH_BINARY)
    if invert:
        bw = 255 - bw
    scribble = np.stack([bw, bw, bw], axis=-1)
    return Image.fromarray(scribble)

def _alpha_to_solid(img_rgba: Image.Image, on=(255,255,255), off=(0,0,0)) -> Image.Image:
    a = np.array(img_rgba.split()[-1])
    rgb = np.zeros((a.shape[0], a.shape[1], 3), dtype=np.uint8)
    rgb[a > 0] = on
    rgb[a == 0] = off
    return Image.fromarray(rgb, mode="RGB")

def _rgba_on_white(img_rgba: Image.Image) -> Image.Image:
    if img_rgba.mode != "RGBA":
        return img_rgba.convert("RGB")
    bg = Image.new("RGBA", img_rgba.size, (255, 255, 255, 255))
    comp = Image.alpha_composite(bg, img_rgba)
    return comp.convert("RGB")

def _clip_safe_prompt(p: str) -> str:
    p = (p or "").strip()
    return p[:300] if len(p) > 300 else p

def _flush_mem():
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    gc.collect()

# ============================================================
# ==== LLM 호출 호환 레이어 + 디버그 수집 ====================
# ============================================================
def _extract_usage_dict(obj: Any) -> Dict[str, Any]:
    if not obj:
        return {}
    u = getattr(obj, "usage", None)
    if not u:
        return {}
    fields = {
        "input_tokens": getattr(u, "input_tokens", None),
        "output_tokens": getattr(u, "output_tokens", None),
        "prompt_tokens": getattr(u, "prompt_tokens", None),
        "completion_tokens": getattr(u, "completion_tokens", None),
        "total_tokens": getattr(u, "total_tokens", None),
    }
    return {k: v for k, v in fields.items() if v is not None}

def _call_llm_json(system: str, user: str, model: str):
    import time as _t, json, re
    t0 = _t.time()
    is_gpt5 = "gpt-5" in (model or "").lower()
    debug = {"model": model, "api": "chat.completions.create"}

    def _extract_json(txt: str) -> dict:
        if not txt:
            return {}
        m = re.search(r"\{.*\}", txt, re.DOTALL)
        return json.loads(m.group(0)) if m else {}

    try:
        kwargs = dict(
            model=model,
            messages=[{"role":"system","content":system},{"role":"user","content":user}],
        )
        if not is_gpt5:
            kwargs["temperature"] = 0.2
            kwargs["response_format"] = {"type":"json_object"}

        cc = openai_client.chat.completions.create(**kwargs)
        latency_ms = int((_t.time()-t0)*1000)
        txt = cc.choices[0].message.content if cc.choices else ""
        data = json.loads(txt) if (not is_gpt5 and kwargs.get("response_format")) else _extract_json(txt)
        debug.update({
            "latency_ms": latency_ms,
            "usage": _extract_usage_dict(cc),
            "raw_sample": (txt[:180] + "…") if (txt and len(txt) > 200) else txt
        })
        return data, debug

    except Exception as e1:
        try:
            cc = openai_client.chat.completions.create(
                model=model,
                messages=[
                    {"role":"system","content": system + "\nReturn STRICT JSON with keys 'positive' and 'negative' only."},
                    {"role":"user","content": user}
                ],
            )
            latency_ms = int((_t.time()-t0)*1000)
            txt = cc.choices[0].message.content if cc.choices else ""
            data = _extract_json(txt)
            debug.update({
                "latency_ms": latency_ms,
                "usage": _extract_usage_dict(cc),
                "raw_sample": (txt[:180] + "…") if (txt and len(txt) > 200) else txt,
                "note": f"retry due to: {e1.__class__.__name__}"
            })
            return data, debug
        except Exception as e2:
            debug.update({"latency_ms": None, "usage": {}, "error": f"{e2.__class__.__name__}: {e2}"})
            return {}, debug

PIPE: Optional[StableDiffusionXLControlNetPipeline] = None
PIPE_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CN_CANNY: Optional[ControlNetModel] = None
CN_SCRIBBLE: Optional[ControlNetModel] = None
# CN_DUAL: Optional[MultiControlNetModel] = None

def _load_pipe() -> Optional[StableDiffusionXLControlNetPipeline]:
    global CN_CANNY, CN_SCRIBBLE, CN_DUAL
    try:
        sdxl_path  = _ensure_model(SDXL_MODEL_ID,   SDXL_LOCAL_DIR)
        vae_path   = _ensure_model(VAE_MODEL_ID,    VAE_LOCAL_DIR)
        canny_path = _ensure_model(CANNY_CN_ID,     CANNY_LOCAL_DIR)
        scrib_path = _ensure_model(SCRIBBLE_CN_ID,  SCRIBBLE_LOCAL_DIR)

        torch_dtype = torch.float16 if PIPE_DEVICE == "cuda" else torch.float32
        CN_CANNY    = ControlNetModel.from_pretrained(canny_path,   torch_dtype=torch_dtype)
        CN_SCRIBBLE = ControlNetModel.from_pretrained(scrib_path,   torch_dtype=torch_dtype)
        # CN_DUAL     = MultiControlNetModel([CN_CANNY, CN_SCRIBBLE])

        vae = AutoencoderKL.from_pretrained(vae_path, torch_dtype=torch_dtype)
        # pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        #     sdxl_path, controlnet=CN_DUAL, vae=vae, torch_dtype=torch_dtype
        # )
        pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            sdxl_path, controlnet=[CN_CANNY, CN_SCRIBBLE], vae=vae, torch_dtype=torch_dtype
        )
        
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
        pipe.enable_attention_slicing("max")
        pipe.enable_vae_slicing()
        pipe.enable_vae_tiling()

        if PIPE_DEVICE == "cuda":
            try: pipe.enable_xformers_memory_efficient_attention()
            except Exception: pass
            if USE_CPU_OFFLOAD:
                pipe.enable_model_cpu_offload()
            else:
                pipe.to("cuda")
        else:
            pipe.to("cpu")

        pipe.set_progress_bar_config(disable=True)
        return pipe
    except Exception as e:
        print("[PIPE LOAD ERROR]", e, flush=True)
        return None

# ---------------------- 데이터 모델 ----------------------
class BriefPayload(BaseModel):
    request_id: str
    cafe_name: str
    copy_text: str
    layout: str
    avoid: str
    strengths: str
    style: str
    notes: str
    model_hint: str
    palette: List[str] = []
    ref_image_present: bool = False

class GenerateRequest(BaseModel):
    brief_id: int
    sketch_png_b64: Optional[str] = None
    text_mask_png_b64: Optional[str] = None
    num_images: int = Field(default=4, ge=1, le=8)
    seed: Optional[int] = None

    prompt: Optional[str] = None
    positive_prompt: Optional[str] = None
    negative_prompt: Optional[str] = None
    use_llm_prompt: Optional[bool] = None
    return_debug: Optional[bool] = False

    preprocess_mode: str = Field(default="canny") # "canny" | "scribble" | "dual"

    canny_low: Optional[int] = None
    canny_high: Optional[int] = None
    canny_dilate: Optional[bool] = None
    canny_ksize: Optional[int] = None

    scribble_thresh: Optional[int] = None
    scribble_invert: Optional[bool] = None

    cn_scale: Optional[float] = None
    cn_start: Optional[float] = None
    cn_end:   Optional[float] = None

    canny_cn_scale: Optional[float] = None
    canny_cn_start: Optional[float] = None
    canny_cn_end:   Optional[float] = None
    scribble_cn_scale: Optional[float] = None
    scribble_cn_start: Optional[float] = None
    scribble_cn_end:   Optional[float] = None

    # 프롬프트 충성도(가이던스) 직접 제어
    guidance_scale: Optional[float] = None

    text_info: Optional[Dict[str, Any]] = None
    llm_inputs: Optional[Dict[str, Any]] = None

class SelectionPayload(BaseModel):
    brief_id: int
    selected_index: int
    total: int

BRIEFS: Dict[int, BriefPayload] = {}
JOBS: Dict[str, Dict[str, Any]] = {}
SELECTIONS: Dict[int, Dict[str, int]] = {}
_lock = threading.Lock()
_brief_seq = 0

class LogoOut(BaseModel):
    id: int
    user_id: int
    image_path: str
    image_url: str
    created_at: str
    class Config:
        from_attributes = True

def _job_view_for_front(j: Dict[str, Any]) -> Dict[str, Any]:
    """프론트 전환 조건을 넉넉히 만족시키도록 응답을 표준화/별칭 추가"""
    status = (j or {}).get("status") or "running"
    imgs_b64 = (j or {}).get("images_b64") or []
    imgs_du  = (j or {}).get("images_data_url") or [f"data:image/png;base64,{s}" for s in imgs_b64]

    # 상태 alias: done|error 외에 completed|failed|processing 등도 함께 제공
    if status == "done":
        state = "completed"; ok = True; progress = 1.0
    elif status in ("error", "failed"):
        state = "failed"; ok = False; progress = 0.0
    else:
        state = "processing"; ok = False; progress = 0.5  # 대략값

    return {
        # 기존 필드 유지
        "status": status,
        "images_b64": imgs_b64,
        "images_data_url": imgs_du,
        "error": j.get("error"),
        "debug": j.get("debug"),
        "used_prompt": j.get("used_prompt"),

        # 하위/상위 호환 alias (프론트 어떤 코드여도 잡히게)
        "ok": ok,
        "state": state,                 # completed | failed | processing
        "progress": progress,           # 0.0~1.0
        "images": imgs_b64,             # alias
        "images_urls": imgs_du,         # alias
        "count": len(imgs_b64),         # 프리뷰 장수 체크용
    }


# ---------------------- 스타트업 ----------------------
def _openai_health_check() -> Dict[str, Any]:
    try:
        t0 = time.time()
        resp = openai_client.chat.completions.create(
            model=PROMPT_LLM_MODEL if "gpt-5" not in PROMPT_LLM_MODEL.lower() else "gpt-4o-mini",
            messages=[{"role":"user","content":"ping?"}],
            max_tokens=1,
            temperature=0.0,
        )
        latency_ms = int((time.time()-t0)*1000)
        return {
            "ok": True,
            "latency_ms": latency_ms,
            "model": PROMPT_LLM_MODEL,
            "usage": _extract_usage_dict(resp),
        }
    except Exception as e:
        return {"ok": False, "model": PROMPT_LLM_MODEL, "error": f"{e.__class__.__name__}: {e}"}

@app.on_event("startup")
def _startup():
    # DB 테이블 생성 보장
    Base.metadata.create_all(bind=engine)

    global PIPE
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(os.getenv("HF_HOME", os.path.join(CHECKPOINT_DIR, ".cache")), exist_ok=True)

    hc = _openai_health_check()
    if not hc.get("ok"):
        log.error("[startup] OpenAI health_check FAILED: %s", json.dumps(hc, ensure_ascii=False))
    else:
        log.info("[startup] OpenAI health_check OK: %s", json.dumps(hc, ensure_ascii=False))

    PIPE = _load_pipe()
    print(f"[startup] device={PIPE_DEVICE}, pipe_loaded={bool(PIPE)}")

# ---------------------- 헬스 ----------------------
@app.get("/logo/health")
def health():
    return {
        "ok": True,
        "local_sdxl": bool(PIPE is not None),
        "device": PIPE_DEVICE,
        "openai": {
            "has_api_key": bool(OPENAI_API_KEY),
            "prompt_llm_model": PROMPT_LLM_MODEL,
            "base_url": OPENAI_BASE_URL or "api.openai.com",
            "org": OPENAI_ORG_ID or None,
            "default_enabled": USE_LLM_PROMPT,
        },
        "paths": {
            "root": CHECKPOINT_DIR,
            "sdxl": os.path.join(CHECKPOINT_DIR, SDXL_LOCAL_DIR),
            "vae": os.path.join(CHECKPOINT_DIR, VAE_LOCAL_DIR),
            "controlnet_canny": os.path.join(CHECKPOINT_DIR, CANNY_LOCAL_DIR),
            "controlnet_scribble": os.path.join(CHECKPOINT_DIR, SCRIBBLE_LOCAL_DIR),
        },
        "controlnets": {
            "canny_loaded": CN_CANNY is not None,
            "scribble_loaded": CN_SCRIBBLE is not None,
            "dual_ready": CN_DUAL is not None,
        },
        "defaults": {
            "out_size": OUT_SIZE, "steps": DEFAULT_STEPS, "guide": DEFAULT_GUIDE,
        }
    }

# ---------------------- 브리프 ----------------------
@app.post("/logo/briefs")
def save_brief(payload: BriefPayload):
    global _brief_seq
    with _lock:
        _brief_seq += 1
        BRIEFS[_brief_seq] = payload
        return {"id": _brief_seq}

def _compose_prompt_from_brief(b: BriefPayload) -> str:
    parts = [
        f"Logo for cafe '{b.cafe_name}'. Text: {b.copy_text}.",
        f"Style: {b.style}. Layout: {b.layout}.",
        f"Strengths: {b.strengths}. Avoid: {b.avoid}.",
        "vector, flat, minimal, clean lines, no gradients, no 3D, no photo.",
    ]
    if b.palette:
        parts.append("Palette: " + ", ".join(b.palette))
    return " ".join(parts)

# ============================================================
# ---------------- LLM 프롬프트 생성 (디버그 확장) -------------
# ============================================================
def _gen_logo_prompts_with_llm(
    brief: BriefPayload,
    has_canny: bool,
    has_scribble: bool,
    *,
    text_info: Optional[Dict[str, Any]] = None,
    prompt_overrides: Optional[Dict[str, Any]] = None,
    gpt_prompt_seed: Optional[str] = None,
    gpt_messages: Optional[List[Dict[str, str]]] = None,
) -> Tuple[str, str, str, Dict[str, Any]]:
    default_neg = "3d, photo, realistic, gradient, shadow, glossy, clutter, watermark, text artifacts"

    cond = []
    if has_canny: cond.append("text layout via Canny edges")
    if has_scribble: cond.append("shape guidance via Scribble")
    cond_txt = "; ".join(cond) if cond else "no external guidance"

    text_info_snippet = ""
    if text_info:
        try: text_info_snippet = json.dumps(text_info, ensure_ascii=False)[:1200]
        except Exception: text_info_snippet = str(text_info)[:1200]

    overrides_snippet = ""
    if prompt_overrides:
        try: overrides_snippet = json.dumps(prompt_overrides, ensure_ascii=False)[:1200]
        except Exception: overrides_snippet = str(prompt_overrides)[:1200]

    sys = ("You are a senior prompt engineer specialized in SDXL cafe logo generation."
           "Always respond in English only. Use US English."
    )   
    seed_hint = (gpt_prompt_seed or "").strip()
    seed_block = f"\n\nPrompt seed:\n{seed_hint}\n" if seed_hint else ""

    usr = f"""
Create two fields as strict JSON for SDXL logo diffusion:
{{
  "positive": "<one line English prompt>",
  "negative": "<one line English negative prompt>"
}}
Brand: {brief.cafe_name}
Logo Text: {brief.copy_text}
Style: {brief.style}
Layout: {brief.layout}
Strengths: {brief.strengths}
Avoid: {brief.avoid}
Notes: {brief.notes}
Palette: {", ".join(brief.palette) if brief.palette else "N/A"}
Guidance: {cond_txt}
Text guide (from step 2): {text_info_snippet or "N/A"}
Overrides (from step 3): {overrides_snippet or "N/A"}{seed_block}

Constraints:
- Vector/flat/minimal logo keywords only; no photography.
- Keep exactly one concise line per field. No markdown/backticks/explanations.
- **Both fields must be English-only; do not include any non-English or non-ASCII characters.**

"""

    data, debug = _call_llm_json(sys, usr, PROMPT_LLM_MODEL)
    pos = (data.get("positive") or "").strip()
    neg = (data.get("negative") or "").strip() or default_neg

    if not pos:
        pos_fallback = _compose_prompt_from_brief(brief)
        debug = {**debug, "fallback_used": True, "fallback_positive": pos_fallback}
        return _clip_safe_prompt(pos_fallback), _clip_safe_prompt(neg), f"{PROMPT_LLM_MODEL} (fallback)", debug

    return _clip_safe_prompt(pos), _clip_safe_prompt(neg), f"{PROMPT_LLM_MODEL} (generated)", debug

# ---------------------- 이미지 생성 ----------------------
def _generate_images_local(
    *,
    positive_prompt: str,
    negative_prompt: str,
    num_images: int,
    seed: Optional[int],
    sketch_b64: Optional[str],
    text_mask_b64: Optional[str],
    preprocess_mode: str,
    canny_low: int, canny_high: int, canny_dilate: bool, canny_ksize: int,
    scribble_thresh: int, scribble_invert: bool,
    cn_scale: float, cn_start: float, cn_end: float,
    canny_scale: float = DEFAULT_CN_SCALE_CANNY, canny_start: float = DEFAULT_CN_START_CANNY, canny_end: float = DEFAULT_CN_END_CANNY,
    scribble_scale: float = DEFAULT_CN_SCALE_SCRIBBLE, scribble_start: float = DEFAULT_CN_START_SCRIBBLE, scribble_end: float = DEFAULT_CN_END_SCRIBBLE,
    guidance_scale: float = DEFAULT_GUIDE,
) -> Tuple[List[str], Dict[str, Any]]:

    if PIPE is None:
        ph = Image.new("RGBA", (OUT_SIZE, OUT_SIZE), (220,220,220,255))
        return [_pil_to_b64_png(ph)], {"mode": preprocess_mode}

    m_img = _b64_to_pil_rgba(text_mask_b64) if text_mask_b64 else None
    s_img = _b64_to_pil_rgba(sketch_b64)    if sketch_b64    else None

    src_mask   = _resize_square(_alpha_to_solid(m_img), OUT_SIZE) if m_img else None
    src_sketch = _resize_square(_rgba_on_white(s_img), OUT_SIZE)  if s_img else None
    fallback   = Image.new("RGB", (OUT_SIZE, OUT_SIZE), (255, 255, 255))
    blank      = Image.new("RGB", (OUT_SIZE, OUT_SIZE), (255, 255, 255))

    mode = (preprocess_mode or "canny").lower().strip()

    # 존재 플래그
    has_mask = src_mask is not None
    has_sketch = src_sketch is not None

    if mode == "dual":
        # dual: 각 입력이 있으면 그걸 사용, 없으면 dummy(blank) + 해당 scale을 0으로
        canny_src = src_mask if has_mask else fallback
        scribble_src = src_sketch if has_sketch else fallback

        canny_img = _make_canny_image(canny_src, low=canny_low, high=canny_high, dilate=canny_dilate, ksize=canny_ksize) if has_mask else blank
        scribble_img = _make_scribble_image(scribble_src, thresh=scribble_thresh, invert=scribble_invert) if has_sketch else blank

        scale_arg = [canny_scale if has_mask else 0.0, scribble_scale if has_sketch else 0.0]
        start_arg = [canny_start if has_mask else 0.0, scribble_start if has_sketch else 0.0]
        end_arg   = [canny_end if has_mask else 0.0, scribble_end if has_sketch else 0.0]

    elif mode == "scribble":
        # scribble 모드: scribble은 오직 스케치가 있을 때만 만들고 사용.
        canny_img = blank  # dummy
        if has_sketch:
            src = src_sketch
            scribble_img = _make_scribble_image(src, thresh=scribble_thresh, invert=scribble_invert)
            scale_arg = [0.0, cn_scale]   # cn_scale은 호출자가 scribble용으로 세팅해 둠
            start_arg = [0.0, cn_start]
            end_arg   = [0.0, cn_end]
        else:
            # 스케치가 없으면 둘 다 비활성
            scribble_img = blank
            scale_arg = [0.0, 0.0]
            start_arg = [0.0, 0.0]
            end_arg   = [0.0, 0.0]

    else:  # canny
        # canny 모드: canny은 오직 text mask가 있을 때만 만들고 사용.
        scribble_img = blank  # dummy
        if has_mask:
            src = src_mask
            canny_img = _make_canny_image(src, low=canny_low, high=canny_high, dilate=canny_dilate, ksize=canny_ksize)
            scale_arg = [cn_scale, 0.0]   # cn_scale은 호출자가 canny용으로 세팅해 둠
            start_arg = [cn_start, 0.0]
            end_arg   = [cn_end,   0.0]
        else:
            # text mask가 없으면 둘 다 비활성
            canny_img = blank
            scale_arg = [0.0, 0.0]
            start_arg = [0.0, 0.0]
            end_arg   = [0.0, 0.0]
            
    start_arg, end_arg = _fix_cn_windows(scale_arg, start_arg, end_arg, eps=EPS)

    # ---------- 여기가 핵심 추가/수정 부분 ----------
    # 1) 파이프라인에 넘길 조건 이미지 리스트 생성 (항상 RGB)
    cond_list = [
        canny_img.convert("RGB") if isinstance(canny_img, Image.Image) else Image.new("RGB", (OUT_SIZE, OUT_SIZE), (255,255,255)),
        scribble_img.convert("RGB") if isinstance(scribble_img, Image.Image) else Image.new("RGB", (OUT_SIZE, OUT_SIZE), (255,255,255)),
    ]

    # 2) ControlNet 활성 판단 (평균 밝기 0~1 정규화하여 임계치와 비교)
    m1 = np.array(cond_list[0].convert("L"), dtype=np.float32).mean() / 255.0
    m2 = np.array(cond_list[1].convert("L"), dtype=np.float32).mean() / 255.0
    cn_active = (
        (scale_arg[0] > 0.0 and m1 >= CN_ACTIVE_THRESH) or
        (scale_arg[1] > 0.0 and m2 >= CN_ACTIVE_THRESH)
    )

    # 3) (선택) 디버그 저장
    try:
        os.makedirs("/home/uv-env/outputs", exist_ok=True)
        cond_list[0].save("/home/uv-env/outputs/_debug_dual_canny.png")
        cond_list[1].save("/home/uv-env/outputs/_debug_dual_scribble.png")
    except Exception:
        pass
    # ---------- 핵심 변경 끝 ----------

    pos = _clip_safe_prompt(positive_prompt)
    neg = _clip_safe_prompt(negative_prompt or "3d, photo, realistic, gradient, shadow, glossy, clutter, watermark, text artifacts")

    outs: List[str] = []
    total = max(1, int(num_images))
    base_seed = seed if seed is not None else int(time.time() * 1000) % 2_147_483_647

    for i in range(total):
        the_seed = (base_seed + i) & 0x7fffffff
        generator = torch.Generator(device=PIPE_DEVICE).manual_seed(the_seed)
        _flush_mem()
        img = None
        try:
            with torch.inference_mode():
                kwargs = dict(
                    prompt=pos,
                    negative_prompt=neg,
                    image=cond_list,
                    controlnet_conditioning_scale=(scale_arg if cn_active else [0.0, 0.0]),
                    control_guidance_start=start_arg,
                    control_guidance_end=end_arg,
                    num_inference_steps=DEFAULT_STEPS,
                    guidance_scale=float(guidance_scale),
                    generator=generator,
                    width=OUT_SIZE, height=OUT_SIZE,
                )
                if PIPE_DEVICE == "cuda":
                    with torch.autocast("cuda"):
                        result = PIPE(**kwargs)
                else:
                    result = PIPE(**kwargs)
            if not hasattr(result, "images") or not result.images:
                raise RuntimeError("pipeline returned no images")
            img = result.images[0]
        finally:
            try: del result
            except Exception: pass
            _flush_mem()
        if img is None:
            raise RuntimeError("generation produced empty image")
        outs.append(_pil_to_b64_png(img))

    debug = {
        "mode": mode,
        "controls": {
            "canny": {
                "scale": scale_arg[0], "start": start_arg[0], "end": end_arg[0],
                "low": canny_low, "high": canny_high, "dilate": canny_dilate, "ksize": canny_ksize
            },
            "scribble": {
                "scale": scale_arg[1], "start": start_arg[1], "end": end_arg[1],
                "threshold": scribble_thresh, "invert": scribble_invert
            }
        },
        "seed": base_seed,
        "steps": DEFAULT_STEPS,
        "guidance": float(guidance_scale),
        "canny_preview_b64": _pil_to_b64_png(canny_img) if isinstance(canny_img, Image.Image) else None,
        "scribble_preview_b64": _pil_to_b64_png(scribble_img) if isinstance(scribble_img, Image.Image) else None,
    }
    return outs, debug


# ---------------------- 작업 스레드 ----------------------
def _clamp(v: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, v)))

def _clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))

def _clamp01_open_interval(x: float, eps: float = EPS) -> float:
    # [0, 1) 로 강제 (control_guidance_end 보호)
    x = _clamp01(x)
    return min(x, 1.0 - eps)

def _run_job(job_id: str, req: GenerateRequest):
    try:
        brief = BRIEFS.get(req.brief_id)
        pos_from_client = (req.positive_prompt or "").strip()
        neg_from_client = (req.negative_prompt or "").strip()
        prompt_source = "rule-based"
        llm_debug: Dict[str, Any] = {}

        text_info        = req.text_info or None
        llm_inputs       = req.llm_inputs or {}
        prompt_overrides = llm_inputs.get("prompt_overrides")
        gpt_prompt_seed  = llm_inputs.get("gpt_prompt_seed")
        gpt_messages     = llm_inputs.get("gpt_messages")

        if pos_from_client or neg_from_client:
            pos = pos_from_client or (req.prompt or "").strip() or (_compose_prompt_from_brief(brief) if brief else "minimal vector logo")
            neg = neg_from_client or "3d, photo, realistic, gradient, shadow, glossy, clutter, watermark, text artifacts"
            prompt_source = "client"
        else:
            will_llm = req.use_llm_prompt if req.use_llm_prompt is not None else USE_LLM_PROMPT
            if brief and will_llm:
                pos, neg, prompt_source, llm_debug = _gen_logo_prompts_with_llm(
                    brief,
                    has_canny=bool(req.text_mask_png_b64),
                    has_scribble=bool(req.sketch_png_b64),
                    text_info=text_info,
                    prompt_overrides=prompt_overrides,
                    gpt_prompt_seed=gpt_prompt_seed,
                    gpt_messages=gpt_messages,
                )
            else:
                pos = (req.prompt or "").strip() or (_compose_prompt_from_brief(brief) if brief else "minimal vector logo")
                neg = "3d, photo, realistic, gradient, shadow, glossy, clutter, watermark, text artifacts"
                prompt_source = "rule-based"

        pos = _clip_safe_prompt(pos); neg = _clip_safe_prompt(neg)

        mode = (req.preprocess_mode or "canny").lower().strip()
        if (req.text_mask_png_b64 and req.sketch_png_b64):
            mode = "dual"

        # 스케일/구간
        canny_scale = req.canny_cn_scale if req.canny_cn_scale is not None else DEFAULT_CN_SCALE_CANNY
        canny_scale = _clamp(float(canny_scale), 0.0, 2.0)

        scr_scale = req.scribble_cn_scale if req.scribble_cn_scale is not None else DEFAULT_CN_SCALE_SCRIBBLE
        scribble_scale = _clamp(float(scr_scale), 0.0, 2.0)

        def _clamp01(x: float) -> float: return float(max(0.0, min(1.0, x)))
        canny_start = _clamp01(req.canny_cn_start if req.canny_cn_start is not None else DEFAULT_CN_START_CANNY)
        canny_end   = _clamp01_open_interval(req.canny_cn_end   if req.canny_cn_end   is not None else DEFAULT_CN_END_CANNY)
        scribble_start = _clamp01(req.scribble_cn_start if req.scribble_cn_start is not None else DEFAULT_CN_START_SCRIBBLE)
        scribble_end   = _clamp01_open_interval(req.scribble_cn_end   if req.scribble_cn_end   is not None else DEFAULT_CN_END_SCRIBBLE)

        guidance_used = float(req.guidance_scale) if (req.guidance_scale is not None) else DEFAULT_GUIDE

        if mode == "canny":
            cn_scale = canny_scale; cn_start = canny_start; cn_end = canny_end
        elif mode == "scribble":
            cn_scale = scribble_scale; cn_start = scribble_start; cn_end = scribble_end
        else:
            cn_scale = (req.cn_scale if req.cn_scale is not None else DEFAULT_CN_SCALE)
            cn_start = _clamp01(req.cn_start if req.cn_start is not None else DEFAULT_CN_START)
            cn_end   = _clamp01_open_interval(req.cn_end   if req.cn_end   is not None else DEFAULT_CN_END)

        images_b64, debug = _generate_images_local(
            positive_prompt=pos,
            negative_prompt=neg,
            num_images=req.num_images,
            seed=req.seed,
            sketch_b64=req.sketch_png_b64,
            text_mask_b64=req.text_mask_png_b64,
            preprocess_mode=mode,
            canny_low =  req.canny_low    if req.canny_low    is not None else CANNY_LOW,
            canny_high = req.canny_high   if req.canny_high   is not None else CANNY_HIGH,
            canny_dilate = req.canny_dilate if req.canny_dilate is not None else CANNY_DILATE,
            canny_ksize = req.canny_ksize if req.canny_ksize  is not None else CANNY_KSIZE,
            scribble_thresh = req.scribble_thresh if req.scribble_thresh is not None else SCRIBBLE_THRESH,
            scribble_invert = req.scribble_invert if req.scribble_invert is not None else SCRIBBLE_INVERT,
            cn_scale=cn_scale, cn_start=cn_start, cn_end=cn_end,
            canny_scale=canny_scale, canny_start=canny_start, canny_end=canny_end,
            scribble_scale=scribble_scale, scribble_start=scribble_start, scribble_end=scribble_end,
            guidance_scale=guidance_used,
        )

        with _lock:
            JOBS[job_id]["status"] = "done"
            JOBS[job_id]["images_b64"] = images_b64
            JOBS[job_id]["images_data_url"] = [_to_data_url(s) for s in images_b64]  # ★ data URL 동시 제공
            # 하위 호환 alias
            JOBS[job_id]["images"] = JOBS[job_id]["images_b64"]
            JOBS[job_id]["images_urls"] = JOBS[job_id]["images_data_url"]

            if req.return_debug:
                JOBS[job_id]["debug"] = debug
            JOBS[job_id]["used_prompt"] = {
                "positive": pos,
                "negative": neg,
                "model": prompt_source,
                "llm_debug": llm_debug,
                "llm_inputs": (req.llm_inputs or {}),
                "text_info": (req.text_info or {}),
            }

        log.info(f"[JOB:{job_id}] generated {len(images_b64)} image(s)")
        if images_b64:
            log.info(f"[JOB:{job_id}] first image b64 len={len(images_b64[0])}")

    except Exception as e:
        with _lock:
            JOBS[job_id]["status"] = "error"
            JOBS[job_id]["error"] = str(e)
        log.exception(f"[JOB:{job_id}] generation failed")

# ---------------------- 생성 시작 ----------------------
@app.post("/logo/generate")
def start_generate(req: GenerateRequest):
    job_id = uuid.uuid4().hex
    with _lock:
        JOBS[job_id] = {
            "status": "running",
            "images_b64": [],
            "error": None,
        }

    th = threading.Thread(target=_run_job, args=(job_id, req), daemon=True)
    th.start()

    return {"job_id": job_id, "status": "running"}

# ---------------------- 폴링 ----------------------
from fastapi.responses import JSONResponse

@app.get("/logo/generate/{job_id}")
def get_job(job_id: str):
    with _lock:
        j = JOBS.get(job_id)
        if not j:
            return {"status": "not_found"}
        # 캐시 방지 헤더 + 호환 형태로 반환
        payload = _job_view_for_front(j)
    return JSONResponse(payload, headers={"Cache-Control": "no-store"})


# ---------------------- 대표 선택 ----------------------
@app.post("/logo/selection")
def save_selection(p: SelectionPayload):
    if p.brief_id not in BRIEFS:
        return {"ok": False, "error": "brief_id not found"}
    SELECTIONS[p.brief_id] = {"idx": p.selected_index, "total": p.total}
    return {"ok": True, "saved": SELECTIONS[p.brief_id]}

# ============================================================
# ==== DEBUG 엔드포인트: OpenAI 핑 / 프롬프트 단독 생성 =======
# ============================================================
@app.get("/_debug/health")
def debug_health():
    return {
        "openai": _openai_health_check(),
        "pipe_loaded": bool(PIPE),
        "device": PIPE_DEVICE,
        "prompt_model": PROMPT_LLM_MODEL,
        "llm_default_enabled": USE_LLM_PROMPT,
    }

def _brief_from_any(d: Dict[str, Any]) -> BriefPayload:
    def pick(*names, default=""):
        for n in names:
            v = d.get(n)
            if v:
                return v
        return default
    palette = d.get("palette") or d.get("colors") or []
    if isinstance(palette, str):
        palette = [s.strip() for s in palette.split(",") if s.strip()]
    return BriefPayload(
        request_id = d.get("request_id","debug"),
        cafe_name  = pick("cafe_name","brand_name","name"),
        copy_text  = pick("copy_text","text","wordmark"),
        layout     = pick("layout"),
        avoid      = pick("avoid"),
        strengths  = pick("strengths"),
        style      = pick("style","styles"),
        notes      = d.get("notes",""),
        model_hint = d.get("model_hint",""),
        palette    = palette,
        ref_image_present = bool(d.get("ref_image_present", False)),
    )

@app.post("/_debug/prompt")
def debug_prompt(payload: Dict[str, Any] = Body(...)):
    b = _brief_from_any(payload)
    pos, neg, src, dbg = _gen_logo_prompts_with_llm(
        b,
        has_canny=bool(payload.get("text_mask_png_b64")),
        has_scribble=bool(payload.get("sketch_png_b64")),
        text_info=payload.get("text_info"),
        prompt_overrides=(payload.get("llm_inputs") or {}).get("prompt_overrides"),
        gpt_prompt_seed=(payload.get("llm_inputs") or {}).get("gpt_prompt_seed"),
        gpt_messages=(payload.get("llm_inputs") or {}).get("gpt_messages"),
    )
    return {
        "ok": True,
        "source": src,
        "positive": pos,
        "negative": neg,
        "debug": dbg,
    }

# ============ DB 세션 DI & 유틸 ============
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def _user_id_from_username(db: SessionLocal, username: Optional[str], user_id: Optional[int]) -> int:
    """username 또는 user_id를 받아 실제 users.id를 반환 (없으면 400/404)"""
    if user_id is not None:
        u = db.query(User).filter(User.id == int(user_id)).first()
        if not u: raise HTTPException(404, "User not found")
        return u.id
    if username:
        u = db.query(User).filter(User.username == username).first()
        if not u: raise HTTPException(404, "User not found")
        return u.id
    raise HTTPException(400, "Either username or user_id is required")

# -------- 경로 정규화(레거시 data/ 접두어 제거 - 반복 제거 강화) --------
def _normalize_rel_path(p: str) -> str:
    rel = (p or "").lstrip("/").replace("\\", "/")
    while rel.startswith("data/"):
        rel = rel[5:]
    return rel

# ============================================================
# ================ Logos API 3종 (로그인 필수) ================
# ============================================================
@app.post("/logos/upload", response_model=LogoOut)
async def upload_logo(
    file: UploadFile = File(None, alias="file"),
    image: UploadFile = File(None, alias="image"),
    username: Optional[str] = Form(None),
    user_id: Optional[int] = Form(None),
    db: SessionLocal = Depends(get_db),
):
    Path(STORAGE_DIR).mkdir(parents=True, exist_ok=True)
    LOGO_ROOT.mkdir(parents=True, exist_ok=True)

    uid = _user_id_from_username(db, username, user_id)
    upload = file or image
    if upload is None:
        raise HTTPException(status_code=400, detail="file 또는 image 필드로 파일을 업로드하세요.")

    user_dir = LOGO_ROOT / str(uid)
    user_dir.mkdir(parents=True, exist_ok=True)

    orig_ext = Path(upload.filename or "").suffix.lower()
    if orig_ext not in [".png", ".jpg", ".jpeg", ".webp"]:
        orig_ext = ".png"
    filename = f"{uuid.uuid4().hex}{orig_ext}"
    disk_path = user_dir / filename

    try:
        blob = await upload.read()
        with open(disk_path, "wb") as f:
            f.write(blob)
    except Exception as e:
        raise HTTPException(500, f"파일 저장 실패: {e}")

    rel_path = str(Path("logos") / str(uid) / filename).replace("\\", "/")
    rel_path = _normalize_rel_path(rel_path)

    row = Logo(user_id=uid, image_path=rel_path)
    db.add(row); db.commit(); db.refresh(row)

    return LogoOut(
        id=row.id, user_id=row.user_id, image_path=row.image_path,
        image_url=f"/static/{row.image_path}", created_at=str(row.created_at),
    )

@app.get("/logos", response_model=List[LogoOut])
def list_my_logos(
    username: Optional[str] = Query(None),
    user_id: Optional[int] = Query(None),
    db: SessionLocal = Depends(get_db),
):
    uid = _user_id_from_username(db, username, user_id)
    rows = db.query(Logo).filter(Logo.user_id == uid).order_by(Logo.id.desc()).all()
    return [
        LogoOut(
            id=r.id, user_id=r.user_id,
            image_path=_normalize_rel_path(r.image_path),
            image_url=f"/static/{_normalize_rel_path(r.image_path)}",
            created_at=str(r.created_at),
        )
        for r in rows
    ]

@app.delete("/logos/{logo_id}")
def delete_logo(
    logo_id: int,
    username: Optional[str] = Query(None),
    user_id: Optional[int] = Query(None),
    db: SessionLocal = Depends(get_db),
):
    uid = _user_id_from_username(db, username, user_id)
    row = db.query(Logo).filter(Logo.id == logo_id, Logo.user_id == uid).first()
    if not row:
        raise HTTPException(404, "Logo not found")

    try:
        (Path(STORAGE_DIR) / _normalize_rel_path(row.image_path)).unlink(missing_ok=True)
    except Exception:
        pass

    db.delete(row); db.commit()
    return {"ok": True, "deleted_id": logo_id}



# ============================================================
# 인스타그램 파트 (APIRouter로 분리)
# ============================================================

class CaptionOption(BaseModel):
    theme: str
    content: str

class Hashtags(BaseModel):
    representative: list[str]
    location: list[str]
    trending: list[str]

class EngagementPrediction(BaseModel):
    score: str
    reason: str

class InstagramPostResponse(BaseModel):
    caption_options: list[CaptionOption]
    hashtags: Hashtags
    engagement_prediction: EngagementPrediction

IMAGE_ANALYST_PROMPT = """
# ROLE
당신은 브랜드 콘텐츠 전략가입니다. 인스타그램 마케팅 관점에서 이미지를 분석합니다.
# INSTRUCTION
주어진 이미지를 분석하여, 잠재 고객의 흥미를 유발할 만한 마케팅 포인트를 중심으로 아래 항목에 맞춰 텍스트로 요약해주세요. 각 항목은 명확하고 간결하게 작성해주세요.
- **주요 피사체 (Main Subject)**: 가장 눈에 띄는 메뉴나 상품, 그리고 그 특징.
- **감성 및 분위기 (Mood & Tone)**: 사진이 전달하는 전체적인 느낌 (예: 아늑함, 활기참, 고급스러움)과 지배적인 색감.
- **마케팅 포인트 (Marketing Point)**: 고객의 방문이나 구매를 유도할 수 있는 매력적인 디테일 (예: 예쁜 플레이팅, 특별한 인테리어 소품, 특정 시간대의 채광).
"""

COPYWRITER_PROMPT = """
# ROLE
당신은 '{brand_persona}' 페르소나를 가진, 독창적인 문체를 구사하는 전문 인스타그램 카피라이터입니다.
# CONTEXT
- 핵심 소재: {product_info}
- 가게 주소: {store_address}
- 이미지 분석 정보: {visual_info}
# INSTRUCTION
위 모든 정보를 종합하여, 아래 3가지 다른 테마의 인스타그램 캡션 초안을 JSON 형식으로 작성해주세요. **이미지 분석 정보를 각 테마에 자연스럽게 녹여내야 합니다.**
### 테마 가이드라인
1.  **감성적인 버전**: 한 편의 짧은 수필처럼, 고객의 경험과 감정에 초점을 맞춘 부드러운 문체.
2.  **정보 전달 버전**: 친한 전문가가 설명해주듯, 메뉴의 특징과 장점을 알기 쉽게 설명하는 문체.
3.  **재치있는 버전**: 독특한 관점이나 언어유희를 사용하여 고객이 미소 지을 수 있는 개성있는 문체.
### 공통 준수사항
- 글의 흐름이 자연스럽게 이어지도록, **초반에는 호기심을 자극하고, 중간에는 상세한 묘사와 스토리를 담고, 마지막에는 행동을 유도**하며 마무리할 것.
- 진부하거나 상투적인 표현(예: "맛이 없을 수 없는 조합")은 피할 것.
- **결과물에는 '[도입부]'와 같은 구분자를 절대 포함하지 말 것.**
- 반드시 아래 JSON 구조를 지켜주세요.
{{
  "options": [
    {{"theme": "감성적인 버전", "content": "여기에 구분자 없이 자연스럽게 이어진 전체 글을 작성해주세요."}},
    {{"theme": "정보 전달 버전", "content": "여기에 구분자 없이 전체 글을 작성해주세요."}},
    {{"theme": "재치있는 버전", "content": "여기에 구분자 없이 전체 글을 작성해주세요."}}
  ]
}}
"""

TARGET_MARKETER_PROMPT = """
# ROLE
당신은 세대별, 그룹별 마케팅 언어에 능통한 MZ세대 타겟 마케팅 전문가입니다.
# CONTEXT
- 캡션 옵션 (JSON): {initial_caption_options_json}
- 타겟 고객층: {target_audience}
# INSTRUCTION
'{target_audience}'가 가장 열광할 만한 **어휘, 문화 코드, 소통 방식**을 사용하여 주어진 3가지 캡션 옵션을 각각 수정해주세요.
- 원본의 핵심 메시지는 유지하되, 타겟 고객층의 눈높이에 맞춰 문체를 완전히 바꿔주세요.
- 수정된 결과도 반드시 원본과 동일한 JSON 구조로 반환해주세요.
"""

HASHTAG_MARKETER_PROMPT = """
# ROLE
당신은 인스타그램 로직을 이해하고, 소상공인을 위한 해시태그 전략을 수립하는 전문가입니다.
# CONTEXT
- 최종 캡션: {final_caption}
- 가게 주소: {store_address}
# INSTRUCTION
위 정보를 바탕으로, 아래 JSON 형식에 맞춰 최적의 해시태그 조합을 생성해주세요.
- **대표/메뉴 (Brand/Menu)**: 브랜드와 메뉴의 정체성을 나타내는 핵심 해시태그 2~3개.
- **지역/장소 (Location)**: 주소를 분석하여 **동네, 근처 지하철역, 유명 거리 등** 매우 구체적인 지역 기반 해시태그 3~4개.
- **감성/라이프스타일 (Mood/Lifestyle)**: 타겟 고객이 공감하고 검색할 만한 감성, 트렌드, 라이프스타일 관련 해시태그 3~4개.
- **제약 조건**: #카페, #커피, #맛집 처럼 너무 광범위한 태그는 피하고, 세분화된 태그를 사용할 것.
{{
  "representative": [],
  "location": [],
  "trending": []
}}
"""

ANALYST_PROMPT = """
# ROLE
당신은 소셜 미디어 데이터를 기반으로 콘텐츠의 성공 가능성을 예측하는 데이터 분석가입니다.
# CONTEXT
- 최종 캡션: {final_caption}
- 해시태그: {hashtags}
# INSTRUCTION
위 콘텐츠의 예상 고객 반응률을 분석하고, 아래 JSON 형식을 반드시 지켜 결과를 반환해주세요.
{{
  "score": "높음, 중간, 낮음 중 하나로 평가",
  "analysis": {{
    "strength": "이 콘텐츠가 고객의 긍정적인 반응을 유도할 수 있는 가장 큰 강점 1~2가지",
    "suggestion": "반응률을 더 높이기 위해 시도해 볼 만한 개선 아이디어 1가지 (선택 사항)"
  }}
}}
"""

# ----- 헬퍼 -----
def _parse_json_str(s: str, what: str) -> dict:
    """OpenAI JSON 모드 가정. 혹시 섞여오면 정규식으로 보정."""
    try:
        return json.loads(s)
    except Exception:
        try:
            m = re.search(r"\{.*\}", s, re.DOTALL)
            return json.loads(m.group(0)) if m else {}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"{what} JSON 파싱 실패: {e}")

async def call_llm_api(prompt: str) -> str:
    try:
        resp = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            response_format={"type": "json_object"},
        )
        return resp.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI 모델 호출 실패: {e}")

async def call_multimodal_llm_api(prompt: str, image_bytes: bytes) -> str:
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    try:
        resp = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
                ],
            }],
            temperature=0.7,
        )
        return resp.choices[0].message.content  # 자유 텍스트 OK
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI 멀티모달 모델 호출 실패: {e}")

# ----- 라우터 -----
ig_router = APIRouter(prefix="/v1/instagram", tags=["instagram"])

@ig_router.post("/generate", response_model=InstagramPostResponse)
async def generate_instagram_post(
    image: UploadFile = File(...),
    brand_persona: str = Form(...),
    product_info: str = Form(...),
    store_address: str = Form(...),
    target_audience: Optional[str] = Form(None),
):
    # 1) 이미지 분석
    image_bytes = await image.read()
    visual_info = await call_multimodal_llm_api(IMAGE_ANALYST_PROMPT, image_bytes)

    # 2) 캡션 초안
    initial_prompt = COPYWRITER_PROMPT.format(
        brand_persona=brand_persona,
        product_info=product_info,
        store_address=store_address,
        visual_info=visual_info,
    )
    initial_json = _parse_json_str(await call_llm_api(initial_prompt), "캡션 초안")
    caption_options = [CaptionOption(**opt) for opt in initial_json.get("options", [])]
    if not caption_options:
        raise HTTPException(status_code=500, detail="캡션 옵션이 비어 있습니다.")

    # 3) 타겟팅(옵션)
    if target_audience and target_audience != "전체":
        refine_prompt = TARGET_MARKETER_PROMPT.format(
            initial_caption_options_json=json.dumps(initial_json, ensure_ascii=False),
            target_audience=target_audience,
        )
        refined_json = _parse_json_str(await call_llm_api(refine_prompt), "타겟팅 캡션")
        try:
            ropts = [CaptionOption(**opt) for opt in refined_json.get("options", [])]
            if ropts:
                caption_options = ropts
        except Exception:
            pass

    # 4) 해시태그
    final_caption = caption_options[0].content
    hashtags_json = _parse_json_str(
        await call_llm_api(
            HASHTAG_MARKETER_PROMPT.format(final_caption=final_caption, store_address=store_address)
        ),
        "해시태그",
    )
    hashtags = Hashtags(**hashtags_json)

    # 5) 반응률 예측
    prediction_json_raw = _parse_json_str(
        await call_llm_api(
            ANALYST_PROMPT.format(
                final_caption=final_caption,
                hashtags=json.dumps(hashtags_json, ensure_ascii=False, indent=2),
            )
        ),
        "반응률 예측",
    )
    score = prediction_json_raw.get("score", "중간")
    analysis = prediction_json_raw.get("analysis", {}) or {}
    reason = analysis.get("strength", "") or analysis.get("suggestion", "") or ""
    prediction = EngagementPrediction(score=score, reason=reason)

    # === 응답 객체 먼저 구성 ===
    response_data = InstagramPostResponse(
        caption_options=caption_options,
        hashtags=hashtags,
        engagement_prediction=prediction,
    )

    # === DB 저장 (return 이전) ===
    try:
        with SessionLocal() as db:
            new_post = GeneratedPost(
                brand_persona=brand_persona,
                product_info=product_info,
                store_address=store_address,
                target_audience=target_audience or "전체",
                generated_captions=[c.model_dump() for c in response_data.caption_options],
                generated_hashtags=response_data.hashtags.model_dump(),
                engagement_prediction=response_data.engagement_prediction.model_dump(),
            )
            db.add(new_post)
            db.commit()
            # db.refresh(new_post)  # 필요 시
            print("INFO: 생성 결과가 DB에 성공적으로 저장되었습니다.")
    except Exception as e:
        # 정책에 따라: 저장 실패 시 500으로 알림
        raise HTTPException(status_code=500, detail=f"DB 저장에 실패했습니다: {e}")

    # === 응답 ===
    return response_data

app.include_router(ig_router)


# --- cartoon ---
import os
os.environ.setdefault("CARTOON_PF_REF_POLICY", "content")   # none|content|style
os.environ.setdefault("CARTOON_STYLE_LOCK", "1")
os.environ.setdefault("CARTOON_REF_MODE", "medium")
os.environ.setdefault("CARTOON_FORCE_PERSON_FOOD", "1")
os.environ.setdefault("CARTOON_FORCE_PF_POS", "any")   # any|last
os.environ.setdefault("CARTOON_FORCE_PF_RELAX", "0")
os.environ.setdefault("CARTOON_PF_CTRL_MAX", "0.82")
os.environ.setdefault("CARTOON_PF_IPW_MAX", "1.15")
os.environ.setdefault("CARTOON_PF_NO_ERASE", "1")
os.environ.setdefault("CARTOON_PF_IPW_MAX_FACE", "0.55")
os.environ.setdefault("CARTOON_PF_CTRL_MAX_FACE", "0.75")

import subprocess, os, asyncio, signal
from pathlib import Path
from contextlib import suppress
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import httpx, asyncio

from fastapi import FastAPI
from cartoon.src.profiles_sync import router as profiles_router
from cartoon.src.comfy_api_sync import router as comfy_router
from cartoon.src.cartoon_diffusers_sync import router as cartoon_router
from cartoon.src.caption_overlay_sync import router as caption_router
from cartoon.src.files_sync import router as files_router
from cartoon.src.post_ocr_erase_sync import router as ocr_router
from cartoon.src.prompt_builder_sync import router as prompts_router
from cartoon.src.story_suggester_sync import router as story_router

app.include_router(profiles_router, prefix="/profiles")
app.include_router(comfy_router,    prefix="/comfy")
app.include_router(cartoon_router,  prefix="/cartoon")
app.include_router(files_router,    prefix="/files")
app.include_router(ocr_router,      prefix="/ocr")
app.include_router(story_router,   prefix="/story")
app.include_router(prompts_router,  prefix="/prompts")
app.include_router(caption_router,  prefix="/caption")


COMFYUI_DIR = Path(os.getenv("COMFYUI_DIR", "/home/uv-env/poster/ComfyUI")).resolve()
COMFYUI_PORT = int(os.getenv("COMFYUI_PORT", "8188"))
COMFYUI_AUTOSTART = os.getenv("COMFYUI_AUTOSTART", "false").lower() in ("1","true","yes")
COMFYUI_AUTOSTOP  = os.getenv("COMFYUI_AUTOSTOP",  "false").lower() in ("1","true","yes")
_COMFY_PROC = None

LOG_DIR = Path("/home/uv-env/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
stdout_f = open(LOG_DIR / "comfyui.out", "ab", buffering=0)
stderr_f = open(LOG_DIR / "comfyui.err", "ab", buffering=0)

def _spawn_comfyui():
    if not COMFYUI_DIR.exists():
        raise RuntimeError(f"COMFYUI_DIR not found: {COMFYUI_DIR}")
    py = os.environ.get("PYTHON", "python")
    cmd = [py, "main.py", "--listen", "0.0.0.0", "--port", str(COMFYUI_PORT)]
    return subprocess.Popen(
        cmd,
        cwd=str(COMFYUI_DIR),
        env=os.environ.copy(),
        start_new_session=True,
        stdout=open("/home/uv-env/logs/comfyui.out", "ab", buffering=0),
        stderr=open("/home/uv-env/logs/comfyui.err", "ab", buffering=0),
    )

@app.on_event("startup")
async def _on_start():
    if COMFYUI_AUTOSTART:
        global _COMFY_PROC
        _COMFY_PROC = _spawn_comfyui()

@app.on_event("shutdown")
def _on_stop():
    global _COMFY_PROC
    if COMFYUI_AUTOSTOP and _COMFY_PROC:
        with suppress(Exception):
            os.killpg(os.getpgid(_COMFY_PROC.pid), signal.SIGTERM)


# ComfyUI 프락시 요청/응답 모델
class FourcutsGenerateReq(BaseModel):
    workflow: Dict[str, Any]
    client_id: Optional[str] = None
    wait: bool = True
    timeout_seconds: int = 60

class FourcutsGenerateRes(BaseModel):
    client_id: str
    prompt_id: Optional[str] = None
    outputs: Optional[Dict[str, Any]] = None
    image_urls: Optional[List[str]] = None

_env_comfy = os.getenv("COMFYUI_URL")

COMFYUI_URL = (_env_comfy if _env_comfy else f"http://127.0.0.1:{COMFYUI_PORT}").rstrip("/")

async def _post_prompt(workflow: Dict[str, Any], client_id: str) -> Dict[str, Any]:
    async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as client:
        r = await client.post(f"{COMFYUI_URL}/prompt", json={"prompt": workflow, "client_id": client_id})
        r.raise_for_status()
        return r.json()

async def _wait_history_any(client_id: str, prompt_id: Optional[str], timeout: int):
    deadline = asyncio.get_event_loop().time() + timeout
    async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
        last = None
        while asyncio.get_event_loop().time() <= deadline:
            # 1) client_id 방식
            r1 = await client.get(f"{COMFYUI_URL}/history/{client_id}")
            if r1.status_code == 200:
                try:
                    hist = r1.json()
                except Exception:
                    hist = None
                if hist:
                    try:
                        k = sorted(hist.keys())[-1]
                        last = hist[k]
                        if last and last.get("outputs"):
                            return last
                    except Exception:
                        pass

            # 2) prompt_id 방식
            if prompt_id:
                r2 = await client.get(f"{COMFYUI_URL}/history/{prompt_id}")
                if r2.status_code == 200:
                    try:
                        hist = r2.json()
                    except Exception:
                        hist = None
                    if hist:
                        try:
                            k = sorted(hist.keys())[-1]
                            last = hist[k]
                            if last and last.get("outputs"):
                                return last
                        except Exception:
                            pass
            await asyncio.sleep(1.0)
        raise TimeoutError("Timed out waiting for ComfyUI history.")


def _extract_urls(outputs: Dict[str, Any]) -> List[str]:
    urls = []
    for node_id, node_out in (outputs or {}).items():
        for im in (node_out.get("images") or []):
            fn = im.get("filename")
            sub = im.get("subfolder", "")
            if fn:
                urls.append(f"{COMFYUI_URL}/view?filename={fn}&subfolder={sub}&type=output")
    return urls

@app.get("/cartoon/healthz")
def cartoon_healthz():
    return {"ok": True, "comfyui_url": COMFYUI_URL}

@app.post("/cartoon/comfyui/generate", response_model=FourcutsGenerateRes)
async def cartoon_comfyui_generate(payload: FourcutsGenerateReq):
    client_id = payload.client_id or f"fourcuts-{uuid.uuid4().hex[:12]}"
    try:
        resp = await _post_prompt(payload.workflow, client_id)
    except httpx.HTTPError as e:
        raise HTTPException(502, f"ComfyUI error: {e}")

    prompt_id = resp.get("prompt_id") if isinstance(resp, dict) else None
    outputs = None
    image_urls = None

    if payload.wait:
        try:
            hist = await _wait_history_any(client_id, prompt_id, payload.timeout_seconds)
            outputs = hist.get("outputs") if isinstance(hist, dict) else None
            image_urls = _extract_urls(outputs) if outputs else None
        except TimeoutError as e:
            raise HTTPException(504, str(e))


    # 선택: usersdata에 간단 기록
    try:
        with SessionLocal() as db:
            db.add(UsersData(client_id=client_id, prompt_id=prompt_id, status=("done" if outputs else "queued")))
            db.commit()
    except Exception:
        pass

    return FourcutsGenerateRes(
        client_id=client_id,
        prompt_id=prompt_id,
        outputs=outputs,
        image_urls=image_urls,
    )
