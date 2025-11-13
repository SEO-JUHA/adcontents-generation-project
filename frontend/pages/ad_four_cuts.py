# ad_four_cuts.py
# FastAPI 변환 버전 (원본 Streamlit 런처를 서버 API로 래핑)
# - 파일명 규칙: *_async.py
# - DB 엔진: asyncmy (연결 풀)
# - ComfyUI API: /prompt 로 워크플로우 JSON 전송, /history 폴링 옵션
# - 원본 함수명(_load_module, _natural_key, _stem_to_title, _set_step) 유지

from __future__ import annotations

import os
# person_food 컷 레퍼런스 정책 (none|content|style)
os.environ.setdefault("CARTOON_PF_REF_POLICY", "style")
# 스타일 일관성
os.environ.setdefault("CARTOON_STYLE_LOCK", "1")
os.environ.setdefault("CARTOON_REF_MODE", "medium")  

# 최소 1컷 person_food 보장 정책
os.environ.setdefault("CARTOON_FORCE_PERSON_FOOD", "1")
os.environ.setdefault("CARTOON_FORCE_PF_POS", "any") # any|last
os.environ.setdefault("CARTOON_FORCE_PF_RELAX", "0")  # 0이면 음식 힌트 있을 때만


os.environ.setdefault("CARTOON_PF_CTRL_MAX", "0.42") 
os.environ.setdefault("CARTOON_PF_IPW_MAX", "0.8")
os.environ.setdefault("CARTOON_PF_NO_ERASE", "1")

import sys
import re
import uuid
import asyncio
import unicodedata
import importlib.util
from pathlib import Path
from typing import Any, Dict, List, Optional


# auth_guard import for login check
sys.path.append(str(Path(__file__).parent.parent))
from auth_guard import require_login

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
import httpx
from dotenv import load_dotenv, find_dotenv
import streamlit as st

import asyncmy


# 환경 변수 로드
load_dotenv(find_dotenv(filename=".env", usecwd=True))


# 원본과 동일한 경로/모듈 래핑 (fourcuts_shared 별칭 주입)
PAGES_DIR = Path(__file__).resolve().parent / "fast_4cuts_pages"
SHARED_ALIAS = "fourcuts_shared"
shared_path = PAGES_DIR / "_shared.py"
if shared_path.exists():
    spec = importlib.util.spec_from_file_location(SHARED_ALIAS, str(shared_path))
    shared_mod = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(shared_mod)
    sys.modules[SHARED_ALIAS] = shared_mod  # fourcuts_shared 라는 이름으로 노출


# 원본과 동일한 유틸 함수들(함수명 유지)

def _load_module(py_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, str(py_path))
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(mod)
    return mod


def _natural_key(file_stem: str):
    m = re.match(r"(\d+)[_\-]?(.*)", file_stem)
    return (int(m.group(1)) if m else 9999, file_stem.lower())


def _stem_to_title(stem: str) -> str:
    s = unicodedata.normalize("NFC", stem)
    s = re.sub(r"^\d+[_-]?", "", s)
    return s.replace("_", " ").replace("-", " ").strip()


def _set_step(i: int):
    st.session_state["fourcuts_step_async"] = int(i)
    st.query_params["step_async"] = str(int(i))


# 하위 페이지(모듈) 동적 로드
files = sorted(PAGES_DIR.glob("[!_]*.py"), key=lambda p: _natural_key(p.stem))
pages: List[dict] = []
for i, p in enumerate(files):
    mod = _load_module(p, f"fourcuts_{p.stem}_{i}")
    title = getattr(mod, "TITLE", None) or _stem_to_title(p.stem)
    pages.append({"index": i, "file": p.name, "title": title, "module": mod})



# FastAPI 앱/설정
app = FastAPI(title="4cuts API", version="1.0.0")

# asyncmy 풀 설정
MYSQL_HOST = os.getenv("MYSQL_HOST", "127.0.0.1")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
MYSQL_USER = os.getenv("MYSQL_USER", "user1")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "4321")
MYSQL_DB = os.getenv("MYSQL_DB", "fourcuts")

DB_POOL: Optional[asyncmy.Pool] = None

# ComfyUI 엔드포인트
COMFYUI_URL = os.getenv("COMFYUI_URL", "http://127.0.0.1:8188")


# 요청/응답 모델
class GenerateRequest(BaseModel):
    workflow: Dict[str, Any]  # ComfyUI 워크플로우 JSON(Graph)
    client_id: Optional[str] = None
    wait: bool = True  # True면 /history 폴링하여 결과까지 반환
    timeout_seconds: int = 60  # wait=True일 때 대기 시간


class GenerateResponse(BaseModel):
    client_id: str
    prompt_id: Optional[str] = None
    outputs: Optional[Dict[str, Any]] = None
    image_urls: Optional[List[str]] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global DB_POOL
    DB_POOL = await asyncmy.create_pool(
        host=MYSQL_HOST,
        port=MYSQL_PORT,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        db=MYSQL_DB,
        minsize=1,
        maxsize=10,
        autocommit=True,
    )
    app.state.db_pool = DB_POOL

    # 선택: 간단한 usersdata 테이블 생성(없으면)
    async with DB_POOL.acquire() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                CREATE TABLE IF NOT EXISTS usersdata (
                    id BIGINT PRIMARY KEY AUTO_INCREMENT,
                    client_id VARCHAR(64),
                    prompt_id VARCHAR(64),
                    status VARCHAR(32),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
                """
            )
    try:
        yield
    finally:
        if DB_POOL:
            DB_POOL.close()
            await DB_POOL.wait_closed()

# FastAPI에 lifespan 적용
app.router.lifespan_context = lifespan



# 기본 헬스체크/메타
@app.get("/healthz")
async def healthz():
    return {"ok": True}


@app.get("/pages")
async def list_pages():
    return [{"index": p["index"], "file": p["file"], "title": p["title"]} for p in pages]


@app.get("/pages/{idx}")
async def get_page(idx: int):
    if idx < 0 or idx >= len(pages):
        raise HTTPException(404, "page index out of range")
    p = pages[idx]
    return {"index": p["index"], "file": p["file"], "title": p["title"]}


# ComfyUI 연동
async def _post_prompt_to_comfyui(workflow: Dict[str, Any], client_id: str) -> Dict[str, Any]:
    async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as client:
        r = await client.post(f"{COMFYUI_URL}/prompt", json={"prompt": workflow, "client_id": client_id})
        r.raise_for_status()
        return r.json()


async def _wait_for_history(client_id: str, timeout: int) -> Dict[str, Any]:
    """/history/{client_id} 를 폴링하여 최신 결과를 획득.
    환경/버전에 따라 /history/{prompt_id} 를 쓰는 경우도 있으니 필요 시 수정하세요.
    """
    deadline = asyncio.get_event_loop().time() + timeout
    last: Optional[Dict[str, Any]] = None
    async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
        while True:
            if asyncio.get_event_loop().time() > deadline:
                raise TimeoutError("Timed out waiting for ComfyUI history.")
            hr = await client.get(f"{COMFYUI_URL}/history/{client_id}")
            if hr.status_code == 200:
                hist = hr.json()
                # hist 예시: { prompt_id: { "outputs": { node_id: { "images": [...] } } }, ... }
                if hist:
                    last_key = sorted(hist.keys())[-1]
                    last = hist[last_key]
                    if last and last.get("outputs"):
                        return last
            await asyncio.sleep(1.0)


def _extract_image_urls(outputs: Dict[str, Any]) -> List[str]:
    urls: List[str] = []
    for node_id, node_out in outputs.items():
        images = node_out.get("images") or []
        for im in images:
            filename = im.get("filename")
            subfolder = im.get("subfolder", "")
            if filename:
                # \"type=output\" 고정
                u = f"{COMFYUI_URL}/view?filename={filename}&subfolder={subfolder}&type=output"
                urls.append(u)
    return urls


@app.post("/comfyui/generate", response_model=GenerateResponse)
async def comfyui_generate(req: GenerateRequest):
    client_id = req.client_id or f"fourcuts-{uuid.uuid4().hex[:12]}"

    try:
        res = await _post_prompt_to_comfyui(req.workflow, client_id)
    except httpx.HTTPError as e:
        raise HTTPException(502, f"ComfyUI error: {e}")

    prompt_id = None
    if isinstance(res, dict):
        prompt_id = res.get("prompt_id")  # 일부 버전에서만 포함

    outputs = None
    image_urls: Optional[List[str]] = None
    if req.wait:
        try:
            hist = await _wait_for_history(client_id, req.timeout_seconds)
            outputs = hist.get("outputs") if isinstance(hist, dict) else None
            if outputs:
                image_urls = _extract_image_urls(outputs)
        except TimeoutError as e:
            raise HTTPException(504, str(e))

    # DB에 간단 기록
    if DB_POOL:
        async with DB_POOL.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "INSERT INTO jobs (client_id, prompt_id, status) VALUES (%s, %s, %s)",
                    (client_id, prompt_id, "done" if outputs else "queued"),
                )

    return GenerateResponse(
        client_id=client_id,
        prompt_id=prompt_id,
        outputs=outputs,
        image_urls=image_urls,
    )


# 선택: 하위 페이지 모듈이 FastAPI 라우터를 제공하면 자동 등록
for p in pages:
    mod = p["module"]
    # 1) module에 'router' (APIRouter) 가 있으면 mount
    router = getattr(mod, "router", None)
    if router is not None:
        app.include_router(router, prefix=f"/pages/{p['index']}")
    # 2) module에 register_routes(app) 가 있으면 호출
    reg = getattr(mod, "register_routes", None)
    if callable(reg):
        reg(app, prefix=f"/pages/{p['index']}")



# Streamlit 렌더링 (타이틀/사이드바 네비·페이지 전환)
if "streamlit" in sys.modules:
    # 로그인 체크 (로그인하지 않으면 app.py로 리다이렉트)
    require_login(redirect="app.py", dest="AD Four-cuts_async")
    
    # 타이틀을 가장 먼저 출력(원본과 동일 UX)
    try:
        st.set_page_config(page_title="광고네컷 (async)", layout="wide")
    except Exception:
        pass

    labels = [p["title"] for p in pages]

    # 현재 인덱스 결정(세션 → 쿼리파라미터)
    idx = st.session_state.pop("fourcuts_step_async", None)
    if idx is None:
        qp = st.query_params
        try:
            idx = int(qp.get("step_async", 0))
        except Exception:
            idx = 0
    idx = max(0, min(int(idx), len(pages) - 1))

    # 사이드바 네비게이션 버튼
    st.sidebar.subheader("4cuts (async)")
    if st.query_params.get("step_async") != str(idx):
        st.query_params["step_async"] = str(idx)

    for i, label in enumerate(labels):
        st.sidebar.button(
            f"{i+1}) {label}",
            key=f"fourcuts_nav_btn_async_{i}",
            width='stretch',
            on_click=lambda i=i: _set_step(i),
        )

    # 타이틀 + 현재 페이지 렌더(타이틀을 상단에 유지)
    st.title(f"4cuts · {labels[idx]}")
    cur = pages[idx]
    mod = cur["module"]
    if hasattr(mod, "render") and callable(mod.render):
        mod.render()
    else:
        st.info("이 페이지 파일에 render() 함수가 없어, import 시 실행된 코드만 표시됩니다.")
