
from __future__ import annotations
import json, zipfile, os
from pathlib import Path
from tempfile import TemporaryDirectory, NamedTemporaryFile
from typing import List, Iterable, Optional, Dict, Any
import requests
import streamlit as st
from PIL import Image
import urllib.parse
import yaml
import traceback

# ---------------------- FastAPI base & HTTP helpers ----------------------
def _read_input_dir() -> Path:
    here = Path(__file__).resolve()
    cfg_path = here.parents[2] / "cartoon" / "configs" / "routing.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    inp = cfg.get("input_dir")
    return Path(inp) if inp else Path(".")


def _api_base() -> str:
    return os.getenv("FOURCUTS_API_BASE", "http://127.0.0.1:8000").rstrip("/")

def file_url(server_path: str) -> str:
    base = _api_base()
    if not server_path:
        return ""
    p = Path(server_path)
    if not p.is_absolute():
        p = _read_input_dir() / server_path
    return f"{base}/files/get?path=" + urllib.parse.quote(str(p), safe="")


def _get(path: str, **params):
    headers = {}
    uid = os.getenv("FOURCUTS_USER_ID") or ((st.session_state.get("auth_user") or {}).get("id")) or (st.session_state.get("user_id") if "user_id" in st.session_state else None)
    if uid:
        headers["X-User-Id"] = str(uid)
    timeout_sec = 15 if path.startswith("/profiles/") else 30
    r = requests.get(f"{_api_base()}{path}", params=params, headers=headers, timeout=timeout_sec)
    r.raise_for_status()
    return r.json()

def _post(path: str, payload: Dict[str, Any] | None = None):
    headers = {"Content-Type": "application/json"}
    uid = os.getenv("FOURCUTS_USER_ID") or ((st.session_state.get("auth_user") or {}).get("id")) \
          or (st.session_state.get("user_id") if "user_id" in st.session_state else None)
    if uid:
        headers["X-User-Id"] = str(uid)

    # 경로별 타임아웃
    if path.startswith("/cartoon/"):
        timeout_sec = 1800  # 30분
    elif path.startswith("/profiles/"):
        timeout_sec = 10    # 10초
    else:
        timeout_sec = 3000

    url = f"{_api_base()}{path}"
    data = payload or {}

    try:
        r = requests.post(url, json=data, headers=headers, timeout=timeout_sec)
        if not r.ok:
            # 실패 시: 상태/URL/요청 페이로드/응답 본문을 노출
            body_text = r.text or ""
            msg_lines = [
                f"[POST] {url} -> {r.status_code}",
                f"Request headers: { {k: v for k, v in headers.items() if k != 'Authorization'} }",
                f"Request payload: {json.dumps(data, ensure_ascii=False)[:1500]}",
                f"Response body: {body_text[:2000]}",
            ]
            msg = "\n".join(msg_lines)

            # Streamlit에도, 콘솔에도 노출
            try:
                st.warning(msg)
            except Exception:
                pass
            print(msg)

            # JSON 본문이면 구조화해서 한 번 더 표시
            try:
                j = r.json()
                try:
                    st.json(j)
                except Exception:
                    print("[Response JSON]", j)
            except Exception:
                pass

            r.raise_for_status()  # 최종적으로 기존 동작 유지: 예외 발생
        return r.json()
    except requests.exceptions.RequestException as e:
        # 네트워크/타임아웃 등
        emsg = f"[POST] {url} request failed: {e}\nRequest payload: {json.dumps(data, ensure_ascii=False)[:1500]}"
        try:
            st.error(emsg)
        except Exception:
            pass
        print(emsg)
        raise

# ---------------------- Session / files / misc ---------------------------
def ensure_tmp_dir() -> Path: # 세션에 yaml의 경로 전달 
    if "_tmp_dir" not in st.session_state:
        st.session_state["_tmp_dir"] = TemporaryDirectory()
    return Path(st.session_state["_tmp_dir"].name)

def ensure_thumb_px(default: int = 240) -> int:
    if "thumb_px" not in st.session_state:
        st.session_state["thumb_px"] = int(default)
    return int(st.session_state["thumb_px"])

def save_one(uploaded_file, tmp_dir: Path) -> Optional[str]:
    if not uploaded_file:
        return None
    try:
        Image.open(uploaded_file).verify()
    except Exception:
        return None
    uploaded_file.seek(0)
    suffix = Path(uploaded_file.name).suffix or ".png"
    with NamedTemporaryFile(dir=tmp_dir, suffix=suffix, delete=False) as tf:
        tf.write(uploaded_file.read())
        return tf.name

def save_one_dir(uploaded_file, tmp_dir: Path) -> Optional[str]:
    if not uploaded_file:
        return None
    try:
        Image.open(uploaded_file).verify()
    except Exception:
        return None
    uploaded_file.seek(0)
    with open(tmp_dir / uploaded_file.name, "wb") as f:
        f.write(uploaded_file.read())
    return tmp_dir / uploaded_file.name

def save_logo_to_comfyui_input(logo_file, user_id, tmp_dir: Path) -> Optional[str]:
    """로고 파일을 ComfyUI input 디렉토리에 영구 저장"""
    if not logo_file:
        return None
    
    # routing.yaml에서 input_dir 경로 가져오기
    here = Path(__file__).resolve()
    cfg_path = here.parents[2] / "cartoon" / "configs" / "routing.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    input_dir = cfg.get("input_dir")
    
    if not input_dir:
        st.error("routing.yaml에서 input_dir을 찾을 수 없습니다.")
        return None
    
    input_path = Path(input_dir)
    input_path.mkdir(parents=True, exist_ok=True)
    
    # data/logos 디렉토리 구조로 저장 (기존 구조와 일치)
    data_logos_dir = input_path / "data" / "logos"
    data_logos_dir.mkdir(parents=True, exist_ok=True)
    
    # 파일 확장자 추출
    file_ext = Path(logo_file.name).suffix or ".png"
    
    # 고유한 파일명 생성 (타임스탬프 + 원본명)
    import time
    timestamp = int(time.time())
    safe_name = Path(logo_file.name).stem.replace(" ", "_")
    filename = f"logo_{timestamp:02d}_{safe_name}{file_ext}"
    
    # 파일 저장
    logo_path = data_logos_dir / filename
    
    try:
        # PIL로 이미지 검증 후 저장
        logo_file.seek(0)
        img = Image.open(logo_file)
        img.verify()  # 이미지 검증
        
        logo_file.seek(0)
        with open(logo_path, "wb") as f:
            f.write(logo_file.read())
        
        # ComfyUI에서 사용할 상대 경로 반환 (data/logos/ 구조)
        relative_path = f"data/logos/{filename}"
        return relative_path
        
    except Exception as e:
        st.error(f"로고 저장 실패: {e}")
        return None

def save_logo_from_existing_path(existing_path: str, user_id, tmp_dir: Path) -> Optional[str]:
    """기존 파일 경로를 ComfyUI input 디렉토리로 복사하여 영구 저장"""
    if not existing_path or not Path(existing_path).exists():
        return None
    
    # routing.yaml에서 input_dir 경로 가져오기
    here = Path(__file__).resolve()
    cfg_path = here.parents[2] / "cartoon" / "configs" / "routing.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    input_dir = cfg.get("input_dir")
    
    if not input_dir:
        st.error("routing.yaml에서 input_dir을 찾을 수 없습니다.")
        return None
    
    input_path = Path(input_dir)
    input_path.mkdir(parents=True, exist_ok=True)
    
    # data/logos 디렉토리 구조로 저장
    data_logos_dir = input_path / "data" / "logos"
    data_logos_dir.mkdir(parents=True, exist_ok=True)
    
    # 파일 확장자 추출
    file_ext = Path(existing_path).suffix or ".png"
    
    # 고유한 파일명 생성 (타임스탬프 + 원본명)
    import time
    timestamp = int(time.time())
    safe_name = Path(existing_path).stem.replace(" ", "_")
    filename = f"logo_{timestamp:02d}_{safe_name}{file_ext}"
    
    # 파일 복사
    logo_path = data_logos_dir / filename
    
    try:
        # PIL로 이미지 검증 후 복사
        img = Image.open(existing_path)
        img.verify()  # 이미지 검증
        
        import shutil
        shutil.copy2(existing_path, logo_path)
        
        # ComfyUI에서 사용할 상대 경로 반환 (data/logos/ 구조)
        relative_path = f"data/logos/{filename}"
        return relative_path
        
    except Exception as e:
        st.error(f"로고 복사 실패: {e}")
        return None

def save_many(uploaded_files, tmp_dir: Path) -> List[str]:
    return [p for f in (uploaded_files or []) if (p := save_one(f, tmp_dir))]

def read_manifest(manifest_path: str | Path) -> Dict[str, Any]:
    try:
        return json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    except Exception:
        return {}

def goto(delta: int = 1):
    # async 버전에서는 step_async 사용
    cur = int(st.query_params.get("step_async", 0))
    nxt = max(0, cur + delta)
    st.query_params["step_async"] = str(nxt)
    st.session_state["fourcuts_step_async"] = nxt
    st.rerun()

def require_inputs() -> bool:
    need = ["brand_bi", "core_msg", "layout_id", "images"]
    missing = [k for k in need if not st.session_state.get(k)]
    if missing:
        st.warning("먼저 ‘업로드 & 설정’ 페이지에서 정보를 입력하세요.")
        return False
    return True

def require_bases() -> bool:
    if not st.session_state.get("base_panels"):
        st.warning("먼저 ‘생성’ 페이지에서 이미지를 생성하세요.")
        return False
    return True

def prefill_caps_from_suggest():
    if st.session_state.get("_refill_caps", False):
        for i, s in enumerate(st.session_state.get("suggested_caps", []), start=1):
            st.session_state[f"cap_{i}"] = s or f"Panel {i}"
        st.session_state["_refill_caps"] = False

def overlay_all_from_bases(caps: Iterable[str]) -> List[str]:
    overlay_dir = Path(st.session_state["manifest"]).parent / "overlays" if st.session_state.get("manifest") else ensure_tmp_dir() / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    out_paths = []
    for i, (base_src, txt) in enumerate(zip(st.session_state["base_panels"], caps), start=1):
        t = (txt or "").strip() or f"Panel {i}"
        res = _post("/caption/overlay", {
            "base_path": base_src, "text": t, "out_dir": str(overlay_dir),
            "bar_mode": "tight", "max_chars_per_line": 25, "stroke_width": 3
        })
        out_paths.append(res.get("out_path", base_src))
    return out_paths

def zip_paths(paths: List[str], out_zip: Path) -> Path:
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
        for p in paths:
            p = Path(p)
            z.write(p, arcname=p.name)
    return out_zip

def output_dir_for(kind: str = "exports") -> Path:
    root = Path(st.session_state["manifest"]).parent if st.session_state.get("manifest") else ensure_tmp_dir()
    out = root / kind
    out.mkdir(parents=True, exist_ok=True)
    return out

def make_collage(captioned: bool, side: int = 2160, pad: int = 16) -> str:
    imgs = (st.session_state.get("last_panels") if captioned else st.session_state.get("base_panels")) or []
    if not imgs: return ""
    out_dir = output_dir_for("exports")
    fn = "comic_2x2_captioned.jpg" if captioned else "comic_2x2_base.jpg"
    out_path = out_dir / fn
    _post("/caption/compose-2x2", {"images": imgs, "out_path": str(out_path), "final_side": int(side), "pad_px": int(pad)})
    return str(out_path)

def _normalize_path(path: str) -> str:
    """경로를 정규화하고 파일 존재 여부 확인
    
    서버에서 생성된 절대 경로를 그대로 사용합니다.
    경로 변환이나 하드코딩된 경로 매핑 없이 단순하게 처리합니다.
    """
    if not path:
        return path
    
    import os
    from pathlib import Path
    
    # 경로를 Path 객체로 변환하여 정규화
    try:
        normalized = Path(path).resolve()
        # 파일이 존재하면 정규화된 경로 반환
        if normalized.exists():
            return str(normalized)
    except Exception:
        # 경로 파싱 실패 시 원본 반환
        pass
    
    # 파일이 없거나 경로가 잘못되었으면 원본 반환
    return path

def gallery_2x2(
    images: list[str] = None,
    paths: list[str] = None,
    *,
    labels: list[str] | None = None,
    gap: str = "medium",
    width=None,
    with_text_keys: list[str] | None = None,
    text_height: int = 80,
):
    """2x2 그리드로 이미지 표시 (4cuts_pages 호환 + 경로 정규화)"""
    # 하위 호환성: paths 또는 images 파라미터 지원
    imgs = (images or paths or [])[:4]
    if len(imgs) < 4 and imgs:
        imgs = imgs + [imgs[-1]] * (4 - len(imgs))
    elif not imgs:
        imgs = [None, None, None, None]
    
    def _show(col, idx):
        cap = (labels[idx] if labels and idx < len(labels) else f"Panel {idx+1}")
        with col:
            if idx < len(imgs) and imgs[idx]:
                p = imgs[idx]
                try:
                    # 경로 정규화
                    normalized_path = _normalize_path(p)
                    
                    # 파일 존재 여부 확인 및 표시
                    if os.path.exists(normalized_path):
                        # width 파라미터 처리
                        if width is None:
                            st.image(normalized_path, caption=cap, use_container_width=True)
                        else:
                            st.image(normalized_path, caption=cap, width=width)
                    else:
                        st.error(f"파일 없음: {cap}")
                        st.info(f"경로: {normalized_path}")
                except Exception as e:
                    st.error(f"이미지 로드 실패: {e}")
                    st.info(f"경로: {p}")
                    st.code(traceback.format_exc())
            
            # 텍스트 입력 필드 (옵션)
            if with_text_keys and idx < len(with_text_keys):
                key = with_text_keys[idx]
                st.text_area(f"{cap} caption", key=key, height=text_height)
    
    # 2x2 레이아웃
    row1 = st.columns(2, gap=gap)
    _show(row1[0], 0)
    _show(row1[1], 1)
    row2 = st.columns(2, gap=gap)
    _show(row2[0], 2)
    _show(row2[1], 3)
