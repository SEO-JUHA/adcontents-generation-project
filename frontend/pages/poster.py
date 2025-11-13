# /home/uv-env/pages/poster.py
from pathlib import Path
import importlib.util
import streamlit as st

# 🔧 추가
import re, unicodedata

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(filename=".env", usecwd=True)) 

from auth_guard import require_login
require_login(dest="pages/insta.py")


st.set_page_config(page_title="Poster", layout="wide")
ok = True
POSTER_DIR = Path(__file__).resolve().parent / "poster_pages"

def _load_module(py_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, str(py_path))
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(mod)
    return mod
 
def _natural_key(file_stem: str):
    # 파일명이 "숫자_제목"이면 숫자 기준 정렬
    m = re.match(r"(\d+)[_\-]?(.*)", file_stem)
    return (int(m.group(1)) if m else 9999, file_stem.lower())

def _stem_to_title(stem: str) -> str:
    s = unicodedata.normalize("NFC", stem)
    s = re.sub(r"^\d+[_-]?", "", s)   # 앞 숫자_ 제거
    return s.replace("_", " ").replace("-", " ").strip()

def _set_step(i: int):
    st.session_state["poster_step"] = int(i)      # 세션에 반영
    st.query_params["step"] = str(int(i))         # URL도 동기화(딥링크 유지)

# 하위 페이지 수집
files = sorted(POSTER_DIR.glob("[!_]*.py"), key=lambda p: _natural_key(p.stem))
if not files:
    st.error(f"하위 페이지가 없습니다: {POSTER_DIR}")
    st.stop()

# 모듈 로드 + 라벨 결정
pages = []
for i, p in enumerate(files):
    mod = _load_module(p, f"poster_{p.stem}_{i}")  # 캐시 충돌 방지용 유니크 이름
    title = getattr(mod, "TITLE", None) or _stem_to_title(p.stem)
    pages.append((p, title, mod))

labels = [t for _, t, _ in pages]

# 쿼리파라미터로 현재 단계 유지 (직접 링크/뒤로가기 UX 개선)
idx = st.session_state.pop("poster_step", None)  # 세션 우선
if idx is None:
    qp = st.query_params
    try:
        idx = int(qp.get("step", 0))
    except Exception:
        idx = 0
idx = max(0, min(idx, len(pages) - 1))

init_label = labels[idx]
st.session_state.setdefault("poster_step_idx", init_label)

# 사이드바 내부 내비게이션
st.sidebar.subheader("📂 Poster")

if st.query_params.get("step") != str(idx):
    st.query_params["step"] = str(idx)

for i, label in enumerate(labels):
    st.sidebar.button(
        f"{i+1}) {label}",
        key=f"poster_nav_btn_{i}",
        use_container_width=True,
        on_click=lambda i=i: _set_step(i),
    )
    
# 헤더
st.title(f"Poster · {labels[idx]}")
if not ok:
    st.warning("백엔드(예: ComfyUI)가 오프라인일 수 있어요. (진행은 가능)")

# 선택된 페이지 렌더링
_, _, mod = pages[idx]
if hasattr(mod, "render") and callable(mod.render):
    mod.render()
else:
    st.info("이 페이지 파일에 render() 함수가 없어, import 시 실행된 코드만 표시됩니다.")

