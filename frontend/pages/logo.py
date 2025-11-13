# /home/uv-env/pages/logo.py
from pathlib import Path
import importlib.util
import streamlit as st
import re, unicodedata
from auth_guard import require_login

require_login(dest="pages/insta.py")

try:
    st.set_page_config(page_title="Logo", page_icon="🎨", layout="wide")
except Exception:
    pass

OK = True  # 외부 백엔드 상태 표기 같은 곳에 쓰려면 사용
LOGO_DIR = Path(__file__).resolve().parent / "logo_pages"

def _load_module(py_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, str(py_path))
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(mod)
    return mod

def _natural_key(file_stem: str):
    # "01_brief.py", "02_masking.py" 처럼 숫자 프리픽스가 있으면 그 순서로 정렬
    m = re.match(r"(\d+)[_\-]?(.*)", file_stem)
    return (int(m.group(1)) if m else 9999, file_stem.lower())

def _stem_to_title(stem: str) -> str:
    s = unicodedata.normalize("NFC", stem)
    s = re.sub(r"^\d+[_-]?", "", s)   # 앞 숫자_ 제거
    return s.replace("_", " ").replace("-", " ").strip()

def _set_step(i: int):
    st.session_state["logo_step"] = int(i)   # 세션 반영
    st.query_params["step"] = str(int(i))    # URL 동기화(딥링크)

# 하위 단계(.py) 수집 (언더스코어로 시작하는 파일은 제외해도 되지만 여기서는 포함)
files = sorted(LOGO_DIR.glob("[!_]*.py"), key=lambda p: _natural_key(p.stem))
if not files:
    st.error(f"하위 페이지가 없습니다: {LOGO_DIR}")
    st.stop()

# 모듈 로드 + 라벨 생성
pages = []
for i, p in enumerate(files):
    mod = _load_module(p, f"logo_{p.stem}_{i}")  # 충돌 방지용 유니크 이름
    title = getattr(mod, "TITLE", None) or _stem_to_title(p.stem)
    pages.append((p, title, mod))

labels = [t for _, t, _ in pages]

# 쿼리파라미터/세션으로 현재 단계 유지
idx = st.session_state.pop("logo_step", None)  # 세션 우선
if idx is None:
    qp = st.query_params
    try:
        idx = int(qp.get("step", 0))
    except Exception:
        idx = 0
idx = max(0, min(idx, len(pages) - 1))

if st.query_params.get("step") != str(idx):
    st.query_params["step"] = str(idx)

# --- 사이드바 내비게이션 ---
st.sidebar.subheader("🎨 Logo")
for i, label in enumerate(labels):
    st.sidebar.button(
        f"{i+1}) {label}",
        key=f"logo_nav_btn_{i}",
        use_container_width=True,
        on_click=lambda i=i: _set_step(i),
    )

# --- 헤더/상태 ---
st.title(f"Logo · {labels[idx]}")
if not OK:
    st.warning("백엔드가 오프라인일 수 있어요. (진행은 가능)")

# --- 선택된 단계 렌더링 ---
_, _, mod = pages[idx]
if hasattr(mod, "render") and callable(mod.render):
    mod.render()
else:
    st.info("이 페이지 파일에 render() 함수가 없어, import 시 실행된 코드만 표시됩니다.")
