import sys, pathlib
import os
from datetime import datetime

import streamlit as st
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

LOGOS_DIR = ROOT / "poster" / "data" / "logos"

from poster.src.poster_utils import goto_poster_step, run_pending_nav, go_to_home


def _reset_downstream_states():
    """
    1페이지 입력(로고/브랜드 소개)이 바뀌면,
    이후 단계(2~4페이지)에서 사용하는 상태를 안전하게 초기화.
    """
    for k in [
        "selected_layout_id",
        "selected_layout_path",
        "slots_mask_path",
        "generated_paths",
        "chosen_path",
        "chosen_version",
        "gen_page_fp",
        "edited_image",
        "layers",
        "hist",
        "redo",
        "last_design_fp",
    ]:
        st.session_state.pop(k, None)


def _fp_stat(p):
    try:
        return f"{p}:{int(os.path.getmtime(p))}" if (p and os.path.exists(p)) else str(p)
    except Exception:
        return str(p)


def _inputs_fingerprint(logo_path: str | None, brand_intro: str | None) -> str:
    """
    1페이지 입력의 지문. (로고 파일 경로+mtime, 브랜드 텍스트 해시)
    뒤 단계의 초기화 트리거에 사용.
    """
    return "|".join([
        _fp_stat(logo_path),
        str(hash(brand_intro or "")),
    ])


def render():
    st.set_page_config(page_title="1) 브랜드/로고 입력", layout="wide")
    st.title("1) 브랜드/로고 입력")
    st.caption("로고 이미지를 업로드하고, BI(브랜드 소개)를 입력하세요.")

    # 현재 저장된 값 표시 (있다면)
    cur_logo = st.session_state.get("logo_path")
    cur_intro = st.session_state.get("brand_intro", "딥그린과 베이지를 사용하는 미니멀 카페. 고급스럽고 차분한 무드.")

    with st.form("brand_form"):
        col1, col2 = st.columns([1, 2], gap="large")

        with col1:
            uploaded = st.file_uploader("로고 이미지(PNG/JPG)", type=["png", "jpg", "jpeg"])
            # 기존 로고가 있으면 미리보기
            if uploaded:
                st.image(uploaded, caption="업로드 미리보기", width='stretch')
            elif cur_logo and os.path.exists(cur_logo):
                st.image(cur_logo, caption="현재 저장된 로고", width='stretch')

        with col2:
            brand_intro = st.text_area(
                "브랜드 소개 (무드/컬러 단서 포함)",
                value=cur_intro,
                height=160,
            )

        saved = st.form_submit_button("저장")

    if saved:
        new_logo_path = cur_logo  # 기본값: 기존 유지
        # 로고 새 업로드가 있으면 저장
        if uploaded:
            os.makedirs(LOGOS_DIR, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            new_logo_path = str(LOGOS_DIR / f"ui_{ts}.png")
            Image.open(uploaded).convert("RGBA").save(new_logo_path)

        # 지문 비교로 변경 여부 판단
        old_fp = st.session_state.get("inputs_fp")
        new_fp = _inputs_fingerprint(new_logo_path, brand_intro)

        # 세션 업데이트
        st.session_state["logo_path"] = new_logo_path
        st.session_state["brand_intro"] = brand_intro
        st.session_state["inputs_fp"] = new_fp

        if old_fp != new_fp:
            # 입력 변경 → 뒤 단계 상태 초기화
            _reset_downstream_states()
            st.success("저장 완료! (입력이 변경되어 뒤 단계 상태를 초기화했습니다.)")
        else:
            st.info("변경 사항이 없습니다. 현재 입력을 유지합니다.")

    st.divider()

    # 하단 네비게이션: 이전=홈, 다음=2페이지
    has_inputs = bool(st.session_state.get("logo_path")) and bool(st.session_state.get("brand_intro"))
    prev_col, next_col = st.columns([1, 1])
    with prev_col:
        st.button("← 홈", use_container_width=True, on_click=lambda: go_to_home())

    with next_col:
        st.button("다음 → 2단계", type="primary", disabled=not has_inputs,
                  use_container_width=True, on_click=lambda: goto_poster_step(1))
    run_pending_nav()


if __name__ == "__main__":
    render()