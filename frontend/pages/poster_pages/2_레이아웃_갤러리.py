# -*- coding: utf-8 -*-
import sys, os, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

INDEX_JSON = ROOT / "poster" / "data" / "layouts" / "index.json"

import json
import streamlit as st
from poster.src.poster_utils import goto_poster_step, flash, consume_flash, run_pending_nav


def _abs_path(p: str | os.PathLike) -> str:
    """data/... 형태의 상대경로를 poster 루트 기준 절대경로로 변환"""
    p = str(p)
    return p if os.path.isabs(p) else str((ROOT / p).resolve())


def _reset_downstream_states():
    """
    레이아웃 변경 시, 3·4페이지에서 사용 중인 상태를 안전하게 초기화.
    (비파괴: 필요한 키만 정리)
    """
    for k in [
        "generated_paths",
        "chosen_path",
        "chosen_version",
        "last_design_fp",
        "gen_page_fp",
        "edited_image", "layers", "hist", "redo",
    ]:
        st.session_state.pop(k, None)


def render():
    st.set_page_config(page_title="2) 레이아웃 갤러리", layout="wide")
    st.title("2) 레이아웃 갤러리")
    st.caption("원하는 레이아웃을 하나 선택하세요.")

    if not st.session_state.get("logo_path") or not st.session_state.get("brand_intro"):
        flash("먼저 로고와 브랜드 소개를 저장하세요.", level="warning")
        goto_poster_step(0)
        run_pending_nav()
        st.stop()

    consume_flash()

    # 레이아웃 카탈로그 로드
    if not INDEX_JSON.exists():
        st.error(f"`{INDEX_JSON}` 가 없습니다.")
        st.stop()

    catalog = json.loads(INDEX_JSON.read_text(encoding="utf-8"))
    layouts = catalog.get("layouts", [])
    if not layouts:
        st.error("index.json에 레이아웃이 없습니다.")
        st.stop()

    # 현재 선택 상태(있다면) 표시
    cur_sel = st.session_state.get("selected_layout_id")
    if cur_sel:
        st.info(f"현재 선택된 레이아웃: **{cur_sel}**")

    # 갤러리 표시 & 선택
    cols = st.columns(3, gap="large")
    for i, item in enumerate(layouts):
        with cols[i % 3]:
            # index.json 내 값(thumb/preview/path)이 상대경로일 수 있으므로 절대경로로 정규화
            thumb_rel = item.get("thumb") or item.get("preview") or item["path"]
            thumb_abs = _abs_path(thumb_rel)

            # 파일 실제 존재 여부 체크(문제 시 경로 출력)
            if not os.path.exists(thumb_abs):
                st.error(f"썸네일을 찾을 수 없습니다:\n{thumb_abs}")
            else:
                st.image(thumb_abs, caption=item.get("name", item["id"]), width='stretch')

            if st.button(f"선택: {item['id']}", key=f"pick_{item['id']}", use_container_width=True):
                # 핵심: 선택 변경 → downstream 상태 초기화
                _reset_downstream_states()

                # 선택 동기화
                st.session_state["selected_layout_id"] = item["id"]
                st.session_state["selected_layout_path"] = _abs_path(item["path"])

                # index.json에 slots_mask가 있으면 함께 세션에 기록 (없으면 제거)
                slots_mask_rel = item.get("slots_mask") or item.get("slots_mask_path")
                if slots_mask_rel:
                    st.session_state["slots_mask_path"] = _abs_path(slots_mask_rel)
                else:
                    st.session_state.pop("slots_mask_path", None)

                # 피드백
                st.success(f"선택됨: {item['id']}")

    st.divider()

    # 하단 네비게이션: 이전=1페이지, 다음=3페이지(선택 시 활성화)
    has_selection = bool(st.session_state.get("selected_layout_id"))
    prev_col, next_col = st.columns([1, 1])
    with prev_col:
        st.button("← 이전: 1단계", use_container_width=True,
                  on_click=lambda: goto_poster_step(0))
    with next_col:
        st.button("다음 → 3단계", type="primary", disabled=not has_selection,
                  use_container_width=True, on_click=lambda: goto_poster_step(2))
    run_pending_nav()


if __name__ == "__main__":
    render()