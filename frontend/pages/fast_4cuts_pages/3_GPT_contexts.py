
TITLE = "문구 & 오버레이"

import streamlit as st
import traceback
import os
from fourcuts_shared import (
    require_bases, prefill_caps_from_suggest,
    overlay_all_from_bases, goto, gallery_2x2, _post, _normalize_path
)

def render():
    if not require_bases():
        return

    st.caption("베이스 이미지를 기준으로 문구를 작성/수정하고, 바로 오버레이까지 처리합니다. (FastAPI 사용)")

    prefill_caps_from_suggest()

    base_imgs = st.session_state["base_panels"]
    cols = st.columns(4, gap="medium")
    for i, col in enumerate(cols, start=1):
        with col:
            try:
                normalized_path = _normalize_path(base_imgs[i-1])
                
                if os.path.exists(normalized_path):
                    st.image(normalized_path, caption=f"Panel {i}", use_container_width=True)
                else:
                    st.error(f"파일을 찾을 수 없습니다: Panel {i}")
                    st.info(f"원본 경로: {base_imgs[i-1]}")
                    st.info(f"변환된 경로: {normalized_path}")
            except Exception as e:
                st.error(f"이미지 로드 실패: {e}")
                st.info(f"원본 경로: {base_imgs[i-1] if i-1 < len(base_imgs) else 'N/A'}")
                st.code(traceback.format_exc())
            
            st.text_area(f"Panel {i} caption", key=f"cap_{i}", height=80)

    c1, c2, c3 = st.columns([1,1,1])

    with c1:
        if st.button("GPT 추천 문구 재생성", use_container_width=True):
            try:
                with st.spinner("추천 문구 생성 중..."):
                    sugg = _post("/story/suggest", {
                        "brand_intro": st.session_state.get("brand_bi",""),
                        "core_message": st.session_state.get("core_msg",""),
                        "panel_image_paths": base_imgs,
                        "language": "ko",
                        "max_chars": 42,
                    }).get("lines", [])
                st.session_state["suggested_caps"] = sugg
                st.session_state["_refill_caps"] = True
                st.success("추천 문구를 재생성했어요.")
                st.rerun()
            except Exception as e:
                st.error(f"GPT 추천 문구 생성 실패: {e}")
                st.code(traceback.format_exc())

    with c2:
        if st.button("현재 문구로 베이스에 오버레이", use_container_width=True):
            try:
                caps = [st.session_state.get(f"cap_{i}", f"Panel {i}") for i in range(1,5)]
                outs = overlay_all_from_bases(caps)
                st.session_state["last_panels"] = outs
                st.success("베이스 기준으로 오버레이했습니다.")
                st.rerun()
            except Exception as e:
                st.error(f"오버레이 실패: {e}")
                st.code(traceback.format_exc())

    with c3:
        st.button("다음 단계 → 미리보기/다운로드", on_click=lambda: goto(+1), use_container_width=True)

    st.divider()
    st.markdown("### 현재 결과 미리보기 (2×2)")
    gallery_2x2(
        st.session_state.get("last_panels", base_imgs),
        gap="medium",
    )
