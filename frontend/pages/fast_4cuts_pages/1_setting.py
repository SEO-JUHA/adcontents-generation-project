TITLE = "업로드 & 설정"

import streamlit as st
from fourcuts_shared import (
    ensure_tmp_dir, save_many,
    save_logo_to_comfyui_input, save_logo_from_existing_path,
    goto, _post, _get, file_url
)

def render():
    st.set_page_config(page_title="업로드 & 설정", layout="wide", initial_sidebar_state="expanded")
    st.caption("브랜드 소개/핵심 메시지와 이미지(로고/메뉴/매장)를 업로드하세요.")
    tmp_dir = ensure_tmp_dir()

    brand_bi_def = st.session_state.get("brand_bi", "")
    core_msg_def = st.session_state.get("core_msg", "시그니처 블렌드 출시!")
    user = st.session_state.get("auth_user") or {}
    user_id = user.get("id")

    server_logo_path = st.session_state.get("server_logo_path")

    with st.form("cfg_form_fast"):
        colA, colB = st.columns(2)

        with colA:
            brand_bi = st.text_area("브랜드 소개(BI)", value=brand_bi_def)
            core_msg = st.text_input("핵심 메시지", value=core_msg_def)

        with colB:
            default_toggle = st.session_state.get("_use_saved_logo", bool(server_logo_path))
            use_saved_logo = st.toggle(
                "저장된 로고 사용",
                value=default_toggle,
                help="ON이면 DB에 저장된 로고를 자동으로 불러와 사용합니다. OFF면 새 파일을 업로드하세요.",
                disabled=not bool(user_id)
            )
            st.session_state["_use_saved_logo"] = use_saved_logo

            if use_saved_logo and user_id and not server_logo_path:
                with st.spinner("서버 저장 로고를 불러오는 중..."):
                    try:
                        res = _get("/profiles/logo", user_id=str(user_id))
                        server_logo_path = res.get("logo_path")
                        if server_logo_path:
                            st.session_state["server_logo_path"] = server_logo_path
                        else:
                            st.info("서버에 저장된 로고가 없습니다. 스위치를 끄고 새 파일을 업로드하세요.")
                    except Exception as e:
                        st.warning(f"로고 조회 실패: {e}")

            disable_uploader = bool(use_saved_logo and server_logo_path)
            logo_file = st.file_uploader("로고 첨부", type=["png","jpg","jpeg","webp"], disabled=disable_uploader)

            if use_saved_logo and server_logo_path:
                st.caption("서버 저장 로고 미리보기")
                st.image(file_url(server_logo_path), width=240)

            menu_files = st.file_uploader("메뉴 이미지들 첨부", type=["png","jpg","jpeg","webp"], accept_multiple_files=True)
            store_files= st.file_uploader("매장 이미지들 첨부", type=["png","jpg","jpeg","webp"], accept_multiple_files=True)

        ok = st.form_submit_button("저장")

    if ok:
        logo = None
        try:
            if use_saved_logo:
                if not server_logo_path:
                    st.error("‘저장된 로고 사용’이 켜져 있지만 서버 로고가 없습니다. 스위치를 끄고 새 파일을 업로드하세요.")
                    return
                logo = save_logo_from_existing_path(server_logo_path, user_id, tmp_dir)
            else:
                if logo_file is not None:
                    logo = save_logo_to_comfyui_input(logo_file, user_id, tmp_dir)
        except Exception as e:
            st.warning(f"로고 처리 중 문제: {e}")

        menus = save_many(menu_files, tmp_dir)
        store = save_many(store_files, tmp_dir)

        if not (menus or store or logo):
            st.error("최소 한 장 이상의 이미지를 업로드해 주세요.")
            return

        st.session_state.update({
            "brand_bi": brand_bi,
            "core_msg": core_msg,
            "layout_id": "default_2x2",
            "seed": None,
            "images": {"logo": logo, "menus": menus, "store": store, "layout_id": "default_2x2"},
        })

        if user_id and logo:
            try:
                _post("/profiles/save-logo", {"user_id": str(user_id), "logo_path": logo})
            except Exception as e:
                st.warning(f"서버에 로고 경로 저장 실패: {e}")

        st.success("설정을 저장했어요.")
        goto(+1)