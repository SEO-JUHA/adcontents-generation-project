# app.py
import os
import sys, pathlib
import streamlit as st

# ==== 경로 세팅 ====
ROOT = pathlib.Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# (선택) 시스템 상태 모듈이 있으면 사용
try:
    from poster.src.comfy_api import system_stats  # 필요 없으면 주석 처리
except Exception:
    system_stats = None

# DB 유틸 (users 테이블, 회원가입/로그인)
from auth_db import init_db, create_user, verify_user
from auth_guard import set_session_persistence, _check_session_persistence

# ==== 공통 페이지 설정 ====
try:
    st.set_page_config(page_title="AI 콘텐츠 생성 스튜디오", page_icon="✨", layout="wide")
except Exception:
    # set_page_config는 세션당 1회만 가능
    pass


# ---------- 로그인 UI ----------
def render_login_ui():
    st.title("🔐 로그인 / 회원가입")

    tabs = st.tabs(["로그인", "회원가입"])

    # 로그인 탭
    with tabs[0]:
        st.subheader("로그인")
        with st.form("login_form", border=True):
            u = st.text_input("아이디")
            p = st.text_input("비밀번호", type="password")
            btn = st.form_submit_button("로그인", type="primary", use_container_width=True)

        if btn:
            username = u.strip()
            success, user_id = verify_user(username, p)
            if success:
                # 영속 세션 저장
                set_session_persistence({"id": user_id, "username": username})

                # ✅ 다른 페이지(예: generate.py)에서 바로 쓰도록 표준 키도 저장
                st.session_state["auth_user"] = {"id": user_id, "username": username}
                st.session_state["user_id"] = user_id
                st.session_state["username"] = username

                st.success("로그인 성공! 잠시만요…")
                st.rerun()  # 최신 API
            else:
                st.error("아이디 또는 비밀번호가 올바르지 않습니다.")

    # 회원가입 탭
    with tabs[1]:
        st.subheader("회원가입")
        with st.form("signup_form", border=True):
            u2 = st.text_input("아이디 (3자 이상)")
            p1 = st.text_input("비밀번호 (6자 이상)", type="password")
            p2 = st.text_input("비밀번호 확인", type="password")
            sbtn = st.form_submit_button("회원가입", type="primary", use_container_width=True)

        if sbtn:
            if p1 != p2:
                st.error("비밀번호 확인이 일치하지 않습니다.")
            else:
                ok, msg = create_user(u2.strip(), p1)
                st.success(msg) if ok else st.error(msg)


# ---------- 로그인 후 홈 화면 ----------
def render_main_home():
    user = st.session_state["auth_user"]

    # 상단/사이드바
    st.sidebar.success(f"안녕하세요, {user['username']} 님 👋")

    # 로그아웃 버튼
    col_sp, col_btn = st.columns([1, 0.16])
    with col_btn:
        if st.button("로그아웃", use_container_width=True):
            # 세션 상태 정리
            st.session_state.pop("auth_user", None)
            st.session_state.pop("session_expiry", None)

            # ✅ 표준 키도 정리
            st.session_state.pop("user_id", None)
            st.session_state.pop("username", None)

            # 세션 파일도 제거
            session_file = os.path.join(os.path.expanduser("~"), ".streamlit_session")
            try:
                if os.path.exists(session_file):
                    os.remove(session_file)
            except Exception:
                pass

            st.rerun()  # 최신 API

    # 원래 app.py가 보여주던 내용
    st.title("✨ AI 콘텐츠 생성 스튜디오에 오신 것을 환영합니다!")
    st.sidebar.success("위에서 작업할 메뉴를 선택하세요.")

    st.markdown(
        """
        이 앱은 소상공인을 위한 AI 기반 콘텐츠 생성 도구입니다.
        왼쪽 사이드바에서 원하는 작업을 선택하여 시작하세요.

        ### 제공 기능:
        - **🚀 로고 생성**: 당신의 브랜드를 위한 독창적인 로고를 만듭니다.
        - **📱 인스타그램 게시물 생성**: 시선을 사로잡는 SNS 게시물 이미지와 캡션을 생성합니다.
        - **🎨 포스터 생성**: 이벤트나 신제품을 위한 홍보 포스터를 디자인합니다.
        - **📷 광고 네컷**: 브랜드 감성 맞춤 4컷 SNS 광고, AI로 손쉽게 제작합니다.

        **👈 사이드바에서 메뉴를 선택하여 시작하세요!**
        """
    )

    # (선택) 시스템 상태 표시
    if system_stats:
        with st.expander("🖥️ 시스템 상태", expanded=False):
            try:
                stats = system_stats()
                st.json(stats)
            except Exception as e:
                st.caption(f"(system_stats 호출 실패: {e})")


# ---------- 엔트리 ----------
def main():
    # users 테이블 보장 (최초 1회)
    init_db()

    # 세션 유지 체크(영속 세션이 있으면 auth_user 복원)
    _check_session_persistence()

    # 영속 세션에서 복원된 auth_user가 있으면 표준 키도 동기화
    if "auth_user" in st.session_state and isinstance(st.session_state["auth_user"], dict):
        au = st.session_state["auth_user"]
        if "user_id" not in st.session_state and ("id" in au):
            st.session_state["user_id"] = au["id"]
        if "username" not in st.session_state and ("username" in au):
            st.session_state["username"] = au["username"]

    # 세션 상태에 따라 로그인/홈 전환
    if "auth_user" not in st.session_state:
        render_login_ui()
    else:
        render_main_home()


if __name__ == "__main__":
    main()
