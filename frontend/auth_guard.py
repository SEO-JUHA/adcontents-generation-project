# auth_guard.py
import streamlit as st
import time
import json
import base64
import os

def require_login(redirect="pages/00_login.py", remember_dest: bool = True, dest: str | None = None):
    """
    세션에 로그인 정보가 없으면 로그인 페이지로 보냄.
    - remember_dest=True 이면 로그인 후 돌아올 경로를 세션에 기록.
    - dest를 안 주면, 간단히 현재 페이지 이름만 저장하거나 생략해도 무방.
    """
    # 세션 유지를 위한 체크
    _check_session_persistence()
    
    user = st.session_state.get("auth_user")
    is_valid = isinstance(user, dict) and bool(user.get("id"))
    # if "auth_user" not in st.session_state:
    if not is_valid:
        st.warning("로그인이 필요합니다.")
        if remember_dest and dest:
            st.session_state["next_page"] = dest
        try:
            st.switch_page(redirect)
        except Exception:
            # 구버전 폴백: 페이지 이동이 안 되면 실행 중단으로 접근 차단
            st.stop()

def _check_session_persistence():
    """
    세션 유지를 위한 체크 함수
    - 세션에 로그인 정보가 있으면 유지
    - 파일에서 세션 복원 시도
    - 세션 만료 시간 체크만 수행 (자동 연장 제거)
    """
    # 현재 시간
    current_time = time.time()
    
    # 세션 상태가 없으면 파일에서 복원 시도
    if "auth_user" not in st.session_state:
        _restore_session_from_cookie()
    
    session_expiry = st.session_state.get("session_expiry", 0)
    if session_expiry > 0 and current_time > session_expiry:
        # 세션 만료
        st.session_state.pop("auth_user", None)
        st.session_state.pop("session_expiry", None)
        st.warning("세션이 만료되었습니다. 다시 로그인해주세요.")
        return
    
    # 세션 자동 연장 제거 - 새로고침 시에도 로그인 상태 유지

def set_session_persistence(user_data: dict):
    """
    로그인 성공 시 세션 유지 설정 (3시간)
    """
    st.session_state["auth_user"] = user_data
    st.session_state["session_expiry"] = time.time() + 10800
    
    # 쿠키에도 세션 정보 저장
    _save_session_to_cookie(user_data)

def _save_session_to_cookie(user_data: dict):
    """
    세션 정보를 파일에 저장 (새로고침 시에도 유지)
    """
    try:
        session_data = {
            "auth_user": user_data,
            "session_expiry": time.time() + 10800
        }
        session_json = json.dumps(session_data)
        session_b64 = base64.b64encode(session_json.encode()).decode()
        
        # 임시 파일에 세션 저장
        session_file = os.path.join(os.path.expanduser("~"), ".streamlit_session")
        with open(session_file, "w") as f:
            f.write(session_b64)
    except Exception:
        pass

def _restore_session_from_cookie():
    """
    파일에서 세션 정보 복원
    """
    try:
        session_file = os.path.join(os.path.expanduser("~"), ".streamlit_session")
        if os.path.exists(session_file):
            with open(session_file, "r") as f:
                session_b64 = f.read().strip()
            
            session_json = base64.b64decode(session_b64).decode()
            session_data = json.loads(session_json)
            
            # 세션 만료 체크
            current_time = time.time()
            session_expiry = session_data.get("session_expiry", 0)
            
            if session_expiry > 0 and current_time < session_expiry:
                # 세션이 유효하면 복원
                st.session_state["auth_user"] = session_data.get("auth_user")
                st.session_state["session_expiry"] = session_expiry
            else:
                # 세션 만료된 경우 파일 제거
                os.remove(session_file)
    except Exception:
        # 복원 실패 시 파일 제거
        try:
            session_file = os.path.join(os.path.expanduser("~"), ".streamlit_session")
            if os.path.exists(session_file):
                os.remove(session_file)
        except Exception:
            pass
