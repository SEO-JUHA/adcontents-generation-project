import streamlit as st

def goto_poster_step(step_idx: int):
    st.session_state["_nav_to"] = ("poster_step", int(step_idx)) # rerun 방지를위해 세션에 이동 기록

def go_to_home():
    st.session_state["_nav_to"] = ("home", None)

def run_pending_nav():
    """렌더링 끝에서 한 번 호출해 실제 전환 실행 (콜백 바깥)"""
    nav = st.session_state.pop("_nav_to", None)
    if not nav:
        return
    kind, arg = nav
    if kind == "home":
        st.switch_page("app.py")
    elif kind == "poster_step":
        st.session_state["poster_step"] = int(arg)
        st.query_params["step"] = str(int(arg))
        st.switch_page("pages/poster.py")

def flash(message: str, level: str = "info"):
    # level: "info" | "warning" | "success" | "error"
    st.session_state["flash"] = {"message": message, "level": level}

def consume_flash():
    data = st.session_state.pop("flash", None)
    if data:
        fn = getattr(st, data.get("level", "info"), st.info)
        fn(data.get("message", ""))