import sys, os, pathlib, time
import streamlit as st
from PIL import Image, ImageFile

ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ImageFile.LOAD_TRUNCATED_IMAGES = True

from poster.src.poster_utils import goto_poster_step, flash, run_pending_nav
from poster.src.adposter_diffusers import load_grid_cfg, _run_single_config_multi_designs


def render():
    st.set_page_config(page_title="3) 생성·선택", layout="wide")
    st.title("3) 생성 · 선택")

    logo_path   = st.session_state.get("logo_path")
    brand_intro = st.session_state.get("brand_intro")
    layout_path = st.session_state.get("selected_layout_path")
    slots_mask  = st.session_state.get("slots_mask_path")

    if not (logo_path and brand_intro and layout_path):
        try:
            flash("먼저 레이아웃을 선택하세요.", level="warning")
            goto_poster_step(1)
            run_pending_nav()
        except Exception:
            pass
        st.stop()

    # --- 입력 변경 감지: 레이아웃/로고/브리프가 바뀌면 이전 생성물/선택 초기화 ---
    def _fp_stat(p):
        try:
            return f"{p}:{int(os.path.getmtime(p))}" if (p and os.path.exists(p)) else str(p)
        except Exception:
            return str(p)

    cur_gen_fp = "|".join([
        _fp_stat(layout_path),
        _fp_stat(logo_path),
        str(hash(brand_intro)),
    ])
    if st.session_state.get("gen_page_fp") != cur_gen_fp:
        st.session_state["gen_page_fp"] = cur_gen_fp
        st.session_state.pop("generated_paths", None)
        st.session_state.pop("chosen_path", None)
        st.session_state.pop("last_design_fp", None)
        try:
            flash("입력(레이아웃/로고/브리프) 변경을 감지했습니다. 새로 생성해주세요.", level="info")
        except Exception:
            st.info("입력 변경을 감지했습니다. 새로 생성해주세요.")

    grid_cfg = load_grid_cfg()
    st.session_state["grid_cfg"] = grid_cfg

    def show_image_safe(path: str):
        for _ in range(3):
            try:
                with Image.open(path) as im:
                    im.load()
                    st.image(im, width='stretch')
                    return
            except Exception:
                time.sleep(0.25)
        try:
            with Image.open(path) as im:
                im = im.convert("RGBA")
                st.image(im, width='stretch')
        except Exception:
            st.warning(f"이미지를 표시할 수 없습니다: {os.path.basename(path)}")

    def _first(arr, default=None):
        try:
            return (arr or [default])[0]
        except Exception:
            return default

    def _recipe_from_grid(gc):
        return dict(
            sampler_name=_first(gc.get("sampler_name"), "dpmpp_2m"),
            scheduler=_first(gc.get("scheduler"), "karras"),
            steps=int(_first(gc.get("steps"), 22)),
            cfg=float(_first(gc.get("cfg"), 7.0)),
            ip_w=float(_first(gc.get("ipadapter_weight"), 1.0)),
        )

    def _control_from_grid(gc):
        cs = gc.get("control_strength_defaults", [0.8])
        if isinstance(cs, (list, tuple)):
            if len(cs) == 1:
                return float(cs[0])
            if len(cs) >= 2:
                lo, hi = float(cs[0]), float(cs[1])
                return (lo + hi) / 2.0
        return float(cs) if isinstance(cs, (int, float)) else 0.8

    LOCK_UI = bool(grid_cfg.get("lock_ui")) or bool(os.getenv("POSTER_LOCK_UI"))
    SHOW_ADVANCED = bool(int(os.getenv("POSTER_SHOW_ADVANCED", "0")))

    recipe  = _recipe_from_grid(grid_cfg)
    ctrl_strength = _control_from_grid(grid_cfg)
    num_designs = int(grid_cfg.get("designs_per_layout", 3))

    st.caption("선택한 레이아웃과 BI로 이미지를 생성합니다. 이 페이지는 생성 및 선택까지만 진행합니다.")

    base_seed = None

    if LOCK_UI:
        st.info("운영 모드: **고정 레시피(편집 불가)**")
        if SHOW_ADVANCED:
            with st.expander("고급 설정 미리보기", expanded=False):
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("Sampler", recipe["sampler_name"]) ; st.metric("Steps", recipe["steps"])
                with c2:
                    st.metric("Scheduler", recipe["scheduler"]) ; st.metric("CFG", recipe["cfg"])
                with c3:
                    st.metric("IP-Adapter weight", recipe["ip_w"]) ; st.metric("Control strength", ctrl_strength)
                st.metric("디자인 개수", num_designs)
        do_generate = st.button("디자인 생성하기", type="primary", width='stretch')
    else:
        show_form = st.toggle("고급 설정 열기 (편집 가능)", value=SHOW_ADVANCED)
        if show_form:
            with st.form("gen_form"):
                st.subheader("생성 설정 (편집 가능)")
                c1, c2, c3 = st.columns(3)
                with c1:
                    num_designs = st.number_input("디자인 개수", 1, 12, num_designs, 1)
                    seed_base = st.number_input("기준 시드(옵션, 0=무작위)", 0, 999999999, 0, 1)
                with c2:
                    recipe["sampler_name"] = st.selectbox("Sampler", ["dpmpp_2m", "euler", "lcm"], index=["dpmpp_2m","euler","lcm"].index(recipe["sampler_name"]) if recipe["sampler_name"] in ["dpmpp_2m","euler","lcm"] else 0)
                    recipe["scheduler"]    = st.selectbox("Scheduler", ["karras"], index=0)
                with c3:
                    recipe["steps"] = st.number_input("Steps", 1, 80, int(recipe["steps"]), 1)
                    recipe["cfg"]   = st.number_input("CFG", 0.0, 20.0, float(recipe["cfg"]), 0.5)
                recipe["ip_w"] = st.slider("IP-Adapter weight", 0.0, 2.0, float(recipe["ip_w"]), 0.05)
                ctrl_strength  = st.slider("ControlNet strength", 0.0, 1.5, float(ctrl_strength), 0.05)
                do_generate = st.form_submit_button("디자인 생성하기", width='stretch')
            base_seed = int(seed_base) if show_form and seed_base and seed_base > 0 else None
        else:
            do_generate = st.button("디자인 생성하기", type="primary", width='stretch')

    if do_generate:
        try:
            with st.spinner("생성 중… 잠시만 기다려 주세요."):
                results = _run_single_config_multi_designs(
                    logo_path=str(logo_path),
                    brand_intro=brand_intro,
                    layout_path=str(layout_path),
                    grid_cfg=grid_cfg,
                    recipe=recipe,
                    control_strength=float(ctrl_strength),
                    num_designs=int(num_designs),
                    base_seed=base_seed if not LOCK_UI else None,
                    upscale=False,
                )
            st.session_state["generated_paths"] = results
            st.success(f"생성 완료! ({len(results)}장)")
        except Exception as e:
            st.error(f"생성 실패: {e}")

    st.divider()

    paths = st.session_state.get("generated_paths", [])
    if not paths:
        st.info("아직 생성된 이미지가 없습니다. 상단에서 ‘디자인 생성하기’를 눌러주세요.")
    else:
        st.subheader("생성 결과 (클릭해서 선택)")
        cols = st.columns(3, gap="large")
        for i, p in enumerate(paths):
            with cols[i % 3]:
                if os.path.exists(p):
                    show_image_safe(p)
                else:
                    st.write(p)
                if st.button("이 이미지 선택", key=f"choose_{i}", width='stretch'):
                    st.session_state["chosen_path"] = p
                    st.session_state["selected_layout_path"] = layout_path
                    if slots_mask:
                        st.session_state["slots_mask_path"] = slots_mask
                    st.session_state["chosen_version"] = st.session_state.get("chosen_version", 0) + 1
                    st.success("선택됨")

    st.divider()
    col_prev, col_next = st.columns(2)
    with col_prev:
        go_prev = st.button("← 이전: 2) 레이아웃 갤러리", type="secondary", width='stretch')
        if go_prev:
            goto_poster_step(1)
            run_pending_nav()
            
    with col_next:
        go_next = st.button("다음: 4) 후처리 · 편집", type="primary", width='stretch',
                            disabled=not bool(st.session_state.get("chosen_path")))
        if go_next:
            if slots_mask:
                st.session_state["slots_mask_path"] = slots_mask
            goto_poster_step(4)
            run_pending_nav()


if __name__ == "__main__":
    render()