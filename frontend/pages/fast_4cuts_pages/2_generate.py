# <frontend/pages/fast_4cuts_pages/2_generate.py>
TITLE = "이미지 생성"

import os
import traceback
import streamlit as st
from fourcuts_shared import _post, _get, require_inputs, read_manifest, goto, ensure_thumb_px, _normalize_path


# helpers: images 스키마 변환/검증
def _coerce_images_to_schema(images_obj):
    """
    기존 구조(logo/menus/store...)를 서버 요구 스키마(images: dict[str,str])로 변환.
    우선순위: menus[0..2] -> panel1..3, store[0] -> panel4, logo -> logo
    이미 dict[str,str] 형태면 그대로 반환.
    """
    if isinstance(images_obj, dict) and images_obj and all(isinstance(v, str) for v in images_obj.values()):
        return dict(images_obj)

    out = {}
    logo_path = None
    menus = []
    store = []

    if isinstance(images_obj, dict):
        logo_path = images_obj.get("logo")
        menus = images_obj.get("menus") or []
        store = images_obj.get("store") or []

    # panel1~3: menus에서 채우기
    if len(menus) > 0: out["panel1"] = menus[0]
    if len(menus) > 1: out["panel2"] = menus[1]
    if len(menus) > 2: out["panel3"] = menus[2]

    # panel4: store가 있으면 하나 사용
    if len(store) > 0: out["panel4"] = store[0]

    # logo
    if logo_path: out["logo"] = logo_path

    return out


def _assert_files_readable(images_map: dict):
    missing = []
    for k, p in images_map.items():
        if k == "layout_id":
            continue
        if not p or not isinstance(p, str) or not os.path.isabs(p) or not os.path.exists(p):
            missing.append((k, p, "not_exists_or_not_abs"))
            continue
        if not os.access(p, os.R_OK):
            missing.append((k, p, "no_read_permission"))
            continue
        try:
            with open(p, "rb"):
                pass
        except Exception as e:
            missing.append((k, p, f"open_failed: {e}"))
    return missing


def _default_captions(core_message: str):
    """서버가 4줄 캡션을 기대할 수 있어 기본 4줄을 제공(필요시 제거 가능)."""
    cm = (core_message or "").strip() or "신메뉴 출시!"
    return [cm, "오늘만 10% OFF", "따뜻한 라떼와 스콘", "블루문 카페에서 만나요"][:4]


def render():
    st.caption("입력값으로 4컷 베이스 이미지를 생성합니다.")

    # 이전 이미지 불러오기 버튼
    user_id = (st.session_state.get("auth_user") or {}).get("id")

    col_load, col_info = st.columns([1, 2])

    with col_load:
        if st.button("🔄 이전 이미지 불러오기", use_container_width=True, disabled=not bool(user_id),
                     help="DB에 저장된 최근 4개 패널 이미지를 불러옵니다"):
            try:
                with st.spinner("이전 이미지 불러오는 중..."):
                    res = _get("/profiles/recent-images", user_id=str(user_id), limit=4)
                    images = res.get("images", [])

                    if images:
                        # 최근 4개를 정렬 (panel_01, 02, 03, 04 순으로 추정되는 이름이면 정렬이 유의미)
                        images_sorted = sorted(images, key=lambda x: x)

                        st.session_state["base_panels"] = images_sorted
                        st.session_state["last_panels"] = images_sorted[:]
                        st.session_state["grid_image"] = None

                        st.session_state.setdefault("brand_bi", "불러온 이미지")
                        st.session_state.setdefault("core_msg", "이전 작업")
                        st.session_state.setdefault("layout_id", "default_2x2")

                        if not st.session_state.get("images"):
                            st.session_state["images"] = {
                                "panel1": images_sorted[0] if len(images_sorted) > 0 else None,
                                "panel2": images_sorted[1] if len(images_sorted) > 1 else None,
                                "panel3": images_sorted[2] if len(images_sorted) > 2 else None,
                                "logo":   None,
                            }

                        st.success(f"이전 이미지 {len(images_sorted)}개를 불러왔어요! (생성: {res.get('created_at', '알 수 없음')})")
                        st.rerun()
                    else:
                        st.warning("저장된 이미지가 없습니다. 먼저 이미지를 생성해주세요.")
            except Exception as e:
                st.error(f"불러오기 실패: {e}")
                st.code(traceback.format_exc())

    with col_info:
        if not user_id:
            st.info("💡 로그인하면 이전 작업을 불러올 수 있습니다.")
        else:
            st.info("💡 새로고침 후 이전 이미지를 빠르게 복원할 수 있습니다.")

    st.divider()

    # 설정 확인 (새로 생성할 때만 필요)
    has_inputs = bool(st.session_state.get("brand_bi") and st.session_state.get("images"))
    if not has_inputs:
        st.warning("⚠️ 새로 생성하려면 먼저 '업로드 & 설정' 페이지에서 정보를 입력하세요.")
        st.markdown("**또는 위의 '📁 이전 이미지 불러오기' 버튼을 클릭하세요.**")
        return

    with st.expander("미리보기 설정", expanded=False):
        cur = ensure_thumb_px()
        thumb = st.slider("썸네일 너비(px)", 160, 360, value=cur, step=10)
        st.session_state["thumb_px"] = int(thumb)

    if st.button("생성하기", use_container_width=True):
        try:
            with st.spinner("생성 중..."):
                # 1) images 변환
                raw_images = st.session_state.get("images", {})
                images_map = _coerce_images_to_schema(raw_images)

                # 2) layout_id 주입 (★ 서버 요구사항)
                layout_id = st.session_state.get("layout_id") or "default_2x2"
                images_map["layout_id"] = layout_id

                # 3) None/빈값 제거(단, layout_id는 보존)
                images_map = {k: v for k, v in images_map.items() if (k == "layout_id") or (v)}

                # 4) 파일 존재/읽기 권한 검증
                missing = _assert_files_readable(images_map)
                if missing:
                    st.error("다음 파일에 접근할 수 없습니다:")
                    for k, p, why in missing:
                        st.write(f"- {k}: {p} ({why})")
                    st.info("경로/권한(디렉터리 x 권한 포함)을 확인해주세요.")
                    return

                # 5) payload 생성 (서버 스키마에 맞춤)
                core_msg = f"{st.session_state.get('brand_bi','').strip()} | {st.session_state.get('core_msg','').strip()}".strip(" |")
                payload = {
                    "core_message": core_msg or "기본 메시지",
                    "images": images_map,
                    "captions": _default_captions(core_msg),
                    "seed": None,
                    "upscale": False,
                    "make_grid": False,
                    "grid_side": 2160,
                    "grid_pad_px": 16,
                }

                # 6) 호출
                res = _post("/cartoon/generate-4cut-from-assets", payload)

        except Exception as e:
            st.error(f"생성 실패: {e}")
            st.code(traceback.format_exc())
            return

        # 디버깅: 요청/응답 확인
        with st.expander("🔍 요청/응답 확인 (디버깅)", expanded=False):
            st.subheader("Request Payload")
            try: st.json(payload)
            except Exception: st.write(payload)
            st.subheader("Response JSON")
            try: st.json(res)
            except Exception: st.write(res)

        st.session_state["manifest"] = res.get("manifest_path")
        mf = read_manifest(st.session_state["manifest"]) if st.session_state.get("manifest") else {}

        # 우선순위: API 응답의 panel_bases > manifest의 panel_bases > API 응답의 panel_images
        base_paths = res.get("panel_bases") or mf.get("panel_bases") or res.get("panel_images") or []

        # 디버깅: 경로 확인
        with st.expander("🔍 이미지 경로 확인 (디버깅)", expanded=False):
            st.write("**추출된 경로들:**")
            for i, p in enumerate(base_paths, 1):
                exists = os.path.exists(p) if p else False
                st.write(f"Panel {i}: `{p}` - 존재: {'✅' if exists else '❌'}")

        st.session_state["base_panels"] = base_paths
        st.session_state["last_panels"] = base_paths[:]
        st.session_state["grid_image"] = None
        st.success("생성이 완료됐어요.")
        goto(+1)

    # 미리보기(베이스)
    if st.session_state.get("last_panels"):
        st.markdown("### 미리보기 (베이스)")
        cols = st.columns(4, gap="small")
        tpx = ensure_thumb_px()
        for i, (col, p) in enumerate(zip(cols, st.session_state["last_panels"]), start=1):
            with col:
                try:
                    normalized_path = _normalize_path(p)
                    if os.path.exists(normalized_path):
                        st.image(normalized_path, caption=f"Panel {i}", width=tpx)
                    else:
                        st.error(f"파일을 찾을 수 없습니다: Panel {i}")
                        st.info(f"원본 경로: {p}")
                        st.info(f"변환된 경로: {normalized_path}")
                        st.info("이미지가 아직 생성 중이거나 경로가 잘못되었습니다.")
                except Exception as e:
                    st.error(f"이미지 로드 실패: {e}")
                    st.info(f"원본 경로: {p}")
                    st.code(traceback.format_exc())