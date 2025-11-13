
TITLE = "미리보기 & 다운로드"

import os
import traceback
from pathlib import Path
import streamlit as st
from fourcuts_shared import (
    require_bases, ensure_tmp_dir, zip_paths, make_collage,
    gallery_2x2, ensure_thumb_px, _post
)

def _open_bytes(path: str) -> bytes:
    # URL인 경우 requests로 다운로드
    if path.startswith(('http://', 'https://')):
        import requests
        response = requests.get(path)
        response.raise_for_status()
        return response.content
    # 로컬 파일인 경우
    from fourcuts_shared import _normalize_path
    normalized_path = _normalize_path(path)
    
    if not os.path.exists(normalized_path):
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {normalized_path}")
    return Path(normalized_path).read_bytes()

def _upscale_paths_via_api(paths, long_side: int) -> list[str]:
    out = []
    for p in paths:
        try:
            res = _post("/upscaler/if-needed", {"path": p, "final_wh": [int(long_side), int(long_side)]})
            out.append(res.get("out_path", p))
        except Exception:
            out.append(p)  # fallback
    return out

def render():
    if not require_bases():
        return

    st.caption("최신 오버레이 결과를 확인하고, 개별 또는 콜라주(4컷 합성)·ZIP로 다운로드할 수 있습니다. (FastAPI 사용)")

    imgs = st.session_state.get("last_panels") or st.session_state["base_panels"]

    st.markdown("#### 최종 이미지")
    with st.expander("미리보기 설정", expanded=False):
        cur = ensure_thumb_px()
        thumb = st.slider("썸네일 너비(px)", 160, 360, value=cur, step=10)
        st.session_state["thumb_px"] = int(thumb)
    gallery_2x2(imgs, width=ensure_thumb_px())

    cols = st.columns(4, gap="small")
    for i, (col, p) in enumerate(zip(cols, imgs), start=1):
        with col:
            try:
                st.download_button(
                    label="다운로드",
                    data=_open_bytes(p),
                    file_name=Path(p).name,
                    mime="image/jpeg",
                    use_container_width=True
                )
            except FileNotFoundError as e:
                st.error(f"파일 없음: {Path(p).name}")
                st.info("이미지 파일이 생성되지 않았습니다.")
            except Exception as e:
                st.error(f"다운로드 실패: {e}")
                st.info("파일을 읽을 수 없습니다.")

    st.divider()

    st.markdown("#### 4컷 콜라주 만들기 (2×2)")
    left, right = st.columns([2,1])
    with left:
        with st.expander("옵션", expanded=True):
            caption_mode = st.radio(
                "콜라주에 캡션 포함 여부",
                options=["캡션 포함", "캡션 없음"],
                horizontal=True,
                index=0 if st.session_state.get("last_panels") else 1
            )
            side = st.number_input("정사각 사이즈(px) — 업스케일 겸용", min_value=1024, max_value=4096, value=2160, step=128)
            pad = st.slider("패널 간격(px)", min_value=0, max_value=64, value=16, step=2)

        want_captioned = (caption_mode == "캡션 포함")

        if st.button("콜라주 생성", use_container_width=True):
            try:
                path = make_collage(captioned=want_captioned, side=int(side), pad=int(pad))
                if path:
                    st.session_state["collage_path"] = path
                    st.success("콜라주 이미지를 생성했습니다.")
                else:
                    st.warning("이미지가 없어 콜라주를 만들 수 없습니다.")
            except Exception as e:
                st.error(f"콜라주 생성 실패: {e}")
                st.code(traceback.format_exc())

    with right:
        cp = st.session_state.get("collage_path")
        if cp and Path(cp).exists():
            st.image(cp, caption=Path(cp).name, use_container_width=True)
            st.download_button(
                label="콜라주 다운로드",
                data=_open_bytes(cp),
                file_name=Path(cp).name,
                mime="image/jpeg",
                use_container_width=True
            )
        else:
            st.info("왼쪽에서 ‘콜라주 생성’을 눌러주세요.")

    st.divider()

    st.markdown("#### 일괄 ZIP 다운로드")
    zip_target = st.radio(
        "ZIP에 포함할 이미지",
        ["최신 4컷(오버레이/없으면 베이스)", "베이스 4컷"],
        horizontal=True
    )
    targets = imgs if zip_target.startswith("최신") else st.session_state["base_panels"]

    up_en = st.checkbox("패널 업스케일 후 ZIP 생성", value=False)
    up_side = st.number_input("업스케일 해상도(긴 변 px)", min_value=1024, max_value=4096, value=2160, step=128, disabled=not up_en)

    if st.button("ZIP 만들기", use_container_width=True):
        try:
            targets_use = _upscale_paths_via_api(targets, long_side=int(up_side)) if up_en else targets
            tmp = ensure_tmp_dir()
            out_zip = tmp / ("fourcuts_latest.zip" if zip_target.startswith("최신") else "fourcuts_bases.zip")
            zip_paths(targets_use, out_zip)
            st.session_state["zip_latest"] = str(out_zip)
            st.success("ZIP 파일을 만들었습니다.")
        except Exception as e:
            st.error(f"ZIP 생성 실패: {e}")
            st.code(traceback.format_exc())

    if st.session_state.get("zip_latest"):
        zp = Path(st.session_state["zip_latest"])
        if zp.exists():
            st.download_button(
                "ZIP 다운로드",
                data=zp.read_bytes(),
                file_name=zp.name,
                mime="application/zip",
                use_container_width=True
            )
