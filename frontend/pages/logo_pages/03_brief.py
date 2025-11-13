# pages/logo_pages/brief.py
# TITLE: 📋 Step 3/4. Brief — 요구사항 입력 (스케치/마스크 감지 → SDXL 프롬프트 자동 분기 + 팔레트 색상 제한 + 컨트롤 힌트)

from __future__ import annotations

TITLE = "📋 Step 3/4. Brief — 요구사항 입력"

import os, io, json, base64, uuid, colorsys, time, re
from typing import List, Optional, Tuple, Dict, Any
from pydantic import BaseModel, Field
from PIL import Image, ImageOps
import streamlit as st
import requests

# =============================
# 환경 변수
# =============================
BACKEND_BASE      = os.environ.get("LOGO_BACKEND_URL", "http://127.0.0.1:8000")
BRIEF_ENDPOINT    = os.environ.get("LOGO_BRIEF_ENDPOINT", "/logo/briefs")
GENERATE_ENDPOINT = os.environ.get("LOGO_GENERATE_ENDPOINT", "/logo/generate")
JOB_ENDPOINT      = os.environ.get("LOGO_JOB_ENDPOINT", "/logo/generate/{job_id}")
BRIEF_POST_URL    = f"{BACKEND_BASE.rstrip('/')}{BRIEF_ENDPOINT}"
GENERATE_URL      = f"{BACKEND_BASE.rstrip('/')}{GENERATE_ENDPOINT}"
NEXT_PAGE_PATH    = os.environ.get("LOGO_NEXT_PAGE", "pages/logo_pages/04_generate.py")

# =============================
# 유틸
# =============================
def _k(name: str) -> str:
    return f"brief::{name}"

def file_to_pil(uploaded) -> Image.Image:
    img = Image.open(uploaded).convert("RGB")
    return ImageOps.exif_transpose(img)

def pil_to_b64_png(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

@st.cache_data(show_spinner=False)
def extract_palette(img: Image.Image, n: int = 5) -> List[str]:
    im = img.copy()
    im.thumbnail((200, 200))
    colors = im.getcolors(maxcolors=2_000_000) or []
    if not colors:
        return ["#000000", "#FFFFFF"]
    colors.sort(key=lambda x: x[0], reverse=True)
    hexes, seen = [], set()
    for _, rgb in colors:
        r, g, b = rgb[:3]
        h, s, v = colorsys.rgb_to_hsv(r/255.0, g/255.0, b/255.0)
        if (v > 0.97 and s < 0.10) or (v < 0.08) or (s < 0.08):
            continue
        hcode = f"#{r:02X}{g:02X}{b:02X}"
        if hcode not in seen:
            hexes.append(hcode); seen.add(hcode)
        if len(hexes) >= n: break
    if not hexes:
        hexes = ["#3B3B3B", "#BEA38A", "#6B4F36", "#C8B09A", "#8F6D52"][:n]
    return hexes

def render_palette_swatches(hexes: List[str]):
    cols = st.columns(len(hexes))
    for c, h in zip(cols, hexes):
        with c:
            st.markdown(
                f"""
                <div style="border-radius:10px;border:1px solid #ddd;height:40px;background:{h};"></div>
                <div style="text-align:center;font-size:12px;margin-top:6px;">{h}</div>
                """,
                unsafe_allow_html=True,
            )

def _nav_to_next_step():
    try:
        st.switch_page(NEXT_PAGE_PATH); return
    except Exception:
        pass
    try:
        st.query_params.update({"step":"4"})
    except Exception:
        st.experimental_set_query_params(step="4")
    finally:
        st.session_state["logo_step"] = 4
        st.rerun()

# =============================
# 데이터 모델
# =============================
class PromptBrief(BaseModel):
    cafe_name: str = Field(description="카페명(브랜드명)")
    copy_text: str = Field(description="생성하고 싶은 텍스트")
    layout: str    = Field(description="배경/구도")
    avoid: str     = Field(description="피해야 할 것")
    strengths: str = Field(description="핵심 장점")
    style: str     = Field(description="원하는 스타일")
    notes: str     = Field(description="참고 사항")
    model_hint: str = Field(description="사용할 모델 힌트 (예: SDXL Base)")

# =============================
# 프롬프트 빌더(백엔드 LLM이 참고할 수 있도록 기본 번들 생성)
# =============================
def _infer_logo_type(copy_text: str, layout: str, style: str) -> str:
    text = f"{copy_text} {layout} {style}".lower()
    if any(k in text for k in ["엠블럼", "emblem", "badge", "round", "seal", "원형"]):
        if any(k in text for k in ["텍스트", "wordmark", "타이포", "type"]):
            return "combination mark (emblem + wordmark)"
        return "emblem"
    if any(k in text for k in ["wordmark", "타이포", "type", "서체", "lettering"]):
        return "wordmark"
    return "combination mark"

def _normalize_phrase(s: str) -> str:
    s = re.sub(r"\s+", " ", s or "").strip()
    return s

def _build_positive_prompt(brief: PromptBrief, palette: Optional[List[str]], logo_type: str,
                           sketch_present: bool, mask_present: bool) -> str:
    core = [
        f"{logo_type} logo for cafe '{brief.cafe_name}'",
        f"logo text: {brief.copy_text}",
        "clean vector aesthetic, sharp edges, smooth bezier curves",
        "minimal, professional brand identity, balanced spacing, consistent stroke weight",
        "high contrast, print-ready, flat colors, no gradients unless essential",
    ]
    if brief.layout: core.append(_normalize_phrase(brief.layout))
    if brief.style:  core.append(_normalize_phrase(brief.style))
    if brief.strengths: core.append(f"brand cues: {_normalize_phrase(brief.strengths)}")
    if brief.notes:  core.append(f"tonality: {_normalize_phrase(brief.notes)}")

    if palette:
        hex_join = ", ".join(palette)
        core.append(f"restrict color palette to: {hex_join}")

    if sketch_present and mask_present:
        core += [
            "respect provided sketch silhouette and composition strictly",
            "align text to provided text mask: same baseline, curvature, kerning, tracking",
            "preserve relative scale and placement from guides",
        ]
    elif sketch_present and not mask_present:
        core += [
            "follow provided sketch for silhouette and composition",
            "typeset logo text centered relative to sketch focal point",
        ]
    elif (not sketch_present) and mask_present:
        core += [
            "follow provided text mask for baseline, curvature, arc radius and alignment",
            "wrap lettering along the guide curve if indicated",
        ]
    else:
        core += ["centered composition, strong focal hierarchy"]

    core += [
        "sdxl-friendly descriptors, logo design focus, graphic design, vector art look",
        "2D, plain background, studio lighting not applicable",
    ]
    return ", ".join(core)

def _build_negative_prompt(brief: PromptBrief) -> str:
    avoid_user = _normalize_phrase(brief.avoid)
    neg = [
        "photo, photorealistic, 3d render, depth of field, shadows, reflections",
        "noise, artifacts, blur, low-res, pixelation, aliasing, messy edges",
        "complex background, textured background, busy pattern",
        "too many colors, neon glow, bevel, emboss, chrome, gradient mesh",
        "drop shadow, lens flare, watermark, signature, stock icon",
        "illegible text, warped letters, inconsistent kerning, misaligned baseline",
    ]
    if avoid_user:
        neg.append(avoid_user)
    return ", ".join(neg)

def build_prompt_bundle(brief: PromptBrief, palette: Optional[List[str]],
                        sketch_present: bool, mask_present: bool) -> dict:
    logo_type = _infer_logo_type(brief.copy_text, brief.layout, brief.style)
    positive = _build_positive_prompt(brief, palette, logo_type, sketch_present, mask_present)
    negative = _build_negative_prompt(brief)

    control_hints = []
    if sketch_present:
        control_hints.append("Enable ControlNet(Scribble or Canny) with medium weight (e.g., 0.6–0.8)")
    if mask_present:
        control_hints.append("Compose text exactly along mask; keep baseline/arc/kerning identical")

    return {
        "model": "SDXL Base",
        "logo_type": logo_type,
        "positive": positive,
        "negative": negative,
        "control_hints": control_hints,
        "sampler": "DPM++ 2M Karras",
        "cfg_scale": 6.5,
        "steps": 30,
        "size": "1024x1024",
    }

# =============================
# 백엔드 호출
# =============================
def start_generate(
    brief_id: int,
    sketch_b64: Optional[str],
    mask_b64: Optional[str],
    text_info: Optional[Dict[str, Any]],
    prompt_overrides: Optional[dict],
    gpt_prompt_seed: Optional[str],
    gpt_messages: Optional[List[Dict[str, str]]],
    num_images: int = 4,
    seed: Optional[int] = None,
) -> Optional[str]:
    payload = {
        "brief_id": brief_id,
        "sketch_png_b64": sketch_b64,
        "text_mask_png_b64": mask_b64,
        "text_info": text_info,                         # 🔑 2단계에서 만든 구조화 텍스트 정보
        "use_llm_prompt": True,                         # 🔑 백엔드가 GPT로 프롬프트 생성
        "llm_inputs": {                                 # 🔑 백엔드 LLM에 전달할 힌트 번들
            "prompt_overrides": prompt_overrides or {},
            "gpt_prompt_seed": gpt_prompt_seed or "",
            "gpt_messages": gpt_messages or [],         # (백엔드가 지원하면 사용)
        },
        "num_images": int(num_images),
        "seed": seed,
        "return_debug": True,
    }
    try:
        r = requests.post(GENERATE_URL, json=payload, timeout=300)
        r.raise_for_status()
        data = r.json()
        return data.get("job_id")
    except Exception as e:
        st.error(f"이미지 생성 시작 실패: {e}")
        return None

def poll_result(job_id: str, timeout_sec: int = 300, interval_sec: float = 2.0) -> Optional[List[str]]:
    t0 = time.time()
    with st.spinner("이미지 생성 중…"):
        while True:
            try:
                url = f"{BACKEND_BASE.rstrip('/')}{JOB_ENDPOINT.format(job_id=job_id)}"
                r = requests.get(url, timeout=300)
                if r.status_code == 404:
                    time.sleep(interval_sec)
                    if time.time() - t0 > timeout_sec:
                        st.error("작업을 찾을 수 없어요(404)."); return None
                    continue
                r.raise_for_status()
                data = r.json()
                status = data.get("status", "pending")
                if status == "done":
                    imgs = data.get("images_b64") or []
                    return imgs if imgs else None
                if status == "error":
                    st.error(f"생성 오류: {data.get('error')}"); return None
            except Exception as e:
                st.warning(f"폴링 에러: {e}")
            if time.time() - t0 > timeout_sec:
                st.warning("생성 대기 시간이 초과되었습니다."); return None
            time.sleep(interval_sec)

# =============================
# 메인 렌더
# =============================
def render():
    try:
        st.set_page_config(page_title=TITLE, page_icon="📝", layout="wide")
    except Exception:
        pass

    st.progress(75, text="Step 3/4 — Brief")
    st.title(TITLE)
    st.subheader("📝 요구사항을 상세하게 입력해 주세요")

    st.info(
        "이 단계에서는 **스케치/텍스트 마스크 유무**를 감지해 SDXL용 **LLM 프롬프트**를 생성하도록 백엔드에 필요한 정보를 모두 전달합니다.\n"
        "참고 이미지를 올리면 **팔레트**를 추출해 색상을 제한합니다."
    )

    # 세션 상태 기본값
    st.session_state.setdefault("brief_payload", None)
    st.session_state.setdefault("brief_id", None)
    st.session_state.setdefault("palette", None)
    st.session_state.setdefault("ref_img_b64", None)
    st.session_state.setdefault("gpt_prompt_seed", None)
    st.session_state.setdefault("prompt_bundle", None)

    # ===== 스케치/마스크 감지 배지 =====
    sketch_b64 = None
    for k in ("sketch_png_b64","sketch_final_png_b64","sketch_canvas_b64","sketch_result_b64","sketch_rgba_b64","sketch_bytes_b64"):
        if st.session_state.get(k):
            sketch_b64 = st.session_state.get(k); break
    mask_b64 = st.session_state.get("mask_text_png_b64")

    col_badge1, col_badge2 = st.columns([1, 5])
    with col_badge1:
        st.caption("가이드 감지")
    with col_badge2:
        s = "✅ 스케치 있음" if sketch_b64 else "⬜ 스케치 없음"
        m = "✅ 텍스트 마스크 있음" if mask_b64 else "⬜ 텍스트 마스크 없음"
        st.markdown(f"- {s}  \n- {m}")

    # 🔑 2단계(Text)에서 만든 구조화 정보
    text_info: Optional[Dict[str, Any]] = st.session_state.get("text_info")
    if text_info is None:
        st.info("참고: 2단계(Text)에서 만든 구조화 텍스트 정보가 없습니다. (직선/원형 텍스트 가이드가 프롬프트에 반영되지 않을 수 있어요)")

    # ===== 폼 =====
    try:
        form_ctx = st.form(_k("form"), border=True)
    except TypeError:
        form_ctx = st.form(_k("form"))

    with form_ctx:
        col1, col2 = st.columns(2)
        with col1:
            cafe_name = st.text_input("카페명 *", value="BlueMoon", key=_k("cafe_name"))
            # '생성하고 싶은 텍스트'은 텍스트 마스크가 있는 경우 표시하지 않습니다.
            mask_present = bool(st.session_state.get("mask_text_png_b64"))
            if not mask_present:
                copy_text = st.text_input("생성하고 싶은 텍스트 *", value="BLUE MOON CAFE", key=_k("copy_text"))
            else:
                copy_text = ""
        with col2:
            strengths = st.text_input("핵심 장점 *", value="스페셜티 원두, 당일 로스팅", key=_k("strengths"))
            style = st.text_input("원하는 스타일 *", value="미니멀, 벡터, 베이지/브라운", key=_k("style"))
            notes = st.text_area("생성하고 싶은 이미지를 설명해 주세요 *", value="깔끔하고 단순하게. 인쇄 적합.", height=90, key=_k("notes"))
            st.text_input("사용할 모델 (자동 고정)", value="SDXL Base", key=_k("model_hint"), disabled=True)

        # 이미지 업로드 UI 및 팔레트 추출은 프론트에서 제거되었습니다.

        generate_now = st.form_submit_button("이미지 생성 ▶", type="primary", use_container_width=True)

    # ===== 입력 검증/프리뷰/프롬프트 생성 =====
    def _prepare_and_preview() -> Tuple[Optional[PromptBrief], Optional[dict], Optional[str], Optional[dict], Optional[List[Dict[str,str]]]]:
        missing = []
        if not cafe_name.strip():   missing.append("카페명")
        mask_present_local = bool(mask_b64)
        if not mask_present_local:
            if not copy_text.strip():   missing.append("생성하고 싶은 텍스트")
        if not strengths.strip():   missing.append("핵심 장점")
        if not style.strip():      missing.append("원하는 스타일")
        if not notes.strip():       missing.append("생성하고 싶은 이미지를 설명해 주세요")
        if missing:
            st.error(f"필수 항목을 입력해 주세요: {', '.join(missing)}")
            return None, None, None, None, None

        brief = PromptBrief(
            cafe_name=cafe_name, copy_text=copy_text, layout="", avoid="",
            strengths=strengths, style=style, notes=notes, model_hint="SDXL Base"
        )

        ref_b64: Optional[str] = None
        palette_vals: Optional[List[str]] = None

        # 참고 이미지 업로드 UI를 제거했습니다 — 프론트에서 참조 이미지/팔레트는 사용하지 않습니다.
        ref_b64 = None
        palette_vals = None

        st.session_state.ref_img_b64 = ref_b64
        st.session_state.palette     = palette_vals

        # ===== 브리프 저장 페이로드 (백엔드에 기록) =====
        payload = {
            "request_id": str(uuid.uuid4()),
            "cafe_name": cafe_name,
            "copy_text": copy_text,
            "layout": "",
            "avoid": "",
            "strengths": strengths,
            "style": style,
            "notes": notes,
            "model_hint": "SDXL Base",
            "palette": [],
            "ref_image_present": False,
            "ref_img_b64": None,
            "guides": {
                "sketch_present": bool(sketch_b64),
                "text_mask_present": bool(mask_b64),
            },
            # 🔑 2단계에서 만든 구조화 텍스트 정보 (LLM 프롬프트 컨텍스트로 중요)
            "text_info": st.session_state.get("text_info"),
        }
        st.session_state["brief_payload"] = payload

        # === 기본 프롬프트 번들(LLM 힌트/오버라이드용) ===
        prompt_bundle = build_prompt_bundle(
            brief=brief,
            palette=palette_vals,
            sketch_present=bool(sketch_b64),
            mask_present=bool(mask_b64),
        )
        st.session_state["prompt_bundle"] = prompt_bundle

        # === LLM 입력 시드 및 메시지 (백엔드에서 그대로 사용 가능) ===
        gpt_prompt_seed = (
            "You are a prompt engineer. Create a single best SDXL Base prompt for a cafe logo.\n"
            "Use concise visual keywords; vector/flat aesthetics; print-ready.\n"
            "Output only the prompt string, no explanations.\n\n"
            f"Cafe Name: {cafe_name}\n"
            f"Logo Text: {copy_text}\n"
            f"Key Strengths: {strengths}\n"
            f"Desired Style: {style}\n"
            f"Notes: {notes}\n"
            f"Palette (hex): {', '.join(palette_vals) if palette_vals else 'N/A'}\n"
            f"Guides: sketch={bool(sketch_b64)}, text_mask={bool(mask_b64)}\n"
            "Model: SDXL Base\n"
        )
        st.session_state["gpt_prompt_seed"] = gpt_prompt_seed

        # 🔑 백엔드에서 chat.completions로 바로 쓸 수 있는 messages
        gpt_messages = [
            {
                "role": "system",
                "content": "You are a branding assistant that writes crisp, SDXL-friendly logo prompts. Respond in JSON {\"prompt\": string, \"neg\": string}."
            },
            {
                "role": "user",
                "content": json.dumps({
                    "brief": payload,
                    "text_info": st.session_state.get("text_info"),
                    "sketch_present": bool(sketch_b64),
                    "text_mask_present": bool(mask_b64),
                    "palette": [],
                }, ensure_ascii=False)
            }
        ]

        with st.expander("🔎 전송/생성 데이터 미리보기", expanded=False):
            st.markdown("**브리프 저장 페이로드(JSON)**")
            st.code(json.dumps(payload, ensure_ascii=False, indent=2), language="json")
            st.markdown("**프롬프트 번들 (positive / negative / control hints)**")
            st.code(json.dumps(prompt_bundle, ensure_ascii=False, indent=2), language="json")
            st.markdown("**LLM messages (백엔드용 참고)**")
            st.code(json.dumps(gpt_messages, ensure_ascii=False, indent=2), language="json")

        return brief, payload, ref_b64, prompt_bundle, gpt_messages

    # ===== 생성 (저장 → 생성 → 폴링 → 네비) =====
    if generate_now:
        brief, payload, _, prompt_bundle, gpt_messages = _prepare_and_preview()
        if brief is None:
            st.stop()

        # 1) 저장
        brief_id = None
        try:
            resp = requests.post(BRIEF_POST_URL, json=payload, timeout=300)
            if resp.ok:
                data = resp.json()
                brief_id = data.get("id")
                st.session_state["brief_id"] = brief_id
                st.toast(f"백엔드 저장 완료 (id={brief_id})", icon="✅")
            else:
                st.error(f"백엔드 저장 실패: {resp.status_code} {resp.text}"); st.stop()
        except Exception as e:
            st.error(f"백엔드 연결 실패: {e}"); st.stop()

        # 2) 생성 시작 — 🔑 LLM 프롬프트 생성에 필요한 모든 컨텍스트를 전달
        job_id = start_generate(
            brief_id=brief_id,
            sketch_b64=sketch_b64,
            mask_b64=mask_b64,
            text_info=st.session_state.get("text_info"),
            prompt_overrides=prompt_bundle,             # 백엔드가 우선 오버라이드로 사용하거나 LLM 힌트로 병합
            gpt_prompt_seed=st.session_state.get("gpt_prompt_seed"),
            gpt_messages=gpt_messages,                  # 백엔드가 지원 시 바로 사용
            num_images=4,
            seed=None,
        )
        if not job_id: st.stop()
        st.session_state["last_job_id"] = job_id

        # 3) 폴링
        images_b64 = poll_result(job_id=job_id, timeout_sec=300, interval_sec=2.0)
        if images_b64 and isinstance(images_b64, list) and len(images_b64) > 0:
            st.session_state["gen_images_b64"] = images_b64
            st.success("이미지 생성 완료! 다음 페이지에서 결과를 확인하세요.")
            _nav_to_next_step()
        else:
            st.warning("생성된 이미지가 없습니다. 다음 페이지에서 다시 시도할 수 있어요.")
            _nav_to_next_step()

if __name__ == "__main__":
    render()
