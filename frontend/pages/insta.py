# frontend_app/pages/2_📱_인스타그램_게시물_생성.py
import streamlit as st
import requests
from enum import Enum

# --- 페이지에 필요한 Enum 정의 ---
class TargetAudience(Enum):
    ALL = "전체"
    UNIVERSITY_STUDENT = "20대 대학생"
    OFFICE_WORKER = "30-40대 직장인"
    COUPLE_DATE = "기념일/주말 데이트 커플"
    FAMILY_KIDS = "아이와 함께하는 가족"

# --- 페이지 기본 설정 ---
st.set_page_config(page_title="인스타그램 게시물 생성", layout="wide")
st.title("📱 인스타그램 게시물 생성")
st.info("이미지와 정보를 바탕으로 AI가 3가지 버전의 인스타그램 캡션과 해시태그를 생성합니다.")

# --- FastAPI 서버 주소 ---
BACKEND_URL = "http://127.0.0.1:8000/v1/instagram/generate"

# --- 세션 상태 초기화 ---
if "insta_result" not in st.session_state:
    st.session_state.insta_result = None

# --- UI 입력 폼 ---
with st.form("instagram_post_form"):
    st.subheader("1. 콘텐츠 정보 입력")
    col1, col2 = st.columns(2)
    with col1:
        brand_persona = st.text_input("브랜드 페르소나", "따뜻한 감성의 동네 친구 같은 바리스타")
        product_info = st.text_area("핵심 소재", "가을 신메뉴, 단호박 크림 라떼 출시")
    with col2:
        store_address = st.text_input("가게 주소", "서울시 마포구 연남동 223-14")
        target_audience = st.selectbox("타겟 고객층", [t.value for t in TargetAudience])
    uploaded_image = st.file_uploader("이미지를 업로드하세요.", type=["png", "jpg", "jpeg"])
    submitted = st.form_submit_button("게시물 생성 요청", type="primary")

# --- 메인 실행 로직 ---
if submitted:
    if not all([brand_persona, product_info, store_address, uploaded_image]):
        st.error("모든 필수 정보와 이미지를 입력해주세요.")
    else:
        with st.spinner("백엔드 AI 서버에 요청 중입니다..."):
            try:
                form_data = {
                    'brand_persona': (None, brand_persona),
                    'product_info': (None, product_info),
                    'store_address': (None, store_address),
                    'target_audience': (None, target_audience)
                }
                files = {'image': (uploaded_image.name, uploaded_image.getvalue(), uploaded_image.type)}
                response = requests.post(BACKEND_URL, files=files, data=form_data)
                
                if response.status_code == 200:
                    st.session_state.insta_result = response.json()
                    st.success("게시물 생성 완료!")
                else:
                    st.error(f"서버 오류: {response.status_code}")
                    try:
                        st.json(response.json())
                    except:
                        st.text(response.text)
            except requests.exceptions.ConnectionError:
                st.error(f"백엔드 서버({BACKEND_URL})에 연결할 수 없습니다. 서버가 실행 중인지 확인하세요.")
            except Exception as e:
                st.error(f"알 수 없는 오류 발생: {e}")

# --- 결과 표시 ---
if st.session_state.insta_result:
    st.divider()
    st.subheader("✨ AI 생성 결과")
    result = st.session_state.insta_result
    
    caption_options = result.get("caption_options", [])
    hashtags = result.get("hashtags", {})
    prediction = result.get("engagement_prediction", {})

    if caption_options:
        caption_tabs = st.tabs([opt.get('theme', f'옵션 {i+1}') for i, opt in enumerate(caption_options)])
        for i, tab in enumerate(caption_tabs):
            with tab:
                st.text_area(
                    label=f"캡션 내용 ({caption_options[i].get('theme', '')})", 
                    value=caption_options[i].get('content', ''),
                    height=250,
                    key=f"caption_{i}"
                )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### #️⃣ 추천 해시태그")
        st.code(
            f"#대표/메뉴\n{' '.join(['#' + h for h in hashtags.get('representative', [])])}\n\n"
            f"#지역/장소\n{' '.join(['#' + h for h in hashtags.get('location', [])])}\n\n"
            f"#감성/트렌드\n{' '.join(['#' + h for h in hashtags.get('trending', [])])}"
        )

    with col2:
        st.markdown("#### 📈 예상 반응률")
        st.metric(label="예상 점수", value=prediction.get('score', 'N/A'))
        st.caption(f"**분석 이유**: {prediction.get('reason', 'N/A')}")