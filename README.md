# 🪄 AdMaster
> **소상공인도 클릭 몇 번으로 나만의 광고 콘텐츠를 만드는 AI 디자인 플랫폼**

<br>

## 🔗 서비스 시연 영상
아래 이미지를 클릭하시면 유튜브에서 시연 영상을 보실 수 있습니다.

[![시연영상](http://img.youtube.com/vi/YgOqr8V3YCU/0.jpg)](https://youtu.be/YgOqr8V3YCU)

<br>

## 💡 프로젝트 개요

**‘광명을 찾아서’** 팀이 개발한, **카페 등 소상공인을 위한 생성형 AI 기반 인스타그램 광고 콘텐츠 제작 서비스 – AdMaster**

현업 디자이너나 복잡한 도구 없이도 매장 주인이 짧은 텍스트(브리프)와 간단한 스케치만으로 
<br>브랜드에 어울리는 로고, 포스터, 4컷 이미지, 인스타그램 내용/태그 등을 빠르게 만들 수 있도록 설계

> **핵심 워크플로우**  
> 입력(브리프 / 톤·무드 / 간단 스케치) → LLM 기반 브리핑 보정 →  
> SDXL 기반 이미지 합성 → ControlNet 계열 어댑터로 형태 제어  
> → ‘아이디어 제안 → 후보 생성 → 선택·세부 편집’ 전체 플로우 자동화

(이 과정을 통해 디자인 경험이 없는 사용자도 일관된 퀄리티의 시각물을 짧은 시간에 얻을 수 있습니다.)  

많은 비즈니스는 마케팅·디자인 리소스가 부족하고, 외주 비용이나 툴 학습 시간이 부담입니다.
<br>Ad Master는 이 격차를 메우기 위해 **효율성(시간·비용 절감)과 창의성을 동시에 제공**하여, 
<br>카페 운영자가 경쟁력 있는 광고 이미지 리소스를 손쉽게 얻고 사용할 수 있도록 도와줍니다.

<br>

## ⚙️ 기술 스택
#### ► **Backend**
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
<img src="https://img.shields.io/badge/Uvicorn-499848?style=for-the-badge&logo=gunicorn&logoColor=white">
<img src="https://img.shields.io/badge/Pydantic-E92063?style=for-the-badge&logo=pydantic&logoColor=white">
<img src="https://img.shields.io/badge/SQLAlchemy-D71F00?style=for-the-badge&logo=sqlalchemy&logoColor=white">
![MySQL](https://img.shields.io/badge/MySQL-4479A1?style=for-the-badge&logo=mysql&logoColor=white)


#### ► **Frontend**
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)


#### ► **Prompt Engine**
<img src="https://img.shields.io/badge/OpenAI-412991?style=for-the-badge&logo=openai&logoColor=white">


#### ► **Processing / Environment**
<img src="https://img.shields.io/badge/uv-3A86FF?style=for-the-badge&logo=python&logoColor=white"> <img src="https://img.shields.io/badge/Google%20Cloud-4285F4?style=for-the-badge&logo=googlecloud&logoColor=white"> <img src="https://img.shields.io/badge/NVIDIA%20L4-76B900?style=for-the-badge&logo=nvidia&logoColor=white">

<br>

## ⚡️ 핵심 기능

### ✓ 로고 생성

> 브랜드를 대표하는 로고를 직관적으로, 쉽게 제작

1. **사용자 맞춤 로고** — 로고 브리핑 → 자동 프롬프트 생성(LLM 보조) → SDXL 기반 로고 생성  
2. **심볼 스케치 제어** — 스케치 입력 → ControlNet(Scribble)으로 형태 제어 → 다양한 심볼 생성  
3. **텍스트 형태 지원** — 직선과 원형 형태의 텍스트 마스크 입력 기능 제공  
4. **정확한 텍스트 이미지 생성** — 텍스트 마스크를 입력 받아 ControlNet(Canny)으로 윤곽선 고정(타 플랫폼 대비 텍스트 이미지 품질 우수)

<br>

### ✓ 인스타그램 글 생성

> 이미지와 브랜드에 어울리는 SNS 콘텐츠를 자동으로 작성

1. **이미지 컨텍스트 분석** — 업로드된 이미지의 분위기, 피사체, 마케팅 포인트를 AI가 자동으로 분석  
2. **다중 캡션 제안** — 하나의 주제에 대해 '감성적', '정보 전달', '재치있는' 3가지 테마의 캡션 동시 생성 
3. **타겟 고객 맞춤** — 선택된 고객층(예: 20대 대학생)의 언어와 스타일에 맞춰 캡션 자동 변환  
4. **전략적 해시태그 생성** — 가게 위치, 메뉴, 감성 등을 조합한 최적의 해시태그 추천
5. **예상 반응률 분석** — 생성된 콘텐츠의 마케팅 효과를 '높음/중간/낮음'으로 예측하고 근거 제시

<br>

### ✓ 4컷 스토리보드 생성

> 브랜드 스토리를 시각적으로 전달하는 스토리형 광고 콘텐츠 제작 기능

1. **BI 기반 연속 이미지 생성** — 브랜드 대표 이미지와 문장을 분석해 컷 간 콘셉트 일관성 유지  
2. **AI 캡션 자동 생성 및 오버레이** — GPT가 컷별 상황에 맞는 자막을 자동 생성 및 배치  
3. **콜라주 기능** — 생성된 4컷을 하나의 시각적 스토리보드로 합성, SNS 공유 최적화

<br>

### ✓ 브랜드 포스터 생성

> 브랜드 아이덴티티(BI)에 맞춘 광고 포스터 자동 생성 기능

1. **BI 기반 포스터 디자인 생성** — 로고, 색상, 문구를 분석해 브랜드의 BI를 반영한 스타일의 포스터 생성  
2. **다양한 레이아웃 제공** — Streamlit UI에서 원하는 레이아웃 선택 → ControlNet에 반영  
   (UX 기능을 통해 제어되는 AI 기능)  
3. **실시간 편집 및 오버레이** — Streamlit UI에서 실시간으로 불필요한 부분 지우기, 텍스트 추가, 이미지 합성(개발 중) 등 지원.

<br>

## 📁 디렉터리 구조
```bash
.
├─ checkpoint/                # 모델 가중치 (.safetensors)
│  ├─ sdxl/ 
│  ├─ controlnet/
│  ├─ ipadapter/
│  └─ ...
│
├─ backend/
│  ├─ main.py                 # FastAPI 메인 엔트리포인트
│  ├─ database.py             # MySQL 연동 및 ORM 설정
│  └─ ...
│
├─ frontend/
│  ├─ pages/                  # Streamlit 페이지 구성
│  ├─ app.py                  # Streamlit 메인 앱
│  ├─ auth_*.py               # 로그인 / 인증 로직
│  └─ load_env.py             # 환경 변수 로더
│
├─ cartoon/
│  ├─ src/                    # 4컷 만화 생성 로직
│  ├─ configs/                # 만화 워크플로우 설정
│  └─ ...
│
├─ poster/
│  ├─ src/                    # 포스터 생성 로직
│  ├─ configs/                # 포스터 워크플로우 설정
│  └─ ...
│
├─ hf-cache/                  # 폰트 및 모델 캐시
│  ├─ ...
│
├─ requirements.txt           # 의존성 설치 목록
└─ .env                       # 환경 설정 파일
```

<br>

## ⚙️ 환경 변수 (.env) 및 설정
루트 디렉터리에 .env 파일을 생성합니다.
```bash
OPENAI_API_KEY=             # OpenAI API 키
MYSQL_HOST=localhost        # localhost 주소
MYSQL_PORT=3306             # 주소의 port 번호
MYSQL_USER=                 # MySQL 작성을 위한 사용자
MYSQL_PASSWORD=             # MySQL 비밀번호
MYSQL_DB=                   # 생성한db 이름
```

<br>

## 데모 (로컬 실행)
>로컬 환경에서 AdMaster를 실행하는 단계별 가이드입니다.<br>Python 3.11.13 + uv 가상환경을 기준으로 작성되었습니다.

### STEP. 0 : uv 설치
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# 새 셸을 열거나, 당장 PATH 반영:
export PATH="$HOME/.local/bin:$PATH"
uv --version
```

### STEP. 1 : 리포지토리 클론
```bash
git clone https://github.com/Ad-Master/AdMaster.git
cd AdMaster
```

### STEP. 2 : Python 3.11.13 & 가상환경(uv) 세팅
```bash
# Python 3.11.13 설치(없으면 uv가 내려받아 설치)
uv python install 3.11.13

# venv 생성 & 활성화
uv venv --python 3.11.13 .venv
source .venv/bin/activate     # Windows PowerShell: .\.venv\Scripts\Activate.ps1

python --version              # → 3.11.13 확인
```

### STEP. 3 : 의존성 설치
```bash
uv pip install -U pip
uv pip install -r requirements.txt
```

### STEP. 4 : ComfyUI 설치 (이미지 생성 엔진)
> AdMaster의 일부 기능은 Stable Diffusion 기반 ComfyUI 워크플로우를 사용합니다.<br> poster/ComfyUI 디렉터리로 이동하여 설치를 진행합니다.
```bash
cd poster/ComfyUI

# 의존성 설치
uv pip install -r requirements.txt

# ComfyUI 실행 테스트
python main.py --help
```
ComfyUI가 정상적으로 실행되면 다음 메시지가 표시됩니다:
```bash
Starting ComfyUI server at http://127.0.0.1:8188
```

### STEP. 5 : 실행
#### STEP. 5-1 : 백엔드(FastAPI/Uvicorn)
```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```
#### STEP. 5-2 : 프론트엔드 (Streamlit)
새 터미널에서 가상환경을 다시 활성화한 후 실행합니다.
```bash
source .venv/bin/activate
streamlit run frontend/app.py --server.address=0.0.0.0 --server.port=8501
```
접속 주소: http://127.0.0.1:8501

#### STEP. 5-3 : ComfyUI 서버
새 터미널에서 가상환경을 다시 활성화한 후 실행합니다.
```bash
cd poster/ComfyUI
python main.py --listen 0.0.0.0 --port 8188
```
접속 주소: http://127.0.0.1:8188

<br>

## 🧠 Model & Resource(hub)
<table>
  <thead>
    <tr>
      <th style="width:12%">구분</th>
      <th style="width:28%">모델명</th>
      <th style="width:60%">설명</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>Base</strong></td>
      <td><code>sd_xl_base_1.0.safetensors</code></td>
      <td>AdMaster의 모든 이미지 생성은 이 모델을 중심으로 렌더링됩니다.</td>
    </tr>
    <tr>
      <td><strong>ControlNet</strong></td>
      <td><code>xinsir/controlnet-union-sdxl-1.0</code>,<br><code>xinsir/controlnet-canny-sdxl-1.0</code>,<br><code>xinsir/controlnet-scribble-sdxl-1.0</code></td>
      <td>ControlNet 기반의 <strong>구도 및 외곽선 제어</strong> 모델로, 포스터·4컷 만화에서 레이아웃 안정성과 윤곽 보존을 담당합니다.</td>
    </tr>
    <tr>
      <td><strong>IP-Adapter</strong></td>
      <td><code>InvokeAI/ip_adapter_sdxl_vit_h</code></td>
      <td>로고 · 음식 · 인테리어 등 <strong>브랜드 시그니처 이미지</strong>를 일관되게 반영하여 색감·스타일의 통일성을 유지합니다.</td>
    </tr>
    <tr>
      <td><strong>Prompt Engine</strong></td>
      <td><code>OpenAI gpt-4.1-mini</code>,<br><code>OpenAI gpt-5</code></td>
      <td>사용자의 <strong>브랜드 소개 문장</strong>을 분석해 핵심 키워드를 추출하고, 생성용 프롬프트를 보정·확장합니다. (인스타 글/캡션·해시태그 생성에 활용)</td>
    </tr>
  </tbody>
</table>

<br>

## 👥 팀원 소개
| 이름 | 파트 | 담당 업무 |
|------|------|-------|
| **전혜정 (팀장)** | 포스터 디자인 / 4컷 만화 이미지 생성 | - SDXL + IP-Adapter + ControlNet 기반 포스터 자동화 및 Streamlit 인터페이스 구축<br>- 컷별 프롬프트 → 이미지 자동 생성 및 Caption Overlay 기능 완성 |
| **조용원** | 포스터 디자인 / 4컷 만화 이미지 생성 | - SDXL + IP-Adapter + ControlNet 기반 포스터 자동화 및 Streamlit 인터페이스 구축<br>- 컷별 프롬프트 → 이미지 자동 생성 및 Caption Overlay 기능 완성 |
| **이현도** | 인스타그램 글 생성 | - 브랜드 소개문을 기반으로 자연스러운 홍보 글·캡션 생성 (GPT-5 활용) |
| **이대석** | 로고 생성 | - SDXL + ControlNet 기반 브랜드 맞춤 로고 생성 파이프라인 완성, 브랜드 컬러 일관성 확보<br>- 로고 텍스트 안정성 개선 및 텍스트 레이아웃 구현 |
| **서주하** | 로고 생성 | - SDXL + ControlNet 기반 브랜드 맞춤 로고 생성 파이프라인 완성, 브랜드 컬러 일관성 확보<br>- 로고 텍스트 안정성 개선 및 텍스트 레이아웃 구현 |


<br>


## Docs
[🔗 협업일지 링크](https://www.notion.so/26a81b926d7281f6ae49e785b2ed4d30?source=copy_link)

[🔗 보고서 pdf](https://drive.google.com/file/d/1cB7_KEVGVFZE-juB8G912NVWupJ5gng0/view?usp=sharing)

