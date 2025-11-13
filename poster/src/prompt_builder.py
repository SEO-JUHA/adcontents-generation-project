from typing import Tuple

def build_prompts(brand_intro: str, category_hint: str = "cafe") -> Tuple[str, str]:
    """
    입력: 브랜드 소개 텍스트(국문/영문 상관없음)
    출력: (positive, negative) 프롬프트
    - 배경+레이아웃 전용. 텍스트/로고/음식/사람 제외.
    """
    bi = (brand_intro or "").strip().replace("\n", " ")
    feature = brand_intro.strip()
    
    base = [
        f"{category_hint} background",
        "professional design",
        # "realistic",
        "high quality",
        f"emphasizing unique features: {feature}"
    ]
    if bi:
        base.append(f"reflect brand: {bi}")

    positive = ", ".join(base)

    negative = ", ".join([
        "text",
        "typography",
        "labels",
        
        "logos",
        "watermarks",
        "low quality",
    ])
    return positive, negative
