from __future__ import annotations
from typing import List, Tuple, Dict, Any
import os
from PIL import Image, ImageOps
import numpy as np
from scipy.ndimage import laplace
import base64
import json
from openai import OpenAI

def _local_aesthetic_score(path: str) -> float:
    """
    외부 API 불가 시 간단한 품질 근사치:
    - 샤프니스(라플라시안 분산)
    - 명암 대비(표준편차)
    - 과도한 빈 픽셀/완전 흰/검 검출에 페널티
    """
    try:
        im = Image.open(path).convert("RGB")
        gray = ImageOps.grayscale(im)
        arr = np.array(gray, dtype=np.float32)
        # 샤프니스 근사 : 라플라시안 분산
        lap = laplace(arr)
        sharp = float(np.var(lap))
        # 대비
        contrast = float(np.std(arr))
        # 극단 값 비율
        zeros = float((arr < 5).mean())
        whites = float((arr > 250).mean())
        penalty = zeros*0.3 + whites*0.3
        score = sharp*0.7 + contrast*0.3 - penalty*50.0
        return score
    except Exception:
        return -1e9

def _rank_local(paths: List[str], top_k: int) -> List[str]:
    scored = [(p, _local_aesthetic_score(p)) for p in paths]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [p for p,_ in scored[:max(1, top_k)]]

def _rank_gpt(paths: List[str], top_k: int, brand_intro: str, model: str="gpt-4o-mini") -> List[str]:
    """
    OpenAI 비전 호출 (있으면 사용). 실패 시 예외 던짐 → 상위 레벨에서 폴백.
    최신 Chat Completions 스펙은 image_url만 허용.
    """
    try:
        client = OpenAI()
        MAX = 12
        cand = paths[:MAX]

        # 경로-순서 매핑을 텍스트로 명시(모델이 그대로 돌려주도록)
        ordered_list_text = "IMAGE_ORDER = [\n" + ",\n".join([f'  "{p}"' for p in cand]) + "\n]"

        rubric = (
            "Rate each image 0~100 using these criteria:\n"
            "A) No visible text/logo/watermarks/numbers (penalize heavily if present).\n"
            "B) Strong alignment to the brand intro (mood, color harmony, style).\n"
            "C) Clean composition with clear negative space suitable for copy.\n"
            'Return ONLY a JSON array of objects like: [{"path": <exact path from IMAGE_ORDER>, "score": 0-100}].'
        )

        # 메시지: text + image_url(data URL)
        content = [{"type": "text", "text": f"Brand intro:\n{brand_intro}\n\n{ordered_list_text}\n\n{rubric}"}]
        for p in cand:
            with open(p, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"}
            })

        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": content}],
            temperature=0.0,
            response_format={"type": "json_object"},
        )

        txt = resp.choices[0].message.content
        data = json.loads(txt)

        # 유연 파싱: results | scores | items | array
        arr = data.get("results") or data.get("scores") or data.get("items") or data
        if isinstance(arr, dict) and "items" in arr:
            arr = arr["items"]
        if not isinstance(arr, list):
            raise RuntimeError(f"Unexpected GPT response format: {type(arr)}")

        scored = []
        cand_set = set(cand)
        for it in arr:
            p = it.get("path")
            try:
                s = float(it.get("score", 0))
            except Exception:
                s = 0.0
            if p in cand_set:
                scored.append((p, s))

        # cand에 포함됐으나 응답에 누락된 것들 보정(0점)
        remain = [p for p in cand if p not in {x[0] for x in scored}]
        scored += [(p, 0.0) for p in remain]

        scored.sort(key=lambda x: x[1], reverse=True)
        return [p for p,_ in scored[:max(1, top_k)]]

    except Exception as e:
        raise RuntimeError(f"GPT rank failed: {e}")

def rank_images(paths: List[str], brand_intro: str, *, method: str="gpt", top_k: int=3, model: str="gpt-4o-mini") -> List[str]:
    """
    공용 엔트리. method='gpt' 우선 시도, 실패 시 local 로 폴백.
    """
    if method == "gpt" and os.getenv("OPENAI_API_KEY"):
        try:
            return _rank_gpt(paths, top_k, brand_intro, model=model)
        except Exception as e:
            print("[WARN]", e)
    # 폴백
    return _rank_local(paths, top_k)
