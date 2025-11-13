import json
from typing import Dict, Any, Tuple

def load_prompt(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    prompt = data["prompt"] if isinstance(data, dict) and "prompt" in data else data

    # 연결 캐시 구축
    _index_links(prompt)

    # 역할별 노드 ID 해석 (한 번만)
    prompt["_roles"] = _resolve_roles(prompt)
    return prompt


def _index_links(prompt: Dict[str, Any]) -> None:
    # 역인덱스: 어떤 노드가 어떤 노드로부터 참조되는지
    back = {}
    for nid, node in prompt.items():
        if not isinstance(node, dict):
            continue
        ins = node.get("inputs", {})
        for k, v in ins.items():
            if isinstance(v, list) and len(v) == 2 and isinstance(v[0], str):
                src = v[0]
                # 직렬화 친화적으로 list 사용
                back.setdefault(src, []).append((nid, k))
    prompt["_backlinks"] = back

def _get(prompt: Dict[str, Any], nid: str) -> Dict[str, Any]:
    return prompt[nid]

def _class(prompt: Dict[str, Any], nid: str) -> str:
    return _get(prompt, nid).get("class_type", "")

def _input(prompt: Dict[str, Any], nid: str, key: str):
    return _get(prompt, nid).get("inputs", {}).get(key)

def _link_target(val) -> str | None:
    if isinstance(val, list) and len(val) == 2 and isinstance(val[0], str):
        return val[0]
    return None

def _find_first_by_class(prompt: Dict[str, Any], cls: str) -> str | None:
    for nid, node in prompt.items():
        if not isinstance(node, dict):
            continue
        if node.get("class_type") == cls:
            return nid
    return None

def _resolve_roles(prompt: Dict[str, Any]) -> Dict[str, str]:
    roles: Dict[str, str] = {}

    # 1) ksampler
    ksampler = _find_first_by_class(prompt, "KSampler")
    if not ksampler:
        raise KeyError("[workflow_ops] KSampler not found")
    roles["ksampler"] = ksampler

    # 2) latent_base: ksampler.inputs.latent_image -> target
    latent = _link_target(_input(prompt, ksampler, "latent_image"))
    if not latent or _class(prompt, latent) != "EmptyLatentImage":
        # fallback: 첫 EmptyLatentImage
        latent = _find_first_by_class(prompt, "EmptyLatentImage")
    if not latent:
        raise KeyError("[workflow_ops] EmptyLatentImage not found")
    roles["latent_base"] = latent

    # 3) control_apply
    c_apply = _find_first_by_class(prompt, "ControlNetApplyAdvanced")
    if not c_apply:
        raise KeyError("[workflow_ops] ControlNetApplyAdvanced not found")
    roles["control_apply"] = c_apply

    # 4) control_loader: c_apply.inputs.control_net -> target
    c_loader = _link_target(_input(prompt, c_apply, "control_net"))
    if not c_loader or _class(prompt, c_loader) != "ControlNetLoader":
        # fallback
        c_loader = _find_first_by_class(prompt, "ControlNetLoader")
    if not c_loader:
        raise KeyError("[workflow_ops] ControlNetLoader not found")
    roles["control_loader"] = c_loader

    # 5) layout_image (LoadImage feeding c_apply.image)
    layout = _link_target(_input(prompt, c_apply, "image"))
    if not layout or _class(prompt, layout) != "LoadImage":
        # fallback: 아무 LoadImage
        layout = _find_first_by_class(prompt, "LoadImage")
    if not layout:
        raise KeyError("[workflow_ops] layout LoadImage not found")
    roles["layout_image"] = layout

    # 6) ipadapter
    ipa = _find_first_by_class(prompt, "IPAdapterAdvanced")
    if not ipa:
        raise KeyError("[workflow_ops] IPAdapterAdvanced not found")
    roles["ipadapter"] = ipa

    # 7) logo_image (LoadImage feeding ipadapter.image)
    logo = _link_target(_input(prompt, ipa, "image"))
    if not logo or _class(prompt, logo) != "LoadImage":
        # fallback: 또 다른 LoadImage (layout과 다르면 우선)
        for nid, node in prompt.items():
            if isinstance(node, dict) and node.get("class_type") == "LoadImage" and nid != roles["layout_image"]:
                logo = nid
                break
    if not logo:
        raise KeyError("[workflow_ops] logo LoadImage not found")
    roles["logo_image"] = logo

    # 8) pos/neg encoders: from control_apply.positive/negative
    pos = _link_target(_input(prompt, c_apply, "positive"))
    neg = _link_target(_input(prompt, c_apply, "negative"))
    # fallback: 아무 CLIPTextEncode 2개를 긍/부로
    if not pos or _class(prompt, pos) != "CLIPTextEncode":
        pos = None
    if not neg or _class(prompt, neg) != "CLIPTextEncode":
        neg = None
    if (pos is None) or (neg is None):
        found = [nid for nid, n in prompt.items() if isinstance(n, dict) and n.get("class_type") == "CLIPTextEncode"]
        if len(found) >= 2:
            pos = pos or found[0]
            neg = neg or found[1]
        elif len(found) == 1:
            pos = pos or found[0]
            neg = neg or found[0]
        else:
            raise KeyError("[workflow_ops] CLIPTextEncode nodes not found")
    roles["pos_text"] = pos
    roles["neg_text"] = neg

    return roles

def _role_node(prompt: Dict[str, Any], role: str) -> Dict[str, Any]:
    nid = prompt["_roles"][role]
    return prompt[nid]


def set_prompts(prompt: Dict[str, Any], positive: str, negative: str) -> None:
    _role_node(prompt, "pos_text")["inputs"]["text"] = positive
    _role_node(prompt, "neg_text")["inputs"]["text"] = negative

def set_layout_image(prompt: Dict[str, Any], image_path_or_name: str) -> None:
    _role_node(prompt, "layout_image")["inputs"]["image"] = image_path_or_name

def set_logo_image(prompt: Dict[str, Any], image_path_or_name: str) -> None:
    _role_node(prompt, "logo_image")["inputs"]["image"] = image_path_or_name

def set_control_strength(prompt: Dict[str, Any], strength: float,
                         start: float = 0.0, end: float = 1.0) -> None:
    n = _role_node(prompt, "control_apply")["inputs"]
    n["strength"] = float(strength)
    n["start_percent"] = float(start)
    n["end_percent"] = float(end)

def set_controlnet_model(prompt: Dict[str, Any], model_name: str) -> None:
    _role_node(prompt, "control_loader")["inputs"]["control_net_name"] = model_name

def set_ipadapter_weight(prompt: Dict[str, Any], weight: float,
                         weight_type: str = None,
                         embeds_scaling: str = None) -> None:
    n = _role_node(prompt, "ipadapter")["inputs"]
    n["weight"] = float(weight)
    if weight_type is not None:
        n["weight_type"] = weight_type
    if embeds_scaling is not None:
        n["embeds_scaling"] = embeds_scaling

def set_base_resolution(prompt: Dict[str, Any], w: int, h: int, batch: int = 1) -> None:
    n = _role_node(prompt, "latent_base")["inputs"]
    n["width"] = int(w); n["height"] = int(h); n["batch_size"] = int(batch)

def set_sampler(prompt: Dict[str, Any], seed: int, steps: int, cfg: float,
                sampler_name: str, scheduler: str, denoise: float = 1.0) -> None:
    n = _role_node(prompt, "ksampler")["inputs"]
    n["seed"] = int(seed); n["steps"] = int(steps); n["cfg"] = float(cfg)
    n["sampler_name"] = str(sampler_name); n["scheduler"] = str(scheduler)
    n["denoise"] = float(denoise)