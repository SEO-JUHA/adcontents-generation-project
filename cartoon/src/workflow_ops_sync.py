from __future__ import annotations

import json
from typing import Dict, Any

# Core Ops
def load_prompt(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    prompt = data["prompt"] if isinstance(data, dict) and "prompt" in data else data

    _index_links(prompt)
    prompt["_roles"] = _resolve_roles(prompt)
    return prompt


def _index_links(prompt: Dict[str, Any]) -> None:
    back = {}
    for nid, node in prompt.items():
        if not isinstance(node, dict):
            continue
        ins = node.get("inputs", {})
        for k, v in ins.items():
            if isinstance(v, list) and len(v) == 2 and isinstance(v[0], str):
                src = v[0]
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

    # 1) KSampler
    ksampler = _find_first_by_class(prompt, "KSampler")
    if not ksampler:
        raise KeyError("[workflow_ops] KSampler not found")
    roles["ksampler"] = ksampler

    # 2) EmptyLatentImage (from ksampler.inputs.latent_image)
    latent = _link_target(_input(prompt, ksampler, "latent_image"))
    if not latent or _class(prompt, latent) != "EmptyLatentImage":
        latent = _find_first_by_class(prompt, "EmptyLatentImage")
    if not latent:
        raise KeyError("[workflow_ops] EmptyLatentImage not found")
    roles["latent_base"] = latent

    # 3) ControlNetApplyAdvanced
    c_apply = _find_first_by_class(prompt, "ControlNetApplyAdvanced")
    if not c_apply:
        raise KeyError("[workflow_ops] ControlNetApplyAdvanced not found")
    roles["control_apply"] = c_apply

    # 4) ControlNetLoader (wired to c_apply.control_net)
    c_loader = _link_target(_input(prompt, c_apply, "control_net"))
    if not c_loader or _class(prompt, c_loader) != "ControlNetLoader":
        c_loader = _find_first_by_class(prompt, "ControlNetLoader")
    if not c_loader:
        raise KeyError("[workflow_ops] ControlNetLoader not found")
    roles["control_loader"] = c_loader

    # 5) Layout LoadImage (wired to c_apply.image)
    layout = _link_target(_input(prompt, c_apply, "image"))
    if not layout or _class(prompt, layout) != "LoadImage":
        layout = _find_first_by_class(prompt, "LoadImage")
    if not layout:
        raise KeyError("[workflow_ops] layout LoadImage not found")
    roles["layout_image"] = layout

    # 6) IPAdapterAdvanced
    ipa = _find_first_by_class(prompt, "IPAdapterAdvanced")
    if not ipa:
        raise KeyError("[workflow_ops] IPAdapterAdvanced not found")
    roles["ipadapter"] = ipa

    # 7) Logo LoadImage (wired to ipadapter.image)
    logo = _link_target(_input(prompt, ipa, "image"))
    if not logo or _class(prompt, logo) != "LoadImage":
        # pick another LoadImage different from layout
        for nid, node in prompt.items():
            if isinstance(node, dict) and node.get("class_type") == "LoadImage" and nid != roles["layout_image"]:
                logo = nid
                break
    if not logo:
        raise KeyError("[workflow_ops] logo LoadImage not found")
    roles["logo_image"] = logo

    # 8) CLIPTextEncode (pos/neg) — ideally from c_apply inputs
    pos = _link_target(_input(prompt, c_apply, "positive"))
    neg = _link_target(_input(prompt, c_apply, "negative"))
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

# mutators
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
                         weight_type: str | None = None,
                         embeds_scaling: str | None = None) -> None:
    n = _role_node(prompt, "ipadapter")["inputs"]
    n["weight"] = float(weight)
    if weight_type is not None:
        n["weight_type"] = weight_type
    if embeds_scaling is not None:
        n["embeds_scaling"] = embeds_scaling

def set_base_resolution(prompt: Dict[str, Any], w: int, h: int, batch: int = 1) -> None:
    n = _role_node(prompt, "latent_base")["inputs"]
    n["width"] = int(w)
    n["height"] = int(h)
    n["batch_size"] = int(batch)

def set_sampler(prompt: Dict[str, Any], seed: int, steps: int, cfg: float,
                sampler_name: str, scheduler: str, denoise: float = 1.0) -> None:
    n = _role_node(prompt, "ksampler")["inputs"]
    n["seed"] = int(seed)
    n["steps"] = int(steps)
    n["cfg"] = float(cfg)
    n["sampler_name"] = str(sampler_name)
    n["scheduler"] = str(scheduler)
    n["denoise"] = float(denoise)

# FastAPI
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Mapping

router = APIRouter(tags=["workflow-ops"])

@router.get("/ping")
def ping():
    return {"ok": True}

def _ensure_roles(prompt: Dict[str, Any]) -> None:
    if "_backlinks" not in prompt:
        _index_links(prompt)
    if "_roles" not in prompt:
        prompt["_roles"] = _resolve_roles(prompt)

class LoadRequest(BaseModel):
    path: str

class PromptIn(BaseModel):
    prompt: Dict[str, Any]

class PromptOut(BaseModel):
    prompt: Dict[str, Any]

@router.post("/load", response_model=PromptOut)
def api_load(req: LoadRequest):
    try:
        return PromptOut(prompt=load_prompt(req.path))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"invalid prompt: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"load_prompt failed: {e}")

class SetPromptsRequest(PromptIn):
    positive: str
    negative: str

@router.post("/set-prompts", response_model=PromptOut)
def api_set_prompts(req: SetPromptsRequest):
    try:
        _ensure_roles(req.prompt)
        set_prompts(req.prompt, req.positive, req.negative)
        return PromptOut(prompt=req.prompt)
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"invalid prompt roles: {e}")

class SetImageRequest(PromptIn):
    image_path_or_name: str

@router.post("/set-layout-image", response_model=PromptOut)
def api_set_layout_image(req: SetImageRequest):
    try:
        _ensure_roles(req.prompt)
        set_layout_image(req.prompt, req.image_path_or_name)
        return PromptOut(prompt=req.prompt)
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"invalid prompt roles: {e}")

@router.post("/set-logo-image", response_model=PromptOut)
def api_set_logo_image(req: SetImageRequest):
    try:
        _ensure_roles(req.prompt)
        set_logo_image(req.prompt, req.image_path_or_name)
        return PromptOut(prompt=req.prompt)
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"invalid prompt roles: {e}")

class SetControlStrengthRequest(PromptIn):
    strength: float
    start: float = 0.0
    end: float = 1.0

@router.post("/set-control-strength", response_model=PromptOut)
def api_set_control_strength(req: SetControlStrengthRequest):
    try:
        _ensure_roles(req.prompt)
        set_control_strength(req.prompt, req.strength, req.start, req.end)
        return PromptOut(prompt=req.prompt)
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"invalid prompt roles: {e}")

class SetControlNetModelRequest(PromptIn):
    model_name: str

@router.post("/set-controlnet-model", response_model=PromptOut)
def api_set_controlnet_model(req: SetControlNetModelRequest):
    try:
        _ensure_roles(req.prompt)
        set_controlnet_model(req.prompt, req.model_name)
        return PromptOut(prompt=req.prompt)
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"invalid prompt roles: {e}")

class SetIPAdapterWeightRequest(PromptIn):
    weight: float
    weight_type: Optional[str] = None
    embeds_scaling: Optional[str] = None

@router.post("/set-ipadapter-weight", response_model=PromptOut)
def api_set_ipadapter_weight(req: SetIPAdapterWeightRequest):
    try:
        _ensure_roles(req.prompt)
        set_ipadapter_weight(req.prompt, req.weight, req.weight_type, req.embeds_scaling)
        return PromptOut(prompt=req.prompt)
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"invalid prompt roles: {e}")

class SetBaseResolutionRequest(PromptIn):
    w: int
    h: int
    batch: int = 1

@router.post("/set-base-resolution", response_model=PromptOut)
def api_set_base_resolution(req: SetBaseResolutionRequest):
    try:
        _ensure_roles(req.prompt)
        set_base_resolution(req.prompt, req.w, req.h, req.batch)
        return PromptOut(prompt=req.prompt)
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"invalid prompt roles: {e}")

class SetSamplerRequest(PromptIn):
    seed: int
    steps: int
    cfg: float
    sampler_name: str
    scheduler: str
    denoise: float = 1.0

@router.post("/set-sampler", response_model=PromptOut)
def api_set_sampler(req: SetSamplerRequest):
    try:
        _ensure_roles(req.prompt)
        set_sampler(
            req.prompt,
            seed=req.seed,
            steps=req.steps,
            cfg=req.cfg,
            sampler_name=req.sampler_name,
            scheduler=req.scheduler,
            denoise=req.denoise,
        )
        return PromptOut(prompt=req.prompt)
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"invalid prompt roles: {e}")

__all__ = [
    "load_prompt",
    "set_prompts",
    "set_layout_image",
    "set_logo_image",
    "set_control_strength",
    "set_controlnet_model",
    "set_ipadapter_weight",
    "set_base_resolution",
    "set_sampler",
    "router",
]