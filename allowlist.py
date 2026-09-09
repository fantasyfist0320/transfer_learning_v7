"""Render and screen the shipped model.py.

The submission analyzer's implementation is not public; Safetensors.md is the
contract. This mirror enforces: import roots limited to the documented set,
and none of the blocked builtin calls anywhere in the file -- `getattr` and
`json` being the two that shape the template's design (all architecture
constants are substituted as literals, never looked up or parsed at load
time).
"""
from __future__ import annotations

import ast
from pathlib import Path

ALLOWED_ROOTS = {
    "torch", "torchvision", "torchaudio", "transformers", "timm", "einops",
    "safetensors", "flash_attn", "PIL", "cv2", "skimage", "decord", "numpy",
    "scipy", "fvcore", "ultralytics", "math", "functools", "typing",
    "collections", "dataclasses", "enum", "abc", "pathlib",
}
BLOCKED_CALLS = {"eval", "exec", "compile", "__import__", "getattr",
                 "setattr", "globals", "locals"}

_TEMPLATE = Path(__file__).resolve().parent / "templates" / "inference_model.py"

# CLIPVisionConfig constructor keys worth carrying from the trained config.
_CLIP_KEYS = ("hidden_size", "intermediate_size", "num_hidden_layers",
              "num_attention_heads", "num_channels", "image_size",
              "patch_size", "hidden_act", "layer_norm_eps",
              "attention_dropout", "initializer_range", "initializer_factor",
              "projection_dim")


def scan_allowlist(src: str) -> list[str]:
    tree = ast.parse(src)
    bad: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] not in ALLOWED_ROOTS:
                    bad.append(f"import {a.name}")
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root and root not in ALLOWED_ROOTS:
                bad.append(f"from {node.module} import ...")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in BLOCKED_CALLS:
                bad.append(f"{node.func.id}()")
    return bad


def render_template(specs, weights, image_size: int,
                    clip_cfg: dict | None = None) -> str:
    """Substitute every architecture constant as a literal.

    `specs` are BranchSpec instances; `clip_cfg` is the trained CLIP vision
    config's to_dict() (required iff a clip branch ships) so the offline
    rebuild uses the exact trained architecture, not library defaults.
    """
    defs = []
    for s in specs:
        defs.append({
            "name": s.name, "family": s.family,
            "timm_id": s.model_id if s.family in ("timm", "dct") else "",
            "native": int(s.native_size), "dim": int(s.feat_dim),
            "head": s.head, "mean": list(s.norm_mean),
            "std": list(s.norm_std),
        })
    has_timm = any(s.family in ("timm", "dct") for s in specs)
    has_clip = any(s.family == "hf_clip" for s in specs)
    if has_clip:
        if not clip_cfg:
            raise ValueError("clip branch shipped but clip_cfg not provided")
        clip_literal = repr({k: clip_cfg[k] for k in _CLIP_KEYS
                             if k in clip_cfg})
    else:
        clip_literal = "{}"

    src = _TEMPLATE.read_text()
    src = (src.replace("__BRANCH_DEFS__", repr(defs))
              .replace("__FUSION_W__", repr([float(w) for w in weights]))
              .replace("__IMAGE_SIZE__", str(int(image_size)))
              .replace("__CLIP_CFG__", clip_literal)
              .replace("__TIMM_IMPORT__",
                       "import timm" if has_timm else ""))
    bad = scan_allowlist(src)
    if bad:
        raise ValueError(f"rendered template violates the allowlist: {bad}")
    compile(src, "model.py", "exec")   # syntax gate (build-side only)
    return src
