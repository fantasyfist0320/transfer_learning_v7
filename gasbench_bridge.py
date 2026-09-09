"""Single import point for the gasbench code that defines what evaluation does.

The benchmark's own transform stack and scorer live in this repo. Reusing them
-- rather than reimplementing an augmentation policy, as v1/v2/v4 each did --
makes the training distribution match the evaluation distribution by
construction. Two things fall out of that which no previous version had:

  * vertical flips, applied to 75% of eval samples (levels 1-3)
  * a WebP roundtrip in the robustness chain; VP8 intra coding leaves a
    different artifact family than JPEG's DCT, and a detector that survives
    repeated JPEG can still collapse on it

Loading is deliberately awkward for one reason: `gasbench.processing.__init__`
eagerly imports `.archive`, which needs `huggingface_hub`, and `.media`, which
needs the constants module. `transforms.py` itself has no relative imports and
needs only math/random/scipy/numpy/cv2/PIL/torch/torchvision, so we load it
straight from its file and skip the package __init__ entirely. The normal
import is tried first so an installed gasbench wins.

We do NOT vendor a copy. The benchmark scores submissions with *its* version of
these transforms; a vendored snapshot would silently drift out of sync and the
training distribution would stop matching eval without anything failing loudly.
`describe()` records the resolved version so drift is at least visible in run
metadata -- the local checkout follows gasbench main (0.8.3 as of 2026-08-10)
while the root pyproject pins an older git tag, so the two can differ.

The checkout is resolved through paths.py ($GASBENCH_SRC -> ancestor walk ->
pip package) because the checkout moved into bitmind-subnet/ on 2026-08-09 and
the old parents[2]-relative constant silently pointed at nothing.
"""
from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path

import paths


def _gasbench_src() -> Path:
    src = paths.find_gasbench_src()
    if src is None:
        raise ImportError(f"gasbench checkout not found. {paths.GASBENCH_HINT}")
    return src


def _load_module_by_path(mod_name: str, file_path: Path):
    """Execute a single module file without importing its parent package."""
    spec = importlib.util.spec_from_file_location(mod_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot build a module spec for {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


def _import_transforms():
    try:
        return importlib.import_module("gasbench.processing.transforms")
    except Exception:
        # transforms.py has no relative imports (only math/random/scipy/numpy/
        # cv2/os/io/PIL/torch/torchvision), so it can be executed straight from
        # its file, skipping the package __init__ that eagerly pulls
        # huggingface_hub.
        return _load_module_by_path(
            "_gb_transforms",
            _gasbench_src() / "gasbench" / "processing" / "transforms.py")


def _import_metrics():
    try:
        return importlib.import_module("gasbench.benchmarks.utils.metrics")
    except Exception:
        pass
    # metrics.py needs only numpy and `from ...logger import get_logger`
    # (stdlib logging), but importing it by its dotted path initialises the
    # parent packages, and the current checkout's benchmarks/__init__ eagerly
    # pulls image_bench/video_bench -> dataset config -> modelscope +
    # onnxruntime. gasbench's ROOT __init__ is lazy (PEP 562), so: import the
    # root, then register __init__-free stub modules for the two intermediate
    # packages and import just the metrics leaf. The relative `...logger`
    # import resolves through the real root package, untouched.
    import types

    src = _gasbench_src()
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    importlib.import_module("gasbench")
    for name, rel in (("gasbench.benchmarks", "benchmarks"),
                      ("gasbench.benchmarks.utils", "benchmarks/utils")):
        if name not in sys.modules:
            stub = types.ModuleType(name)
            stub.__path__ = [str(src / "gasbench" / rel)]
            stub.__package__ = name
            sys.modules[name] = stub
    return importlib.import_module("gasbench.benchmarks.utils.metrics")


_transforms = _import_transforms()
_metrics = _import_metrics()

# --- the eval-time transform stack -----------------------------------------
# apply_random_augmentations(inputs, target_size, mask=None, level_probs=None,
#                            level=None, crop_prob=0.5, seed=None)
#     -> (aug_image_hwc_uint8, aug_mask, level, params)
#   With prob crop_prob a RandomCrop at per-axis scale U(0.35, 0.99) (floored at
#   224/target), then ResizeShortestEdge(target) = center-crop to the target
#   aspect then resize. Then a difficulty level drawn 25% each:
#     0 nothing | 1 +hflip +vflip | 2 +CS(0-1) +CC(0-1)
#     3 +CS(0-2) +CC(0-2) +GNC(0-2) +GB(0-2)
apply_random_augmentations = _transforms.apply_random_augmentations

# apply_robustness_augmentations(image_array, target_size, seed=None,
#                                jpeg_quality=55, scale_factor=0.5,
#                                webp_quality=75) -> same 4-tuple
#   The deterministic chain behind the benchmark's aug_binary_* metrics:
#   downscale 0.5 (INTER_AREA, floored at 256px) -> upscale (INTER_LINEAR)
#   -> JPEG q55 -> WebP q75 -> JPEG q80 -> base transforms.
apply_robustness_augmentations = _transforms.apply_robustness_augmentations

get_base_transforms = _transforms.get_base_transforms
compress_image_jpeg_pil = _transforms.compress_image_jpeg_pil
compress_image_webp_pil = _transforms.compress_image_webp_pil

# --- the canonical scorer ---------------------------------------------------
# Metrics.compute_sn34_score(alpha=1.2, beta=1.8, agg="geomean"):
#   mcc_norm    = clip((mcc + 1) / 2, 0, 1) ** alpha
#   brier_score = max(0, (0.25 - brier) / 0.25) ** beta
#   score       = sqrt(mcc_norm * brier_score)
# beta > alpha, so calibration outweighs discrimination. This is why v4's
# `mcc - brier` selection rule is wrong and why temperature scaling matters.
Metrics = _metrics.Metrics

__all__ = [
    "apply_random_augmentations",
    "apply_robustness_augmentations",
    "get_base_transforms",
    "compress_image_jpeg_pil",
    "compress_image_webp_pil",
    "Metrics",
    "describe",
]


def describe() -> dict:
    """Provenance for run metadata, so transform drift is visible after the fact."""
    try:
        import gasbench

        version = getattr(gasbench, "__version__", "unknown")
    except Exception:
        version = "unimported (loaded by path)"
    return {
        "gasbench_version": version,
        "transforms_file": getattr(_transforms, "__file__", "?"),
        "metrics_file": getattr(_metrics, "__file__", "?"),
    }
