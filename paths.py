"""Filesystem resolution for the one external checkout v7 depends on.

The pipeline runs on (at least) two machines with different layouts:

  * local orchestration VM: ~/Documents/new-training/{training/
    transfer_learning_v7, bitmind-subnet/gasbench}
  * the rented GPU box: /workspace/training/ holding these files flat, with
    the checkout wherever it was cloned.

The gasbench checkout MOVED on 2026-08-09 from the old sibling
`~/Documents/new-training/gasbench` into `bitmind-subnet/gasbench/`, which
silently broke every `parents[2] / "gasbench"` in this pipeline. So the one
external dependency -- the benchmark itself -- now resolves as:

  1. an environment variable, absolute and explicit:
       GASBENCH_SRC = .../gasbench/src   (the dir CONTAINING the gasbench pkg)
  2. otherwise an ancestor walk from this file's directory, checking the
     known layouts at every level up to filesystem root;
  3. for the dataset configs only, an installed `pip install gasbench`.

Resolvers return None when nothing is found; callers raise with the hint
below so every failure names the fix instead of a bare FileNotFoundError.

(Same design as video_v1/paths.py; kept as a copy so each pipeline stays
standalone. Divergence is visible in the probe filenames: this one checks
real_images.yaml.)
"""
from __future__ import annotations

import os
from pathlib import Path

_HERE = Path(__file__).resolve().parent

GASBENCH_HINT = (
    "Set GASBENCH_SRC=/path/to/gasbench/src (the directory containing the "
    "`gasbench` package, e.g. <bitmind-subnet>/gasbench/src), or clone "
    "bitmind-subnet / gasbench into any ancestor of this directory, or "
    "`pip install gasbench`.")


def _ancestors(limit: int = 8):
    p = _HERE
    for _ in range(limit):
        yield p
        if p.parent == p:
            return
        p = p.parent


def find_gasbench_src() -> Path | None:
    """The directory that CONTAINS the `gasbench` package (…/gasbench/src)."""
    env = os.environ.get("GASBENCH_SRC")
    if env:
        p = Path(env).expanduser().resolve()
        if (p / "gasbench" / "processing" / "transforms.py").exists():
            return p
        raise FileNotFoundError(
            f"$GASBENCH_SRC={env} does not contain "
            f"gasbench/processing/transforms.py -- it must be the src dir "
            f"holding the `gasbench` package, not the repo root.")
    for anc in _ancestors():
        for cand in (anc / "bitmind-subnet" / "gasbench" / "src",
                     anc / "gasbench" / "src"):
            if (cand / "gasbench" / "processing" / "transforms.py").exists():
                return cand
    return None


def find_config_dir() -> Path | None:
    """gasbench's dataset-config dir (holds real_images.yaml etc.).

    Tries, in order: $GASBENCH_SRC / ancestor-walk checkout, then an
    INSTALLED gasbench package (find_spec on the ROOT package only -- its
    __init__ is lazy PEP 562; probing a submodule spec would execute the
    heavy intermediate __init__s)."""
    src = find_gasbench_src()
    if src:
        cand = src / "gasbench" / "dataset" / "configs"
        if (cand / "real_images.yaml").exists():
            return cand
    try:
        import importlib.util
        spec = importlib.util.find_spec("gasbench")
        if spec and spec.submodule_search_locations:
            cand = Path(list(spec.submodule_search_locations)[0]) \
                / "dataset" / "configs"
            if (cand / "real_images.yaml").exists():
                return cand
    except Exception:
        pass
    return None
