"""Walk a gasbench cache directory and emit one manifest row per image.

Why this exists rather than v4's make_manifest.py: that script rglob'd two
directories and produced `path,label,media_type,dataset`. Everything the
sampler and the split logic need -- generator_family, content_category, source
repo, dimensions, container -- was absent, so v4 could not balance or split on
any of it.

Three facts about the cache drive the implementation:

  1. Layout is FLAT: <cache_dir>/datasets/<config.name>/{dataset_info.json,
     sample_metadata.json, samples/}. `media_type` is a JSON field, never a
     path component, so the tree cannot be used to classify. gasstation
     datasets insert an ISO-week level (<name>/2025W40/...), which is why we
     probe for dataset_info.json instead of globbing a fixed depth.

  2. dataset_info.json does NOT carry generator_family, generator_variant or
     content_category -- the cache writer drops them. They live only in the
     YAML registry and must be joined on `name`. `name` is the primary key;
     `path` is not unique (bitmind/FakeClue backs 9 separate datasets).

  3. sample_metadata.json carries no width, no height and no original format.
     Dimensions require opening every file with PIL. Filenames are
     img_{index:06d}{ext} with a monotonic index that survives eviction, so
     indices are neither contiguous nor equal to the count, and extensions vary
     within one dataset (.jpg/.png/.webp/.mpo all occur). Never assume either.

Note on container forensics: the cache writer re-encodes through PIL with
default settings, so original JPEG quantization tables are destroyed and
homogenized across labels. The file extension still reflects the *original*
detected format, and the original encode's artifacts survive in the pixels, so
`ext` remains a real signal even though `file_bytes` and qtables do not measure
what they would on the source data. audit_shortcuts.py quantifies which of
these actually reach the model.
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import yaml
from PIL import Image

import paths

Image.MAX_IMAGE_PIXELS = 512_000_000  # the corpus contains ~100 Mpx images

_OVERRIDES = Path(__file__).resolve().parent / "overrides.yaml"

FOLD_TARGET = "diverse"  # single-label categories are folded here


def load_overrides(path: Path | None = None) -> dict:
    """Load the overrides file. A MISSING EXPLICIT PATH IS FATAL.

    Silently falling back to {} loses every semisynthetic correction, the
    category fixes and extra_datasets, and the resulting manifest trains a
    materially different model with no error anywhere -- exactly the failure
    that produced an unusable manifest_ship.parquet on 2026-08-06.
    """
    explicit = path is not None
    path = Path(path) if path else _OVERRIDES
    if not path.exists():
        if explicit:
            raise SystemExit(
                f"--overrides {path} does not exist. Refusing to build a "
                f"manifest without the corrections it carries (semisynthetic "
                f"kinds, category fixes, extra_datasets). Copy the file to "
                f"this machine and re-run.")
        print(f"[warn] no overrides file at {path}; building with registry "
              f"values only")
        return {}
    return yaml.safe_load(path.read_text()) or {}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def load_registry(config_dir: Path | None = None) -> dict[str, dict]:
    """Return {dataset_name: yaml_entry} for the image modality.

    Parsed straight from YAML rather than through
    gasbench.dataset.config.load_benchmark_datasets_from_yaml() for two
    reasons: it needs no gasbench install (only pyyaml), and the gasbench
    dataclass whitelists its fields and silently drops `generator_variant`,
    which appears on 29 image entries.
    """
    config_dir = Path(config_dir) if config_dir else paths.find_config_dir()
    if config_dir is None:
        raise FileNotFoundError(
            f"gasbench dataset configs not found. {paths.GASBENCH_HINT}")
    registry: dict[str, dict] = {}
    for fname in ("real_images.yaml", "synthetic_images.yaml"):
        path = config_dir / fname
        if not path.exists():
            raise FileNotFoundError(
                f"registry file not found: {path}\n"
                f"Pass --config-dir if the gasbench checkout lives elsewhere."
            )
        for entry in yaml.safe_load(path.read_text())["datasets"]:
            name = entry["name"]
            if name in registry:
                raise ValueError(f"duplicate dataset name in registry: {name}")
            registry[name] = entry
    return registry


def build_release_map(registry: dict[str, dict], overrides: dict) -> dict[str, str]:
    """Map each dataset to its sampling GROUP, collapsing shards of one release.

    The sampler's cascade hands every named dataset an equal (sqrt-weighted)
    share of its cell, so a release split into nine `name` entries takes nine
    shares. Measured on this registry, human-faces-dataset-r1..r9 is 9 of the
    43 fake-face groups and draws 20.9% of all fake-face sampling mass. That is
    one release dominating a fifth of the signal for the largest cell in the
    corpus.

    Only the sampler's group key changes; `dataset` stays intact for splitting
    and reporting.
    """
    mapping = {n: n for n in registry}
    for shard in (overrides or {}).get("shard_groups") or []:
        present = [n for n in shard if n in registry]
        missing = [n for n in shard if n not in registry]
        if missing:
            print(f"[warn] shard_groups names not in the registry: {missing}")
        for n in present:
            mapping[n] = present[0]
    return mapping


def apply_category_overrides(registry: dict[str, dict],
                             overrides: dict) -> dict[str, str]:
    """Return {dataset_name: corrected_category}, leaving the registry untouched.

    Several registry entries are miscategorised in a way their own
    `include_paths` disproves -- fakeclue-*-satellite is filed as `diverse`
    while extracting data/{real,fake}/satellite/. The satellite fix is not
    cosmetic: it turns `aerial` from 3 real / 0 fake into 4 real / 1 fake, i.e.
    from a category that must be folded away into one that can be balanced.
    """
    fixes = (overrides or {}).get("category_overrides") or {}
    unknown = sorted(set(fixes) - set(registry))
    if unknown:
        print(f"[warn] category_overrides names not in the registry: {unknown}")
    return {name: fixes.get(name, e.get("content_category") or "unknown")
            for name, e in registry.items()}


def resolve_kind(registry: dict[str, dict], overrides: dict) -> dict[str, str]:
    """Return {dataset_name: 'real' | 'synthetic' | 'semisynthetic'}.

    The registry's own `media_type` is authoritative for `real`, and unreliable
    for the split within the fake half: it tags 2 of 91 synthetic datasets as
    semisynthetic while `generator_family: mixed-generation` lumps 48 datasets
    of both kinds together. `overrides.yaml:semisynthetic_datasets` supplies the
    corrections; anything unlisted keeps its registry value.

    This is the axis the sampler balances on in `data.py:balanced_weights`. A
    dataset landing on the wrong side is a real cost -- it either dilutes the
    upweight or hands replay to an image that did not need it -- so the list is
    reviewed, not inferred.
    """
    listed = set((overrides or {}).get("semisynthetic_datasets") or [])
    unknown = sorted(listed - set(registry))
    if unknown:
        print(f"[warn] semisynthetic_datasets names not in the registry: {unknown}")
    out: dict[str, str] = {}
    promoted = 0
    for name, e in registry.items():
        mt = (e.get("media_type") or "").strip()
        if mt == "real":
            out[name] = "real"
        elif name in listed:
            out[name] = "semisynthetic"
            promoted += mt != "semisynthetic"
        else:
            out[name] = "semisynthetic" if mt == "semisynthetic" else "synthetic"
    n_semi = sum(1 for v in out.values() if v == "semisynthetic")
    print(f"[registry] kind: {sum(1 for v in out.values() if v == 'real')} real, "
          f"{sum(1 for v in out.values() if v == 'synthetic')} synthetic, "
          f"{n_semi} semisynthetic ({promoted} promoted by overrides.yaml)")
    return out


def compute_category_folding(registry: dict[str, dict],
                             fixed: dict[str, str] | None = None) -> dict[str, str]:
    """Map each content_category to the category the sampler should balance on.

    A category carrying only one label is a perfect shortcut: on the current
    registry, food/aerial/medical/vehicles/action/plants/currency hold 18 real
    datasets and zero fakes, so "is this aerial?" answers "is this real?" with
    no error. It also breaks global balance -- the cascade in data.py splits
    each category's mass evenly across the labels *present*, so a single-label
    category donates all of its mass to one side and P(fake) lands at 0.45
    instead of 0.50.

    Folding those categories into `diverse` fixes both: P(fake) becomes 0.5000
    and P(fake|category) becomes 0.5000 for every surviving category.

    Computed from the registry rather than hardcoded so it stays correct when
    the registry gains or loses datasets.
    """
    labels_per_cat: dict[str, set[int]] = {}
    for name, entry in registry.items():
        cat = (fixed or {}).get(name) or entry.get("content_category") or "unknown"
        labels_per_cat.setdefault(cat, set()).add(media_type_to_label(entry["media_type"]))
    mapping = {}
    for cat, labels in labels_per_cat.items():
        mapping[cat] = FOLD_TARGET if len(labels) < 2 else cat
    return mapping


def media_type_to_label(media_type: str) -> int:
    """real -> 0, synthetic and semisynthetic -> 1.

    semisynthetic (receipts-i2i, face-swap) is partially generated content and
    the benchmark scores it as the positive class, so it belongs with fake.
    """
    return 0 if str(media_type).strip().lower() == "real" else 1


def normalize_source_format(value) -> str:
    """The registry uses both `jpg` and `jpeg`, and leaves 16 entries null.

    Note this names the *archive* format on HuggingFace (parquet/zip/tar/...),
    not the image codec. The codec is `file_format`/`ext`, measured from disk.
    """
    if value is None:
        return "unknown"
    v = str(value).strip().lower()
    return "jpeg" if v == "jpg" else (v or "unknown")


def normalize_ext(suffix: str) -> str:
    e = suffix.lower().lstrip(".")
    return "jpg" if e == "jpeg" else e


def parse_file_index(fname: str) -> int:
    """img_000123.jpg -> 123, else -1.

    Used by build_splits.py as a fallback block key. The cache writer assigns
    indices in archive order, so contiguous index ranges are contiguous source
    material -- which is what makes block-wise splitting work when
    `source_file` is uninformative. The index is monotonic and survives
    eviction, so it is neither contiguous nor equal to the sample count.
    """
    stem = Path(fname).stem
    if "_" not in stem:
        return -1
    tail = stem.rsplit("_", 1)[-1]
    return int(tail) if tail.isdigit() else -1


# ---------------------------------------------------------------------------
# Cache discovery
# ---------------------------------------------------------------------------

def find_dataset_dirs(cache_dir: Path, max_depth: int = 2) -> list[tuple[str, Path]]:
    """Locate every directory holding a dataset_info.json.

    Returns (dataset_name, dir) pairs. The name comes from the directory
    immediately under datasets/, so the gasstation ISO-week layout
    (<name>/2025W40/) resolves to the same dataset name as its siblings and its
    weeks accumulate into one logical dataset.
    """
    root = cache_dir / "datasets"
    if not root.is_dir():
        # Tolerate being pointed straight at a directory of datasets.
        root = cache_dir
    if not root.is_dir():
        raise FileNotFoundError(f"no datasets directory under {cache_dir}")

    found: list[tuple[str, Path]] = []
    for top in sorted(p for p in root.iterdir() if p.is_dir()):
        if (top / "dataset_info.json").exists():
            found.append((top.name, top))
            continue
        # Probe one level down for the ISO-week variant.
        for sub in sorted(p for p in top.iterdir() if p.is_dir()):
            if (sub / "dataset_info.json").exists():
                found.append((top.name, sub))
        if max_depth > 2:  # reserved; the writer never nests deeper than this
            pass
    return found


def read_sample_metadata(dataset_dir: Path) -> dict[str, dict]:
    """Authoritative {filename: metadata} map, from sample_metadata.json.

    Preferred over listing samples/ because it is what gasbench itself counts
    (_get_cached_count uses len(metadata)), so a mismatch between the two is a
    real signal worth reporting rather than papering over.

    `source_file` is the only field here that earns its keep downstream: it
    names the parquet shard or archive the image came from, and build_splits.py
    uses it to carve val_id by source block instead of by row. Several datasets
    in this corpus are video-derived (inst-it-dataset-videos-*, vtuav,
    bdd100k, FDDB, fakeclue-*-ffpp), so consecutive rows are near-duplicate
    frames; a row-level split puts frame t in train and frame t+1 in val and
    the resulting val number is meaningless. v4's make_manifest.py does exactly
    that.
    """
    meta_path = dataset_dir / "sample_metadata.json"
    if not meta_path.exists():
        return {}
    try:
        meta = json.loads(meta_path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    return meta if isinstance(meta, dict) else {}


# ---------------------------------------------------------------------------
# Per-image probing
# ---------------------------------------------------------------------------

def probe_image(path_str: str) -> dict | None:
    """Header-only read of one image. Returns None if unreadable.

    Image.open parses the header lazily, so .size and .format come back without
    decoding pixel data -- fast enough to run over the whole corpus.
    """
    try:
        p = Path(path_str)
        size_bytes = p.stat().st_size
        with Image.open(p) as im:
            w, h = im.size
            fmt = (im.format or "").upper()
            mode = im.mode or ""
    except Exception:
        return None
    if w <= 0 or h <= 0:
        return None
    return {
        "path": path_str,
        "width": int(w),
        "height": int(h),
        "file_format": fmt,
        "pil_mode": mode,   # 'L' is its own shortcut: greyscale corpora skew real
        "file_bytes": int(size_bytes),
    }


def probe_many(paths: list[str], workers: int) -> tuple[dict[str, dict], list[str]]:
    """Probe in parallel. Returns ({path: info}, [failed paths])."""
    info: dict[str, dict] = {}
    failed: list[str] = []
    if workers <= 1:
        for p in paths:
            r = probe_image(p)
            (info.__setitem__(p, r) if r else failed.append(p))
        return info, failed

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(probe_image, p): p for p in paths}
        done = 0
        for fut in as_completed(futures):
            p = futures[fut]
            try:
                r = fut.result()
            except Exception:
                r = None
            if r:
                info[p] = r
            else:
                failed.append(p)
            done += 1
            if done % 5000 == 0:
                print(f"  probed {done:,}/{len(paths):,}", flush=True)
    return info, failed


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def build_manifest(cache_dir: Path, config_dir: Path | None = None,
                   workers: int = 8, overrides_path: Path | None = None,
                   image_size: int = 512) -> pd.DataFrame:
    registry = load_registry(config_dir)
    overrides = load_overrides(overrides_path)
    # Locally generated datasets (e.g. generate_swaps.py output) are unknown
    # to the gasbench registry; overrides.yaml:extra_datasets registers them
    # so they join the manifest with a label instead of being skipped.
    for e in overrides.get("extra_datasets") or []:
        entry = {
            "media_type": e.get("media_type", "synthetic"),
            "generator_family": e.get("generator_family", "unknown"),
            "content_category": e.get("content_category", "unknown"),
            "path": e.get("path", f"local/{e['name']}"),
            "source_format": e.get("source_format", "image_folder"),
        }
        registry.setdefault(e["name"], entry)
        print(f"[registry] +local dataset {e['name']} "
              f"({entry['media_type']}, {entry['content_category']})")
    fixed = apply_category_overrides(registry, overrides)
    release = build_release_map(registry, overrides)
    n_col = len(set(release.values()))
    if n_col != len(registry):
        print(f"[registry] {len(registry)} datasets collapse to {n_col} sampling "
              f"groups (shards of one release merged)")
    n_fixed = sum(1 for n, c in fixed.items()
                  if c != (registry[n].get("content_category") or "unknown"))
    if n_fixed:
        print(f"[registry] applied {n_fixed} content_category corrections")
    kinds = resolve_kind(registry, overrides)
    fold = compute_category_folding(registry, fixed)
    folded = sorted({c for c, t in fold.items() if c != t})
    if folded:
        print(f"[registry] single-label categories folded into '{FOLD_TARGET}': "
              f"{', '.join(folded)}")

    dataset_dirs = find_dataset_dirs(Path(cache_dir))
    print(f"[cache] {len(dataset_dirs)} dataset directories under {cache_dir}")

    on_disk = {name for name, _ in dataset_dirs}
    unknown = sorted(on_disk - set(registry))
    missing = sorted(set(registry) - on_disk)
    if unknown:
        print(f"[warn] on disk but not in the registry ({len(unknown)}): "
              f"{', '.join(unknown[:10])}{' ...' if len(unknown) > 10 else ''}")
    if missing:
        print(f"[warn] in the registry but not downloaded ({len(missing)}): "
              f"{', '.join(missing[:10])}{' ...' if len(missing) > 10 else ''}")

    rows: list[dict] = []
    absent_files = 0
    for name, ddir in dataset_dirs:
        entry = registry.get(name)
        if entry is None:
            continue  # unknown datasets carry no label; skip rather than guess
        meta = read_sample_metadata(ddir)
        if not meta:
            print(f"[warn] {name}: sample_metadata.json empty or unreadable")
            continue
        samples_dir = ddir / "samples"
        raw_category = entry.get("content_category") or "unknown"
        cat_fixed = fixed.get(name, raw_category)
        # The ISO-week directory name for gasstation, "" otherwise.
        iso_week = ddir.name if ddir.name != name else ""
        for fname in sorted(meta):
            fpath = samples_dir / fname
            if not fpath.is_file():
                absent_files += 1
                continue
            m = meta[fname] if isinstance(meta[fname], dict) else {}
            rows.append({
                "image_id": f"{name}/{iso_week}/{fname}" if iso_week else f"{name}/{fname}",
                "path": str(fpath),
                "dataset": name,
                "label": media_type_to_label(entry["media_type"]),
                "media_type": entry["media_type"],
                # real / synthetic / semisynthetic, after overrides.yaml. The
                # sampler balances on this; see data.py:balanced_weights.
                "kind": kinds.get(name, "synthetic"),
                "generator_family": entry.get("generator_family") or "unknown",
                "generator_variant": entry.get("generator_variant") or "",
                "content_category": raw_category,          # YAML verbatim
                "content_category_fixed": cat_fixed,        # after overrides.yaml
                "category": fold.get(cat_fixed, cat_fixed),  # + single-label folding
                # Sampler grouping key. Shards of one release share a group so a
                # single release cannot take N shares of its cell; `dataset`
                # stays intact for splitting and reporting.
                "group": release.get(name, name),
                "source_path": entry["path"],
                "source_format": normalize_source_format(entry.get("source_format")),
                "source_file": str(m.get("source_file") or ""),
                "iso_week": iso_week,
                "file_index": parse_file_index(fname),
                "ext": normalize_ext(fpath.suffix),
            })

    if absent_files:
        print(f"[warn] {absent_files:,} files listed in sample_metadata.json "
              f"are missing from disk")
    if not rows:
        raise RuntimeError(
            f"no images found under {cache_dir}. The metadata may describe an "
            f"interrupted download -- check that samples/ directories are populated."
        )

    df = pd.DataFrame(rows)
    print(f"[probe] reading headers for {len(df):,} images with {workers} workers")
    info, failed = probe_many(df["path"].tolist(), workers)
    if failed:
        print(f"[warn] {len(failed):,} images unreadable, dropped "
              f"(e.g. {failed[0] if failed else ''})")
    df = df[df["path"].isin(info)].reset_index(drop=True)

    probe_df = pd.DataFrame([info[p] for p in df["path"]])
    for col in ("width", "height", "file_format", "pil_mode", "file_bytes"):
        df[col] = probe_df[col].to_numpy()

    df["min_side"] = df[["width", "height"]].min(axis=1)
    df["max_side"] = df[["width", "height"]].max(axis=1)
    df["aspect"] = df["max_side"] / df["min_side"]
    df["megapixels"] = df["width"] * df["height"] / 1e6
    df["bytes_per_pixel"] = df["file_bytes"] / (df["width"] * df["height"])

    # The resample factor the benchmark will apply to this image.
    #
    # The base eval pass is `augment_level=0, crop_prob=0.0`
    # (image_bench.py:245-246, not overridable from the CLI), so eval is
    # exactly ResizeShortestEdge(S): centre-crop to the target aspect, then
    # cv2.resize to (S, S). For a square target the crop is min_side x
    # min_side, so
    #
    #     m_eval = S / min_side          exactly, with no randomness.
    #
    # Two consequences worth stating plainly. First, log m = log S - log
    # min_side, so any gap between P(log m | real) and P(log m | fake) is
    # translation-invariant in S -- choosing a different input resolution
    # moves both distributions by the same constant and cannot reduce the
    # separation. The resolution shortcut is not tunable via S; only the
    # training augmentation can make the model stop relying on it. Second,
    # m > 1 means the benchmark upsamples the image, destroying nothing but
    # adding a spectral cliff; m < 1 means it downsamples with cv2
    # INTER_LINEAR, which does NOT antialias and therefore aliases the
    # generator's high-frequency fingerprint into the visible band.
    df["m_eval"] = image_size / df["min_side"]
    df["split"] = ""  # written by build_splits.py

    return df


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _overlap_report(df: pd.DataFrame, n_bins: int = 12) -> None:
    """How much of the resample-factor range can be served with both labels.

    This is the honest price ticket for the resolution shortcut. Training can
    only make the model m-invariant on the range where both labels supply
    images; outside that range there is nothing to be invariant between, and
    m alone identifies the label. Printing the covered mass here means the
    trade is a measured number rather than an argument.
    """
    import numpy as np

    lo, hi = df.m_eval.quantile(0.01), df.m_eval.quantile(0.99)
    if not (lo > 0 and hi > lo):
        return
    edges = np.geomspace(lo, hi, n_bins + 1)
    idx = np.clip(np.digitize(df.m_eval.to_numpy(), edges) - 1, 0, n_bins - 1)
    both, covered = 0, 0.0
    print(f"\n-- m-bin coverage ({n_bins} log-spaced bins over "
          f"[{lo:.2f}, {hi:.2f}]) --")
    print(f"{'bin':>4}{'m_lo':>9}{'m_hi':>9}{'real':>9}{'fake':>9}   status")
    for k in range(n_bins):
        sel = idx == k
        r = int(((df.label == 0) & sel).sum())
        f = int(((df.label == 1) & sel).sum())
        ok = r >= 50 and f >= 50
        both += ok
        covered += sel.mean() if ok else 0.0
        print(f"{k:>4}{edges[k]:9.2f}{edges[k+1]:9.2f}{r:9,}{f:9,}"
              f"   {'both' if ok else 'SINGLE-LABEL -> undroppable shortcut'}")
    print(f"  {both}/{n_bins} bins carry both labels; they hold {covered:.1%} "
          f"of the corpus.")
    print("  Only that fraction can be made resample-invariant by training. "
          "The rest is\n  where m alone identifies the label, and no sampler "
          "or augmentation can fix it.")


def report(df: pd.DataFrame) -> None:
    """The first real measurement of the corpus. Everything downstream --
    the split design, the sampler, the augmentation policy -- is a response to
    what these tables say, so they are printed unconditionally."""
    print(f"\n=== manifest: {len(df):,} images, {df.dataset.nunique()} datasets ===")
    print(f"label balance: real={int((df.label == 0).sum()):,}  "
          f"fake={int((df.label == 1).sum()):,}  "
          f"P(fake)={df.label.mean():.4f}")

    print("\n-- datasets and images per (category, label) --")
    piv = df.pivot_table(index="category", columns="label", values="image_id",
                         aggfunc="count", fill_value=0)
    ds = df.drop_duplicates("dataset").pivot_table(
        index="category", columns="label", values="dataset",
        aggfunc="count", fill_value=0)
    print(f"{'category':14}{'real imgs':>11}{'fake imgs':>11}"
          f"{'real ds':>9}{'fake ds':>9}{'P(fake)':>10}")
    for cat in sorted(piv.index):
        r, f = int(piv.get(0, {}).get(cat, 0)), int(piv.get(1, {}).get(cat, 0))
        rd, fd = int(ds.get(0, {}).get(cat, 0)), int(ds.get(1, {}).get(cat, 0))
        p = f / (r + f) if (r + f) else 0.0
        flag = "  <-- SINGLE LABEL" if not (r and f) else ""
        print(f"{cat:14}{r:11,}{f:11,}{rd:9}{fd:9}{p:10.4f}{flag}")

    print("\n-- resolution by label (min_side quantiles) --")
    print(f"{'label':8}{'p05':>8}{'p25':>8}{'p50':>8}{'p75':>8}{'p95':>8}{'mean':>9}")
    for lab, sub in df.groupby("label"):
        q = sub.min_side.quantile([.05, .25, .5, .75, .95])
        print(f"{'real' if lab == 0 else 'fake':8}"
              + "".join(f"{int(q[x]):8,}" for x in (.05, .25, .5, .75, .95))
              + f"{sub.min_side.mean():9.0f}")

    print("\n-- eval resample factor m = S / min_side, by label --")
    print(f"{'label':8}{'p05':>9}{'p50':>9}{'p95':>9}{'  P(upsampled)':>15}")
    for lab, sub in df.groupby("label"):
        q = sub.m_eval.quantile([.05, .5, .95])
        print(f"{'real' if lab == 0 else 'fake':8}"
              + "".join(f"{q[x]:9.2f}" for x in (.05, .5, .95))
              + f"{(sub.m_eval > 1).mean():15.3f}")
    _overlap_report(df)

    print("\n-- container by label (share of images) --")
    ct = pd.crosstab(df.label, df.ext, normalize="index")
    cols = list(ct.columns)[:8]
    print(f"{'label':8}" + "".join(f"{c:>10}" for c in cols))
    for lab in sorted(ct.index):
        print(f"{'real' if lab == 0 else 'fake':8}"
              + "".join(f"{ct.loc[lab, c]:10.3f}" for c in cols))

    print("\n-- images per dataset --")
    counts = df.groupby("dataset").size()
    print(f"  min={counts.min():,}  median={int(counts.median()):,}  "
          f"max={counts.max():,}  (CACHE_MAX_SAMPLES caps most at 500)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache-dir", required=True,
                    help="gasbench cache root (contains datasets/)")
    ap.add_argument("--config-dir", default=None,
                    help="override the gasbench dataset configs directory")
    ap.add_argument("--overrides", default=None,
                    help="override overrides.yaml (category fixes, pair groups)")
    ap.add_argument("--out", default="data/manifest.parquet")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--image-size", type=int, default=512,
                    help="declared model input; only used to precompute m_eval")
    args = ap.parse_args()

    df = build_manifest(Path(args.cache_dir),
                        Path(args.config_dir) if args.config_dir else None,
                        args.workers,
                        Path(args.overrides) if args.overrides else None,
                        args.image_size)
    report(df)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"\nwrote {len(df):,} rows -> {out}")


if __name__ == "__main__":
    sys.exit(main())
