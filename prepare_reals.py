"""Download hidden-real-style photo corpora and stage them for the pipeline.

    python prepare_reals.py --cache-dir <CACHE>                  # everything
    python prepare_reals.py --cache-dir <CACHE> --sources laion-aes65
    python prepare_reals.py --ood-only                           # probe pool only

Produces two disjoint things:

  1. ood_pool/real/            -- the calibration/measurement probe slice
                                  (measure_margins.py), from a DIFFERENT repo
                                  than any training source, so the probe never
                                  grades data the model will train on.
  2. <CACHE>/datasets/<name>/  -- gasbench-cache-layout training datasets
                                  (dataset_info.json + sample_metadata.json +
                                  samples/), picked up by build_manifest via
                                  overrides.yaml:extra_datasets. The YAML block
                                  to paste is printed at the end.

Why these sources (2026-08-12 coverage audit + DFLIP-3K post-mortem): hidden
holdout REALS are aesthetic/stylized photography -- DFLIP-3K's reals are
literally LAION filtered at aesthetic >= 6.5 -- and that style is ~absent from
the 96 registry real datasets. This is the cheapest attack on the largest
measured weakness (hidden reals; also the reigning winner's weak axis).

Label hygiene: every source here is camera-native photography from pre-AI or
curated-human collections. Do NOT add DeviantArt/ArtStation-style sources
without a pre-2022 date filter -- post-2022 art platforms are full of AI
images, and one contaminated "real" source poisons the class.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path

# repo, split, images to take, min shorter side. Probe source is separate.
PROBE_SOURCE = ("recoilme/aesthetic_photos_xs", "train", 600, 256)
TRAIN_SOURCES = {
    # THE distribution dflip3k reals came from: LAION @ aesthetic >= 6.5
    "laion-aes65": ("bhargavsdesai/laion_improved_aesthetics_6.5plus_with_images",
                    "train", 8000, 256),
    # professional photography breadth, cleanly licensed
    "unsplash-lite": ("1aurent/unsplash-lite", "train", 6000, 256),
    # 2024 smartphone computational photography (HDR/processed look)
    "phone-photos-2024": ("EarthnDusk/Photography_2024", "train", 500, 256),
}


def _to_pil(value):
    from PIL import Image
    if hasattr(value, "convert"):
        return value
    if isinstance(value, dict) and value.get("bytes"):
        return Image.open(io.BytesIO(value["bytes"]))
    return None


def _fetch_url_image(url: str):
    import requests
    from PIL import Image
    if "images.unsplash.com" in url:
        # imgix CDN: request a bounded size instead of the ~5MB original
        url = url + ("&" if "?" in url else "?") + "w=1024&fm=jpg&q=90"
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    return Image.open(io.BytesIO(r.content))


def _find_image(row):
    for k, v in row.items():
        img = _to_pil(v)
        if img is not None:
            return img
    # Metadata-only rows (e.g. 1aurent/unsplash-lite): the photo lives behind
    # a CDN link inside a nested struct (photo.image_url). Fetch failures
    # raise and are counted as bad rows by harvest().
    for v in row.values():
        if isinstance(v, dict):
            url = v.get("image_url") or v.get("img_url") or v.get("photo_image_url")
            if isinstance(url, str) and url.startswith("http"):
                return _fetch_url_image(url)
    return None


def harvest(repo: str, split: str, n: int, min_side: int, save_fn,
            streaming: bool = True, max_consecutive_errors: int = 25) -> int:
    """Save up to n de-duplicated RGB images via save_fn(idx, pil).

    The iterator itself can raise (zip-backed imagefolders stream over HTTP
    and die mid-archive with 'I/O operation on closed file' -- observed
    2026-08-12 on EarthnDusk/Photography_2024), so errors are caught at the
    next() level with a consecutive-failure cap, not just around the body.
    stage_train falls back to streaming=False (full download) when the
    streamed pass dies early.
    """
    from datasets import load_dataset
    ds = load_dataset(repo, split=split, streaming=streaming)
    it = iter(ds)
    seen, kept, bad, consec = set(), 0, 0, 0
    while kept < n:
        try:
            row = next(it)
        except StopIteration:
            break
        except Exception as e:
            bad += 1
            consec += 1
            if bad == 1:
                # A generator that raises is closed -- the next next() is
                # StopIteration -- so without this the cause is invisible.
                print(f"  [warn] {repo}: iterator error "
                      f"({type(e).__name__}: {e})")
            if consec >= max_consecutive_errors:
                print(f"  [warn] {repo}: iterator failed {consec}x in a row "
                      f"({type(e).__name__}); abandoning this pass at "
                      f"{kept}/{n}")
                break
            continue
        consec = 0
        try:
            img = _find_image(row)
            if img is None:
                bad += 1
                if bad == 1:
                    print(f"  [warn] {repo}: no image column in row with keys "
                          f"{list(row)[:8]}")
                continue
            img = img.convert("RGB")
            if min(img.size) < min_side:
                continue
            h = hashlib.sha1(img.resize((64, 64)).tobytes()).hexdigest()
            if h in seen:            # exact/near-exact duplicate
                continue
            seen.add(h)
            save_fn(kept, img)
            kept += 1
            if kept % 500 == 0:
                print(f"  {kept}/{n}", flush=True)
        except Exception:
            bad += 1
            if bad % 100 == 0:
                print(f"  [warn] {bad} unreadable rows so far")
    print(f"  {repo}: kept {kept}, skipped {bad} bad rows"
          f"{' (streamed)' if streaming else ' (full download)'}")
    return kept


def stage_probe(ood_root: Path) -> int:
    repo, split, n, min_side = PROBE_SOURCE
    out = ood_root / "real"
    out.mkdir(parents=True, exist_ok=True)
    print(f"[probe] {repo} -> {out}")
    return harvest(repo, split, n, min_side,
                   lambda i, im: im.save(out / f"probe_{i:05d}.jpg", quality=95))


def stage_train(name: str, cache_dir: Path, force: bool = False) -> int:
    repo, split, n, min_side = TRAIN_SOURCES[name]
    root = cache_dir / "datasets" / name
    samples = root / "samples"

    # Skip-if-done: a crash on a later source must not force re-downloading
    # the earlier ones on the rerun.
    meta_path = root / "sample_metadata.json"
    if meta_path.exists() and not force:
        have = len(json.loads(meta_path.read_text()))
        if have >= 0.9 * n or have >= 500:
            print(f"[train] {name}: {have} images already staged -- skipping "
                  f"(--force to redo)")
            return have

    samples.mkdir(parents=True, exist_ok=True)
    print(f"[train] {repo} -> {root}")
    meta: dict[str, dict] = {}

    def save(i, im):
        fname = f"img_{i:06d}.jpg"
        im.save(samples / fname, quality=95)
        # source_file groups ~500-image blocks so build_splits can carve
        # val_id by block instead of by row.
        meta[fname] = {"source_file": f"{repo}#shard{i // 500}"}

    kept = harvest(repo, split, n, min_side, save)
    if kept < min(n, 200):
        # Streamed pass died early (zip-over-HTTP sources). Full download is
        # the reliable path for small imagefolder repos; restart clean so
        # indices stay contiguous.
        print(f"  [fallback] retrying {repo} with streaming=False")
        for f in samples.glob("*"):
            f.unlink()
        meta.clear()
        try:
            kept = harvest(repo, split, n, min_side, save, streaming=False)
        except Exception as e:
            print(f"  [warn] full-download pass also failed "
                  f"({type(e).__name__}: {e}); keeping {kept} images")
    (root / "dataset_info.json").write_text(json.dumps(
        {"name": name, "media_type": "real", "source": repo}, indent=2))
    meta_path.write_text(json.dumps(meta))
    return kept


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", help="gasbench cache root (for training sets)")
    ap.add_argument("--ood-root", default="ood_pool")
    ap.add_argument("--sources", nargs="+", default=sorted(TRAIN_SOURCES),
                    choices=sorted(TRAIN_SOURCES))
    ap.add_argument("--ood-only", action="store_true")
    ap.add_argument("--skip-probe", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="re-download sources that are already staged")
    args = ap.parse_args()

    if not args.skip_probe:
        stage_probe(Path(args.ood_root))
    if args.ood_only:
        return
    if not args.cache_dir:
        raise SystemExit("--cache-dir is required unless --ood-only")

    counts = {}
    for name in args.sources:
        try:
            counts[name] = stage_train(name, Path(args.cache_dir), args.force)
        except Exception as e:
            print(f"[warn] {name} failed entirely ({type(e).__name__}: {e}); "
                  f"continuing with the remaining sources")
            counts[name] = 0

    print("\n" + "=" * 60)
    print("Paste into overrides.yaml AND overrides-ship.yaml:")
    print("\nextra_datasets:")
    for name in args.sources:
        if counts.get(name):
            print(f"  - {{name: {name}, media_type: real, "
                  f"content_category: diverse}}")
    print("\nThen: build_manifest.py -> audit_data.py -> build_splits.py "
          "--write -> retrain.")
    print("Probe pool and training sources are from DISJOINT repos by "
          "construction -- keep it that way if you edit SOURCES.")


if __name__ == "__main__":
    main()
