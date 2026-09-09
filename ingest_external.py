"""Ingest an external image source into the gasbench cache layout.

    # probe role: held-out forever, scored by eval_probes.py
    python ingest_external.py --data-root <CACHE> --role probe \
        --name probe-streetview-mapillary --label real \
        --src ~/downloads/mapillary_sample/ \
        --source-url https://www.mapillary.com/dataset --collected-before 2020-06

    # train role: enters training via overrides.yaml:extra_datasets
    python ingest_external.py --data-root <CACHE> --role train \
        --name local-film-photos --label real --src hf:some/repo:train \
        --max-n 2000 --source-url https://hf.co/datasets/some/repo \
        --collected-before 2021-12

One tool for both halves of the probe-suite design (see probes.yaml):

  * `--role probe` requires a `probe-` name prefix. Probe datasets are NEVER
    registered in extra_datasets, so build_manifest.py skips them (unknown to
    the registry) and they cannot leak into training. eval_probes.py finds
    them straight off the disk.
  * `--role train` requires a `local-` prefix and prints the extra_datasets
    stanza to paste into BOTH overrides files.

Sources: a local folder (recursive), `hf:repo[:split[:column]]` (HuggingFace,
streamed), or a path to a .zip archive (e.g. a Kaggle download). Folder and
zip bytes are copied VERBATIM -- no re-encode, so the source codec mix
survives (the training prechain launders codecs label-independently anyway).
HF rows are saved verbatim when the row carries original bytes, else
re-encoded to PNG with a warning.

Label hygiene (`--label real`): --source-url and --collected-before are
required, and a collection date of 2022-01 or later prints a loud
AI-contamination warning -- post-2022 "real" scrapes routinely contain
generated images, and one polluted source poisons the class. `--screen`
additionally scores the ingested set with the current ensemble and prints the
most fake-looking files for manual review (report-only).

Provenance lands in dataset_info.json (a marker file the pipeline never
parses -- the right home for free-form audit trail). sample_metadata.json
records a `source_file` per image so build_splits.py can carve val_id by
source block; filenames follow img_{i:06d}.<ext> so the file_index fallback
block key works too.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import zipfile
from datetime import date, datetime, timezone
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
ROLE_PREFIX = {"probe": "probe-", "train": "local-"}


# ---------------------------------------------------------------------------
# Source iterators: each yields (bytes | PIL.Image, ext_hint, source_file)
# ---------------------------------------------------------------------------

def iter_folder(root: Path, rng: random.Random):
    files = sorted(p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS)
    if not files:
        raise SystemExit(f"no images with {sorted(IMG_EXTS)} under {root}")
    rng.shuffle(files)          # deterministic subsample when --max-n < total
    for p in files:
        rel = p.relative_to(root)
        src = str(rel.parent) if str(rel.parent) != "." else root.name
        yield p.read_bytes(), p.suffix.lower(), src


def iter_zip(zip_path: Path, rng: random.Random):
    zf = zipfile.ZipFile(zip_path)
    names = sorted(n for n in zf.namelist()
                   if Path(n).suffix.lower() in IMG_EXTS
                   and not n.endswith("/"))
    if not names:
        raise SystemExit(f"no images in {zip_path}")
    rng.shuffle(names)
    for n in names:
        parent = str(Path(n).parent)
        src = parent if parent != "." else zip_path.stem
        yield zf.read(n), Path(n).suffix.lower(), src


def iter_hf(spec: str, rng: random.Random):
    """spec: repo[:split[:column]]. Streamed; rng unused (stream order)."""
    from datasets import load_dataset
    parts = spec.split(":")
    repo, split = parts[0], (parts[1] if len(parts) > 1 else "train")
    column = parts[2] if len(parts) > 2 else None
    ds = load_dataset(repo, split=split, streaming=True)
    shard = 0
    for i, row in enumerate(ds):
        val = row.get(column) if column else None
        if val is None and not column:
            for v in row.values():
                if hasattr(v, "convert") or (isinstance(v, dict) and v.get("bytes")):
                    val = v
                    break
        if val is None:
            continue
        shard = i // 500
        if isinstance(val, dict) and val.get("bytes"):
            ext = Path(str(val.get("path") or "")).suffix.lower()
            yield val["bytes"], (ext if ext in IMG_EXTS else ".png"), \
                f"{repo}#shard{shard}"
        elif hasattr(val, "convert"):
            yield val, ".png", f"{repo}#shard{shard}"


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

def ingest(items, out_dir: Path, max_n: int, min_side: int) -> tuple[dict, int, int]:
    """Write up to max_n images; returns (metadata, n_kept, n_skipped).

    Dedup by 64x64 thumbnail sha1 (the prepare_reals.py rule); min-side
    filter; verbatim bytes when the source provided bytes, else PNG.
    """
    from PIL import Image

    samples = out_dir / "samples"
    samples.mkdir(parents=True, exist_ok=True)
    meta: dict[str, dict] = {}
    seen: set[str] = set()
    kept = skipped = 0
    for item, ext, src in items:
        if kept >= max_n:
            break
        try:
            if isinstance(item, (bytes, bytearray)):
                img = Image.open(io.BytesIO(item))
                img.load()
                raw: bytes | None = bytes(item)
            else:
                img, raw = item, None
            rgb = img.convert("RGB")
        except Exception:
            skipped += 1
            continue
        if min(rgb.size) < min_side:
            skipped += 1
            continue
        h = hashlib.sha1(rgb.resize((64, 64)).tobytes()).hexdigest()
        if h in seen:
            skipped += 1
            continue
        seen.add(h)
        fname = f"img_{kept:06d}{ext if ext in IMG_EXTS else '.png'}"
        if raw is not None:
            (samples / fname).write_bytes(raw)
        else:
            rgb.save(samples / fname, format="PNG")
        meta[fname] = {"source_file": src}
        kept += 1
        if kept % 200 == 0:
            print(f"  {kept}/{max_n}", flush=True)
    return meta, kept, skipped


def screen_with_ensemble(out_dir: Path, runs_dir: str, branch_names: str,
                         image_size: int, top_k: int = 20) -> None:
    """Score the ingested set with the current ensemble; print the most
    fake-looking files for manual review. Report-only, never gates."""
    import numpy as np
    import torch

    from branches import resolve, fusion_weights
    from model import BranchModel, binary_margin

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    specs = resolve([n.strip() for n in branch_names.split(",") if n.strip()])
    w = np.asarray(fusion_weights(specs), dtype=float)
    models = []
    for spec in specs:
        ck = torch.load(Path(runs_dir) / spec.name / "best.pt",
                        map_location="cpu", weights_only=False)
        m = BranchModel(spec, lora_r=ck["config"]["lora"]["r"],
                        lora_alpha=ck["config"]["lora"]["alpha"],
                        lora_dropout=0.0, gradient_checkpointing=False)
        m.load_state_dict(ck["state"], strict=True)
        m.eval()
        m.merge_and_strip()
        models.append(m.to(dev))

    import cv2
    from gasbench_bridge import apply_random_augmentations

    paths = sorted((out_dir / "samples").iterdir())
    scores: list[tuple[float, str]] = []
    buf, buf_paths = [], []

    def flush():
        nonlocal buf, buf_paths
        if not buf:
            return
        x = torch.stack(buf).to(dev)
        with torch.no_grad():
            zs = [m(x).to(torch.float32) for m in models]
        d = torch.stack([binary_margin(z) for z in zs], dim=1).cpu().numpy()
        p = 1.0 / (1.0 + np.exp(-(d * w[None, :]).sum(axis=1)))
        scores.extend(zip(p.tolist(), buf_paths))
        buf, buf_paths = [], []

    for p in paths:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        aug, _, _, _ = apply_random_augmentations(
            img, (image_size, image_size), level=0, crop_prob=0.0, seed=34)
        buf.append(torch.from_numpy(np.ascontiguousarray(aug)).permute(2, 0, 1))
        buf_paths.append(p.name)
        if len(buf) == 32:
            flush()
    flush()
    scores.sort(reverse=True)
    frac = sum(1 for s, _ in scores if s > 0.5) / max(1, len(scores))
    print(f"\n[screen] ensemble calls {frac:.1%} of the set fake "
          f"({len(scores)} scored). Highest-P(fake) files -- eyeball these:")
    for s, name in scores[:top_k]:
        print(f"  {s:6.3f}  {name}")


# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True,
                    help="gasbench cache root (build_manifest --cache-dir)")
    ap.add_argument("--role", required=True, choices=("probe", "train"))
    ap.add_argument("--name", required=True,
                    help="dataset name; must start with probe- (probe role) "
                         "or local- (train role)")
    ap.add_argument("--label", required=True,
                    choices=("real", "fake", "semisynthetic"))
    ap.add_argument("--src", required=True,
                    help="image folder | hf:repo[:split[:column]] | path.zip")
    ap.add_argument("--max-n", type=int, default=500)
    ap.add_argument("--min-side", type=int, default=256)
    ap.add_argument("--seed", type=int, default=34)
    ap.add_argument("--source-url", default=None,
                    help="where the data came from (REQUIRED for --label real)")
    ap.add_argument("--collected-before", default=None,
                    help="YYYY-MM upper bound on collection date "
                         "(REQUIRED for --label real)")
    ap.add_argument("--license", default="")
    ap.add_argument("--notes", default="")
    ap.add_argument("--type", default="",
                    help="probes.yaml type key this dataset belongs to "
                         "(printed in the snippet)")
    ap.add_argument("--screen", action="store_true",
                    help="score the ingested set with the current ensemble "
                         "and print the most fake-looking files")
    ap.add_argument("--runs-dir", default="runs/v7")
    ap.add_argument("--branches", default="dinov3,clip,convnext,eva,dct")
    ap.add_argument("--image-size", type=int, default=384)
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing dataset dir of the same name")
    args = ap.parse_args()

    prefix = ROLE_PREFIX[args.role]
    if not args.name.startswith(prefix):
        raise SystemExit(f"--role {args.role} requires a {prefix!r} name "
                         f"prefix, got {args.name!r}. The prefix is what "
                         f"keeps probe data out of training (build_manifest "
                         f"skips registry-unknown dirs) and marks train data "
                         f"as locally added.")

    # Real-label contamination guards.
    if args.label == "real":
        if not args.source_url or not args.collected_before:
            raise SystemExit("--label real requires --source-url and "
                             "--collected-before: unprovenanced 'real' data "
                             "is how one contaminated source poisons the "
                             "whole class.")
        try:
            y, m = (int(x) for x in args.collected_before.split("-")[:2])
            cutoff = date(y, m, 1)
        except ValueError:
            raise SystemExit(f"--collected-before must be YYYY-MM, got "
                             f"{args.collected_before!r}")
        if cutoff >= date(2022, 1, 1):
            print("=" * 72)
            print("WARNING: --collected-before is 2022-01 or later. Post-2022")
            print("online 'real' collections routinely contain AI-generated")
            print("images. Use --screen and review before training on this.")
            print("=" * 72)

    data_root = Path(args.data_root)
    out_dir = data_root / "datasets" / args.name
    if not (data_root / "datasets").is_dir():
        out_dir = data_root / args.name
    if out_dir.exists() and not args.force:
        raise SystemExit(f"{out_dir} already exists (--force to overwrite)")

    # Refuse names the registry already owns -- an extra_datasets setdefault
    # would silently lose, and a probe sharing a registry name would collide
    # with a training dataset.
    try:
        from build_manifest import load_registry
        registry = load_registry(None)
        if args.name in registry:
            raise SystemExit(f"{args.name!r} is already a registry dataset; "
                             f"pick a different --name")
    except SystemExit:
        raise
    except Exception as e:
        print(f"[warn] registry check skipped ({type(e).__name__}: {e}) -- "
              f"make sure {args.name!r} is not a registry dataset name")

    rng = random.Random(args.seed)
    src = args.src
    if src.startswith("hf:"):
        items = iter_hf(src[3:], rng)
        src_kind = "huggingface"
    elif src.endswith(".zip"):
        items = iter_zip(Path(src).expanduser(), rng)
        src_kind = "zip"
    else:
        items = iter_folder(Path(src).expanduser(), rng)
        src_kind = "folder"

    print(f"[ingest] {args.name} <- {src} ({src_kind}, max {args.max_n}, "
          f"min side {args.min_side})")
    meta, kept, skipped = ingest(items, out_dir, args.max_n, args.min_side)
    if not kept:
        raise SystemExit("nothing ingested; check --src")

    n_sources = len({m["source_file"] for m in meta.values()})
    (out_dir / "sample_metadata.json").write_text(json.dumps(meta))
    (out_dir / "dataset_info.json").write_text(json.dumps({
        "name": args.name,
        "media_type": args.label if args.label != "fake" else "synthetic",
        "role": args.role,
        "type": args.type,
        "source": src,
        "source_url": args.source_url,
        "license": args.license,
        "collected_before": args.collected_before,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "n_ingested": kept,
        "n_skipped": skipped,
        "seed": args.seed,
        "ingest_tool": "ingest_external.py",
        "notes": args.notes,
    }, indent=2))
    print(f"[ingest] wrote {kept} images ({skipped} skipped: unreadable/"
          f"small/duplicate) -> {out_dir}")
    print(f"[ingest] {n_sources} distinct source_file values", end="")
    if n_sources < 8:
        print("  <-- WARNING: under 8, build_splits falls back to file_index "
              "blocks (fine for probes; for train data, correlated frames "
              "may straddle val_id blocks)")
    else:
        print()

    print("\n" + "=" * 60)
    if args.role == "train":
        media = args.label if args.label != "fake" else "synthetic"
        print("Paste into overrides.yaml AND overrides-ship.yaml "
              "(keep them in sync):\n")
        print("extra_datasets:")
        print(f"  - {{name: {args.name}, media_type: {media}, "
              f"content_category: diverse}}")
        if args.label == "semisynthetic":
            print(f"\n...and add {args.name!r} to semisynthetic_datasets.")
        print("\n# content_category: diverse is the safe default. Adding a")
        print("# FAKE dataset to a real-only category un-folds that category")
        print("# (folding is registry-driven) and shifts sampler mass -- do")
        print("# that only deliberately, e.g. to complete a real/fake pair.")
        print("# If this set shares source images with anything else, add a")
        print("# pair_groups clique; multi-shard releases need shard_groups.")
        print("\nThen: build_manifest.py -> build_splits.py (dry-run, then "
              "--write) -> audit_data.py -> retrain.")
    else:
        lbl = "real" if args.label == "real" else "fake"
        print("Paste under the matching type in probes.yaml:\n")
        print(f"      - name: {args.name}")
        print(f"        label: {lbl}")
        print(f"        source_url: {args.source_url or '...'}")
        print(f"        added: {date.today().isoformat()}")
        print("\nThen: python eval_probes.py --probes probes.yaml "
              f"--data-root {args.data_root} --runs-dir {args.runs_dir}")
    print("=" * 60)

    if args.screen:
        screen_with_ensemble(out_dir, args.runs_dir, args.branches,
                             args.image_size)


if __name__ == "__main__":
    main()
