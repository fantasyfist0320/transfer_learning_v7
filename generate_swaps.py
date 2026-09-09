"""Generate insightface-inswapper face swaps as local training data.

    python generate_swaps.py --data-root /path/to/datasets --n 3000 \
        --swap-model /path/to/inswapper_128.onnx

The benchmark's `face-swap` dataset is insightface-inswapper output
(per its gasbench registry notes), and hidden holdouts can carry the same
generator. This script runs the SAME tool over real face sources already in
the download cache, writing a new dataset `local-inswapper-swaps` in the
gasbench cache layout so build_manifest.py picks it up via the
`extra_datasets` entry in overrides-ship.yaml.

Requirements (remote GPU box): pip install insightface onnxruntime-gpu
Weights: inswapper_128.onnx (search "inswapper_128" on HF; place anywhere and
pass --swap-model). buffalo_l detection weights download automatically on
first FaceAnalysis() use.

Design choices:
  - Source and target are drawn from DIFFERENT datasets so identity and
    background statistics decorrelate.
  - Output is saved as JPEG with per-image quality drawn from [LO, HI]
    (default 88-96): a swap of a JPEG-sourced photo should not become a PNG,
    and a CONSTANT quality would make one quantisation table mean "swap".
  - sample_metadata.json's source_file records the TARGET dataset (bucketed
    into blocks of 64), so build_splits' block logic groups swaps by their
    pixel provenance.
  - --target restricts swap targets to one dataset so the output can be
    registered per-source (local-swap-<target>) and pair_grouped 1:1 with it.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2

DEFAULT_SOURCES = [
    "ffhq-256", "celeb-a-hq", "lfw", "UTKFace", "fairface", "AgeDB",
    "affectnet", "imdb-crop", "real-human-faces-data-set", "CACD",
]
EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def collect_images(data_root: Path, datasets: list[str]) -> dict[str, list[Path]]:
    out: dict[str, list[Path]] = {}
    for name in datasets:
        for base in (data_root / "datasets" / name, data_root / name):
            samples = base / "samples"
            if samples.is_dir():
                files = [p for p in samples.iterdir() if p.suffix.lower() in EXTS]
                if files:
                    out[name] = sorted(files)
                break
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True,
                    help="dataset cache root (same as build_manifest --cache-dir)")
    ap.add_argument("--swap-model", required=True,
                    help="path to inswapper_128.onnx")
    ap.add_argument("--out-name", default="local-inswapper-swaps")
    ap.add_argument("--sources", default=",".join(DEFAULT_SOURCES),
                    help="comma-separated real face datasets to draw from")
    ap.add_argument("--target", default=None,
                    help="restrict swap TARGETS to this one dataset (identity "
                         "donors still come from the other --sources). Use with "
                         "--out-name local-swap-<target> so the output can be "
                         "pair_grouped 1:1 with its pixel source.")
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=34)
    ap.add_argument("--jpeg-quality", type=int, nargs=2, default=(88, 96),
                    metavar=("LO", "HI"),
                    help="per-image quality is drawn uniformly from [LO, HI]; "
                         "a constant quality would make one quantisation table "
                         "mean 'swap'")
    args = ap.parse_args()

    import insightface
    from insightface.app import FaceAnalysis

    rng = random.Random(args.seed)
    data_root = Path(args.data_root)
    wanted = [s.strip() for s in args.sources.split(",")]
    if args.target and args.target not in wanted:
        wanted.append(args.target)
    pools = collect_images(data_root, wanted)
    if len(pools) < 2:
        raise SystemExit(f"need >=2 source datasets with samples/, found: "
                         f"{sorted(pools)}")
    if args.target and args.target not in pools:
        raise SystemExit(f"--target {args.target} has no samples/ under "
                         f"{data_root}")
    print(f"[swap] drawing from {len(pools)} datasets: "
          + ", ".join(f"{k}({len(v)})" for k, v in pools.items()))

    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=0, det_size=(640, 640))
    swapper = insightface.model_zoo.get_model(args.swap_model,
                                              download=False, download_zip=False)

    out_dir = data_root / "datasets" / args.out_name
    if not (data_root / "datasets").is_dir():
        out_dir = data_root / args.out_name
    samples_dir = out_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    meta: dict[str, dict] = {}
    names = sorted(pools)
    made = attempts = 0
    donor_names = [n for n in names if n != args.target] if args.target else names
    while made < args.n and attempts < args.n * 8:
        attempts += 1
        if args.target:
            dst_ds = args.target
            src_ds = rng.choice(donor_names)
        else:
            src_ds, dst_ds = rng.sample(names, 2)
        src_p = rng.choice(pools[src_ds])
        dst_p = rng.choice(pools[dst_ds])
        src_img = cv2.imread(str(src_p))
        dst_img = cv2.imread(str(dst_p))
        if src_img is None or dst_img is None:
            continue
        src_faces = app.get(src_img)
        dst_faces = app.get(dst_img)
        if not src_faces or not dst_faces:
            continue
        # Swap the largest detected face in the target.
        dst_face = max(dst_faces, key=lambda f: (f.bbox[2] - f.bbox[0])
                       * (f.bbox[3] - f.bbox[1]))
        try:
            swapped = swapper.get(dst_img, dst_face, src_faces[0], paste_back=True)
        except Exception:
            continue
        # img_ prefix so build_manifest's parse_file_index works and
        # build_splits gets stable positional blocks even for a single-target
        # run, where source_file alone would be near-constant.
        fname = f"img_{made:06d}.jpg"
        jpeg_q = rng.randint(*args.jpeg_quality)
        ok = cv2.imwrite(str(samples_dir / fname), swapped,
                         [cv2.IMWRITE_JPEG_QUALITY, jpeg_q])
        if not ok:
            continue
        meta[fname] = {"source_file": f"{dst_ds}#blk{made // 64}",
                       "source_image": dst_p.name,
                       "swap_source": src_ds,
                       "jpeg_q": jpeg_q}
        made += 1
        if made % 200 == 0:
            print(f"[swap] {made}/{args.n} (attempts {attempts})", flush=True)

    (out_dir / "sample_metadata.json").write_text(json.dumps(meta))
    (out_dir / "dataset_info.json").write_text(json.dumps({
        "name": args.out_name,
        "generator": "insightface-inswapper_128",
        "media_type": "semisynthetic",
        "note": "generated by generate_swaps.py; targets/sources from the "
                "public real-face roster",
        "n_samples": made,
        "seed": args.seed,
    }, indent=2))
    print(f"[swap] wrote {made} swaps -> {out_dir}")
    if made < args.n:
        print(f"[swap] WARNING: target was {args.n}, face detection failed on "
              f"many pairs (attempts={attempts}). Consider adding sources.")


if __name__ == "__main__":
    main()
