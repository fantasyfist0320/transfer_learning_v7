"""Generate inpainting edits of our own training reals as semisynthetic data.

    python generate_edits.py --manifest data/manifest.parquet \
        --cache-dir /path/to/cache --per-source 750

Round 23 made the image taxonomy 3-class (real / synthetic / semisynthetic),
and the registry carries only 6 semisynthetic datasets -- all face-swap or
inpainting flavored. This script manufactures the third class by running the
SAME inpainting pipelines the validator's own i2i path uses
(gas/generation/media/models.py: sdxl-1.0-inpainting-0.1 and
dreamshaper-8-inpainting) over reals already in the download cache, writing
one dataset per source, `local-inpaint-<source>`, in the gasbench cache
layout. Registration happens via the `extra_datasets` + `pair_groups` blocks
already added to overrides.yaml / overrides-experiment.yaml -- the pair_group
puts each edit set in the same split_unit as its source so clique pairing
serves real/edited pairs in the same batch.

Requirements (remote GPU box): the training venv (torch, diffusers,
transformers) plus insightface + onnxruntime-gpu (already installed for
generate_swaps.py; used here only for face-box masks -- no swap model needed).

Design choices:
  - Sources come from the MANIFEST, filtered to split == "train". Never edit
    a val/holdout row: an edited near-duplicate of a val image in train is
    leakage across the carve.
  - Masks are 50% face-box (insightface detection, bbox grown 10-40%) and
    50% random shapes with the covered-area fraction swept ~0.05-0.6. Small
    edits are the hard case (the mean-pooled head dilutes local evidence) --
    they must be represented, not just the easy half-image repaints.
  - Output keeps the source aspect ratio (dims rounded to /8, capped per
    editor). Emitting native square 512/1024 for every edit would hand the
    geometry shortcut a class label.
  - Saved as JPEG with quality drawn per-image from U{lo..hi}: a constant
    quality would make one quantisation table mean "semisynthetic"
    (audit_data's format-bias check exists for exactly this).
  - sample_metadata.json's source_file buckets by source so build_splits'
    block logic groups edits by pixel provenance (>=8 distinct values).
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

# The default roster: face sources are the class's center of mass (the 6
# registry semi sets are all face manipulation), openfake-real/FantasyID-real
# are the current false-positive weak spots (the hardest real-vs-edited
# boundary), yfcc100m is messy consumer/organic traffic, OCR-Quality gives
# document edits (receipts-i2i analog), fashionpedia gives products.
# wider-face / AgeDB / M6Doc_test are deliberately ABSENT: they are val_xgen
# holdouts in overrides.yaml, and the pair_group would drag the edit set into
# the holdout with them.
DEFAULT_SOURCES = [
    "celeb-a-hq", "FDDB_Dataset", "fakeclue-real-ffpp", "yfcc100m",
    "openfake-real", "FantasyID-real", "Aslan-mingye-OCR-Quality",
    "fashionpedia",
]

# From gas/generation/prompts/model_prompt_styles.py "inpainting" style.
NEGATIVE = "blurry, low quality, bad blending, visible seam, artifact"

FACE_PROMPTS = [
    "a photorealistic human face, natural skin texture, detailed",
    "portrait of a person, natural lighting, photo",
    "a smiling person's face, sharp focus, photograph",
    "a person's face turned slightly, soft daylight, photo",
    "an elderly person's face, realistic skin detail, photo",
    "a young person's face, neutral expression, photograph",
]
GENERIC_PROMPTS = [
    "a small object on a surface, photorealistic",
    "natural scenery blending with the surroundings, photo",
    "a household object, realistic texture, photograph",
    "an animal standing in the scene, photo",
    "a piece of clothing with realistic fabric texture, photo",
    "food on a table, natural light, photograph",
    "a wooden surface with natural grain, photo",
    "a sign with printed text, photorealistic",
]
DOC_PROMPTS = [
    "a block of printed text on paper, document scan",
    "a paragraph of clean printed text, high resolution scan",
    "handwritten text on paper, realistic ink",
    "a table of printed numbers on a page, scan",
    "an official stamp on a printed document, photo",
]
PROMPT_OVERRIDES = {
    "Aslan-mingye-OCR-Quality": DOC_PROMPTS,
    "M6Doc_test": DOC_PROMPTS,
}

EDITORS = {
    # kwargs mirror gas/generation/media/models.py:250-287 (the validator's
    # own i2i configs) so the artefact distribution matches gasstation semis.
    "sdxl": {
        "path": "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        "max_side": 1024,
        "steps": (50, 50),
        "guidance": 7.5,
        "strength": 0.99,
    },
    "dreamshaper": {
        "path": "Lykon/dreamshaper-8-inpainting",
        "max_side": 640,        # SD1.5-class UNet; larger drifts off-manifold
        "steps": (40, 60),
        "guidance": 7.5,
        "strength": 0.99,
    },
}


def random_mask(size: tuple[int, int], rng: random.Random,
                lo: float, hi: float) -> Image.Image:
    """1-3 random rectangles/ellipses covering roughly lo..hi of min-side.

    Adapted from gas/generation/util/image.py:create_random_mask, rewritten
    onto a passed rng so renders are reproducible per (seed, row).
    """
    w, h = size
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    for _ in range(rng.randint(1, 3)):
        side = min(w, h)
        mw = max(48, int(rng.uniform(lo, hi) * side))
        mh = max(48, int(rng.uniform(lo, hi) * side))
        mw, mh = min(mw, w), min(mh, h)
        x = rng.randint(0, w - mw)
        y = rng.randint(0, h - mh)
        if rng.random() < 0.5:
            draw.rectangle([x, y, x + mw, y + mh], fill=255)
        else:
            draw.ellipse([x, y, x + mw, y + mh], fill=255)
    return mask


def face_mask(bgr: np.ndarray, app, rng: random.Random) -> Image.Image | None:
    """Rectangle mask over one detected face, bbox grown 10-40%."""
    faces = app.get(bgr)
    if not faces:
        return None
    f = rng.choice(faces)
    x0, y0, x1, y1 = f.bbox
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    grow = rng.uniform(1.1, 1.4)
    hw, hh = (x1 - x0) / 2 * grow, (y1 - y0) / 2 * grow
    h, w = bgr.shape[:2]
    box = [max(0, int(cx - hw)), max(0, int(cy - hh)),
           min(w - 1, int(cx + hw)), min(h - 1, int(cy + hh))]
    if box[2] - box[0] < 32 or box[3] - box[1] < 32:
        return None
    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).rectangle(box, fill=255)
    return mask


def prep_image(path: str, max_side: int) -> Image.Image | None:
    """Load, RGB, cap the long side, round dims down to multiples of 8."""
    try:
        img = Image.open(path).convert("RGB")
    except OSError:
        return None
    w, h = img.size
    if max(w, h) > max_side:
        s = max_side / max(w, h)
        w, h = int(w * s), int(h * s)
    w, h = max(64, w - w % 8), max(64, h - h % 8)
    if (w, h) != img.size:
        img = img.resize((w, h), Image.LANCZOS)
    return img


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True,
                    help="manifest parquet WITH the split column written "
                         "(build_splits.py --write must have run)")
    ap.add_argument("--cache-dir", required=True,
                    help="dataset cache root (same as build_manifest --cache-dir)")
    ap.add_argument("--sources", default=",".join(DEFAULT_SOURCES))
    ap.add_argument("--per-source", type=int, default=750)
    ap.add_argument("--seed", type=int, default=34)
    ap.add_argument("--jpeg-quality", type=int, nargs=2, default=(88, 96),
                    metavar=("LO", "HI"))
    ap.add_argument("--editors", default="sdxl,dreamshaper",
                    help="comma-separated subset of: " + ",".join(EDITORS))
    ap.add_argument("--force", action="store_true",
                    help="regenerate sources that already look complete")
    args = ap.parse_args()

    import pandas as pd
    import torch
    from diffusers import AutoPipelineForInpainting, DEISMultistepScheduler
    from insightface.app import FaceAnalysis

    editors = [e.strip() for e in args.editors.split(",") if e.strip()]
    unknown = [e for e in editors if e not in EDITORS]
    if unknown:
        raise SystemExit(f"unknown editors {unknown}; choose from {list(EDITORS)}")

    df = pd.read_parquet(args.manifest)
    if "split" not in df.columns or (df["split"] == "").all():
        raise SystemExit("manifest has no split column -- run build_splits.py "
                         "--write first (the train filter is the leakage guard)")

    cache = Path(args.cache_dir)
    ds_root = cache / "datasets" if (cache / "datasets").is_dir() else cache
    device = "cuda" if torch.cuda.is_available() else "cpu"

    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=0, det_size=(640, 640))

    # ---- plan every job up front, deterministically ----------------------
    # jobs[source] = list of dicts; editor assignment alternates so each
    # editor is loaded ONCE and sweeps its half of every source.
    jobs: dict[str, list[dict]] = {}
    meta: dict[str, dict[str, dict]] = {}
    for source in [s.strip() for s in args.sources.split(",") if s.strip()]:
        out_dir = ds_root / f"local-inpaint-{source}"
        done = out_dir / "sample_metadata.json"
        if done.exists() and not args.force:
            have = len(json.loads(done.read_text()))
            if have >= 0.9 * args.per_source:
                print(f"[edit] {source}: {have} already staged -- skip "
                      f"(--force to redo)")
                continue
        rows = df[(df["dataset"] == source) & (df["split"] == "train")]
        if rows.empty:
            print(f"[edit] WARNING: {source} has no train rows in this "
                  f"manifest -- skipped. (Held out in these overrides?)")
            continue
        rows = rows.sort_values("image_id").reset_index(drop=True)
        rng = random.Random(f"{args.seed}|{source}")
        order = rng.sample(range(len(rows)), k=len(rows))
        js = []
        for i in range(args.per_source):
            r = rows.iloc[order[i % len(order)]]
            js.append({
                "fname": f"img_{i:06d}.jpg",
                "path": r["path"],
                "image_id": r["image_id"],
                "editor": editors[i % len(editors)],
                "want_face": rng.random() < 0.5,
                "jpeg_q": rng.randint(*args.jpeg_quality),
                "seed": rng.randrange(2 ** 31),
            })
        jobs[source] = js
        meta[source] = {}
        (out_dir / "samples").mkdir(parents=True, exist_ok=True)
    if not jobs:
        print("[edit] nothing to do")
        return

    # ---- run, one editor resident at a time ------------------------------
    for editor in editors:
        spec = EDITORS[editor]
        print(f"[edit] loading {editor} ({spec['path']}) ...")
        load_kwargs = {"torch_dtype": torch.float16, "variant": "fp16"}
        if editor == "dreamshaper":
            # The SD1.5 safety checker false-positives into black frames on
            # faces; the validator strips it too for local generation.
            load_kwargs.update(safety_checker=None, requires_safety_checker=False)
        else:
            load_kwargs["use_safetensors"] = True
        pipe = AutoPipelineForInpainting.from_pretrained(spec["path"], **load_kwargs)
        if editor == "dreamshaper":
            pipe.scheduler = DEISMultistepScheduler.from_config(pipe.scheduler.config)
        pipe = pipe.to(device)
        pipe.set_progress_bar_config(disable=True)

        for source, js in jobs.items():
            mine = [j for j in js if j["editor"] == editor]
            prompts_generic = PROMPT_OVERRIDES.get(source, GENERIC_PROMPTS)
            made = 0
            for j in mine:
                rng = random.Random(j["seed"])
                img = prep_image(j["path"], spec["max_side"])
                if img is None:
                    continue
                mask = None
                kind = "random"
                if j["want_face"]:
                    bgr = np.array(img)[:, :, ::-1].copy()
                    mask = face_mask(bgr, app, rng)
                    kind = "face-box" if mask is not None else "random-fallback"
                if mask is None:
                    lo = rng.uniform(0.05, 0.30)
                    mask = random_mask(img.size, rng, lo, lo + rng.uniform(0.10, 0.30))
                area = float(np.array(mask).mean() / 255.0)
                prompt = rng.choice(FACE_PROMPTS if kind == "face-box"
                                    else prompts_generic)
                gen = torch.Generator(device).manual_seed(j["seed"])
                try:
                    out = pipe(prompt=prompt, negative_prompt=NEGATIVE,
                               image=img, mask_image=mask,
                               width=img.size[0], height=img.size[1],
                               guidance_scale=spec["guidance"],
                               strength=spec["strength"],
                               num_inference_steps=rng.randint(*spec["steps"]),
                               generator=gen).images[0]
                except Exception as e:
                    print(f"[edit] {source}/{j['fname']}: {type(e).__name__}: {e}")
                    continue
                arr = np.asarray(out)
                if arr.std() < 2.0:          # black/blank failure
                    continue
                out_dir = ds_root / f"local-inpaint-{source}"
                out.convert("RGB").save(out_dir / "samples" / j["fname"],
                                        quality=j["jpeg_q"])
                meta[source][j["fname"]] = {
                    "source_file": f"{source}#blk{int(j['fname'][4:10]) // 64}",
                    "source_image_id": j["image_id"],
                    "editor": editor,
                    "mask_kind": kind,
                    "mask_area_frac": round(area, 4),
                    "jpeg_q": j["jpeg_q"],
                    "prompt": prompt,
                }
                made += 1
                if made % 100 == 0:
                    print(f"[edit] {editor}/{source}: {made}/{len(mine)}",
                          flush=True)
            print(f"[edit] {editor}/{source}: done ({made}/{len(mine)})")

        del pipe
        if device == "cuda":
            torch.cuda.empty_cache()

    # ---- write dataset files ---------------------------------------------
    for source, m in meta.items():
        out_dir = ds_root / f"local-inpaint-{source}"
        (out_dir / "sample_metadata.json").write_text(json.dumps(m))
        (out_dir / "dataset_info.json").write_text(json.dumps({
            "name": f"local-inpaint-{source}",
            "generator": " + ".join(EDITORS[e]["path"] for e in editors),
            "media_type": "semisynthetic",
            "note": "generated by generate_edits.py from this source's "
                    "TRAIN-split rows; masks 50% face-box / 50% random shapes",
            "source_dataset": source,
            "n_samples": len(m),
            "seed": args.seed,
        }, indent=2))
        n_blocks = len({v["source_file"] for v in m.values()})
        flag = "" if n_blocks >= 8 else "  [WARN <8 blocks]"
        print(f"[edit] wrote {len(m):5d} -> {out_dir}  ({n_blocks} blocks){flag}")


if __name__ == "__main__":
    main()
