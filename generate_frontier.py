#!/usr/bin/env python3
"""Generate training images from frontier commercial image APIs.

Why (2026-08-25): the no-degradation experiment proved gemini31-flash-lite
carries almost no detectable signature for our feature stack (AUC 0.75
ceiling even with every degradation removed) while the r22 winner scored
73-81% on it -- they almost certainly trained on self-generated Gemini
images. Rounds are decided by the NEXT hidden-holdout generator; this script
manufactures that data ahead of time from whatever commercial APIs are
current. It writes cache-layout datasets exactly like generate_edits.py
(sample_metadata.json with block-bucketed source_file for the 80/20 carve,
dataset_info.json, samples/), ready for `extra_datasets` registration.

DECISION GATE: this script is provided but NOT wired into any runbook step;
whether to spend API budget is the user's call. It needs only network + CPU,
so it can run on the orchestration VM while the GPU trains.

Usage:
    export GEMINI_API_KEY=...          # or OPENAI_API_KEY for --provider openai
    python generate_frontier.py --provider gemini \
        --model gemini-3.1-flash-lite --n 1500 \
        --cache-dir <CACHE>/datasets [--out-name local-gen-gemini31fl]
    python generate_frontier.py --provider openai --model gpt-image-1.5 ...
    # --dry-run prints the first prompts and writes nothing.

Model ids move fast -- they are ALWAYS passed explicitly, never defaulted.
Idempotent: existing sample files are counted and skipped; rerun after an
interruption. Rate limits: exponential backoff on 429/5xx, --sleep between
calls.

To register the output (user opt-in), add to overrides*.yaml:
    extra_datasets:
      - name: local-gen-<slug>
        media_type: synthetic
        content_category: diverse
then rebuild manifest -> audit -> splits as usual (RUNBOOK-SELFGEN §3 checks
apply: no dead-weight warn, a val_id carve needs >=2 source_file blocks --
this writer buckets 64 images per block).
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import random
import re
import time
from pathlib import Path

import requests
from PIL import Image

# ---------------------------------------------------------------------------
# Prompt synthesis: diverse across the corpus's content categories so the
# data teaches the GENERATOR's signature, not one content niche.
# ---------------------------------------------------------------------------

SUBJECTS = {
    "faces": ["portrait of a middle-aged {adj} person", "close-up face of a {adj} teenager",
              "candid photo of a {adj} elderly person laughing", "passport-style photo of a {adj} adult"],
    "people": ["{adj} street musician performing", "two friends cooking in a small kitchen",
               "a {adj} athlete mid-jump outdoors", "crowd at a rainy bus stop"],
    "animals": ["a {adj} dog running on wet sand", "close-up of a {adj} cat in window light",
                "a bird perched on a rusty fence", "a horse grazing at dusk"],
    "scenes": ["narrow alley in an old city at noon", "supermarket interior with harsh lighting",
               "foggy mountain road with a parked car", "cluttered garage workbench"],
    "documents": ["a crumpled paper receipt on a wooden table", "a handwritten grocery list on lined paper",
                  "an open passport on a scanner bed", "a whiteboard covered in diagrams"],
    "food": ["homemade breakfast plate photographed from above", "street food stall with steam rising",
             "half-eaten birthday cake on a paper plate"],
    "objects": ["a scratched smartphone on concrete", "worn leather boots by a door",
                "a bicycle leaning on a graffiti wall"],
}
ADJ = ["smiling", "serious", "tired", "young", "freckled", "bearded", "windswept"]
STYLES = ["photorealistic, natural light", "shot on a phone camera, slightly noisy",
          "DSLR photo, shallow depth of field", "overcast daylight, muted colors",
          "indoor tungsten lighting", "harsh flash photo at night",
          "webcam quality", "golden hour backlight"]


def make_prompts(n: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    cats = list(SUBJECTS)
    out = []
    for i in range(n):
        c = cats[i % len(cats)]                    # even category coverage
        t = rng.choice(SUBJECTS[c]).format(adj=rng.choice(ADJ))
        out.append(f"{t}, {rng.choice(STYLES)}")
    return out


# ---------------------------------------------------------------------------
# Providers: one function per API, returning raw image bytes or None.
# requests-only on purpose -- vendor SDKs churn; these REST shapes are the
# stable public contracts. Extend by adding to PROVIDERS.
# ---------------------------------------------------------------------------

def _backoff(fn, tries=5):
    for k in range(tries):
        try:
            r = fn()
        except requests.RequestException as e:
            print(f"    network error ({type(e).__name__}); retry {k+1}/{tries}")
            time.sleep(2 ** k)
            continue
        if r.status_code in (429, 500, 502, 503):
            wait = 2 ** k * 5
            print(f"    HTTP {r.status_code}; backing off {wait}s")
            time.sleep(wait)
            continue
        return r
    return None


def gen_gemini(model: str, prompt: str, key: str) -> bytes | None:
    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"{model}:generateContent")
    r = _backoff(lambda: requests.post(
        url, headers={"x-goog-api-key": key},
        json={"contents": [{"parts": [{"text": prompt}]}],
              "generationConfig": {"responseModalities": ["IMAGE"]}},
        timeout=120))
    if r is None or r.status_code != 200:
        if r is not None:
            print(f"    HTTP {r.status_code}: {r.text[:200]}")
        return None
    for part in (r.json().get("candidates") or [{}])[0] \
            .get("content", {}).get("parts", []):
        blob = part.get("inlineData") or part.get("inline_data")
        if blob and blob.get("data"):
            return base64.b64decode(blob["data"])
    return None


def gen_openai(model: str, prompt: str, key: str) -> bytes | None:
    r = _backoff(lambda: requests.post(
        "https://api.openai.com/v1/images/generations",
        headers={"Authorization": f"Bearer {key}"},
        json={"model": model, "prompt": prompt, "n": 1, "size": "1024x1024"},
        timeout=180))
    if r is None or r.status_code != 200:
        if r is not None:
            print(f"    HTTP {r.status_code}: {r.text[:200]}")
        return None
    d = (r.json().get("data") or [{}])[0]
    if d.get("b64_json"):
        return base64.b64decode(d["b64_json"])
    if d.get("url"):
        rr = _backoff(lambda: requests.get(d["url"], timeout=120))
        return rr.content if rr is not None and rr.status_code == 200 else None
    return None


PROVIDERS = {"gemini": ("GEMINI_API_KEY", gen_gemini),
             "openai": ("OPENAI_API_KEY", gen_openai)}


# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", required=True, choices=sorted(PROVIDERS))
    ap.add_argument("--model", required=True,
                    help="exact API model id (they move fast; never defaulted)")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--cache-dir", required=True,
                    help="the gasbench datasets dir the dataset is written into")
    ap.add_argument("--out-name", default=None,
                    help="dataset dir name; default local-gen-<model slug>")
    ap.add_argument("--seed", type=int, default=34)
    ap.add_argument("--sleep", type=float, default=1.0,
                    help="seconds between API calls (rate-limit kindness)")
    ap.add_argument("--jpeg-q", type=int, nargs=2, default=(88, 96),
                    help="JPEG quality jitter on save -- format-bias hygiene, "
                         "same range as generate_edits.py")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    slug = re.sub(r"[^a-z0-9]+", "", args.model.lower())[:24]
    name = args.out_name or f"local-gen-{slug}"
    prompts = make_prompts(args.n, args.seed)
    if args.dry_run:
        for p in prompts[:8]:
            print("  ", p)
        print(f"[dry-run] {args.n} prompts for {args.provider}/{args.model} "
              f"-> {name}; nothing written")
        return

    env_key, fn = PROVIDERS[args.provider]
    key = os.environ.get(env_key)
    if not key:
        raise SystemExit(f"{env_key} is not set")

    out_dir = Path(args.cache_dir) / name
    (out_dir / "samples").mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "sample_metadata.json"
    meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
    rng = random.Random(args.seed)
    made = skipped = failed = 0
    for i, prompt in enumerate(prompts):
        fname = f"img_{i:06d}.jpg"
        if (out_dir / "samples" / fname).is_file():
            skipped += 1
            continue
        raw = fn(args.model, prompt, key)
        if raw is None:
            failed += 1
            continue
        try:
            with Image.open(io.BytesIO(raw)) as im:
                img = im.convert("RGB")
        except Exception as e:
            print(f"    undecodable response ({type(e).__name__})")
            failed += 1
            continue
        import numpy as np
        if np.asarray(img).std() < 2.0:          # blank/black failure
            failed += 1
            continue
        q = rng.randint(*args.jpeg_q)
        img.save(out_dir / "samples" / fname, quality=q)
        meta[fname] = {
            # 64-image blocks: >=2 blocks unlock the val_id carve; a 1000-image
            # run yields ~16 blocks (RUNBOOK-SELFGEN §3 wants >=8).
            "source_file": f"{name}#blk{i // 64}",
            "model_name": args.model,
            "provider": args.provider,
            "prompt": prompt,
            "jpeg_q": q,
        }
        made += 1
        if made % 25 == 0:
            meta_path.write_text(json.dumps(meta))   # checkpoint the metadata
            print(f"[gen] {name}: {made} made / {skipped} skipped / "
                  f"{failed} failed", flush=True)
        time.sleep(args.sleep)

    meta_path.write_text(json.dumps(meta))
    (out_dir / "dataset_info.json").write_text(json.dumps({
        "name": name,
        "generator": f"{args.provider}/{args.model}",
        "media_type": "synthetic",
        "note": "generated by generate_frontier.py (frontier-API coverage; "
                "see docstring for the r22/gemini31 rationale)",
        "n_samples": len(meta),
        "seed": args.seed,
    }, indent=2))
    blocks = len({v["source_file"] for v in meta.values()})
    print(f"[gen] wrote {len(meta)} samples -> {out_dir} ({blocks} blocks; "
          f"{failed} failed this run). Register via extra_datasets (docstring) "
          f"and rebuild the manifest to use.")


if __name__ == "__main__":
    main()
