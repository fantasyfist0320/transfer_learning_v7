# v21 data update (gasbench 0.8.2 → 0.8.3, 2026-08-10)

## What changed upstream

- **13 net-new image datasets** joined the registry as "released v21
  holdouts" — former hidden holdout sets made public. 7 real:
  google-streetview-images, data-art-backup, cddb-real,
  dflip3k-real-{juggernaut,dreamshaper-xl,pixelwave-11},
  feyiamujohuman-palm-images. 6 synthetic: deepfakes-qa-15k, sdfd,
  17-ai-model-images, cddb-fake, dflip3k-fake-{pixelwave-11,dreamshaper-xl}.
  All parquet under the `34data/` HF org. Registry is now 96 real + 96
  synthetic.
- **3 synthetic datasets curated out** (0.8.3): yogastudio-orbit-timeofday,
  synmulti-pie-images, onemillionfaces. If their cache dirs exist they are
  skipped by build_manifest with an "on disk but not in the registry" warn —
  harmless, delete at leisure.
- **Eval contract verified unchanged** in 0.8.3 (augment_level=0,
  crop_prob=0.0, INTER_LINEAR, 0.8/0.2 blend) — no transform drift, existing
  branch checkpoints stay comparable.
- `art` became a two-label category (data-art-backup vs 3 fake art sets) and
  is now balanced on its own instead of folded; `scenes` is real-only and
  folds into `diverse` automatically.

## What changed in this pipeline

- **paths.py** (new, copied from video_v1): the gasbench checkout moved into
  `bitmind-subnet/gasbench/` on 2026-08-09; every `parents[2]/gasbench` was
  silently broken. Resolution is now $GASBENCH_SRC → ancestor walk → pip
  package, used by build_manifest.py, gasbench_bridge.py and verify.py.
- **gasbench_bridge.py**: metrics import now stubs the intermediate
  `benchmarks` packages (this checkout's `benchmarks/__init__` eagerly
  imports modelscope/onnxruntime); transforms load by file path as before.
- **overrides.yaml / overrides-ship.yaml** (kept in sync):
  - pair_groups: `[cddb-real, cddb-fake]`; all five dflip3k-* as one
    same-source clique.
  - shard_groups: the three dflip3k-real-* collapse to one sampling group
    (one real pool split per matched generator); the two dflip3k fakes stay
    separate groups (different generators, the FairFaceGen rule).
  - semisynthetic_datasets: + deepfakes-qa-15k (faceswap on real footage —
    more of exactly what the 24.2% zero-shot face-swap result asked for).
  - holdouts: **all 13 v21 datasets go to TRAIN**, deliberately. They are
    public now; the face-swap lesson says holding out public data the
    leading miner trains on only donates points. The dev/ship holdout panels
    are untouched.

## Gasstation weeks (2026-08-10)

The cache now holds 8 ISO-week dirs under `gasstation-generated-images/`
(2026W26…). Nothing structural is needed to include them: build_manifest's
`find_dataset_dirs` probes one level down and accumulates every week into the
one logical `gasstation-generated-images` dataset (the `iso_week` column
records which week each row came from), and with no `--per-dataset-cap`
(default) every week lands in train.

What DID need changing is the sampler: gasstation was getting a single
dataset-share of its (fake, diverse, synthetic) cell while gasbench samples
it at 5x per dataset (GASSTATION_WEIGHT_MULTIPLIER) and weights it further in
score_composition. Ported video_v1's `gasstation_boost` into
`data.py:balanced_weights` (group-share level, so P(fake)=0.5 and
P(fake|category)=0.5 are exactly preserved — verified numerically), wired
through `train.py`, and set `sampler.gasstation_boost: 5.0` in config.yaml.
The sampler report now prints/records `p_gasstation` — sanity-check it after
the manifest rebuild.

audit_data.py note: `--expected-gasstation` defaults to 5000 (the old
single-snapshot count). With 8 weeks pass the real expectation, e.g.
`--expected-gasstation <total rows you expect>`, or just read the printed
per-week count.

## Runbook (on the GPU box)

```bash
# 0. sync the gasbench checkout (must be ≥ 0.8.3 / commit 52b3b99)
cd <bitmind-subnet>/gasbench && git pull
export GASBENCH_SRC=<bitmind-subnet>/gasbench/src   # if not an ancestor of the pipeline dir

# 1. download the new datasets into the existing cache (~6.5k images at the
#    500/dataset cap). --no-eviction so nothing already downloaded is evicted.
gasbench download --modality image --cache-dir <CACHE> --no-eviction \
  --datasets cddb dflip3k google-streetview-images data-art-backup \
             feyiamujohuman-palm-images deepfakes-qa-15k sdfd 17-ai-model-images
# (--datasets is substring-matched: "cddb" pulls both halves, "dflip3k" all five)

# 2. rebuild the manifest + gate it
python build_manifest.py --cache-dir <CACHE> --out data/manifest.parquet --image-size 384
python audit_data.py --data-root <CACHE>/datasets --manifest data/manifest.parquet

# 3. re-split (dev first; ship split when freezing for submission)
python build_splits.py --manifest data/manifest.parquet            # review matrix
python build_splits.py --manifest data/manifest.parquet --write
#   ship: add --overrides overrides-ship.yaml

# 4. pre-flight, then (re)train branches per README
python verify.py
```

Expected in the build_manifest report: 13 more datasets, `art` no longer in
the single-label fold list, P(fake)=0.5000 preserved, and the dflip3k reals
counted as one group in the faces-real cell.
