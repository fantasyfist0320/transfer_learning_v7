# RUNBOOK: self-generated semisynthetic data (~12k images)

Why: round 23 made the image taxonomy 3-class; the round-22 model measures
multiclass sn34 **0.8589** vs binary 0.9153 (the 2-wide head's dead
semisynthetic class), and the registry has only 6 semisynthetic datasets.
This run manufactures a diverse third class by editing reals already in the
cache. Target after a 3-class head trains on it: multiclass sn34 ≈ 0.93.

Scripts: `generate_edits.py` (inpaint arm, new) and `generate_swaps.py`
(swap arm, extended with `--target` / quality jitter / `img_` filenames).
Registration (`extra_datasets` + `pair_groups`) is ALREADY in
`overrides.yaml` and `overrides-experiment.yaml` — until the directories
exist, `build_manifest` prints a harmless registry-not-on-disk `[warn]`
per name.

All commands: remote GPU box, training venv, from the v7 directory.
Budget ≈ 5–6 GPU-hours total. Run between training arms.

## 0. Preconditions

```bash
pip list | grep -Ei 'insightface|onnxruntime'   # both needed (swap + face masks)
ls data/manifest.parquet                         # must have splits WRITTEN
python -c "import pandas as pd; d=pd.read_parquet('data/manifest.parquet'); print(d.split.value_counts())"
```

The manifest must be built from the SAME overrides you will train with
(the experiment 80/20 manifest is fine — every source has train rows there).
`generate_edits.py` refuses a manifest without a split column: the
`split == "train"` filter is the leakage guard.

## 1. Swap arm (~6k, fast)

```bash
DONORS="celeb-a-hq,CACD,celebs500k,FDDB_Dataset,fakeclue-real-ffpp,MAPIR-Faces,fairface,UTKFace,lfw"
for T in celeb-a-hq CACD celebs500k FDDB_Dataset fakeclue-real-ffpp MAPIR-Faces; do
  python generate_swaps.py --data-root <CACHE> --swap-model <PATH>/inswapper_128.onnx \
      --target "$T" --out-name "local-swap-$T" --sources "$DONORS" --n 1000
done
```

Donor pool is train-side only — never add wider-face/AgeDB (val_xgen
holdouts) or ffhq-256/casia_web_face (val_stress): holdout pixels entering a
train dataset is leakage even as a face crop.

## 2. Inpaint arm (~6k, ~4-5h)

```bash
python generate_edits.py --manifest data/manifest.parquet --cache-dir <CACHE> \
    --per-source 750
```

Defaults: 8 sources (celeb-a-hq, FDDB_Dataset, fakeclue-real-ffpp, yfcc100m,
openfake-real, FantasyID-real, Aslan-mingye-OCR-Quality, fashionpedia),
editors sdxl-inpainting-0.1 + dreamshaper-8 (the validator's own i2i models,
so the artefact distribution matches gasstation semis), masks 50% face-box /
50% random, JPEG q ∈ [88, 96]. Idempotent — rerun after an interruption.

## 3. Rebuild + gate

```bash
python build_manifest.py --overrides overrides-experiment.yaml ...   # as usual
python audit_data.py     --overrides overrides-experiment.yaml ...
python build_splits.py   --overrides overrides-experiment.yaml --val-id-frac 0.20 --write
```

Expect / verify:
- **No "dead weight" warn** for any `local-*` name (proves extra_datasets took).
- **No format-bias warn** >30pp (JPEG jitter should keep qtables mixed).
- **build_splits --write succeeds** — no split_unit straddle error; each
  `local-*` dataset shows a val_id carve (needs ≥2 blocks; the scripts write
  ~12-16 `source_file` blocks per dataset).
- Clique eligibility:
  ```bash
  python - <<'EOF'
  import pandas as pd
  d = pd.read_parquet("data/manifest.parquet")
  t = d[d.split == "train"]
  for u, s in t[t.dataset.str.startswith("local-")].groupby("split_unit"):
      full = t[t.split_unit == u]
      print(u, "labels:", sorted(full.label.unique()), "rows:", len(full))
  EOF
  ```
  Every printed unit must show `labels: [0, 1]` — that is what makes
  clique_pair_frac serve real/edited pairs in one batch.

## 4. QC + baseline (before any retraining)

- Montage ~30 samples per new dataset; check edits are visible, no blank/black
  frames, mask_area_frac in sample_metadata spans ~0.05–0.6.
- Zero-shot the current best checkpoint on the new sets (calibrate.py
  per-dataset AUC table). Expected: swaps score LOW (dinov3 24.2% zero-shot on
  face-swap) — that's the gap this data closes, record it as the baseline.
- `audit_shortcuts.py --mode pairs` — new P_unit rows, no new [warn] beyond
  the known baseline set.

## 5. Afterwards

- Next dinov3 run trains on it (kind=semisynthetic flows automatically).
  Watch: registry semi datasets (face-swap etc.) per-dataset val, the new
  local-* sets, and `multiclass_sn34_score` in benchmark results.json
  (baseline 0.8589).
- Ship freeze: copy the pair_groups + extra_datasets blocks into
  overrides-ship.yaml (sources chosen to be holdout-clean there too).
- Separate decisions, not this runbook: 3-class head; lowering
  `kind_balance` semisynthetic 0.25 → ~0.15 now that the pool is ~5x larger.
