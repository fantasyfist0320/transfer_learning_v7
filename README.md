# v7 — heterogeneous ensemble for SN34

Fresh pipeline built on the design of BitMind's own production model
(arXiv:2607.13234): four backbones that disagree in architecture family,
pretraining objective, and native resolution, trained **independently** and
fused by **fixed logit averaging**. Their per-branch ablation (Table 23) is
the whole argument: every single backbone has failure domains (DINOv3's worst
is 0.211 AUC on FF++ face swaps — exactly our worst datasets), while the
fused model has **zero domains below 0.85**.

```
f(x) = 1/3 · ( 1/2·(g_convnext + g_eva) + g_clip + g_dinov3 )
```

| branch | backbone | native | head | adapt |
|---|---|---|---|---|
| dinov3 | DINOv3 ViT-L/16 | 224 | CLS‖patch-mean MLP | LoRA q/k/v/o |
| dinov3_336 (exp) | DINOv3 ViT-L/16 | 336 | CLS‖patch-mean MLP | LoRA q/k/v/o |
| dinov3_384 (exp) | DINOv3 ViT-L/16 | 384 | CLS‖patch-mean MLP | LoRA q/k/v/o |
| clip | CLIP ViT-L/14 | 224 | dropout+linear | LoRA q/k/v/out |
| convnext | ConvNeXt-L (CLIP soup) | 384 | bounded cosine | head+norms |
| eva | EVA-02-L/14 | 448 | bounded cosine | LoRA q/k/v/proj |
| dct | luma→2D-DCT→log-mag → ResNet-18 | 384 | bounded cosine | full, from scratch |

`dct` (added 2026-08-10, ported from video_v1 / BMF §3.3, registry weight
1/6 — the 5-way fusion renormalises to 2/7·(dinov3, clip) + 1/7·(convnext,
eva, dct)) is the frequency-forensic counterweight: nearly blind to content
and style, so it cannot learn the "stylized aesthetic ⇒ fake" prior that put
dflip3k reals at 15%. Its native size equals the declared 384 so no
antialiased resample ever low-passes the spectrum it reads.

`dinov3_336` / `dinov3_384` (added 2026-08-19, EXPECTED_OUTCOMES.md Δ5) are
experimental resolution variants of `dinov3` — identical except the internal
resample target. Registry weight 1/3 with **swap semantics**: a variant ships
*instead of* `dinov3` in its slot (never alongside it), so the renormalised
5-way vector is unchanged. Adoption is governed by the decision rule below
plus Δ5's acceptance criteria.

Declared submission resolution: **384** (`data.image_size`); every branch
downsamples internally with antialiased bilinear, identically at train and
eval. 4-branch core ≈ 700 GFLOPs — 2.7× cheaper than one DINOv3-L at 768; a
dinov3_336 swap adds ≈80 GFLOPs, dinov3_384 ≈125 — both well inside the
1h25m/5h wall clocks.

Score levers, stated once: MCC comes from heterogeneity (a generator shortcut
learned by one representation rarely survives averaging with three others);
Brier comes from the bounded-cosine heads (per-branch confidence is capped, so
one wrong branch cannot saturate the fused probability), the averaging itself,
and one ensemble-level temperature fitted under export numerics and shipped as
a buffer.

## End-to-end

```bash
# 0. one-time: data (needs the 9 missing registry datasets downloaded first —
#    control-face-10k is the only controlnet dataset and val_xgen holds that
#    family out; without it the cross-fit selection falls back)
python build_manifest.py --cache-dir <CACHE> --out data/manifest.parquet --image-size 384
python build_splits.py --manifest data/manifest.parquet          # review
python build_splits.py --manifest data/manifest.parquet --write

# 1. pre-flight
python verify.py

# 2. train branches (independent runs; order = expected value)
accelerate launch train.py --config config.yaml --branch clip --max-steps 60  # smoke
accelerate launch train.py --config config.yaml --branch clip
accelerate launch train.py --config config.yaml --branch dinov3
accelerate launch train.py --config config.yaml --branch convnext
accelerate launch train.py --config config.yaml --branch eva
accelerate launch train.py --config config.yaml --branch dct   # cheap: ~11M params, no downloads

# 3. fuse + export (any subset; weights renormalise)
python export.py --branches dinov3,clip --out submission/
python export.py --branches dinov3,clip,convnext,eva,dct --out submission/

# 4. gate before submitting
cd submission && zip -r ../v7.zip model_config.yaml model.py model.safetensors config.json
gasbench run --image-model ./submission/ --small
```

**Decision rule:** a fusion ships only if it beats the best single branch on
the same gasbench run. Each branch also gets its own solo benchmark (export
with `--branches <name>`), so every regression is attributable.

## Design decisions and their reasons

- **Independent training, not joint.** Fixed-weight fusion means branches
  never need to co-adapt; independent runs fit one 32 GB card each,
  are individually benchmarkable, and a bad branch is dropped, not debugged.
- **Four-term loss** (CE on both views + KL consistency + ramped Brier).
  The previous teacher/MIL/supcon stack was never individually ablated;
  BMF's production branches carry none of it.
- **Temperature as a buffer, not a weight bake.** tanh in the cosine heads is
  nonlinear, so a bake cannot pass through it; dividing the fused logits by a
  stored scalar is exact for every head type and ships inside the safetensors.
- **Data layer carried over, not rewritten** (`build_manifest.py`,
  `build_splits.py`, `data.py`, `calibrate.py`, `overrides.yaml`): the
  cascade with the kind level, the union-find split units, and the cross-fit
  selection were verified numerically; re-typing them would only add bugs.
- **kind_balance 50/25/25** — the manipulated-photo class (attribute edits,
  face swaps, inpainting, doc tampering — 17 datasets listed in
  overrides.yaml) rises from 17% to 43% of the fake half. `face-swap` is the
  held-out probe: if in-train manipulation datasets improve and face-swap
  does not, the sampler memorised sources instead of learning the concept.

## Submission contract (verified against gasbench source)

- harness hands `forward` one uint8 `[B,3,H,W]` batch at
  `preprocessing.resize`, batch 32, on GPU; expects `[B,2]` logits back
- the whole module is cast to the single declared `dtype` (bf16 here)
- allowlist: torch/torchvision/transformers/**timm**/einops/safetensors/PIL/
  cv2/numpy/scipy/math/functools/typing/collections/dataclasses/enum/abc/
  pathlib; **blocked**: os, sys, json, pickle, importlib, getattr, setattr,
  eval, exec — hence every constant in the shipped model.py is a literal
- `load_model(weights_path, **model_config['model'])` must return an
  `nn.Module`; note the loader retries any TypeError with default args, so
  constructor bugs surface as confusing double failures
- wall clocks: 1h25m entrance exam, 5h full benchmark; no per-image latency
  rule exists in code
