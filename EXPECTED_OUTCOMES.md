# Augmentation deltas (2026-08-07): expected outcomes, written before the code

Four config-gated changes to the view pipeline, each targeting a gap between
what training shows the model and what the evaluator (or the holdout it proxies)
actually contains. Outcomes are declared here first so the A/B afterwards is a
test, not a justification. Baseline for every comparison: the 2026-08-06 full
local gasbench run (`latest_log.txt`): overall 97.69%, sn34 0.9198,
semisynthetic 62.10%.

Evidence recap for why these four and not more: the scored pass is
`apply_random_augmentations(level=0, crop_prob=0.0)` (image_bench.py:245-246);
the production score blends 0.8 base + 0.2 aug-chain (common.py:114); the
ladder's severity tables are unreachable past param index 1 (transforms.py
level_max <= 2, param index = level - 1). Anything beyond eval's reachable
severity re-enters the run_3 high-frequency-erasure failure (face-swap 0.138,
fakeclue-fake-ffpp 0.000 IN-train), so severity is deliberately not touched.

## Δ1 — shifted double-JPEG prechain arm (`laundering.double_jpeg_p: 0.10`)

Re-shared internet content is recompressed after crops/shifts, leaving
**misaligned 8x8 DCT grids**. No current view produces this: the prechain is a
single re-encode, and the robustness chain's JPEG passes land on an aligned
grid. In-the-wild holdout scrapes carry it; a detector that treats "two
misaligned lattices" as out-of-distribution has an avoidable gap.

- Mechanism: JPEG(q1) -> crop shift (dx, dy ~ U{1..7}px) -> JPEG(q2), inside
  `source_prechain`, so it is label-independent by construction. Mass carved
  from the single-JPEG arm: none 0.30 / jpeg 0.35 / double 0.10 / webp 0.25.
- Expected: public overall ±0.2pt (neutral -- public sets mostly lack this
  artifact); in-the-wild-style rows +0-1pt; the real target is holdout, not
  locally measurable.
- Accept if: canaries (`face-swap`, `fakeclue-fake-ffpp` per-eval rows) within
  ±1pt of the paired baseline.
- Revert if: any canary drops >1pt. Revert = `double_jpeg_p: 0.0`.

## Δ2 — JPEG-only robustness-view fraction (`laundering.robust_skip_webp_p: 0.15`)

`apply_robustness_augmentations(webp_quality=None)` is gasbench's own
JPEG-only chain. 100% of view_b currently includes the WebP hop; real
laundering paths are often JPEG-only. 15% of view_b (train path only,
rng-driven) skips WebP; the eval path (`rng=None`) still reproduces the
deployed constants byte-exactly.

- Expected: no public movement; `val_*:robust` (the deterministic exact chain)
  within ±0.5pt -- full-chain exposure stays >=85% of view_b plus the 10%
  robust arm of view_a.
- Accept if: `val_*:robust` holds. Revert if it drops >0.5pt.
  Revert = `robust_skip_webp_p: 0.0`.

## Δ3 — semisynthetic crop guard (`laundering.ladder_crop_guard: true`)

Correctness fix, default ON. Eval crops are mask-aware
(`RandomCropWithParams` keeps the edited region in frame); training ladder
crops are mask-blind because the manifest has no masks. A random crop on a
semisynthetic image can exclude the edit entirely -> a genuinely-real view
trained with label fake, on the class already at 62.1% and now carrying 25% of
sampling mass (`kind_balance`), with the KL term propagating the noise to
view_b. Guard: `kind == "semisynthetic"` rows run the ladder arm with
`crop_prob=0.0` (full frame), mirroring eval's foreground guarantee.

- Noise mass removed: ladder arm 0.20 x crop prob 0.5 x P(crop excludes the
  edit) -> roughly 2-5% of semisynthetic CE mass, plus the KL echo.
- Expected: semisynthetic val accuracy +1-3pt at equal steps; semisynthetic
  Brier down; nothing else moves.
- Accept if: semisynthetic >= +1pt and canaries hold. This is a correctness
  fix -- revert (`ladder_crop_guard: false`) only on a canary break.

## Δ4 — kernel-diverse resample jitter (`resample_jitter.kernels`, default OFF)

In-the-wild content has bicubic/lanczos resize histories; the current jitter
mirrors only the eval chain's AREA-down/LINEAR-up. Kernel diversity is a
plausible holdout win and a plausible run_3-family regression (it can blur the
deterministic INTER_LINEAR aliasing signature that is learnable signal), so it
ships OFF: `kernels: ["area_linear"]` is byte-identical to current behavior
and the single-kernel path consumes no extra rng draw.

- Experiment protocol (only if a dedicated run is budgeted): set
  `kernels: ["area_linear", "area_cubic", "cubic_linear"]`, run the A/B below.
- Adopt only if: val_stress >= +2pt with canaries held. Otherwise it stays off.

## Δ5 — dinov3 native-resolution bump (`dinov3_336` / `dinov3_384` registry variants, 2026-08-19)

The dinov3 branch sees the 384 input antialias-downsampled to 224 (0.583x),
which low-passes part of the generator high-frequency band. 336 (0.875x,
still antialiased) and 384 (no resample at all -- the branch ingests
gasbench's non-antialiased INTER_LINEAR output raw, the dct branch's diet)
expose more of it, at 2.2x / 2.9x token cost. Counter-hypotheses, stated up
front so the A/B is falsifiable:
(a) frozen-DINOv3 detection literature (arXiv 2511.22471, 2602.01738) finds
    global low-frequency structural cues dominate cross-generator transfer,
    so the semantic gain may be ~0;
(b) the HF DINOv3 module applies a random RoPE coordinate rescale in
    [0.5, 2] per training forward (`pos_embed_rescale: 2.0`), so the branch
    already trains under positional-scale jitter that may cover this axis --
    and the same mechanism adds run-to-run noise, hence the margin below;
(c) at 384 the branch loses the ensemble's resolution-heterogeneity rung AND
    gains m_eval-shortcut exposure (build_manifest.py's argument that the
    resolution shortcut is not tunable via S).

- Mechanism: new registry entries only (branches.py) -- exact twins of
  dinov3 except native_size, swap semantics on fusion_weight. NOT config-
  gated: native_size cannot live in YAML (validate_config ignores unknown
  branches: subkeys); reverting = not shipping the variant. Entries are
  inert without a training run.
- Paired 500-step runs, seed 34, config.yaml arm, bs pinned 96 on all arms
  (ckpt flips are numerically neutral; bs changes are NOT -- if an arm must
  drop bs on the OOM ladder, rerun the 224 arm at that bs):
  `accelerate launch train.py --config config.yaml --branch dinov3     --max-steps 500`
  `accelerate launch train.py --config config.yaml --branch dinov3_336 --max-steps 500`
  `accelerate launch train.py --config config.yaml --branch dinov3_384 --max-steps 500`
  Compare the step-500 tables: val_id/val_xgen deploy+robust sn34,
  val_stress accuracy (real-only FPR complement, the stylized-real canary).
- Expected: val_id:deploy sn34 +0.005..+0.015 at step 500 if resolution
  carries signal for this backbone; val_stress within -1pt; robust splits
  within -0.5pt.
- Accept (full 10-epoch run budgeted) if: variant beats the paired
  dinov3@224 step-500 val_id:deploy sn34 by >= +0.005 with val_stress
  >= -1pt and val_*:robust >= -0.5pt.
- 336-vs-384 decision: neither clears -> stop, 224 stands. 384 clears but
  val_stress drops >1pt -> 384 disqualified (aliasing/shortcut exposure);
  proceed with 336 iff 336 clears. Both clear -> prefer 336 unless 384's
  margin >= 2x 336's (336 preserves the resolution-heterogeneity axis and
  costs ~35% less GPU time; 384's raw-aliasing view overlaps dct's niche).
- Ship-swap only per the README decision rule: variant solo gasbench beats
  the same-day dinov3@224 solo export, AND the swapped 5-way fusion beats
  the incumbent 5-way on the same full gasbench run, with the face-swap /
  fakeclue-fake-ffpp canaries within 1pt.
- Revert trigger: any of -- step-500 delta < +0.005; val_stress drop > 1pt;
  fusion-swap canary regression > 1pt. Revert = do not run/ship the variant;
  the registry entries stay (inert without a checkpoint).

## A/B protocol (the "test and verify" part)

1. Cheap structural checks, CPU, ~1 min:
   `python verify_views.py` (new) and `python verify.py` (unchanged).
2. Paired short run, same seed, one branch:
   `accelerate launch train.py --config config.yaml --branch clip --max-steps 500`
   once with the `laundering:` block as shipped, once with it removed.
   Compare the printed per-split tables at step 500: `val_id`/`val_xgen`
   deploy+robust, `val_stress`, and the canary rows.
3. Full check on the winner: export, then
   `gasbench run --image-model <export_dir> --full`; compare overall / sn34 /
   semisynthetic against 97.69 / 0.9198 / 62.10.
4. Every change is config-gated; reverting is editing `config.yaml`, no code.
