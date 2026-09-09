# RUNBOOK: dinov3 epoch-budget plateau probe (2026-09-08)

**Why.** `config-experiment.yaml` trains every branch for 10 epochs, and that
10 is not a derived number. Its own header states the justification: it matches
the 10-entry `degradation_schedule` so the run ends on the deploy-faithful
finale profile. That explains 10 *given* a 10-entry schedule; nothing explains
the schedule's length. Three checks confirm it was never solved for:

- Profiles **0 and 7 are byte-identical** — 9 distinct profiles padded to 10.
- The code never required the match: epoch `e` uses profile `e % len`, so the
  real constraint is `epochs % len == 0`, far weaker than "must equal".
- Nothing else pins it: `warmup_ratio` is a ratio, `cosine_lr` spans `total`,
  `brier_ramp` is a progress fraction. Every schedule-shaped hyperparameter
  auto-stretches. The baseline is `epochs: 2`, so 2-vs-10 is an arm contrast.

At the ~465k-image pool this costs ~37.5k steps/branch and turns a ~14 h
5-branch build into ~40 h. **This probe measures the budget instead of
inheriting it.**

**What it produces.** `S` = the total step budget past which dinov3's `val_id`
sn34 stops improving. Ship `epochs` is then `round(S / steps_per_epoch)`.

## Design, and why it is a LADDER and not one long run

`cosine_lr` decays over `total = args.max_steps or steps_per_epoch * epochs`.
In a single 12-epoch run, late gains mix "more steps" with "LR → 0", so you
cannot read off what a shorter *complete* run would have scored — and killing
it early leaves a high-LR, mid-schedule checkpoint, not a converged one.

So: **three complete runs**, each given its own budget with `--max-steps` and
therefore its own complete cosine. 2+4+6 epochs = 45k steps — the same compute
as one 12-epoch run, but it answers the actual question.

Degradation is **OFF** for this arm, with `laundering` / `resample_jitter` /
`view_arms` set to the exact time-average of the schedule profiles. A
stationary training distribution is what makes a flat curve mean "plateau"
rather than "profile 4 was the hard one".

**Caveat on transfer — the time-average is NOT what a scheduled run trains
on.** Cosine decay weights early epochs far more, so a scheduled run's
effective mixture is the LR-weighted marginal, not the mean. Measured against
this schedule: `double_jpeg_p` reads 0.125 as a mean but 0.1387 at 8 epochs
(+11%), 0.1675 at 4 (+34%) and 0.2503 at 2 (+100%). `S` therefore transfers
approximately, and better the longer the ship run. This does not invalidate the
probe — arm E is stationary by construction and its curve is readable — but do
not claim the two distributions are identical.

This does **not** change the yardstick: `__getitem__`'s eval path uses only
`cfg.image_size` and `eval_mode` — it never reads `prechain`,
`resample_jitter`, `view_arms` or `robust_skip_webp_p`. Eval rendering stays
byte-identical to `config-experiment.yaml`.

| arm | config | writes | recipe |
|---|---|---|---|
| E (probe) | `config-experiment-2.yaml` | `runs/v7-plateau` | schedule OFF, base = profile time-average, `eval_every: 500`, `eval_max_rows: 8000` |

**The schedule was rebuilt for this arm (2026-09-08).** The inherited 10-entry
schedule carried a byte-identical duplicate (profiles 0 and 7), spent variation
on `view_arms` — a near-inert dial, since view_b is unconditionally the
robustness chain so sweeping deploy 0.65→0.80 moves clean CE mass only
32.5%→40.0% — and ordered profiles so its two hardest ran at 9.2% and 6.0% of
the LR-weighted learning while the "deploy-faithful finale" got **0.18%**.

It is now **2 profiles**: `[laundered, deploy-faithful]`. Length 2 is the only
length for which `epochs % len == 0` holds across every even budget the probe
can return (2, 4, 6, 8) — a 4-profile version broke at epochs 2, where the last
two profiles never execute and the run ends mid-schedule. The two profiles were
chosen so their mean is **bit-identical to the previous base**, so this rebuild
does not move the distribution `S` is measured against.

It also varies `prechain_webp_q`, which nothing varied before: WebP at the
default (60,95) is the largest photometric prechain distortion (6.6 MSE against
single-JPEG's 2.3). The marginal stays at the default, so again no drift.

**Correction to an earlier severity claim.** double-JPEG was described as the
most damaging prechain op at 24.9 dB. That measurement was ~99% the 1-7px crop
that misaligns the DCT lattice, not compression: with the shift disabled it
measures 41.7 dB, *milder* than WebP (40.8) and resample jitter (42.6). PSNR is
translation-sensitive. So the double arm is a translation augmentation carrying
a scarce lattice signature — it does not erase high-frequency evidence, and is
therefore safer at high LR than that number implied, but it is not the harshest
operation in the recipe.

`ladder_level_probs: [0.25, 0.0, 0.25, 0.50]` also ships here. Ladder level 1
was dead mass — it adds only h/v flips, and `__getitem__` already flips at
source with `hflip_p = vflip_p = 0.5`, so composing two Bernoulli(0.5) flips is
Bernoulli(0.5). Verified empirically: 600/600 L1 renders are an exact flip of
the same-seed L0 render, with identical pixel statistics. Its share goes to L3,
taking blur/noise from 1.29% to **2.71%** of CE mass at no cost in clean-pixel
mass and with no change in severity.

**S is a property of THIS recipe.** The marginal shifted (double-JPEG and
`robust_skip_webp_p` each +19%, everything else within a few percent, clean CE
mass unchanged at 35.0%). If the recipe changes again — especially the
degradation balance — re-measure the budget; a harder distribution takes longer
to fit.

Arms C/D are in RUNBOOK-NODEGRADE.md. All three `output_dir`s differ; nothing
in the tree guards that, so check it before launching.

## 0. Preconditions

- **transformers MUST equal gasbench's pin (5.2.0)** in the training venv.
  See RUNBOOK-NODEGRADE.md §0 — a mismatch produces a 0-scoring submission and
  checkpoints that do not strict-load after pinning.
- `python verify.py` all green; `python verify_views.py` 10/10.
- 60-step smoke, which must print **no** `degradation schedule ACTIVE` line,
  must print `view_b (consistency view): robust`, and must print the
  `eval_max_rows=8000` line:
  ```bash
  python train.py --config config-experiment-2.yaml --branch dinov3 --max-steps 60
  ```

## 1. Data preparation

```bash
python build_pool.py --cache-dir /data/pool --workers 4 --skip-done
gasbench download --gasstation-only        # build_pool.py skips gasstation

python build_manifest.py --cache-dir /data/pool --overrides overrides-experiment.yaml
python build_splits.py  --overrides overrides-experiment.yaml --val-id-frac 0.20
```

Three landmines:

- **`--val-id-frac 0.20` MUST be passed explicitly.** The CLI default is
  `0.06`; every runbook and `overrides-experiment.yaml` assume `0.20`. Omitting
  it silently yields ~442k train rows instead of ~360k — a 23% shift in
  steps/epoch that corrupts the entire measurement.
- **`val_xgen`, `val_stress` and `test` will all be empty** under
  `overrides-experiment.yaml` (all four `train_gates` carry
  `include_in_train: true`; the surrounding comments are stale). This is fine —
  `selection: {val_id: 1.0}` is already set — but state it plainly: the plateau
  is an **in-distribution** measurement. That is the right yardstick here
  (val_id tracked the public score to 0.008 in the 2026-08-07 dinov3 run), not
  an accident.
- **Do not run `split_holdout.py` against this pool.** Its default
  `--mode link` leaves held-out images in the source, and nothing in this tree
  reads its `CACHE_ROLE` / `held_out_image_ids.txt` markers — the integration
  is not implemented, so those rows would be trained on.

**MEASURED, 2026-09-08 pilot run.** Train split **255,738 rows** →
**2,664 steps/epoch @ bs 96**, at **1.6 s/step** (not the 0.92 originally
assumed, so every budget here is ~1.7x the earlier estimate). Re-read
`steps_per_epoch` from `len(train_loader)` on each launch and rescale if the
manifest changes.

**Before launching, confirm the `local-*` sets are actually in the pool.**
`build_manifest.py` skips any on-disk dataset absent from registry + extras
(`continue  # unknown datasets carry no label; skip rather than guess`), and it
equally cannot serve a registered dataset that is NOT on disk. The pilot served
209 datasets and reported `P(kind): semisynthetic=0.0952` — the exact value the
registry alone produces, which means the `local-*` extras were missing. All 13
registry semis live in `faces`, and `kind_balance` is cell-local, so the only
way to raise the semi share is to supply semis in other categories; the
`local-*` edit2/fluxfill sets cover `diverse` and `documents`. Check the
`[warn] in the registry but not downloaded (...)` line.

## 2. The ladder

**Status 2026-09-08.** The 7500-step run is a **PILOT, not a rung**: it trained
with `use_degradation_schedule: false` (the stationary marginal), and the
schedule is now ON. Its verdict stands regardless — selection was still climbing
at step 5000 (0.9273, +0.0025 per 500 steps, no flattening), so 255k rows are
not saturated at 7500 steps.

```bash
C=config-experiment-2.yaml
accelerate launch train.py --config $C --branch dinov3 --max-steps 15000
mv runs/v7-plateau/dinov3 runs/v7-plateau/dinov3-s15000
# only if 15000 has not flattened:
accelerate launch train.py --config $C --branch dinov3 --max-steps 22500
mv runs/v7-plateau/dinov3 runs/v7-plateau/dinov3-s22500
```

The `mv` is **required**: `out = Path(cfg["output_dir"]) / args.branch`, so the
rungs would otherwise overwrite each other's `best.pt`.

Cost at the measured 1.6 s/step: 15000 ≈ **6.7 h**, 22500 ≈ **10 h**.

With the schedule ACTIVE the run alternates P0 (laundered) / P1
(deploy-faithful) by `epoch % 2`. At 2,664 steps/epoch, 15000 steps ends inside
epoch 5 → profile 1, so the run still finishes deploy-faithful. Rungs whose
final epoch index is even would not — check the last
`epoch N degradation profile` line.

## 3. Reading the result

Compare the **final** `val_id` sn34 of each rung — not the within-run curve,
which is what the ladder exists to avoid. With the schedule active the
within-run curve is confounded THREE ways — more steps, LR → 0, and the profile
switching every epoch — so it cannot be read for a plateau at all. Every rung is scored on the *same*
8000 rows (`cap_eval_frame` ranks by a stable hash of `image_id`, it does not
resample), so this is a paired comparison and ~0.002 differences are readable.

- gain 15000 → 22500 < 0.002 → `S ≈ 15000`, ship `epochs: 6`
- both still rising → add a 30000 rung before concluding
- rungs must share a recipe to be comparable; the schedule-off pilot is not
  one of them

The within-run curves are still worth a look as a sanity check: if a rung's
curve is still climbing steeply at its own final step, that rung was
LR-limited, not data-limited.

Then set the ship `epochs` from `S`. **If the schedule is re-enabled, round up
to a multiple of the schedule length** so the finale still lands last — the
constraint is `epochs % len == 0`, not `epochs == len`. The shipped 2-entry
schedule satisfies it for every even budget, so no rounding is needed unless
`S` lands on an odd epoch count.

## 4. What this does NOT settle

- Only dinov3 was probed. `dct` is a from-scratch resnet18 ("the most
  undertrained at 2") and does not inherit the frozen-backbone saturation
  argument — keep it longer, or probe it separately.
- Schedule-vs-time-average is confounded with budget here. Arm E's checkpoint
  is a legitimate time-average arm, so a same-budget scheduled run would settle
  it — but that is a second experiment, not this one.
