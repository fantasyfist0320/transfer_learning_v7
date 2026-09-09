# RUNBOOK: no-degradation experiment (fragile-fingerprint test, 2026-08-25)

Why: r22 lost 83.52 vs 86.22 blended despite better accuracy AND far better aug
robustness (97.8% vs 77.7%). Hypothesis: the winner trains WITHOUT degradation,
keeping fragile high-frequency generator fingerprints that crater under the aug
pass (their robustness 77.7%) but transfer zero-shot to unseen generators —
they scored 68-81% on the two holdouts that are almost certainly
gemini31-flash-lite (released post-round as 34data/*), where our
degradation-trained model zero-shots 7.69% / 23.08%.

Two dinov3 arms, SAME manifest/splits/seed (34), 10 epochs, bs 96, lr 1.7e-4:

| arm | config | writes | recipe |
|---|---|---|---|
| C (treatment) | `config-clean.yaml` | `runs/v7-clean` | NO degradation: prechain/jitter/schedule off, view_arms deploy 1.0, `view_b: deploy` (new knob), flips kept |
| D (control) | `config-experiment.yaml` | `runs/v7` | per-epoch degradation schedule (unchanged r22 recipe) |

(A third arm E — `config-experiment-2.yaml` → `runs/v7-plateau`, the dinov3
epoch-budget plateau probe — lives in RUNBOOK-PLATEAU.md. It is a separate
experiment; this table is arms C/D only.)

gemini31-flash-lite-train/val are gated OUT of train/val_id/val_xgen into
val_stress (overrides-experiment.yaml `train_gates.gemini31-zeroshot`), so both
arms print a per-epoch zero-shot trajectory on deploy AND robust views while
selection (val_id 0.8·deploy+0.2·robust — NOT pure deploy; pre-registered)
cannot see it. dflip3k is re-gated to `test` as a second probe family with a
stylized-REAL side (fake-leaning-prior check), evaluated post-hoc only.

## 0. Preconditions

- **transformers MUST equal gasbench's pin (5.2.0)** in the training venv
  (`pip install "transformers==5.2.0"`). A fresh venv grabs 5.15.x, which
  builds DINOv3 with an extra `.model.` nesting level: training/export
  round-trip fine against themselves, then gasbench (and the validator)
  refuse every backbone key — a 0-scoring submission. verify.py now has a
  check for this ("transformers version matches gasbench's pin").
  Checkpoints trained under the wrong version do NOT strict-load after
  pinning — retrain, don't remap.

- v23 sync per UPDATE-v23.md (gasbench dev >= 39cdf29; 34data image sets
  downloaded — gemini31 pair REQUIRED), **EXCEPT: do NOT re-pull gasstation
  weeks yet.** The arms train on the frozen <=2026W32 cache so validator
  gemini-3.1 traffic cannot leak the target generator into train. Re-pull
  after both arms finish.
- rsync the WHOLE dir, not a file list — the last 48h touched the 3-class
  migration, the balance audit AND the view_b change, and partial syncs
  produce one observed failure mode: `sampler identities survive the kind
  level` fails at p_fake != 0.5. Seen twice on the box (2026-08-25): stale
  data.py (missing cell-local `_kind_shares`) and stale **overrides.yaml**
  (old category_overrides -> different folding; the check's "(0 promoted)"
  line only proves the semisynthetic_datasets section is current, so a stale
  overrides.yaml can look freshly synced).
  ```bash
  rsync -av --exclude runs/ --exclude data/ --exclude __pycache__/ \
      ~/Documents/new-training/training/transfer_learning_v7/ \
      <box>:/workspace/transfer_learning_v7/
  grep -c _kind_shares /workspace/transfer_learning_v7/data.py   # expect 2; 0 = stale
  ```
- `python verify_views.py` → 10/10 (torch+torchvision+scipy in the venv).
- `python verify.py` all green + 60-step smoke of the 3-logit head
  (UPDATE-v23.md pending item) — run the smoke for BOTH configs:
  `python train.py --config config-clean.yaml --branch dinov3 --max-steps 60`
  (repeat with config-experiment.yaml). The clean run MUST print
  `view_b (consistency view): deploy` — if it says robust, stop.

## 1. Gasstation: gated OUT of the arms (was: freeze + leak scan)

RESOLVED 2026-08-25: the box cache holds ONLY `2026W35` — the live round-23
week — with 2,459/2,785 samples (88%) attributed `model_name: unknown`
(+ seedream-4-5 x326). The scan's "no gemini match" is therefore weak
evidence: the target generator could hide in the unattributed mass of the
exact week it would flow in. The pre-gemini weeks (<=W32) were never
downloaded to this box, so the freeze is unavailable.

Decision: `train_gates.gasstation-w35` in overrides-experiment.yaml parks
gasstation-generated-images in `test` — out of train for BOTH arms
(symmetric, so the C-vs-D contrast is untouched), and post-hoc it becomes a
live-traffic zero-shot readout per arm (seedream-4-5 is itself an unseen
generator). gasstation_boost no-ops with zero gasstation train rows; configs
stay identical. audit_data.py will warn `gasstation ... under floor` — that
warning is EXPECTED for the arms. SHIP: flip the gate true after the
round's gasstation re-pull (~Aug 29).

Still run the census for the record (and re-run after any re-pull):
```bash
python scan_gasstation_generators.py --cache-dir <CACHE> --pattern gemini
```

## 2. Manifest + splits (once, shared by both arms)

**FREEZE RULE: both arms train on the byte-identical manifest.** Rebuilding
between arm C and arm D breaks the paired design (adding datasets can shift
the 80/20 carve of EXISTING datasets). The selfgen data (RUNBOOK-SELFGEN)
interacts with this: as of 2026-08-25 the 6 local-swap-* sets are on disk and
in the manifest, the 8 local-inpaint-* are pending ("[warn] ... not
downloaded (8)" is that). Either finish the inpaint arm FIRST (~4-5h),
rebuild ONCE, rerun the gate/parquet checks below plus RUNBOOK-SELFGEN §3
(clique eligibility: every local-* split_unit shows labels [0,1] in train)
and §4 QC — recommended, it also un-faces-onlys the semi pool — or run both
arms on the swap-only manifest and fold inpaints into the ship build only.

```bash
python build_manifest.py --overrides overrides-experiment.yaml ...  # as usual
python audit_data.py     --overrides overrides-experiment.yaml ...
python build_splits.py   --overrides overrides-experiment.yaml --val-id-frac 0.20 --write
```
Expect THREE `[gate]` lines: `gemini31-zeroshot ... -> val_stress (2 datasets)`,
`dflip3k ... -> test (5 datasets)`, `gasstation-w35 ... -> test (1 dataset)`.
Then prove it:
```bash
python - <<'EOF'
import pandas as pd
d = pd.read_parquet("data/manifest.parquet")
vs = d[d.split == "val_stress"];  te = d[d.split == "test"]
assert sorted(vs.dataset.unique()) == ["gemini31-flash-lite-train", "gemini31-flash-lite-val"], vs.dataset.unique()
assert sorted(te.dataset.unique()) == sorted(["dflip3k-fake-dreamshaper-xl","dflip3k-fake-pixelwave-11","dflip3k-real-juggernaut","dflip3k-real-dreamshaper-xl","dflip3k-real-pixelwave-11","gasstation-generated-images"]), te.dataset.unique()
leak = d[d.dataset.str.startswith("gemini31") & (d.split != "val_stress")]
assert leak.empty, leak
assert not (d[d.split == "train"].dataset.str.contains("gasstation")).any()
print(d.split.value_counts())
EOF
```

## 3. Phase 0b GATE — RESOLVED "hypothesis live" on prior evidence (2026-08-25)

The gate as designed (per-dataset AUC/margins of the current best.pt on
gemini31) is NOT runnable: the fresh box has no runs/ (old instance gone),
and the API's per-sample analytics endpoints (misclassified-samples /
confusion-matrix / summary) hang until timeout for closed r22 runs, so
probabilities are unrecoverable from the dashboard side.

Resolution on banked evidence: the r22 loss decomposition (4-decimal exact
score fit) showed the two gemini31 holdouts were a **confident inversion** —
the fakes sat at confidently-REAL probabilities, interleaved with the
below-floor-calibrated reals (~0.02-0.05). That is the no-signal case, not
the threshold-shifted case → calibration alone cannot fix it → the arms are
justified. OPTIONAL upgrade if the r22 submission zip is still retrievable
(HF model repo / local copy): load it and run measure_margins on gemini31 —
a real AUC read in ~10 min; do it if it is one download away, do not block.

Still valuable pre-training (needs no checkpoint): per-stat AUCs
(audit_shortcuts tensors, or FFT/DCT stats) on clean gemini31 AND nano-banana
vs reals, repeated after the aug chain — direct "easy signal exists and
degradation kills it" evidence.

## 4. Train (arm C first — earlier hypothesis signal), ~10h each

```bash
python train.py --config config-clean.yaml      --branch dinov3 2>&1 | tee train-clean-dinov3.log
python train.py --config config-experiment.yaml --branch dinov3 2>&1 | tee train-sched-dinov3.log
```
- Startup lines to verify: clean prints `view_b ... deploy` and NO
  "degradation schedule ACTIVE"; sched prints the reverse.
- The `kl=` term in the step log is the symKL magnitude — in arm C it is
  R-Drop (dropout-consistency on identical views); record its size vs arm D.
- Per-epoch `val_stress:deploy` accuracy IS the zero-shot gemini31 number
  (all-fake panel → accuracy = recall); `val_stress:robust` = does the signal
  survive degradation. The trajectory lives ONLY in these tee'd logs.
- Early read: arm C flat <= ~30% by epoch 5-6 with healthy val_id → the
  hypothesis is unlikely to rescue; cutting the run is legitimate.

## 5. Benchmark + export per arm

```bash
python export.py --branches dinov3 --runs-dir runs/v7-clean --calib-splits val_id,val_xgen ...
python export.py --branches dinov3 --runs-dir runs/v7       --calib-splits val_id,val_xgen ...
```
`--calib-splits val_id,val_xgen` is MANDATORY here: val_stress is the judged
zero-shot pool and must not enter the temperature fit (the log prints
`calibration fit splits:` — check it). Then the naming-contract proof, in
order:
```bash
# 1. artifact-level: backbone keys must be flat (no `.model.` level)
python -c "
from safetensors import safe_open
f = safe_open('submission/model.safetensors', 'pt')
ks = [k for k in f.keys() if 'backbone' in k]
assert not any('.backbone.model.' in k for k in ks), ks[:3]
print('flat naming ok:', ks[0])"
# 2. environment-level: gasbench itself must get PAST model loading —
#    the small run doubles as this gate; a load failure prints in seconds.
```
Then gasbench small sanity → full run
with `--n-aug-per-dataset 8`. Headline rows: gemini31 x2, nano-banana,
face-swap, gasstation (now ZERO-SHOT for both arms — W35 is test-parked, and
its 88%-unknown generator mix incl. seedream-4-5 makes it a live-traffic
transfer probe); dflip3k via eval_probes/measure_margins (test split).

## 6. Judgment (pre-registered)

Primary: zero-shot gemini31, per-epoch trajectory AND best.pt, both arms.
Anchors: current model 7.69%/23.08%; winner 73-81%. Min effect 15-20pt C-D
(±3-4pt binomial CI); the contrast is only interpretable if arm D <= ~50%.

Prior-shift disambiguators (all three, per arm, at best.pt):
1. AUC of gemini31 margins vs a fixed real pool (val_id reals + dflip3k reals).
2. val_id real-class accuracy (did the clean boundary drift fake-ward?).
3. gemini31 recall at matched FPR (pin val_id real acc ~98%, read recall).
A raw-recall win that vanishes at matched FPR / in AUC = prior shift →
calibration/clamp path, not a recipe change.

| C | D | reading → action |
|---|---|---|
| >=~60% (survives matched FPR) | <=~30% | CONFIRMED → r23 recipe = mixed exposure (clean primary + robust auxiliary view is the natural next arm); ship folds gemini31 into train |
| <=~30% | <=~30% | rejected for our arch → data path (self-gen Gemini-3.1 via API) + probability clamp |
| high | >=~60% | v23 DATA explains it → keep robust recipe + fold gemini31 in |

Secondary: arm C aug-robustness cost (prices the hedge); dflip3k-real acc per
arm (winner-weakness reproduction); face-swap/semis rows (run_3 predicts clean
training lifts HF-dependent manipulation detection); audit_shortcuts
`--mode interventions,slices` on BOTH arms (differential mechanism: C's
highpass/jpeg sensitivity should rise; codec AUC must stay ~0.5 in both — a
format-shortcut win would be a FALSE positive). Mirror-score note: 22/24
released holdouts are IN train, so any "holdout component" must be computed on
gemini31 only.
