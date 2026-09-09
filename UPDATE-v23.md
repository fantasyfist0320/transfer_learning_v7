# UPDATE v23 — registry resync for Round 23 (2026-08-24)

Gasbench dev @ 98498e0. Round 23 runs Aug 24 – Sep 3 20:00 UTC.

## What changed upstream (all verified against gasbench commits)

- **Registry cull** (d659721): 74 duplicate rows -> `legacy_images/videos.yaml`
  (NOT loaded by the loader or by eval). Image side: 18 synthetic + 3 real
  gone from active, incl. SFHQ part2-4, human-faces r2-r9, SDv15R part1/3,
  fakeclue satellite pair + fake-doc.
- **3-class audited taxonomy** (ecb2cea, PR #130): image = real | synthetic |
  semisynthetic, **pixel-based** (captured pixels retained alongside localized
  generated content; editing synthetic media stays synthetic). SN34 scoring
  still collapses to binary real-vs-not-real — a 2-logit head remains valid.
  Excluded as unsplittable: justweirdimages, cg-fake-id, deepfake-insight,
  artifact-bench. receipts-i2i flipped semi -> synthetic.
  Registry-native semis now: face-swap, deepfakeface-inpainting,
  deepfakes-qa-15k, fakeclue-fake-ffpp, deepfake-identity-isolated-fake.
- **v22 holdout release** (39cdf29): 24 image datasets joined as `34data/*`,
  incl. gemini31-flash-lite-train/val, gpt-image-edit-1-5m-hqedit,
  ntire-robustaigendetection pair, deepfake-identity-isolated pair,
  sid-set pair (released as saberzl-sid-*, renamed), medical reals,
  signature genuines, yfcc100m. Note: onemillionfaces-* and 2 of the 3
  ai-generated-ecommerce-* sets were subsequently culled to legacy.
- Net active image registry: **190 datasets (106 real / 79 synthetic /
  5 semisynthetic)**, was 192.

## What was edited here (all three overrides files, kept in sync)

1. `category_overrides`: fakeclue-*-satellite removed (legacy).
2. `pair_groups`: SDv15R and human-faces groups removed (single survivors);
   ADDED ntire pair, deepfake-identity-isolated pair, sid-set pair,
   gemini31-flash-lite train/val.
3. `shard_groups`: human-faces / SFHQ / SDv15R groups removed (cull collapsed
   each to one entry); ADDED gemini31-flash-lite train/val.
4. `semisynthetic_datasets`: **emptied**. The audited registry media_type is
   authoritative; the old list was adopted (3 names), overruled (11 names:
   AttGAN/STARGAN/STGAN, imagepulsev2 x6, digi2real, pica-100k), or referenced
   gone datasets. Revisit the overruled set only if manipulation probes
   degrade — do not edit the registry.
   Consequence: manipulation is now only 5 fake datasets; the sampler's
   manipulation-axis balance rests on those, watch the face-swap probe.

Holdouts and train_gates untouched — every referenced dataset is still active.
Validated: all names in all three files resolve against the v23 registry;
registry loads 190 entries with no duplicate names.

## Remote steps (GPU box) — in order

```
cd bitmind-subnet/gasbench && git checkout dev && git pull   # -> 98498e0+
# download new 34data image sets into the gasbench cache (24 released
# holdouts; prioritize gemini31/gpt-image-edit/ntire/deepfake-identity/sid-set)
# re-pull latest gasstation weeks
cd <v7> && python build_manifest.py                          # expect 190 joined
python audit_data.py                                         # pre-training gate
python build_splits.py --overrides overrides-experiment.yaml --val-id-frac 0.20 --write
#   (or overrides-ship.yaml per arm; expect NO "[warn] ... not in the registry" lines)
```

Expect: manifest count drops (culled sets leave) then rises (new 34data sets
arrive once downloaded). Datasets in the cache but no longer in the registry
(legacy) must NOT enter the manifest — the registry join already enforces this.

## 3-class head migration (2026-08-25, gasbench 0.9.0 / PR #136)

Gasbench added multiclass sn34 (Gorodkin MCC + multiclass Brier over
real/synthetic/semisynthetic) behind `--multiclass-scoring`; the score is
recorded on every run regardless of the flag. A 2-logit head takes a
structural ~7-8% multiclass haircut (semi recall 0, >=1.5 Brier per semi
sample). v7 migrated to a **factorized 3-logit head** that provably preserves
binary behaviour (see `.claude/plans/generic-swinging-penguin.md` for design):

- model.py: heads widened to 3 logits [real, syn, semi], prior-shift init
  (semi logit -3 nats); new helpers `binary_margin` (logsumexp collapse),
  `type_margin`, `collapse_binary` — THE margin convention, everywhere.
- data.py: `y3` derived from label+kind, emitted in every batch.
- train.py: factorized CE (binary term byte-identical incl. 2-class label
  smoothing) + `loss.type_weight` (default 0.3; 0 = binary-only fallback) x
  conditional syn-vs-semi CE on fake rows; brier_loss now 1-softmax[0];
  selection unchanged (binary sn34); new per-split `type` diagnostic column.
- calibrate.py: `multiclass_metrics` (gasbench Metrics num_classes=3 — the
  local multiclass mirror) and `fit_type_posterior` (prior-reweighted (T2,b2)
  fit + q_max grid; returns mc_sn34 before/after = ship/no-ship gate).
- export.py/template: factorized forward — p_fake = sigmoid(fused
  logsumexp-margins / t_eff) with the existing hinge; q = clamp(sigmoid(
  d2/T2+b2), q_min, q_max); output log([1-p_fake, p_fake(1-q), p_fake q]).
  New buffers type_temperature/type_bias/q_min/q_max; model_config
  num_classes: 3. New flags: `--type-prior`, `--binary-equivalent` (pins q —
  the fallback export; binary path identical either way).
- verify.py: factorized-parity check + gasbench 3-class contract check;
  shape asserts (2,3).
- Swept every `z[:,1]-z[:,0]` site (measure_margins, brier_headroom,
  ingest_external, audit_shortcuts, eval_probes 4-tuple unpacks).

Local validation (GPU-less VM, numpy venv, gasbench's real Metrics):
scorer-level binary parity to 1e-12 for arbitrary q; pinned-q export ==
2-logit head under multiclass; fit gate behaves on informative/noise/
degenerate margins. STILL PENDING on the GPU box: `python verify.py` (torch
checks), 60-step smoke per branch family, then the retrain.
Old 2-logit best.pt checkpoints fail strict load loudly — retrain required
(already planned for round 23).

## Dataset balance audit post-v23 (2026-08-25) — values KEPT

`sampler.kind_balance` 50/25/25 was re-derived against the v23 registry.
Because kind shares are cell-local and all 5 surviving semi datasets sit in
`faces` (q_faces ~ 0.39), the realized semi share self-moderated from 43.4%
to ~19.4% of the fake half (~9.7% of total mass, ~3.4x replay vs the 25x
cap) — a healthy 3x lift over natural, safe on replay, well matched to the
type head (which prior-corrects at export) and to eval's semi class (same
faces-manipulation family). No value change. What DID change: stale config/
docstring numbers refreshed; sampler report gained `top_mass` (top-5 dataset
mass + amp, printed every run) and `replay_cap_bound` (loud warning when the
water-fill clip engages — the P(fake|category)=0.5 identity breaks then);
verify.py `_sampler` gained absolute guardrails (P(semi) in [0.05, 0.30],
semi peak replay < max_replay/2, cap unbound) that the old relative-lift
assert could not provide. Verified locally with pandas on a v23-shaped frame
(semi amp 3.37x reproduced; cap-bind path exercised).
