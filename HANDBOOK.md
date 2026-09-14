# SN34 v7 Detector Handbook

This handbook covers how Bittensor SN34 scores image discriminators, and how each stage and setting of the `transfer_learning_v7_2` training pipeline follows from that scorer and from measurements recorded in the repo.

| | |
|---|---|
| **Training repo** | `transfer_learning_v7_2` @ `6596991` (7 commits, 2026-09-09 → 09-13) |
| **Subnet** | `bitmind-subnet` @ `720df63` (gas 5.0.0) |
| **Benchmark** | `bitmind-subnet/gasbench` dev @ `fd21699` (v0.9.5) |
| **Sibling docs used** | `../transfer_learning_v7/` (EXPECTED_OUTCOMES, UPDATE-v21/v23, RUNBOOK-*) |
| **Written** | 2026-09-13 |

---

## Contents

- [How to read this](#how-to-read-this)
- **Part I: The subnet**
  - [I.1 What SN34 is](#i1-what-sn34-is)
  - [I.2 The submission contract](#i2-the-submission-contract)
  - [I.3 How gasbench evaluates an image model](#i3-how-gasbench-evaluates-an-image-model)
  - [I.4 The score: sn34](#i4-the-score-sn34)
- **Part II: From scorer to design**
  - [Nine premises that drive everything](#part-ii-nine-premises-that-drive-everything)
- **Part III: Pipeline stages**
  - [S1 Manifest](#s1--manifest-joining-the-cache-to-the-registry)
  - [S2 Audits](#s2--audits-does-the-data-teach-shortcuts)
  - [S3 Splits](#s3--splits-making-every-later-number-mean-something)
  - [S4 Sampler](#s4--sampler-removing-label-information-from-everything-but-pixels)
  - [S5 Views & degradation](#s5--views--degradation-rendering-what-gets-scored)
  - [S6 Model](#s6--model-heterogeneous-ensemble-factorised-3-class-output)
  - [S7 Loss](#s7--loss-a-few-terms-each-tied-to-the-score)
  - [S8 Training loop & selection](#s8--training-loop--checkpoint-selection)
  - [S9 Calibration & export](#s9--calibration--export)
  - [S10 Gate & operating point](#s10--gate--operating-point)
- **Part IV: Record**
  - [Experiment history](#part-iv-experiment-history)
  - [Experiment configs](#experiment-configs-arms-c-to-g)
  - [Commit log](#commit-log)
- [Part V: Open issues found](#part-v-open-issues-found)
- [Part VI: Tool reference](#part-vi-tool-reference)
- [Part VII: Command runbook](#part-vii-command-runbook)

---

## How to read this

The codebase records its reasoning in code comments, run notes and commit messages, not in one design document. This handbook pulls it together. Every claim carries one of three provenance marks:

| Mark | Meaning |
|---|---|
| **[SRC]** | Checked directly against source code (gasbench, the subnet repo, or the training code). |
| **[RUN]** | A measurement already recorded in the repo: a training run, local benchmark or on-chain result. Quoted, not re-run. |
| **[INF]** | An inference made while writing this handbook: usually a consequence of the code, or a gap between two sources. Verify before acting on it. |

Deductions in Part III use a three-column **ledger**:

> **Observed** (the fact or measurement) → **Inferred** (what it implies) → **Decided** (what the code does about it)

Reading the ledgers top to bottom is the fastest way to learn why the pipeline looks the way it does.

---

# Part I: The subnet

## I.1 What SN34 is

SN34 is BitMind's **GAS (Generative Adversarial Subnet)**: mainnet netuid 34, testnet 379 (`gas/config.py:5-6`). It sets two kinds of miner against each other:

- **Discriminative miners** submit a media-provenance classifier (image, video or audio) as a zipped safetensors package. BitMind's cloud evaluates it offline with `gasbench`, so the miner hosts nothing. **[SRC]** `docs/Discriminative-Mining.md:5-12`
- **Generative miners** run a FastAPI service that renders validator prompts. Media must be C2PA-signed by a trusted issuer. Verified media is uploaded to **GAS-Station**, which becomes discriminator eval data. **[SRC]** `docs/Generative-Mining.md:9-20`

Validators do not benchmark models. Every `epoch_length` of 360 blocks they:

1. fetch the current "kings" from `https://gas.bitmind.ai/api/v1/validator/kings`,
2. fetch generator fool counts,
3. verify generator media, and
4. set weights. **[SRC]** `neurons/validator/validator.py:193-278`

### Emissions: King of the Hill

| Lane | Share | Paid to |
|---|---:|---|
| **Image discriminator** (this codebase) | **0.40** | current king 0.85 · previous king 0.10 · the one before 0.05 |
| Video discriminator | 0.40 | same 85/10/5 residual |
| Audio discriminator | 0.04 | same 85/10/5 residual |
| Generators | 0.16 | proportional to verified volume × pass rate × (1 + fool bonus), EMA α = 0.5 |

- **[SRC]** `gas/koth_weights.py`: `KOTH_SPLIT`, `KOTH_LANE_RESIDUAL=(0.85,0.10,0.05)`.
  - Unused slots roll up to the current king.
  - An unresolvable king's share burns (to hotkey `5HjBSeeoz52CLfvDWDkzupqrYLHz1oToDPHjdmJjc4TF68LQ`).
  - Discriminator lanes burn until the kings payload sets `emissions_enabled` and gives an `emissions_start_at` in the past.
- **[SRC, docs only; crowning happens server-side]** `docs/Incentive.md:154-156`
  - A challenger needs `sn34_score ≥ king + 0.01` on the same `CURRENT_BENCHMARK_VERSION`.
  - Resubmitting the same `file_hash` refreshes the score without resetting the reign.
  - When the benchmark version changes, the king defends with a full re-eval; there is a 48 h timeout.
- **[SRC, docs only]** One counted submission per hotkey, across all modalities. A failed exam does not use it up; a new model needs a new hotkey.
- **[INF]** This is winner-take-most. Second place by 0.009 earns nothing. Local improvements below about 0.01 sn34 are not worth a hotkey. That is why the repo tracks *on-chain* numbers rather than trusting local gasbench (see S10).

---

## I.2 The submission contract

| Aspect | Rule | Consequence in v7 |
|---|---|---|
| Package | `model_config.yaml`, `model.py` with `load_model()`, `*.safetensors`, optional `config.json`. **No ONNX.** **[SRC]** `gasbench/docs/Safetensors.md:9-17` | `export.py` writes exactly these four files. |
| Loader | `load_model(weights_path, **model_config['model'])` must return an `nn.Module`. On `TypeError` it retries with `num_classes=` only. **[SRC]** `custom_model_loader.py:29-106` | Constructor bugs surface as confusing double failures. The template takes no runtime parameters. |
| Input | `uint8 [B,3,H,W]` RGB, on the device, at `preprocessing.resize`. gasbench applies **no normalisation**; the `preprocessing` section is never passed to your code. **[SRC]** `pytorch_session.py:216-222` | Each branch divides by 255, resamples and normalises *inside* `forward`, identically at train and eval. |
| Output | Logits `[B,K]`. Image classes: **0 real · 1 synthetic · 2 semisynthetic** (`constants.py:15-30`). Last dimension must be in {1,2,3,4} or equal `num_classes`. **[SRC]** | 3-logit head. The forward returns `log p`, so softmax recovers the composed probabilities exactly. |
| dtype | The whole module is cast: `model.to(device, dtype).eval()`, LayerNorms included. **[SRC]** `pytorch_session.py:64-126` | Temperature is fitted under a module-wide bf16 cast, not under autocast. |
| Code sandbox | Import allowlist: torch, torchvision, torchaudio, transformers, timm, einops, safetensors, flash_attn, PIL, cv2, skimage, decord, numpy, scipy, … Blocked: `os`, `sys`, `subprocess`, network, `pickle`, `importlib`, `ctypes`, multiprocessing, …; calls `eval`, `exec`, `compile`, `__import__`, `getattr`, `setattr`, `globals`, `locals`. No network at runtime. **[SRC, docs]** `Safetensors.md:114-144` | Hand-rolled LoRA (`peft` is not allowed), merged away at export. No `open_clip`. Every constant is rendered as a literal. `allowlist.py` scans the rendered file. |
| Library pin | gasbench pins `transformers==5.2.0`, which builds DINOv3 with flat keys. **[SRC]** | `export.py` refuses to run under any other version. A fresh venv on 5.15 added a `.model.` key level that "would have scored 0". **[RUN]** |
| Batch | Local default image batch 32. Production batch size cannot be determined. **[SRC]** | Model must be batch-size agnostic. |
| Entrance exam | `gasbench run --small`, **≥ 80% accuracy** (averaged across submitted modalities), **5,100 s**. Accuracy is strict 3-class `pred == label`. **[SRC]** `recording.py:90,529` | A head that never predicts class 2 scores 0% on every semisynthetic dataset. Export warns loudly if q ships pinned. |
| Full benchmark | `--full` plus private holdouts, **18,000 s** per modality. **[SRC, docs]** | 4-branch core ≈ 700 GFLOPs at 384 px: well inside the limit. |
| Errors | Samples whose inference errors are dropped from the metrics, not counted wrong (local gasbench). **[SRC]** `recording.py:459` | Production behaviour unknown; do not rely on it. |
| Push | `gascli d push --image-model x.zip --wallet-name … --wallet-hotkey …`: presigned upload, R2 PUT, confirm. HTTP 409 means the hash was already accepted. Optional on-chain commitment `sha256(file_hash+hotkey)[:16]`. **[SRC]** `gas/cli.py:447-509` | |

---

## I.3 How gasbench evaluates an image model

A model is scored on the pixels gasbench actually produces, not on the original files. Reproducing that rendering exactly is the pipeline's first job.

### Registry and sample budgets (v0.9.5)

| | Real | Synthetic | Semisynthetic | Total |
|---|---:|---:|---:|---:|
| Image datasets | 115 | 84 | 13 | **212** |

- Labels are set **per dataset**, not per image (`metadata_utils.py:30-31`). **[SRC]**
- **Total samples** (`dataset/config.py:25-29`): debug 100 · small 5,000 · **full 72,500** (image).
- **Allocation** (`calculate_weighted_dataset_sampling`, `dataset/config.py:232-291`): `unit = T / (n_regular + 5·n_gasstation)`, with `GASSTATION_WEIGHT_MULTIPLIER = 5.0`.

| Mode | Per regular dataset | Per GAS-Station dataset | Actual total |
|---|---:|---:|---:|
| small | 23 | 115 | 4,968 |
| full | 335 | 1,678 | 72,363 |

- The cache holds at most `CACHE_MAX_SAMPLES = 500` per regular dataset, and up to 10,000 per ISO week for GAS-Station.
- **Regular datasets:** seeded shuffle of the cached files, then truncate.
- **GAS-Station:** **newest first**, current ISO week, falling back up to 4 weeks.
- **Holdouts:** private and name-obfuscated (`{media_type}-{modality}-holdout-{sha1[:8]}`). Many are released into the public registry after each round.

### What happens to an eval image

| Step | Base pass (always) | Robustness pass (only if `n_aug_per_dataset > 0`) |
|---|---|---|
| Cache write | PIL `image.save()` with defaults, so JPEG sources are re-encoded at **q75**, WebP at **q80**, PNG stays lossless, EXIF dropped. **[SRC]** `dataset/cache.py:62-84` | same |
| Decode | `Image.open().convert("RGB")` → uint8 HWC | same |
| Degrade | **none**: `augment_level=0, crop_prob=0.0` (`image_bench.py:245-246`, not overridable from the CLI) | ×0.5 `INTER_AREA` down (floor 256) → `INTER_LINEAR` up → JPEG q55 4:2:0 → WebP q75 m4 → JPEG q80 4:2:0. **No blur**, despite the CLI help. |
| Resize | `ResizeShortestEdge`: centre-crop to the target aspect ratio, then `cv2.resize(INTER_LINEAR)`. **No antialiasing on downscale.** **[SRC]** `transforms.py:1177-1235` | same |

**[SRC] Score combination:**

- Robustness blend: `sn34 = 0.8·base + 0.2·aug`; `aug_weight` defaults to 0.2 (`common.py:750-765`).
- Provenance class: dataset name contains `-holdout-` → holdout; contains `gasstation` → gasstation; otherwise public.
- Provenance weight: `w_c = share_c · N_total / n_c`. Production shares are round configuration and cannot be observed.

### What GAS-Station adds

- **What it contains:** verified generator-miner output, i.e. modern commercial API media (Gemini/GPT image, Seedream, …) at 1K/2K/4K (probabilities 0.40/0.40/0.20). All of it is labelled **synthetic**, edits included.
- **Where it is stored:** HF `gasstation/gs-images-v4`, partitioned by ISO week.
- **How much it counts:** 5× the per-dataset budget, newest week first.
- **How generators are rewarded:** volume `min(n,10) + log2(max(1, n−9))` × pass rate × `sqrt(price/baseline)`, times `(1 + clip(fool_rate·s, 0, 2))`, blended as 0.30 image + 0.70 video. **[SRC]** `gas/evaluation/rewards.py`
- **[INF]** GAS-Station is the only eval source that is both adversarial and changes weekly. That is why the sampler boosts it (S4), and why a leak scan runs before any zero-shot claim about Gemini-like generators (S2).

---

## I.4 The score: sn34

```text
gasbench/benchmarks/utils/metrics.py · compute_sn34_score   [SRC]

mcc_norm   = clip((MCC + 1) / 2, 0, 1) ** 1.2
brier_norm = max(0, (B0 − Brier) / B0) ** 1.8        # B0 = 0.25 binary · (K−1)/K = 2/3 for K=3
sn34       = sqrt(max(1e-12, mcc_norm · brier_norm))   # clipped to [0,1]

binary view : binary_label = (label ≠ 0);  p_notreal = 1 − p[0];  decision  p_notreal > 0.5
              MCC = (TP·TN − FP·FN) / sqrt(...);  Brier = mean (clip(p_notreal) − y)²
multiclass  : Gorodkin R_K over argmax confusion;  Brier = mean Σ_k (p_k − y_k)²
              narrow heads are zero-padded, wide heads truncated
blend       : 0.8 · base + 0.2 · robustness   (when the aug pass runs)
```

- **[SRC]** Every result records both `binary_sn34_score` and `multiclass_sn34_score`.
- **Which one is live:** the docs say image and video "currently use multiclass scoring" (`docs/Incentive.md:127`), but gasbench's default is `multiclass_scoring=False`. The production flag cannot be verified from code, so the pipeline tracks both.

### Sensitivity tables

Multiclass (K=3, B0 = 2/3). Rows are MCC, columns are Brier:

| MCC \ Brier | 0.02 | 0.05 | 0.08 | 0.12 | 0.16 |
|---:|---:|---:|---:|---:|---:|
| 0.80 | 0.9134 | 0.8751 | 0.8367 | 0.7852 | 0.7333 |
| 0.85 | 0.9285 | 0.8896 | 0.8506 | 0.7982 | 0.7454 |
| 0.90 | 0.9435 | 0.9040 | 0.8643 | 0.8111 | 0.7575 |
| 0.95 | 0.9583 | **0.9182** | 0.8779 | 0.8238 | 0.7694 |

Binary (B0 = 0.25):

| MCC \ Brier | 0.02 | 0.05 | 0.08 | 0.12 | 0.16 |
|---:|---:|---:|---:|---:|---:|
| 0.80 | 0.8709 | 0.7679 | 0.6634 | 0.5211 | 0.3743 |
| 0.85 | 0.8853 | 0.7807 | 0.6744 | 0.5298 | 0.3805 |
| 0.90 | 0.8996 | 0.7933 | 0.6853 | 0.5383 | 0.3866 |
| 0.95 | 0.9137 | **0.8057** | 0.6961 | 0.5468 | 0.3927 |

Marginal value of each lever at MCC 0.95, Brier 0.05:

| Mode | +0.01 MCC | −0.01 Brier | Ratio |
|---|---:|---:|---:|
| Multiclass | +0.0028 | +0.0134 | 4.8× |
| Binary | +0.0025 | +0.0362 | 14.6× |

### Three consequences **[INF]**

1. **Calibration outweighs discrimination.**
   - Multiclass: ∂S/∂Brier ≈ −1.34 vs ∂S/∂MCC ≈ +0.28. Binary: ∂S/∂Brier ≈ −3.6.
   - Temperature scaling moves Brier but cannot move binary MCC, so it is nearly free score.
2. **Balance beats error count.** MCC and Brier both punish a lopsided FP:FN ratio. Arm F made more errors than log6 and scored higher (Part IV).
3. **The multiclass argmax is not the binary threshold.**
   - With the composed output `[1−p, p(1−q), pq]`, argmax chooses real only when `p < 1/(1+max(q,1−q))`. That cutoff lies between 0.5 and 0.667 depending on q.
   - Example at q ≈ 0.5: an image with p_fake = 0.6 counts as *fake* in the binary view and *real* in the Gorodkin view.

---

# Part II: Nine premises that drive everything

Almost every choice in Part III traces back to one of these facts about the scorer and the data.

| # | Premise | Evidence | Where it lands |
|---|---|---|---|
| P1 | The scored pixels are gasbench's own centre-crop + non-antialiased `INTER_LINEAR` resize. | **[SRC]** `image_bench.py:245` | Training views call gasbench's function through `gasbench_bridge`; nothing is re-implemented. |
| P2 | 20% of the score may come from the JPEG→WebP→JPEG laundering chain. | **[SRC]** `aug_weight=0.2` | view_b is that chain; KL ties it to view_a; selection blends 0.8/0.2. |
| P3 | Brier matters more than MCC. | **[SRC]** exponents 1.8 vs 1.2 | Brier loss term, bounded heads, temperature, hinge, p-clamp. |
| P4 | The benchmark draws about equally from every dataset, so its class prior is ~50/50. | **[SRC]** `calculate_weighted_dataset_sampling` | Sampler forces P(fake)=0.5; eval reweights to a flat prior. |
| P5 | The public score is mostly in-distribution: released holdouts join the public registry. | **[RUN]** v21/v22/v23 releases | Public data goes to TRAIN; selection on val_id. |
| P6 | Scoring is 3-class, and the semisynthetic pool is small and almost all faces. | **[SRC]** registry · **[RUN]** r22 multiclass 0.8589 vs binary 0.9153 | Factorised 3-logit head, kind weights, export type posterior. |
| P7 | Any metadata that predicts the label (format, resolution, category) is a shortcut. | **[RUN]** fakes 45.3% PNG vs reals 17.3% | Category folding, codec laundering, resample jitter, audits. |
| P8 | Heavy degradation erases the high-frequency evidence that identifies face manipulations. | **[RUN]** run_3: face-swap 0.138, ffpp 0.000 | Deploy arm ≥ 0.65 in every epoch; schedules end on a deploy-faithful mix. |
| P9 | Local gasbench is saturated and can rank models in the opposite order to on-chain. | **[RUN]** dinov3_384: local 0.992, lower on-chain | On-chain A/B comparisons; local gasbench is a smoke test only. |

### Pipeline at a glance

| Stage | Name | Main files |
|---|---|---|
| S1 | Manifest | `build_manifest.py`, `overrides.yaml`, `paths.py`, data generators |
| S2 | Audits | `audit_data.py`, `audit_shortcuts.py`, `verify_shortcuts.py`, `scan_gasstation_generators.py` |
| S3 | Splits | `build_splits.py`, `overrides*.yaml` |
| S4 | Sampler | `data.balanced_weights`, `data.BalancedSampler`, `diagnose_replay.py` |
| S5 | Views & degradation | `data.ManifestDataset`, `gasbench_bridge.py`, `verify_views.py` |
| S6 | Model | `branches.py`, `model.py`, `templates/inference_model.py` |
| S7 | Loss | `train.py` |
| S8 | Train & select | `train.py`, `calibrate.py` |
| S9 | Calibrate & export | `export.py`, `allowlist.py`, `measure_margins.py` |
| S10 | Gate & operating point | `gasbench run`, `eval_probes.py`, `brier_headroom.py`, `tune_operating_point.py` |

---

# Part III: Pipeline stages

## S1 · Manifest: joining the cache to the registry

**Files:** `build_manifest.py` · `overrides.yaml` · `paths.py` · `prepare_reals.py` · `ingest_external.py` · `generate_*.py`

**Job.** Walk the gasbench cache and emit one parquet row per image, carrying every column later stages balance or split on:

- `label`, `kind`, `category` (folded), `group`, `source_file`, `file_index`
- dimensions, `file_format`, `pil_mode`, `bytes_per_pixel`, `m_eval`

**Cache layout facts** **[SRC]**

- **Flat layout:** `<cache>/datasets/<name>/{dataset_info.json, sample_metadata.json, samples/}`. GAS-Station adds an ISO-week level (`<name>/2025W40/`).
- **Unreliable filenames:** files are named `img_{index:06d}{ext}`. Indices are monotonic but *not* contiguous (they survive eviction), and extensions vary within one dataset.
- **No dimensions stored:** `sample_metadata.json` has no width, height or original format, so every header is probed with PIL.

### Ledger

| Observed | Inferred | Decided |
|---|---|---|
| **[SRC]** `dataset_info.json` has no generator family or content category (the cache writer drops them). `path` is not unique: FakeClue backs 9 datasets. | The cache alone cannot classify or balance; the YAML registry is the source of truth. | Join the registry on `name`, the primary key. Parse the YAML directly, because gasbench's dataclass silently drops `generator_variant`. |
| **[RUN]** food / aerial / medical / vehicles / action / plants / currency held 18 real datasets and 0 fakes. Per-category mass splitting pushed P(fake) to 0.45. | A single-label category is a perfect shortcut ("aerial ⇒ real") and also skews the global prior. | Fold single-label categories into `diverse` (computed from the registry). Result: P(fake)=0.5000 and P(fake\|category)=0.5000. |
| **[RUN]** human-faces r1…r9 were 9 of 43 fake-face groups and took 20.9% of all fake-face sampling mass. | One release split into N shards gets N shares. | `shard_groups` collapse the sampler's `group` key; `dataset` stays intact for splits. (The v23 registry cull later removed those shards; current groups: gemini31 train/val, dflip3k reals, swappir, scaledf.) |
| **[SRC]** Base eval is deterministic, so the resample factor is exactly `m = S / min_side`. | `log m = log S − log min_side`: changing S shifts both classes equally, so the resolution shortcut **cannot** be tuned away via S. Only augmentation can break it. `m < 1` aliases the fingerprint (no antialias); `m > 1` adds a spectral cliff. | Store `m_eval` per row and print how much of the m-range carries both labels (`_overlap_report`). This is the case for resample jitter (S5). |
| **[SRC]** The cache writer re-encodes through PIL, which destroys the original JPEG quantisation tables, but the extension and the pixels still carry the original codec. | `ext` remains a real signal; `file_bytes` and qtables do not mean what they would on the source files. | Keep `ext`, `file_format` and `pil_mode` as audit columns; S2 measures what actually reaches the model. |
| **[SRC]** The v23 registry ships an audited pixel-based 3-class taxonomy ("captured pixels retained alongside localised generated content"). | A hand-kept correction list is now redundant and risks contradicting the audit. | `semisynthetic_datasets: []`: registry `media_type` is authoritative. overrides.yaml documents what was adopted (deepfakeface-inpainting, fakeclue-fake-ffpp, deepfakes-qa-15k) and what was overruled (AttGAN, STARGAN, STGAN, imagepulsev2-*, digi2real, pica-100k → synthetic). |
| **[SRC]** Some registry `content_category` values are contradicted by the entry's own `include_paths`. | Category fixes change the folding and balance, so they must be reviewable. | `category_overrides` (fakeclue-*-ffpp → faces, FDDB → faces, 47-plant-species → plants). The manifest keeps both the raw and fixed category. |
| **[SRC]** A missing `--overrides` path used to fall back to `{}` silently; this produced an unusable `manifest_ship.parquet` on 2026-08-06 **[RUN]**. | A missing overrides file trains a materially different model with no error. | A missing *explicit* overrides path is fatal. |

### Data sources beyond the registry

- **`prepare_reals.py`**
  - Why: DFLIP-3K's reals are LAION images filtered at aesthetic ≥ 6.5, a style almost absent from the registry's reals. **[RUN]**
  - Train sets: laion-aes65 (8,000), unsplash-lite (6,000), phone-photos-2024 (500).
  - Probe pool: a disjoint `recoilme/aesthetic_photos_xs` set (600).
- **`generate_swaps.py` / `generate_edits.py`**
  - What: semisynthetic data built with the validator's own i2i models (sdxl-inpainting-0.1, dreamshaper-8) and inswapper.
  - Leakage guard: sources are drawn from `split == "train"` rows only, and each output is pair-grouped with its source.
  - Why: dinov3 zero-shot on face-swap was 24.2%. **[RUN]**
  - Output: `local-inpaint-*`, `local-swap-*`.
- **`generate_frontier.py`**
  - What: self-generated Gemini/OpenAI images.
  - Status: **provided but not wired into any runbook step.** API spend is a user decision.
  - Why it exists: gemini31 showed an **AUC ceiling of 0.75 even with every degradation removed**, and the r22 winner scored 73–81% on it, presumably by training on self-generated Gemini images. **[RUN]**
- **`ingest_external.py`**
  - Enforces the `probe-` / `local-` prefixes and refuses registry names.
  - Dedupes on a 64×64 sha1.
  - A real set needs `--source-url` and `--collected-before`; a date ≥ 2022-01 prints an AI-contamination warning.
  - `--screen` lists the most fake-looking files for manual review.

---

## S2 · Audits: does the data teach shortcuts?

**Files:** `audit_data.py` · `audit_shortcuts.py` · `verify_shortcuts.py` · `scan_gasstation_generators.py`

**Job.** Measure, before and after training, whether anything other than generation artefacts predicts the label, and gate the training launch on hard failures.

### Ledger

| Observed | Inferred | Decided |
|---|---|---|
| **[RUN]** 2026-08-06: fakes 45.3% PNG vs reals 17.3%, with a 6× bytes/pixel tail gap. | A clean PNG (no 8×8 DCT lattice) leans fake. The cache's q75 re-encode homogenises qtables but **not whether a lattice is present**; on the registry 20/91 synthetic datasets declare PNG vs 4/89 real. | `clean_view_recompress: true`: re-encode both classes through the same codec mixture. After rendering, codec survival AUC measured **0.52**. **[RUN]** |
| **[SRC]** Audit numbers can fail silently: a NaN AUC was bucketing as "ok" (2026-08-19 review). | An unvalidated shortcut detector is worse than none. | `verify_shortcuts.py` runs each test with a planted positive and a null control before any audit verdict is trusted. |
| **[RUN]** GAS-Station 2026W35: 2,459/2,785 (88%) `model_name: unknown`, plus seedream-4-5 ×326. | "No Gemini found" is weak evidence when most rows are untagged. | Gate GAS-Station to `test` for zero-shot experiments; the scanner can emit a quarantine `mv` script. |

### `audit_data.py` (pre-training gate, exits 1 on any HARD failure)

| Check | Severity |
|---|---|
| A holdout dataset (or a pair_group mate) missing from disk | HARD |
| Registry datasets not on disk; unknown directories | WARN |
| Datasets under 400 rows (500×0.8); GAS-Station under 4,000 | WARN |
| `file_format` / `pil_mode` class-share gap > 30pp (enable `clean_view_recompress`) | WARN |
| Per-dataset median `m_eval` > 2.0 or < 0.25 | WARN |
| `val_xgen` P(fake) more than 0.15 from 0.5 | WARN |

### `audit_shortcuts.py` modes, cheapest first

| Mode | Needs | Question answered | Thresholds (warn / hard) |
|---|---|---|---|
| `metadata` | manifest | **M1–M6.** Can a blocked-fold GBM predict the label from metadata? Per-feature AUC; naive − blocked gap (memorisation); does val inherit the train shortcut; category MI; did category overrides help? | AUC 0.70 / 0.85 |
| `tensors` | cache + gasbench | **T1–T4.** GBM on ~20 stats of the *rendered* deploy view; do nuisances (is_png, non-RGB, upsampled) survive; AUC delta per laundering knob. Capped at 6,000 rows. | AUC 0.65 / 0.80 |
| `slices` | GPU + checkpoint | **S1–S8.** p_fake vs PNG on reals; Spearman(margin, log m_eval); worst category gap; per-dataset accuracy ledger; semisynthetic deficit; pil_mode; can metadata predict errors? | per slice |
| `pairs` | GPU | **P1–P3.** Fake accuracy inside vs outside cliques; clique margin gap vs PNG/resolution deltas; clique real FPR > 2× global. | FPR 2× |
| `interventions` | GPU | 10 label-blind families (JPEG norm/sweep, WebP, double JPEG aligned/shifted, grey, resize 192/384/768, crop, blur, high-pass, noise); **I11** per-branch attribution. Capped at 4,000 rows. | real flip 0.10 / 0.25 |

- Results go to the `shortcut_history.jsonl` ledger.
- The first row per key becomes the baseline; later "ok" rows escalate to "warn" if they regress (Δ 0.05).
- Exit code is 0 unless `--strict`: measurement first, gating second.

---

## S3 · Splits: making every later number mean something

**Files:** `build_splits.py` · `overrides.yaml` (dev) · `overrides-ship.yaml` · `overrides-experiment.yaml`

**Job.** Assign each row to `train`, `val_id`, `val_xgen`, `val_stress` or `test`, such that no source straddles train and a held-out split.

### Ledger

| Observed | Inferred | Decided |
|---|---|---|
| **[SRC]** Some fakes are built *from* real datasets in the same registry: celeb-a-hq→AttGAN/STARGAN/STGAN, fairface→FairFaceGen(-flux/-sd35), `-real`/`-fake` pairs, dflip3k, swappir, scaledf, local-* edits. | A real and its edited version on opposite sides of a split turns validation into a memorisation test. | **Split units** = union-find over shared HF `path` plus explicit `pair_groups`. Naming one dataset in a holdout pulls its whole clique. `distribution_groups` (StyleGAN trained on FFHQ) are unioned only with `--strict-provenance`, because doing so would drag 4 of 9 StyleGAN datasets out of train for one small canary. |
| **[SRC]** Video-derived datasets (inst-it-videos, vtuav, bdd100k, FDDB, fakeclue-*-ffpp) have near-duplicate consecutive frames. | A row-level split (v4's `make_manifest.py`) puts frame t in train and t+1 in val; val reads ~1.0 while measuring nothing. | Carve `val_id` by **source block** within each dataset: `source_file` if it has ≥ 8 values, otherwise contiguous index blocks. |
| **[RUN]** With fixed 256-row blocks, the 500-image cache cap gave 2 blocks per dataset: val_id covered 22 of 136 datasets and P(fake) drifted 0.57 → 0.64. | An in-distribution split missing 5/6 of the sources is not in-distribution. | Block size derived from dataset size (~16 blocks); take ~`val_id_frac` from **every** dataset. |
| **[RUN]** 2026-08-07 dinov3 at step 1000: **val_id 0.8561** vs **gasbench full 0.8644** (0.008 apart); val_xgen read 0.9172 and peaked early while val_id kept climbing. | The public score is mostly in-distribution; selecting on val_xgen freezes `best.pt` too early. | `selection: {val_id: 1.0}` today; `{val_id: 0.7, val_xgen: 0.3}` for ship runs with a holdout panel. |
| **[RUN]** face-swap held out: dinov3 24.2%, clip 14.8% zero-shot; the reigning winner 98.9%. | Swap detection does not generalise from other manipulation data. Holding out public data only donates points. | **Ship split** (`overrides-ship.yaml`): public data goes to train; only datasets already ≥ 96% zero-shot stay held out. |
| **[RUN]** Holding out seven small real corpora led the benchmark to score ffhq-256 at 0.207 and or-real-id at 0.310. | Partly a resolution shortcut, partly self-inflicted: no small upsampled real faces left to learn from (train had 21 real face sources vs 40 fake). | `val_stress` keeps just 2 canaries (ffhq-256, casia_web_face): **reported, never selected on.** |
| **[SRC]** A holdout real dataset in a category with no held-out fake only skews the prior. MCC and Brier both depend on the prior. | Holdout panels must be category-matched. | val_xgen fake side: diverse ×3, faces ×2, documents ×1, with the real side matched exactly. `min_datasets_to_hold_out: 3`. |
| **[RUN]** v21 released holdouts scored well zero-shot. | Training on them buys little; they are better used as honest probes. | `train_gates` park them (and dflip3k, the zero-shot judge) in `val_stress`. `deepfakes-qa-15k` stays in train: it is swap data, the weak class. |

### Split regimes

| File | Purpose | Holdouts | Selection |
|---|---|---|---|
| `overrides.yaml` | Development: honest generalisation measurement | val_xgen 6 fake + matched reals; test 6 + 6; val_stress 2; gates park dflip3k + v21 in val_stress | val_xgen, cross-fitted |
| `overrides-ship.yaml` | The model that gets submitted | Easy, generator-distinct panel only (≥ 96% zero-shot); test and val_stress empty | val_id 0.7 / val_xgen 0.3 |
| **`overrides-experiment.yaml`** | Train-everything 80/20 (current arms E–G) | None; both gates flipped to train | val_id 1.0; **needs `--val-id-frac 0.20`** |

> **Landmine [RUN]:** `build_splits.py --val-id-frac` defaults to **0.06**, but the experiment configs expect **0.20**. Forgetting the flag silently gives a much thinner selection split.

Validation (`validate()`) blocks writing when:

- a split unit straddles two split families,
- train or val_id is empty or single-label, or
- a train category is single-label.

It warns on an empty val_xgen or test, and on a holdout P(fake) outside [0.35, 0.65].

---

## S4 · Sampler: removing label information from everything but pixels

**Files:** `data.balanced_weights` · `data._kind_shares` · `data.BalancedSampler` · `data.clique_pairs` · `diagnose_replay.py`

```text
Per-row sampling weight · data.py   [SRC]

w_i = label_w(l) · q_c · kshare(kind | l, c) · ( n_g^0.5 · boost_g / Σ_g' n_g'^0.5 · boost_g' ) · 1/n_g

q_c      = sqrt(real_datasets_c · fake_datasets_c) / Σ_c'      (geometric mean)
label_w  = kind_balance mass → real 0.50 / fake 0.50
kshare   = kind_balance share of the kinds present INSIDE the (label, category) cell
boost_g  = gasstation_boost (2.5) if "gasstation" in dataset name, else 1
cap      = water-fill: w_i = min(λ·w_i, max_replay / N), λ by bisection (max_replay 25)

Normalisation: Σ over a group's rows of 1/n_g = 1; Σ over a cell's groups of the share = 1
⇒ each (label, category) cell carries exactly label_w · q_c; P(fake) = 0.5 exactly; P(fake | category) = 0.5.
```

### Ledger

| Observed | Inferred | Decided |
|---|---|---|
| **[RUN]** v2 grouped by `generator_family`, which is the string "real" for all 89 real datasets. v2's documented collapse: or-real-id 0.084, fairface 0.645, birds 0.994. | All reals formed one group: the largest real corpus in each category absorbed that category's real mass while fakes were flattened. | Group key = dataset (after shard collapse), with a √-share inside each (label, category, kind) cell. |
| **[SRC]** v2 split each category's mass across the labels *present*: single-label categories donated all their mass to one side, giving P(fake) = 0.45. | The prior identity only holds if every category carries both labels. | Fold single-label categories (S1); categories that are single-label *in this split* are excluded, with a printed warning. |
| **[INF]** An arithmetic mean would let `animals` (9 real / 2 fake) replay its 2 fake datasets hard enough to memorise them. | Categories thin on one side should be de-weighted. | `q_c` uses the **geometric** mean of the two sides' dataset counts. |
| **[RUN]** run_2: fake datasets *in training* scored 0.631 inside mixed cliques vs 0.941 outside; STARGAN 0.138, fakeclue-fake-ffpp 0.000. | The two halves of a clique (near-duplicate real/fake) almost never share a batch, so CE settles on whichever carries more gradient. | `clique_pair_frac: 0.25`: a quarter of the stream is matched real/fake pairs, kept adjacent so they land in one batch, shuffled in blocks of 2. |
| **[RUN]** 2026-09-08: `kind_balance` requested 0.25 semisynthetic, realised **0.0952**; the log printed "[balanced]" through 15,000 steps. | Kind shares apply *inside* (label, category) cells. Semis exist only in `faces`, so their ceiling is `label_w · q_faces ≈ 0.19`. No sampler dial passes it. | The log now prints target / realised / ceiling and `[TARGET NOT MET]` / `UNREACHABLE`; `verify.py` asserts `P(semi) = ceiling · kb_semi/(kb_semi+kb_syn)`. The objective fix lives in the loss: `kind_class_weights` (S7). |
| **[RUN]** v23 audit: the semi pool shrank to faces-only; realised semi mass self-moderated from 43.4% to ~19.4% of the fake half (~9.7% of total), peak replay ~3.4×. | Still a ~3× lift over the natural 6.5%, safe on replay. | `kind_balance 50/25/25` kept deliberately. |
| **[SRC]** Eval uses `GASSTATION_WEIGHT_MULTIPLIER = 5.0`. **[RUN]** The cache holds 1/4 of the original 8 weeks. | Share scales as √n·boost and per-image replay as boost/√n. Quartering n doubles replay at a fixed boost. | `gasstation_boost: 2.5` restores the validated 8-week per-image replay. Restore 5.0 if all 8 weeks return. |
| **[SRC]** v2 pooled groups under `min_group_rows`: a 49-row cell merged while a 51-row cell was replayed ~60×. | The thing to bound is draws per image, and the bound should be continuous. | Water-fill with `max_replay: 25`. `diagnose_replay.py` names datasets sitting at the cap (the train log does not). Measured deviation when bound: ≤ 0.0008 on P(fake) and P(fake\|category). |
| **[SRC]** `WeightedRandomSampler` is single-process; under accelerate every rank would draw the same indices. | Multi-GPU runs would silently repeat data. | `BalancedSampler` draws one global `(seed, epoch)` stream and slices `[rank::world_size]`. |

> **Re-check after a manifest rebuild [INF]:**
> - The comments assume 5 semisynthetic datasets; gasbench v0.9.5 lists **13** (12 faces + fakeclue-fake-ffpp, which overrides.yaml moves to faces).
> - The ceiling geometry is unchanged because they are all faces.
> - Per-image semi replay drops, and the derived `type_prior` changes. See Part V.

---

## S5 · Views & degradation: rendering what gets scored

**Files:** `data.ManifestDataset.__getitem__` · `data.source_prechain` · `data.source_resample_jitter` · `data.source_effects` · `gasbench_bridge.py` · `verify_views.py`

**Job.** Turn one image into two uint8 views:

- **view_a:** drawn from a three-arm mixture.
- **view_b:** the robustness chain.

Source-level degradation is applied once, before the split, so both views share it.

```text
source image
 ├─ prechain codec    none .30 │ jpeg │ double-JPEG (1–7px shift) │ webp remainder        (label-blind)
 ├─ resample jitter   p≈.4, factor U(.60,1.0), INTER_AREA↓ / INTER_LINEAR↑
 ├─ source effects    per-epoch p; motion/defocus blur, film grain, halftone, oversharpen,
 │                    sensor noise, banding, colour cast, vignette, chroma shift  (always 4 RNG draws)
 ├─ h/v flips         0.5 each (shared by both views)
 ├─ view_a ← deploy  (.70)   gasbench level 0, crop 0        = the scored transform
 │         ← ladder  (.16)   gasbench levels L0..L3, crop .5 (0 for semisynthetic)
 │         ← robust  (.14)   randomised JPEG→WebP→JPEG chain (scale U(.35,1), jpeg 40–75, webp 60–90)
 └─ view_b ← robust  (or deploy in a clean epoch)
eval loaders: deploy = exact scored transform; robust = exact deployed constants (rng=None)
```

### Ledger

| Observed | Inferred | Decided |
|---|---|---|
| **[SRC]** v2 resized with torchvision and v4 with `antialias=True`; gasbench uses non-antialiased `cv2.INTER_LINEAR`. v2/v4 also normalised outside `forward`. | Downscaling a 1024px generator output aliases its high-frequency fingerprint in a deterministic, learnable way; antialiasing destroys the signal the model is scored on. | Call gasbench's own `apply_random_augmentations(level=0, crop_prob=0)` through the bridge, never a vendored copy. Every view is uint8; normalisation lives only inside `forward`. |
| **[RUN]** run_3 at arms 0.55/0.25/0.20: face-swap 0.138 and fakeclue-fake-ffpp 0.000, both **in training**. | Face manipulations and old GANs are identified by HF artefacts at 112–256px; heavy JPEG/blur/down-up erases them. view_b is always degraded, so at deploy 0.70 the CE mass is already ~35% clean / 65% degraded. | Floor `deploy ≥ 0.65` in every epoch; JPEG quality floor 40; schedules end on a deploy-faithful finale. |
| **[RUN]** Jitter on every sample at U(0.35,1.0): val_stress 0.19 → 0.84 (the largest single effect of any change) but ffpp held at 0.000. Jitter off: val_stress halved to 0.37 within 500 steps, while ffpp's loss finally fell. | Both extremes cost score. A mild dose on a fraction of samples protects low-res reals without erasing local manipulation artefacts. | `resample_jitter: p 0.4, factor [0.60, 1.0]`. Kernel diversity stays **OFF** (it may blur the learnable INTER_LINEAR signature). |
| **[SRC]** Eval crops are mask-aware and keep the edit in frame; training has no masks. | A random crop that excludes the edit is a genuinely real view labelled fake (2–5% of semi CE mass was this noise). | `ladder_crop_guard`: never crop semisynthetic rows in the ladder arm. |
| **[SRC]** A single-pass JPEG and the aligned robustness chain never produce misaligned 8×8 lattices; re-shared content does. | The recompress-after-crop signature is absent from training. | Double-JPEG prechain arm: JPEG(q1) → 1–7px crop → JPEG(q2), mass carved from the single-JPEG arm. |
| **[SRC]** Real laundering paths are often JPEG-only; 100% cross-codec exposure was the previous behaviour. | view_b over-represented WebP. | `robust_skip_webp_p: 0.15` for rng-driven views; eval (rng=None) always keeps the WebP hop. |
| **[RUN]** 600/600 ladder L1 renders were exact flips; the source already flips with p 0.5. | Two Bernoulli(0.5) flips compose to Bernoulli(0.5): L1 renders L0's distribution, so 25% of the ladder does nothing. | `ladder_level_probs [.25, 0, .25, .50]`: blur/noise goes 1.29% → 2.71% of CE mass at zero change in severity. |
| **[SRC]** The ladder's DeeperForensics distortions only reach levels 1–2 (blur kernel 7/9, σ 0.001–0.002) on ~2% of CE mass; motion blur, grain and halftone are not in its table. | Real capture and print artefacts are absent from training. A degraded real looks unfamiliar and gets called fake. | `source_effects` at the source stage (reaches 100% of CE mass and both views), with measured PSNR and cost per family. Arm F used 4 families; arm G uses 10. |
| **[RUN]** Perf: an uncapped motion kernel cost 111 ms/img on a 3000px source; full-res grain 525 ms/img; a pure halftone render measured 9.8 dB. | The dataloader budget is ~518 ms/img per worker (bs 64, 16 workers, 2.07 s/step). | Motion/defocus kernels capped at 31px; grain and noise generated at ¼ scale; halftone alpha-blended by strength. |
| **[RUN]** The double-JPEG arm's "24.9 dB" was ~99% its crop shift; 41.7 dB without it. | PSNR is brutal to translations and global tone shifts. | Do not rank degradation families by dB alone (stated in the arm G header). |
| **[SRC]** With `persistent_workers=True`, workers fork the dataset once, so `set_epoch` never reaches them (verified against torch 2.7). | Epoch 2 would render byte-identical augmentations and the schedule would be a no-op. | Train loader `persistent_workers=False`. |
| **[SRC]** Every decision in `__getitem__` consumes a shared per-row RNG stream. | Toggling an enable flag per epoch shifts every downstream draw (flips, arm choice, robust params). | Schedule entries may override probabilities and ranges only, never `enabled` flags. `source_effects` always consumes exactly 4 draws. view_b draws are last, so `view_b` is schedulable. |
| **[SRC]** A uniform effects list divides incumbent mass when families are added (4 → 10 cuts motion_blur 8.75% → 3.5% of rows; measured 2.9 dB milder). | Width and severity were coupled. | `effects` accepts a `{name: weight}` mapping (one draw, cumulative walk); `effects_p` re-solved per epoch to hold mass-weighted MSE. |

### Source-effect families (measured PSNR vs the 384 render, cost per image) **[RUN]**

| Family | PSNR | Cost | Axis it adds |
|---|---:|---:|---|
| motion_blur | 26.8 dB | 97 ms | directional streak |
| defocus_blur | 29.3 dB | 44 ms | out-of-focus, past the ladder's kernel 7/9 |
| film_grain | 32.3 dB | 152 ms | luma-correlated grain + slight desaturation |
| halftone | 15.6 dB | 111 ms | print stipple, alpha-blended |
| oversharpen | 34.5 dB | 18 ms | unsharp halos: consumer ISPs sharpen, generators emit smooth edges |
| sensor_noise | 35.1 dB | 158 ms | signal-dependent per-channel shot noise |
| banding | 41.7 dB | 38 ms | value-space quantisation (the only tone-curve distortion) |
| color_cast | 28.1 dB | 30 ms | white-balance / illuminant error |
| vignette | 25.0 dB | 38 ms | spatial non-uniformity |
| chroma_shift | 24.0 dB | 18 ms | per-channel misregistration |

---

## S6 · Model: heterogeneous ensemble, factorised 3-class output

**Files:** `branches.py` · `model.py` · `templates/inference_model.py`

| Branch | Backbone | Native px | Head | Adaptation | Fusion w |
|---|---|---:|---|---|---:|
| `dinov3` | DINOv3 ViT-L/16 (`facebook/dinov3-vitl16-pretrain-lvd1689m`), self-supervised | 224 | mac_lite: CLS‖patch-mean → LN → MLP(2d→512→3) | LoRA q/k/v/o | 1/3 |
| `clip` | CLIP ViT-L/14 (`openai/clip-vit-large-patch14`), contrastive | 224 | dropout 0.3 + Linear(d,3) | LoRA q/k/v/out | 1/3 |
| `convnext` | ConvNeXt-L CLIP soup (`convnext_large_mlp.clip_laion2b_soup_ft_in12k_in1k_384`) | 384 | bounded cosine | LoRA fc1/fc2 + norms | 1/6 |
| `eva` | EVA-02-L/14 (`eva02_large_patch14_448.mim_m38m_ft_in22k_in1k`), MIM | 448 | bounded cosine | LoRA q/k/v/proj | 1/6 |
| `dct` | luma → orthonormal 2D-DCT → log1p\|·\| → per-sample standardise → ResNet-18 (scratch, in_chans 1) | 384 | bounded cosine | full training (~11M) | 1/6 |
| `dinov3_336` / `dinov3_384` | resolution variants; **swap** semantics (ship instead of dinov3, never alongside) | 336 / 384 | mac_lite | LoRA | 1/3 slot |

- **Fusion weights:** `fusion_weights()` renormalises over the shipped subset; the 5-way ship is `2/7, 2/7, 1/7, 1/7, 1/7`.
- **Resolution:** the declared input is 384. Each branch resamples internally with antialiased bilinear. `dct` and `dinov3_384` run no resample at all.
- **LoRA settings:** r 32, α 64, dropout 0.05. LayerNorm, layer-scale and gamma weights stay trainable.

### The exported forward pass

```text
x uint8 [B,3,384,384]
  for each branch i:   z_i ∈ ℝ³  (own resize + own normalisation inside)
                       d_i  = logsumexp(z_i1, z_i2) − z_i0        # real-vs-not-real margin
                       d2_i = z_i2 − z_i1                          # semi-vs-synthetic margin
  fuse:                d̄  = Σ w_i d_i ;  spread s = sqrt(Σ w_i (d_i − d̄)²)
                       d̄2 = Σ w_i d2_i
  calibrate:           T_eff = T0 · (1 + k · max(0, s − pivot))
                       p = clamp(σ(d̄ / T_eff), p_min, p_max)
                       q = clamp(σ(d̄2 / T2 + b2), q_min, q_max)
  output log p:        [ log(1−p) , log(p(1−q)) , log(p·q) ]      # real, synthetic, semisynthetic
```

**[SRC]** Invariants, proven in `verify.py`:

- The binary collapse `1 − P[0] = p` for **any** q, so the type posterior can never move the binary score.
- Margins are fused per branch and then combined; the code never takes "logsumexp of fused logits".
- Raw logits are never divided by T (for K=3, `softmax(z/T)[0] ≠ 1 − σ(d/T)`).

### Ledger

| Observed | Inferred | Decided |
|---|---|---|
| **[RUN]** BMF production paper (arXiv:2607.13234) Table 23: every single backbone has domains < 0.85 AUC (DINOv3's worst: 0.211 on FF++ swaps); the fused model has none. | A generator shortcut learned by one representation rarely survives averaging with other families. | Branches that disagree in architecture, pretraining objective and native resolution; trained **independently**; fixed logit averaging. A fusion ships only if it beats the best single branch on the same run. |
| **[INF]** Independent runs fit one card each, are individually benchmarkable, and a bad branch can be dropped rather than debugged. | Joint training would couple failures. | One `train.py --branch` run per branch; each gets its own solo export and gasbench number. |
| **[RUN]** dflip3k reals called real only 15% of the time: the semantic branches call the whole stylised-real domain fake. | A branch nearly blind to content and style is the structural counterweight. | `dct` branch (2026-08-10), native size = declared 384 so no low-pass resample runs before the spectrum. |
| **[RUN]** convnext head+norms only: 0.1M params; at step 250 loss 1.25 / acc 0.61 vs dinov3 0.55 / 0.87. | A capacity ceiling, not slow warmup. ConvNeXt MLPs are `nn.Linear`, so they are valid LoRA sites. | LoRA on `fc1`/`fc2`. `inject_lora` counts matches *per target* and raises if any target matches nothing. |
| **[SRC]** Brier punishes a single confidently wrong branch heavily. | Capping per-branch confidence bounds its damage to the fused average. | Bounded cosine head: `15·tanh(30·cos/15) + [0, +0.5, −2.5]`. |
| **[RUN]** 2-logit heads take a structural multiclass haircut of ~7–8% (r22: multiclass 0.8589 vs binary 0.9153). | Class 2 needs its own logit, but the binary path must stay as validated. | 3-logit heads, semi bias initialised at −3: an untrained head says p(semi\|fake) ≈ 0.05 instead of an indecisive 0.5. |
| **[RUN]** DINOv3 register activations reach 154k. | fp16 overflows. | `amp: bf16`, never fp16. |
| **[SRC]** With frozen embeddings, reentrant gradient checkpointing sees no grad-requiring inputs and silently drops LoRA gradients. | The branch would train head-only while looking healthy. | HF path: `use_reentrant=False` + `enable_input_require_grads`. timm branches are checked empirically (step-0 dead-grad check in `train.py`). |
| **[SRC]** timm's default zero-inits each block's last BN gamma, so inner conv gradients are exactly zero at step 0. | The dead-grad check would abort a healthy dct run. | `zero_init_last=False` for the scratch dct net. |
| **[SRC]** No tensor shape depends on `native_size`. | A 224-trained dinov3 checkpoint would strict-load silently into a 384 model. | Variants are separate registry entries with separate run dirs; `best.pt` stores `spec`, and `export.py` cross-checks it. |
| **[RUN]** dinov3_384 solo: local 99.72% acc and sn34 0.992, but lower on-chain. | The extra frequency band was a pipeline-instance fingerprint. | Resolution variants stay experimental; on-chain results decide. |

---

## S7 · Loss: a few terms, each tied to the score

**Files:** `train.py` main loop · config `loss:`

```text
Per step · train.py   [SRC]

L =  CE₂(collapse(z_a), y; kw, ls=.02) + CE₂(collapse(z_b), y; kw, ls=.02)     # binary on logsumexp collapse
  +  0.50 · symKL(z_a, z_b)                                                    # degradation invariance
  +  0.30 · ½[ CE(z_a[fake][:,1:], y3−1; tw) + CE(z_b[fake][:,1:], y3−1; tw) ] # syn-vs-semi, fake rows only
  +  0.50 · ramp(progress; 0.2 → 0.6) · ½[ Brier(z_a; kw) + Brier(z_b; kw) ]   # p_notreal = 1 − softmax[0]

collapse(z) = [z0, logsumexp(z1, z2)]
kw = kind_weights(y3)          real / synthetic / semi each get equal total mass, mean 1
tw = 2-class balance over the fake half
```

### Ledger

| Observed | Inferred | Decided |
|---|---|---|
| **[SRC]** v5 used spectral teacher + MIL + supcon + deep supervision, none ablated. BMF production branches carry none of it. | Unablated terms make per-branch results unattributable. | Four terms, each directly objective-aligned: view_a CE (the scored transform), view_b CE (the aug pass), KL tying them, and Brier (the dominant score term). |
| **[SRC]** 3-class label smoothing leaks `ls/3` onto the semi logit and drags q toward 0.5. | Smoothing belongs on the binary axis only. | Factorise: `CE₃ = CE_bin(collapse) + CE_cond(fake rows)`. The binary term is byte-identical to the 2-logit pipeline. `type_weight` is the single risk dial (0 = binary-only training with a 3-wide head). `type_label_smoothing: 0`. |
| **[RUN]** The sampler realises kinds at ~0.50 / 0.40 / 0.095, while gasbench and `class_balance_weights` score a flat prior. | Training fits one prior and is graded on another. Reweighting rows is the one lever the category ceiling doesn't bound. | `kind_class_weights: true`, normalised to mean 1 so the LR schedule is unchanged; `false` reproduces earlier runs byte for byte. |
| **[SRC]** The type term sees only fake rows, where semi is ~19%. | Corpus-wide kind weights are the wrong weighting there. | Separate 2-class balance `tw` over the fake half. |
| **[INF]** Early in training the probabilities are meaningless; a Brier gradient then mostly flattens logits. | Calibration pressure pays once discrimination exists. | Brier ramps 0 → 1 between 20% and 60% of progress. |
| **[SRC]** With a deploy view_b (clean epochs), the pair is byte-identical. | symKL degrades to dropout consistency (R-Drop), not zero. | Its raw magnitude is logged every 50 steps; do not assume it vanishes. |

---

## S8 · Training loop & checkpoint selection

**Files:** `train.py` · `calibrate.py`

| Knob | Value | Why |
|---|---|---|
| Optimiser | AdamW, weight decay 0.02, grad clip 1.0 | Standard for LoRA + norms. |
| Learning rate | `1e-4 · √(bs/32)` → 64: 1.4e-4 · 96: 1.7e-4 · dct bs 256: 3e-4 | √-scaled from the validated 1e-4 @ 32. |
| Schedule | 5% warmup + cosine to 0 | Everything is a ratio of `total`, so `--max-steps` defines a complete run. |
| EMA | 0.9995, warmup `min(d, (1+t)/(10+t))` | Evals and `best.pt` use the EMA weights. |
| Precision | bf16 via accelerate | See S6. |
| Per-branch overrides | `batch_size`, `grad_accum_steps`, `lr`, `epochs` only | Branches span 11M params to ViT-L@448; everything else stays global so runs are comparable. |
| Eval cadence | `eval_every: 500` (or `epoch`), `eval_max_rows: 8000` | A blake2b-ranked stratified subsample, identical across evals, restarts and runs. Paired comparisons can read ~0.002 differences. Every (dataset, label) cell survives (quota floor 1). |
| Step-0 grad check | abort on dead gradients | Checkpointing can silently drop LoRA grads. LoRA A is zero at step 0 by construction (B is zero-init) and is re-checked after the first update. |
| Config validation | unknown keys fail | `validate_config` rejects keys nothing reads; `build_view` re-validates every schedule profile at startup. |

### Ledger

| Observed | Inferred | Decided |
|---|---|---|
| **[RUN]** run_2 step 250: two halves of val_xgen picked T=4.9989 and T=0.6403; the pooled score was exactly 0.0000. | `brier_norm` clips at 0, so sn34 is flat while the model is weak, and argmax over a flat objective returns noise. | Choose T by **minimising Brier** (smooth, proper) on a 240-point log grid over [0.08, 8]. The floor moved down from 0.25 after a run hit it. Fall back to T=1 if the best Brier is ≥ 0.245. |
| **[RUN]** run_1: fitting T on val_id sharpened it 0.736 → 0.587 as the model overfit; 95% of the val_xgen decline from step 2500 to 4000 was calibration, not discrimination. | Fitting T on the scored split flatters every checkpoint, and unevenly. | `crossfit_temperature` on val_xgen: fold by *dataset* (not generator family, which collapses all reals), fit on one fold, score the other. Falls back to pooled when < 4 groups or fold temperatures differ > 2×. Platt is cross-fitted on the same folds for reporting. |
| **[SRC]** A validation split's raw prior rarely matches the benchmark's ~0.5 (val_id ~0.64, val_xgen ~0.33 under the dev lists). | Calibrating on the raw prior targets the wrong distribution. | `class_balance_weights` reweights to 50/50 before every fit and score. |
| **[SRC]** `compose_3class` makes the binary collapse invariant to q. | Binary sn34 is blind to synthetic-vs-semisynthetic typing, the axis this pipeline loses on. Two checkpoints with equal binary scores and opposite typing look identical. | `multiclass_selection: true`: Gorodkin + multiclass Brier with deployment-share weights (`y3_class_weights`), blended 0.8 deploy / 0.2 robust, mirroring export exactly. Falls back to binary when a split lacks all three kinds. |
| **[SRC]** The multiclass metric needs a deployment P(semi\|fake), not the sampler's. | Selection and export must rank checkpoints the same way. | `type_prior: null` → derived from the manifest's per-dataset semi share, exactly as `export.py` does. |

**Selection formula (current configs).** `selection = mc_sn34_val_id = 0.8 · mc(deploy) + 0.2 · mc(robust)`, where:

- `p = σ(d / T)`, with T fitted on val_id deploy+robust;
- `q = clamp(σ(d2 / T2 + b2), q_min, q_max)`, with (T2, b2, q_max) fitted on val_id;
- `mc` = Gorodkin/multiclass-Brier sn34 under deployment class weights.

`best.pt` is saved (EMA weights, config, spec, effective training dict) whenever selection improves.

> **Worth knowing [INF]:**
> - On the val_id path, T and the type posterior are fitted on the same rows they score.
> - Ranking stays mostly consistent, but the absolute selection number is optimistic.
> - The val_xgen path cross-fits precisely to avoid this, and it currently has weight 0.

**Per-eval log columns:** `split · n · acc · sn34 (binary, class-balanced, own T) · type (raw syn-vs-semi accuracy on fakes) · mc_sn34`. `val_stress` is single-label, so it reports accuracy only (the complement of the false-positive rate).

---

## S9 · Calibration & export

**Files:** `export.py` · `allowlist.py` · `templates/inference_model.py` · `measure_margins.py`

### Steps

1. **Environment guard:** refuse to run unless `transformers == 5.2.0`.
2. **Load branches:** for each branch, load `best.pt` and check the `branch` name and `native_size` provenance. Remap `backbone.model.*` keys against the rebuilt model, strict-load, merge LoRA, and assert the merged vs unmerged probe drift is ≤ 1e-3.
3. **Assemble:** build the `Ensemble` with renormalised weights, cast module-wide to bf16 (exactly as the harness does).
4. **Collect margins:** per-branch binary and type margins on `--calib-splits` (default `val_id,val_xgen,val_stress`), capped at 8,000 rows, under both the deploy and robust views.
5. **Fit the calibration buffers** (table below).
6. **Write artifacts:**
   - `model.safetensors` (calibration buffers kept in fp32) and `config.json` (DINOv3 HF config);
   - `model.py` rendered with literal constants and scanned by the allowlist;
   - `model_config.yaml` (`dtype`, `preprocessing.resize: [384,384]`, `num_classes: 3`).
7. **Round-trip:** re-import the rendered `model.py`, check state-dict shapes, and require argmax to match exactly on 16 random probes, probability drift ≤ 5e-3, and `exp(out)` summing to 1.

### Calibration buffers

| Buffer | How it is set | Protects |
|---|---|---|
| `temperature` T0 | Brier grid on 0.8 deploy + 0.2 robust, class-balanced; or `--override-temperature` | Brier (binary MCC unchanged) |
| `spread_pivot` | max over both views of local p99 spread | Keeps the hinge off locally and on public data |
| `temp_slope` k | **A stated prior, not fitted**: `--spread-slope 1.0` (`--no-spread` → 0) | Hidden-pool disagreement (spread ~8–12) → T_eff ~4–7 → confidence ~0.8–0.9 instead of ~0.99 |
| `p_min` / `p_max` | Default numerical clamp (1e-6); tuned by `tune_operating_point.py`; must satisfy `0 < p_min < 0.5 < p_max < 1` | Confident inversions on hidden pools |
| `type_temperature` T2, `type_bias` b2 | Prior-corrected conditional-Brier grid on fake rows (T2 ∈ geomspace(0.25, 8, 60), b2 ∈ [−6, 1]) | Multiclass score |
| `q_min` / `q_max` | q_min 1e-3; q_max grid 0.2–0.95 on composed multiclass sn34. **Kept only if it beats pinned q**; otherwise q_min = q_max (binary-equivalent). | Multiclass score, binary untouched |

### Ledger

| Observed | Inferred | Decided |
|---|---|---|
| **[SRC]** The bounded-cosine head applies tanh; gasbench casts the whole module to `dtype`. | A temperature baked into weights cannot pass through tanh. A T fitted under autocast is a different function from the deployed cast. | Store T as a **buffer** in the safetensors; fit it under export numerics (module-wide bf16). |
| **[RUN]** 2026-08-12: local spread p95 2.26 at ~99% local accuracy. Scalar T refit against the dashboard mix: +0.002; pool-conditional calibration estimated ~+0.05. | Locally no errors correlate with disagreement, so a local fit always returns k=0. Hidden pools are where branches disagree. | Disagreement-aware hinge: pivot above every local sample, k as a reasoned prior. `measure_margins.py` later measures AUC(spread, error); a null result there argues for `--no-spread`. |
| **[RUN]** r22 lost 83.52 vs 86.22 despite better accuracy and better aug robustness (97.8% vs the winner's 77.7%). Gemini-like holdouts scored 7.69% / 23.08%; ~626 of ~6–7k holdout images (≈10%) were confidently inverted. | The hinge cannot catch *unanimous* inversions; each costs Brier^1.8 at holdout weight. | `--p-min/--p-max` insurance, restricted to a range where binary MCC is provably unchanged. |
| **[RUN]** 2026-09-10: `--override-temperature` skipped the whole calibration pass, leaving q pinned: multiclass sn34 **0.7593** pinned vs **0.9618** fitted. | A pinned q is invisible in binary metrics and in the round-trip check, yet it zeroes class 2 (and the exam's semisynthetic accuracy). | Commit `62f2879`: the T0 fit and type fit are decoupled; the type posterior always runs unless `--binary-equivalent`/`--no-refit-temperature`; a loud WARNING prints if q ships pinned. |
| **[SRC]** The sampler trains the type margin at P(semi\|fake) ~0.5; deployment is ~`type_prior`. | An uncorrected fit would be confidently wrong on the large synthetic class. | Both fit stages reweight to the deployment prior. |
| **[RUN]** Random-noise probes: logit drift 6.8e-3 / prob drift 1.4e-3 at export, vs ~1e-4 on real images. | Noise sits far above the hinge pivot, which amplifies bf16 differences. | The round-trip gate checks what the score consumes: argmax exact, probability drift ≤ 5e-3. |
| **[SRC]** In no-degradation arms, `val_stress` holds the zero-shot judgement set (gemini31). | Fitting T on the judged pool leaks it into the benchmarked export. | `--calib-splits val_id,val_xgen` for those arms; the log prints which splits the fit saw. |

---

## S10 · Gate & operating point

**Files:** `gasbench run` · `eval_probes.py` + `probes.yaml` · `brier_headroom.py` · `measure_margins.py` · `tune_operating_point.py`

### Ledger

| Observed | Inferred | Decided |
|---|---|---|
| **[SRC]** The exam needs ≥ 80% strict accuracy; crowning needs +0.01 on the full benchmark. | Local `--small` is a *correctness* gate (loadable, fast enough, class 2 reachable), not a ranking. | Run `gasbench run --image-model ./submission/ --small` before every push. Solo-export each branch too, so any regression can be traced to one branch. |
| **[RUN]** 2026-08-12: dashboard Brier sat exactly on `a(1−a)`, the constant-confidence floor (a = (1+MCC)/2). A comparable single-DINOv3 model beat its own floor by 26%. | The gap above the floor bounds what monotone recalibration can recover; a gap below it is accuracy-shaped. | `brier_headroom.py`: Stage 1 uses dashboard numbers only; Stage 2 sweeps T on 0.7 val_id + 0.3 OOD margins, anchored to the dashboard Brier. It either prints an `--override-temperature` command or says "needs training data". |
| **[SRC]** Whether calibration can help depends on whether confidence ranks correctness. | That is measurable. | `measure_margins.py`: AUC(\|d\|, correct) ≥ 0.70 → ship the mixture T; 0.60–0.70 weak; < 0.60 → "calibration is a dead end; every point must come from training data". |
| **[RUN]** Hidden holdouts are ordinary public datasets with distinctive styles (the v21 release proved it). | A never-seen style is called fake wholesale. | `eval_probes.py` + `probes.yaml`: a taxonomy of 14 style types (only `stylized-real` populated so far), `probe_history.jsonl` ledger, weak < 0.80 / strong ≥ 0.92. Hard error if a probe leaks into any `extra_datasets`. |
| **[SRC]** Production score shares are not observable. **[RUN]** r22 realised ~10% inversions. | Operating-point tuning must be robust to an unknown holdout mix. | `tune_operating_point.py`: simulate holdout pools with 0/5/10/20% inversions; grid over T multiplier × p_min × p_max; maximise **worst-case** gain subject to ≤ 0.002 local loss; assert MCC constant; parity with gasbench sn34 within 0.005. |
| **[RUN]** Local gasbench is saturated (dinov3_384 local 0.992, lower on-chain). | Local score cannot be the ship criterion. | Compare arms on-chain (Part IV); keep local runs as smoke tests. |

---

# Part IV: Experiment history

The r24 work (commits `330c2ae` → `6596991`) is a controlled search over one question: **how much degradation, of which kinds, at which point in the LR schedule?**

### On-chain results **[RUN]** (config-experiment-4.yaml header)

| Run | FP (real → fake) | FN (fake → real) | FP:FN | Total errors | **SN34** |
|---|---:|---:|---:|---:|---:|
| log7, clean-first | 1,105 | 426 | 2.59 : 1 | 1,531 | 0.8416 |
| oc-1, mild @30k | 985 | 304 | 3.24 : 1 | 1,289 | 0.8709 |
| log6, mild-first | 696 | 395 | 1.76 : 1 | 1,091 | 0.9050 |
| **oc-2 = Arm F, hard-first @15k** | **601** | **524** | **1.15 : 1** | 1,125 | **0.9148** |

- **Arm F vs log6:** arm F made *more* errors (1,125 vs 1,091) and still scored higher, because MCC and Brier reward balance.
- **Pattern:** every weaker run over-called **fake** on real images.
- **Arm G's hypothesis:** a degraded real photo unlike anything seen in training looks unfamiliar, and unfamiliar falls to the fake side (fakes are the more varied class). So widening the **kinds** of degradation should fix the balance.
- **[INF]** Mapping log6/log7/oc-1 to exact commits is inferred from naming; the repo does not record it.

### Timeline

| When | Event | Hypothesis → result → decision |
|---|---|---|
| pre-v7 | **BMF design adopted** | Per-branch failure domains (DINOv3 0.211 on FF++) vs a fused model with none → 4 heterogeneous backbones, independent training, fixed fusion. Ship rule: beat the best single branch. |
| run_1 | Calibration drift | T sharpened 0.736 → 0.587 on val_id while val_xgen fell (95% calibration) → cross-fitted temperature. |
| run_2 | Clique failures, degenerate T | In-clique fakes 0.631 vs 0.941 → clique pair batching. Halves chose T 4.9989 / 0.6403 → Brier-grid fit. |
| run_3 | Over-degradation | Arms 0.55/0.25/0.20 → face-swap 0.138, ffpp 0.000 in-train → deploy ≥ 0.65 guard, "severity is deliberately not touched". |
| 2026-08-06 | **First full benchmark** | 97.69% overall, sn34 0.9198, semisynthetic 62.10%. PNG shortcut (45.3% vs 17.3%) → prechain laundering. face-swap 24.2% held out vs winner 98.9% → ship split. |
| 08-07 | Pre-registered Δ1–Δ4 | Double-JPEG, skip-WebP, semi crop guard, kernel jitter (OFF). A/B protocol: paired 500-step clip runs. val_id tracks public score within 0.008. (No recorded A/B outcomes.) |
| 08-09/10 | **gasbench 0.8.3, v21 holdouts** | 13 released holdouts → all to TRAIN ("holding out public data the leader trains on only donates points"). dct branch added (dflip3k reals 15%). GAS-Station boost introduced. |
| 08-12 | Calibration headroom | Dashboard Brier on the a(1−a) floor → `measure_margins`, `brier_headroom`, spread hinge. Aesthetic-real coverage gap → `prepare_reals`. |
| 08-13 | Probe suite | `probe-aesthetic-reals`; probes.yaml taxonomy. |
| 08-17/18 | Epoch × degradation experiment | 10 epochs, 10-profile schedule (config-experiment.yaml, arm D). dflip3k re-gated as the zero-shot judge. |
| 08-19 | Δ5 resolution variants | dinov3_336/384. dinov3_384 local 99.72% / 0.992 but lower on-chain → local gasbench can invert rankings; the extra band was a pipeline fingerprint. |
| r22 | **Lost on confident inversions** | 83.52 vs 86.22. Gemini-like holdouts 7.69% / 23.08% (winner 68–81%); ~10% inverted → hinge, p-clamp, `tune_operating_point`. Multiclass 0.8589 vs binary 0.9153. |
| 08-24/25 | **Round 23: 3-class taxonomy** | Registry cull (74 duplicates → legacy); 190 datasets (106/79/5). Factorised 3-logit head, type posterior, kind_balance kept. |
| 08-25 | No-degradation arm C vs D | Hypothesis: the winner trains without degradation, keeping fragile HF fingerprints. Result: gemini31 AUC ceiling 0.75 even clean → the data path (self-generated frontier images) is the lever, not augmentation removal. |
| 08-26 | Enhanced reals | Denoised reals FPR 30.7%, beauty-filtered 13.2% (identity 1.2%) → label-blind enhancement of both classes (v7). |
| 08-27/28 | **Fingerprints are per pipeline instance** | Fake2M probes: LDM families transfer 94–99.6%, pixel-space IF 48–84%; IF-dpms50 stayed 48.6% after folding gemini31 → one source per family teaches a source shortcut. CogView2 (80%) is the better Gemini-family proxy. |
| 08-31 → 09-02 | T2I cohort, SELFGEN v2, blind spots (v7) | Zero registry coverage of Qwen-Image/Z-Image/Sana/PixArt-Σ/Lumina-2/HiDream → generate. Hard-arm misses: animal close-ups 17%, night/degraded 7–8%. |
| 09-08 | **Plateau probe · semi shortfall** | 255,738 train rows → 2,664 steps/epoch @ bs 96, 1.6 s/step; selection still climbing at step 5000 (0.9273, +0.0025 per 500). `epochs: 10` was never derived. Semi requested 0.25, realised 0.0952. |
| 09-09 | r24 code, kind weights, multiclass selection | Arm E schedules: 2-entry alternating → mild-first (MILD, MILD, LAUNDERED, LAUNDERED). |
| 09-10 | Export override bug; clean-first | Pinned q 0.7593 → fitted 0.9618. On-chain base Brier 0.0583 vs aug 0.0215 ("calibrated for the wrong pass") → clean-first ramp (log7 → **0.8416**, worst). |
| 09-11 | **Arm F: hard-first + 4 effect families** | LR-weighted learning per epoch 46.5 / 37.0 / 15.1 / 1.4%. On-chain **0.9148**, best to date. |
| 09-13 | **Arm G: width 4 → 10 families** | Severity held equal (22.71 / 25.89 / 36.43 dB). No result yet. |

### Recurring lessons

1. The held-out canaries are **face-swap** and **fakeclue-fake-ffpp**; watch them whenever degradation changes.
2. Local gasbench is saturated and can invert on-chain ordering; A/B on-chain.
3. Fingerprints are per pipeline instance: family coverage needs ≥ 2 sources per family.
4. Weak runs over-call FAKE on reals; FP:FN balance predicts SN34 better than error count.
5. Brier dominates sn34 (effective exponent 0.9 vs MCC's 0.6 after the square root).
6. Released public holdouts go to TRAIN.
7. Real data collected after 2022 is suspect for AI contamination.

---

## Experiment configs (arms C to G)

**Baseline `config.yaml`:**

- `output_dir runs/v7`, seed 34, epochs 2, `use_degradation_schedule: false` (10-profile schedule defined but inert).
- Prechain jpeg 0.35 / double 0.10 / skip-webp 0.15; jitter p 0.4 [0.60, 1.0]; arms 0.70/0.20/0.10.
- `selection {val_id: 1.0}`, `multiclass_selection` and `kind_class_weights` on, `gasstation_boost 2.5`, `max_replay 25`.

| Config | Arm | Changes vs parent | Hypothesis | Result |
|---|---|---|---|---|
| `config-experiment.yaml` | D (control) | Every branch `epochs: 10`; `use_degradation_schedule: true` | Epoch × degradation interaction; end on the deploy-faithful finale | r22 recipe; control for C |
| `config-clean.yaml` | C | `output_dir runs/v7-clean`; `clean_view_recompress: false`; jitter off; `view_arms deploy 1.0`; `view_b: deploy`; schedule off (5 treatment knobs, asserted by `verify_views.py`) | Fragile-fingerprint test | gemini31 AUC ceiling 0.75: not rescued |
| `config-experiment-2.yaml` | E | `runs/v7-plateau`; dinov3 epochs 8 (ceiling; `--max-steps` sets rungs); `eval_max_rows 8000`, `eval_every 500`; base laundering = schedule time-average (jpeg 0.325 q[56,91], double 0.125, skip-webp 0.125, jitter 0.4375 [0.6125,1.0], arms 0.700/0.1625/0.1375); `ladder_level_probs [.25,0,.25,.5]`; schedule evolved: 2-entry → mild-first 4-entry → **clean-first 6-entry** (CLEAN×2 → LAUNDERED×4) | Measure step budget; protect semi evidence at high LR; then fix base-pass calibration | log6 mild-first 0.9050; log7 clean-first 0.8416 **[INF mapping]** |
| `config-experiment-3.yaml` | F | `runs/v7-diverse`; dinov3 bs 96 → 64, lr 1.7e-4 → 1.4e-4; schedule **4 entries hard-first**: e0 heavy (effects_p .35, strength [.40,1], 4 families), e1 hard (effects .25), e2 medium (effects .15, motion+grain), e3 deploy-faithful finale (effects off, arms .80/.10/.10) | Degradation heaviest where the cosine does most learning; add capture/print families the ladder can't produce | **0.9148 @15k, FP:FN 1.15** |
| `config-experiment-4.yaml` | G | `runs/v7-width`; only `degradation_schedule` differs from F: families 4 → 10 via weighted mapping (incumbents 2; sensor_noise/color_cast/chroma_shift 1.5; banding/vignette 1); effects_p re-solved .35→.654, .25→.466, .15→.211 so mass-weighted MSE is identical; e2 keeps 7 families; finale unchanged | Weak runs over-call fake on unfamiliar degraded reals → widen degradation *kinds* at equal severity | pending |

The headers of configs 2 and 3 are stale (they still describe the plateau probe).

---

## Commit log

| Commit | Date | Message | What changed and why |
|---|---|---|---|
| `330c2ae` | 09-09 06:39 | "fist version of r24 training code - hard degradation on high lr period" | Initial import (71 files, including `__pycache__` and 7 docs). config-experiment-2: 2-entry alternating [MILD, LAUNDERED], so the harsh profile sits in the high-LR epoch. `eval_max_rows`, `eval_every 500`, `ladder_level_probs`. |
| `981d9a0` | 09-09 06:44 | "degradation mild->hard applied" | 4 entries MILD, MILD, LAUNDERED, LAUNDERED: same time-average, degradation back-loaded (73.6% of learning on MILD). The laundered 0.475× jitter compounds with view_b's 0.5× (32.0 dB vs mild 35.2 dB). |
| `815ea19` | 09-09 07:36 | "balance dataset weights" | Response to realised semi 0.0952: `KIND_TOL`, target/ceiling reporting, `kind_weights()` in the loss, the type term's own balance, `multiclass_selection`, `type_prior`, verify.py ceiling identity; all configs gain `multiclass_selection`, `type_prior: null`, `kind_class_weights: true`. Backups `*.prefix-bak`, `train.py.mc-bak`. |
| `62f2879` | 09-10 09:54 | "voerride temperature errors in export.py fixed" | `--override-temperature` no longer skips the type posterior (0.7593 → 0.9618); loud WARNING when q ships pinned. |
| `8c252ea` | 09-10 11:16 | "image-degradation renew" | Clean-first 6-entry ramp (LR-weighted clean CE mass 35.8% → 73.8%) after on-chain base Brier 0.0583 vs aug 0.0215. `view_b` becomes schedulable per epoch; loss flags logged. |
| `9fc4da4` | 09-11 07:57 | "arm F newer degradation test" | `config-experiment-3.yaml`; `source_effects` (motion_blur, defocus_blur, film_grain, halftone) with a fixed 4-draw contract and perf caps; unknown effect names rejected. |
| `6596991` | 09-13 11:00 | "degradation keep, diversity wider" | `config-experiment-4.yaml`; six new families; `effects` accepts a weight mapping so width and severity are independent. |

---

# Part V: Open issues found

None of these has been changed; each needs a decision.

### 1. HIGH: registry drift, 13 semisynthetic datasets instead of 5 [SRC]

gasbench v0.9.5 (`dataset/configs/synthetic_images.yaml`) lists **13** semisynthetic image datasets:

- face-swap, deepfakeface-inpainting, deepfakes-qa-15k, deepfake-identity-isolated-fake
- swappir-celeba-hq-roop, swappir-celeba-hq-simswap, swappir-fairface-roop
- scaledf-ffpp-deepfakes, scaledf-ffpp-face2face, scaledf-e4e, scaledf-diffusionclip
- dgm4-styleclip, fakeclue-fake-ffpp

That gives 212 image datasets in total. Sampler comments and `verify.py` guardrails assume the v23 pool of 5.

**Impact:** the derived `type_prior` (semi share of fake datasets) moves from ≈ 0.06 to ≈ 13/97 ≈ **0.13**. That changes both checkpoint selection and the export posterior.

**Action:**
1. Rebuild the manifest.
2. Re-read the sampler's `kind target/realized/ceiling` line and peak replay.
3. Re-run `verify.py`.

### 2. HIGH: the multiclass argmax cutoff is not 0.5 [INF]

- **Where the cutoff sits:** for output `[1−p, p(1−q), pq]`, argmax picks real only when `p < 1/(1+max(q,1−q))`, which lies in [0.5, 0.667].
- **What was only proven for binary:**
  - T0 is fitted on binary Brier.
  - The claim that the p-clamp "cannot move MCC" holds for binary MCC only.
- **Where it can break:** in Gorodkin MCC, a `p_max` below ≈ 0.667 could flip argmax to real when q ≈ 0.5.

**Action:** the current default (1−1e-6) is safe. Keep `tune_operating_point.py`'s p_max grid above 2/3, or add an assertion in `export.py`.

### 3. MEDIUM: two gasbench versions on the machine [SRC]

- The subnet's `.venv` pins **gasbench 0.8.0** (`uv.lock`), which has binary-only metrics and different full-mode sizes (55k vs 72.5k).
- The training bridge loads whichever checkout `paths.py` finds.

**Action:** always `export GASBENCH_SRC=<bitmind-subnet>/gasbench/src` (v0.9.5), so views and scorer match what grades the model.

### 4. MEDIUM: selection on val_id is fitted in-sample [INF]

- T and (T2, b2, q_max) are fitted on the val_id rows they score.
- Ranking is mostly fair, but absolute numbers are optimistic.
- A 2-parameter posterior can favour checkpoints whose miscalibration it happens to fit.

**Action:** cross-fit by dataset on the val_id path, as `crossfit_temperature` already does for val_xgen.

### 5. MEDIUM: six design documents deleted in the working tree [SRC]

`EXPECTED_OUTCOMES.md`, `RUNBOOK-NODEGRADE.md`, `RUNBOOK-PLATEAU.md`, `RUNBOOK-SELFGEN.md`, `UPDATE-v21.md` and `UPDATE-v23.md` show as `D` in `git status`. `RUNBOOK-PLATEAU.md` exists only in this repo's history.

**Action:** `git restore <files>` if the deletion was not intended.

### 6. MEDIUM: the robustness blend is an assumption [SRC]

- The aug pass runs only when `n_aug_per_dataset > 0`; production count and weight are hidden round config.
- view_b, the 0.2 selection blend, and T fitted on the blend all assume it runs.
- The on-chain base Brier (0.0583) vs aug Brier (0.0215) finding suggests it does.

### 7. LOW: stale headers and verification gaps

- `config-experiment-2.yaml` and `-3.yaml` headers describe the plateau probe and contradict their bodies.
- `verify_views.py` does not validate configs 2–4 or the `effects` families.
- `generate_swaps.py` `DEFAULT_SOURCES` includes ffhq-256 and AgeDB (holdouts in the dev split): always pass `--sources`.
- `RUNBOOK-PLATEAU.md` references `build_pool.py` and `split_holdout.py`, which exist nowhere.

### 8. LOW: assumed score composition

`tune_operating_point.py` defaults to `public=0.50, holdout=0.35, gasstation=0.15`. These are guesses, not observed shares, so treat the output as robust over a range, not exact.

### 9. LOW: v7 / v7_2 tool divergence [SRC]

v7_2 carries older copies of `generate_edits.py` (no `--mask-mix v2`, no FLUX-fill), `generate_frontier.py` (no fal/openrouter), `generate_swaps.py` (no `--enhance gfpgan`) and `measure_margins.py` (no `diversity_report`).

The v7-only tools are: `generate_if.py`, `generate_t2i.py`, `generate_enhanced.py`, `probe_blindspot.py`, `probe_enhanced_reals.py`, `predict_folder.py`, `explain_predict.py`, `verify_dataset.py`, `aggregate_image.py`, `prompt_engine.py`.

---

# Part VI: Tool reference

| Tool | Stage | CLI essentials | What it decides |
|---|---|---|---|
| `paths.py` | infra | env `GASBENCH_SRC` | Resolves the gasbench checkout (moved into `bitmind-subnet/` on 2026-08-09). |
| `gasbench_bridge.py` | infra | import only | Loads gasbench `transforms.py` by file path; stubs heavy imports so `Metrics` loads without onnxruntime. Never vendored. |
| `allowlist.py` | S9 | import only | `scan_allowlist` (AST scan), `render_template` (literal substitution + compile). |
| `build_manifest.py` | S1 | `--cache-dir --out --image-size 384 [--overrides --workers]` | Manifest rows + corpus report (category × label, resolution, m_eval coverage, containers). |
| `prepare_reals.py` | S1 | `--cache-dir [--sources … --ood-only --skip-probe]` | Aesthetic real train sets + disjoint probe pool. |
| `ingest_external.py` | S1 | `--data-root --role probe\|train --name --label --src [--screen]` | Safe external ingestion with provenance checks. |
| `generate_edits.py` | S1 | `--manifest --cache-dir --sources --per-source 750 --editors sdxl,dreamshaper` | Self-generated semisynthetic inpaints from train rows. |
| `generate_swaps.py` | S1 | `--data-root --swap-model inswapper_128.onnx --target --sources --n 3000` | Self-generated face swaps. |
| `generate_frontier.py` | S1 | `--provider gemini\|openai --model --n 1000 [--dry-run]` | Frontier-API synthetic images (user budget decision). |
| `scan_gasstation_generators.py` | S2 | `--cache-dir --pattern gemini [--emit-mv OUT.sh]` | Per-week generator census; leak scan. |
| `audit_data.py` | S2 | `--data-root --manifest [--expected-per-dataset 500 --expected-gasstation 5000]` | Pre-training gate (exit 1 on HARD). |
| `audit_shortcuts.py` | S2/S10 | `--mode metadata,tensors,slices,pairs,interventions [--strict]` | Shortcut measurement + ledger. |
| `verify_shortcuts.py` | S2 | none | Planted positives / null controls for the audit. |
| `build_splits.py` | S3 | `--manifest --overrides --val-id-frac --write [--per-dataset-cap --strict-provenance]` | Split assignment + validation. |
| `diagnose_replay.py` | S4 | `--manifest --config --top 25` | Names datasets at the replay cap. |
| `verify.py` | pre-flight | none | Registry weights, allowlist, sampler identities, transformers pin, eval contract, 3-class parity, DCT basis, CLIP e2e. |
| `verify_views.py` | pre-flight | none | Prechain frequencies, double-JPEG shift, skip-webp parity, crop guard, deploy parity, determinism, jitter stream, view_b deploy, clean-vs-experiment knob diff. |
| `train.py` | S5–S8 | `accelerate launch train.py --config --branch [--max-steps]` | One branch per run → `runs/<output_dir>/<branch>/best.pt`. |
| `export.py` | S9 | `--branches --runs-dir --out [--calib-splits --override-temperature --no-spread --spread-slope --p-min --p-max --type-prior --binary-equivalent]` | Fused, calibrated submission. |
| `measure_margins.py` | S9/S10 | `--ood-root --runs-dir --branches --manifest` | AUC(\|d\|, correct): is calibration worth pursuing? |
| `eval_probes.py` | S10 | `--probes probes.yaml --data-root --runs-dir --branches --modes deploy,robust` | Probe accuracy ledger; where the next data goes. |
| `brier_headroom.py` | S10 | `--dashboard-mcc --dashboard-brier [--current-temperature --runs-dir --ood-root]` | Recoverable Brier without retraining. |
| `tune_operating_point.py` | S10 | `--records records.parquet --results-json --composition … --max-local-cost 0.002` | T multiplier, p_min, p_max for re-export. |

---

# Part VII: Command runbook

This order matches the stages above, for the current train-everything experiment arms.

```bash
# 0 · environment: gasbench's transformers pin, explicit benchmark source
export GASBENCH_SRC=~/Documents/new-training/bitmind-subnet/gasbench/src
pip install transformers==5.2.0

# S1–S3 · data
python build_manifest.py --cache-dir <CACHE> --out data/manifest.parquet --image-size 384
python audit_data.py --data-root <CACHE> --manifest data/manifest.parquet
python build_splits.py --manifest data/manifest.parquet \
       --overrides overrides-experiment.yaml --val-id-frac 0.20            # dry run: review
python build_splits.py --manifest data/manifest.parquet \
       --overrides overrides-experiment.yaml --val-id-frac 0.20 --write
python audit_shortcuts.py --mode metadata,tensors

# pre-flight
python verify.py && python verify_views.py
python diagnose_replay.py --manifest data/manifest.parquet --config config-experiment-4.yaml

# S4–S8 · train each branch independently (smoke test first)
accelerate launch train.py --config config-experiment-4.yaml --branch dinov3 --max-steps 60
accelerate launch train.py --config config-experiment-4.yaml --branch dinov3 --max-steps 15000
#   check in the log: "[sampler] kind target/realized/ceiling", "grad check", "view_b", "selection = …"

# S9 · export (solo first, so every regression is attributable)
python export.py --branches dinov3 --runs-dir runs/v7-width --out submission/
python export.py --branches dinov3,clip,convnext,eva,dct --runs-dir runs/v7-width --out submission/
#   check in the log: "type posterior: …" and NO "WARNING: q is PINNED"

# S10 · gate, then operating point
gasbench run --image-model ./submission/ --small
python eval_probes.py --probes probes.yaml --runs-dir runs/v7-width --branches dinov3,clip,convnext,eva,dct
python tune_operating_point.py --records records.parquet            # optional re-export with --p-min/--p-max
cd submission && zip -r ../v7.zip model_config.yaml model.py model.safetensors config.json
gascli d push --image-model v7.zip --wallet-name <W> --wallet-hotkey <H>
```

### Pre-push checklist

- [ ] `verify.py` and `verify_views.py` pass.
- [ ] Sampler log: P(fake) = 0.5000, P(fake\|category) = 0.5000, replay cap not bound (or diagnosed).
- [ ] Every branch passed the step-0 grad check.
- [ ] Export ran under `transformers==5.2.0`; round-trip OK; q **fitted** (not pinned).
- [ ] `--calib-splits` excludes any pool being used as a zero-shot judge.
- [ ] Fusion beats the best single branch on the same gasbench run.
- [ ] `gasbench --small` accuracy ≥ 80%, with non-zero semisynthetic accuracy.
- [ ] Expected improvement over the current king is ≥ 0.01 (one submission per hotkey).

---

*Sources: `transfer_learning_v7_2` (7 commits, 2026-09-09 → 09-13), sibling `transfer_learning_v7` docs, `bitmind-subnet` @ 720df63, `gasbench` @ fd21699. **[RUN]** figures are quoted from repo comments and notes, not re-run. **[INF]** items are the author's analysis.*
