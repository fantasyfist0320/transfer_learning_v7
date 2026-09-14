"""v7 trainer: one branch per run.

    accelerate launch train.py --config config.yaml --branch clip
    accelerate launch train.py --config config.yaml --branch dinov3 --max-steps 60

Branches never share a process. The fused model exists only at export time,
which is what makes each branch's gasbench number attributable and lets a bad
branch be dropped without retraining the rest.

Loss = CE(view_a) + CE(view_b) + kl * symKL(a, b) + brier(ramped) -- four
terms, all directly objective-aligned: view_a is the exact scored transform,
view_b the exact aug_binary_* chain, KL ties them, and Brier is beta=1.8 of
the deployed score. Selection = blended sn34 on val_xgen under a cross-fitted
temperature (calibrate.py, carried over verified from the previous package).
"""
from __future__ import annotations

import argparse
import contextlib
import math
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.utils.data import DataLoader

from branches import BRANCHES, resolve
from build_splits import _stable_frac
from calibrate import (blended_sn34, class_balance_weights, compose_3class,
                       crossfit_temperature, deploy_p_fake, fit_temperature,
                       fit_type_posterior, multiclass_metrics,
                       set_deploy_prior, sn34_from_probs, y3_class_weights)
from data import (BalancedSampler, ManifestDataset, ViewConfig,
                  balanced_weights, clique_pairs, realized_stream_report,
                  worker_init_fn)
from model import BranchModel, binary_margin, collapse_binary, type_margin

REQUIRED = {
    "data": ["manifest", "image_size", "num_workers", "view_arms"],
    "sampler": ["max_replay", "clique_pair_frac", "kind_balance"],
    "training": ["epochs", "batch_size", "grad_accum_steps", "lr",
                 "weight_decay", "warmup_ratio", "max_grad_norm", "amp",
                 "ema_decay", "eval_every"],
    # multiclass_selection / type_prior are OPTIONAL (see OPTIONAL_TRAINING).
    "loss": ["kl_consistency", "brier", "brier_ramp", "label_smoothing",
             "type_weight", "type_label_smoothing", "kind_class_weights"],
    "lora": ["r", "alpha", "dropout"],
}
# Despite the name this is the allowlist for OPTIONAL keys in every section
# validate_config checks, not just `data`.
OPTIONAL_DATA = ("clean_view_recompress", "resample_jitter", "laundering",
                 "selection", "gasstation_boost", "degradation_schedule",
                 "use_degradation_schedule", "view_b", "eval_max_rows",
                 "ladder_level_probs", "multiclass_selection", "type_prior",
                 "objective", "deploy_prior")
OBJECTIVES = ("factorized", "multiclass")


def resolve_prior(tr: dict, df: pd.DataFrame) -> tuple[float, str]:
    """(type_prior = P(semi|fake) for selection, where it came from).

    `training.deploy_prior: [real, synthetic, semisynthetic]` -- the
    score-weighted class shares of the benchmark the model will be graded on
    (panel_check.py prior prints them from a chain run) -- also sets the
    real/fake target of every calibrate.py weighting via set_deploy_prior.
    Without it the old behaviour stands: a flat 50/50 real/fake target and
    the registry's dataset-level semi share.
    """
    dp = tr.get("deploy_prior")
    if dp is not None:
        real, syn, semi = set_deploy_prior(dp)
        derived = semi / (syn + semi)
        if tr.get("type_prior") is not None and \
                abs(float(tr["type_prior"]) - derived) > 1e-3:
            raise SystemExit(
                f"training.type_prior={tr['type_prior']} contradicts "
                f"training.deploy_prior (semi/(syn+semi) = {derived:.4f}); "
                f"set only one")
        return derived, f"deploy_prior {real:.3f}/{syn:.3f}/{semi:.3f}"
    if tr.get("type_prior") is not None:
        return float(tr["type_prior"]), "training.type_prior"
    fake_ds = df[df.label == 1].drop_duplicates("dataset")
    share = (float((fake_ds["kind"] == "semisynthetic").mean())
             if len(fake_ds) else 0.19)
    return share, "registry dataset share (no deploy_prior set)"


def validate_config(cfg: dict) -> None:
    missing = [f"{s}.{k}" for s, ks in REQUIRED.items()
               for k in ks if k not in (cfg.get(s) or {})]
    unknown = [f"{s}.{k}" for s, d in cfg.items()
               if isinstance(d, dict) and s in REQUIRED
               for k in d if k not in REQUIRED[s] and k not in OPTIONAL_DATA]
    problems = []
    if missing:
        problems.append("missing keys:\n    " + "\n    ".join(missing))
    if unknown:
        problems.append("keys nothing reads:\n    " + "\n    ".join(unknown))
    for k in ("seed", "output_dir", "branches"):
        if k not in cfg:
            problems.append(f"missing top-level {k!r}")
    if problems:
        raise SystemExit("config.yaml does not match train.py\n\n"
                         + "\n\n".join(problems))
    arms = cfg["data"]["view_arms"]
    tot = sum(float(arms[a]) for a in ("deploy", "ladder", "robust"))
    if abs(tot - 1.0) > 1e-6:
        raise SystemExit(f"view_arms must sum to 1.0, got {tot}")
    objective = cfg["loss"].get("objective", "factorized")
    if objective not in OBJECTIVES:
        raise SystemExit(f"loss.objective must be one of {OBJECTIVES}, "
                         f"got {objective!r}")
    dp = cfg["training"].get("deploy_prior")
    if dp is not None:
        set_deploy_prior(dp)      # raises on a malformed prior, before training


def build_view(d: dict, overrides: dict | None = None,
               tag: str = "data") -> ViewConfig:
    """Construct a training ViewConfig from cfg["data"], optionally layered
    with one epoch's degradation-schedule overrides.

    Schedule entries may override `laundering`, `resample_jitter` and
    `view_arms` KEYS ONLY -- never the `clean_view_recompress` /
    `resample_jitter.enabled` flags: toggling an enable flag changes how many
    draws __getitem__ consumes from the shared per-row RNG stream
    (data.py:654-676), silently shifting every downstream decision (flips,
    arm selection, robustness params) between epochs. Probabilities and
    ranges are stream-safe; flags are not.

    Every epoch's merged config is re-validated here (arms sum to 1.0; the
    prechain arms leave room for the webp remainder), because the startup
    validate_config() only sees the base config.
    """
    arms = dict(d["view_arms"])
    rj = dict(d.get("resample_jitter") or {})
    la = dict(d.get("laundering") or {})
    ov = overrides or {}
    # `view_b` IS schedulable, unlike the enable flags below. The view_b draws
    # are the LAST rng consumers in __getitem__ -- deploy consumes none, robust
    # consumes some, and nothing reads the stream afterwards -- so switching it
    # per epoch leaves prechain / jitter / flips / arm-selection byte-identical.
    # That is what makes a genuine clean epoch possible: it is the only way to
    # lift clean CE mass above ~50%, since view_b is the robustness chain on
    # 100% of samples and carries half the loss.
    unknown = set(ov) - {"laundering", "resample_jitter", "view_arms", "view_b"}
    if unknown:
        raise SystemExit(f"{tag}: unknown override sections {sorted(unknown)}")
    if "enabled" in (ov.get("resample_jitter") or {}):
        raise SystemExit(f"{tag}: toggling resample_jitter.enabled per epoch "
                         f"shifts the shared RNG stream; vary p/factor_range "
                         f"instead")
    arms.update(ov.get("view_arms") or {})
    rj.update(ov.get("resample_jitter") or {})
    la.update(ov.get("laundering") or {})

    tot = arms["deploy"] + arms["ladder"] + arms["robust"]
    if abs(tot - 1.0) > 1e-6:
        raise SystemExit(f"{tag}: view_arms must sum to 1.0, got {tot:.4f}")
    # The webp arm is the remainder in _prechain_arm, so the named arms must
    # leave room for it: double mass is carved from the jpeg arm, not added.
    named = 0.30 + la.get("prechain_jpeg_p", 0.45) + la.get("double_jpeg_p", 0.0)
    if named > 1.0 + 1e-9:
        raise SystemExit(f"{tag}: prechain arms exceed 1.0 ({named:.2f}); "
                         f"lower prechain_jpeg_p when raising double_jpeg_p")
    # Ladder level mixture (L0..L3). A `data.` key, NOT a laundering one, so
    # the degradation schedule cannot vary it -- deliberate: it changes which
    # augmentation FAMILIES exist, not how strong they are, and a per-epoch
    # swap of that is the kind of non-stationarity the schedule is barred from
    # (build_view rejects enable-flag overrides for the same reason).
    lp = d.get("ladder_level_probs")
    if lp is not None:
        lp = tuple(float(x) for x in lp)
        if len(lp) != 4 or any(x < 0 for x in lp):
            raise SystemExit(f"{tag}: ladder_level_probs must be 4 "
                             f"non-negative numbers (L0..L3), got {lp}")
        if abs(sum(lp) - 1.0) > 1e-6:
            # apply_random_augmentations raises on this too, but inside a
            # dataloader worker mid-run; fail here instead.
            raise SystemExit(f"{tag}: ladder_level_probs must sum to 1.0, "
                             f"got {sum(lp):.4f}")
    known_fx = {"motion_blur", "defocus_blur", "film_grain", "halftone",
                "oversharpen", "sensor_noise", "banding", "color_cast",
                "vignette", "chroma_shift"}
    # `effects` accepts EITHER a list (uniform weights) or a mapping
    # {name: weight}. The mapping form exists because `pick` indexes the menu:
    # under a uniform list, adding families divides every incumbent's mass, so
    # a width increase is silently also a severity CUT. Weights decouple them.
    fx_raw = la.get("effects") or ()
    if isinstance(fx_raw, dict):
        fx_names = tuple(str(k) for k in fx_raw)
        fx_wts = tuple(float(v) for v in fx_raw.values())
        if any(w <= 0.0 for w in fx_wts):
            raise SystemExit(f"{tag}: laundering.effects weights must be > 0, "
                             f"got {dict(zip(fx_names, fx_wts))}")
    else:
        fx_names = tuple(str(x) for x in fx_raw)
        fx_wts = ()
    bad_fx = set(fx_names) - known_fx
    if bad_fx:
        # Loud: an unknown name would silently fall through source_effects'
        # else-branch and return the image untouched, so the arm would look
        # configured but train clean.
        raise SystemExit(f"{tag}: unknown laundering.effects {sorted(bad_fx)}; "
                         f"choose from {sorted(known_fx)}")
    view_b = ov.get("view_b", d.get("view_b", "robust"))
    if view_b not in ("robust", "deploy"):
        # Loud, because the failure mode of a typo here is an arm that LOOKS
        # like the no-degradation treatment but still trains a robust view_b.
        raise SystemExit(f"{tag}: data.view_b must be 'robust' or 'deploy', "
                         f"got {view_b!r}")
    return ViewConfig(image_size=d["image_size"],
                      view_b_mode=view_b,
                      prechain=d.get("clean_view_recompress", False),
                      prechain_jpeg_p=la.get("prechain_jpeg_p", 0.45),
                      prechain_jpeg_q=tuple(la.get("prechain_jpeg_q",
                                                   (55, 98))),
                      prechain_webp_q=tuple(la.get("prechain_webp_q",
                                                   (60, 95))),
                      prechain_double_p=la.get("double_jpeg_p", 0.0),
                      resample_jitter=rj.get("enabled", False),
                      resample_jitter_p=rj.get("p", 0.4),
                      resample_jitter_range=tuple(rj.get("factor_range",
                                                         (0.60, 1.0))),
                      resample_jitter_kernels=tuple(rj.get("kernels",
                                                           ("area_linear",))),
                      robust_skip_webp_p=la.get("robust_skip_webp_p", 0.0),
                      # Source-stage effect families. Live under `laundering`
                      # so the degradation schedule can vary them per epoch;
                      # source_effects consumes a FIXED 4 draws either way, so
                      # neither p nor the menu can shift the row's stream.
                      effects_p=float(la.get("effects_p", 0.0)),
                      effects=fx_names,
                      effects_weights=fx_wts,
                      effects_strength=tuple(la.get("effects_strength",
                                                    (0.25, 1.0))),
                      ladder_crop_guard=la.get("ladder_crop_guard", True),
                      ladder_level_probs=lp,
                      arm_deploy=arms["deploy"], arm_ladder=arms["ladder"],
                      arm_robust=arms["robust"])


def cosine_lr(step: int, total: int, warmup: int) -> float:
    if step < warmup:
        return (step + 1) / max(1, warmup)
    t = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, t)))


class EMA:
    def __init__(self, model, decay: float):
        self.decay = decay
        self.params = [p for p in model.parameters() if p.requires_grad]
        self.shadow = [p.detach().float().clone() for p in self.params]

    @torch.no_grad()
    def update(self, step: int) -> None:
        d = min(self.decay, (1.0 + step) / (10.0 + step))
        torch._foreach_mul_(self.shadow, d)
        torch._foreach_add_(self.shadow,
                            [p.detach().float() for p in self.params],
                            alpha=1.0 - d)

    @contextlib.contextmanager
    def swapped_in(self):
        backup = [p.detach().clone() for p in self.params]
        with torch.no_grad():
            for p, s in zip(self.params, self.shadow):
                p.copy_(s.to(p.dtype))
        try:
            yield
        finally:
            with torch.no_grad():
                for p, b in zip(self.params, backup):
                    p.copy_(b)


def symmetric_kl(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    la, lb = F.log_softmax(a, -1), F.log_softmax(b, -1)
    return 0.5 * (F.kl_div(la, lb.exp(), reduction="batchmean")
                  + F.kl_div(lb, la.exp(), reduction="batchmean"))


def brier_loss(z: torch.Tensor, y: torch.Tensor,
               w: torch.Tensor | None = None) -> torch.Tensor:
    # p_notreal = 1 - softmax[0]: K-generic, matches the deployed binary
    # collapse (gasbench scores binary MCC/Brier on exactly this quantity).
    p = 1.0 - F.softmax(z, -1)[:, 0]
    se = (p - y.float()) ** 2
    if w is None:
        return se.mean()
    return (se * w).sum() / w.sum().clamp_min(1e-12)


def brier3_loss(z: torch.Tensor, y3: torch.Tensor,
                w: torch.Tensor | None = None) -> torch.Tensor:
    """Multiclass Brier sum_k (p_k - 1{y3=k})^2 per row, (weighted) mean.

    gasbench's multiclass Brier term (baseline 2/3 = a uniform guess). Unlike
    brier_loss, a confident 'synthetic' on a semisynthetic row costs as much
    here as calling it real, which is how the chain scores it.
    """
    p = F.softmax(z.float(), -1)
    se = ((p - F.one_hot(y3, p.shape[-1]).float()) ** 2).sum(-1)
    if w is None:
        return se.mean()
    return (se * w).sum() / w.sum().clamp_min(1e-12)


def kind_weights(y3: torch.Tensor, n_classes: int = 3) -> torch.Tensor:
    """Per-row weights giving every kind PRESENT in the batch equal total mass.

    Why this exists. The sampler cannot deliver the configured kind mix:
    `_kind_shares` applies `kind_balance` INSIDE each (label, category) cell,
    so a kind confined to a few categories is capped at
    `label_w * sum(q_c over the cells it occupies)` no matter what the target
    says. On the v23 faces-only semisynthetic pool that ceiling is ~0.19
    against a 0.25 request, and the 2026-09-08 run measured 0.0952 -- semi
    reached ~19% of the fake half and ~9.5% of the corpus.

    That cap is a data-availability fact and no sampler dial removes it. The
    loss, however, is not capped: a scarce row can simply count for more, so
    the type boundary sees enough semisynthetic gradient to be learned.

    CORRECTION 2026-09-13: this was first justified as "gasbench weights the
    kinds equally". It does not -- it samples an equal cap per DATASET and
    weights by provenance (public/holdout/gasstation), never by label; the
    score-weighted v24 prior is ~0.43 / 0.52 / 0.06. Balanced kinds remain a
    reasonable training choice, but matching the scorer's prior is the job of
    `training.deploy_prior` in selection and calibration, not of this.

    Normalised to mean 1, so the loss scale -- and therefore the LR schedule
    -- is unchanged on a batch that is already balanced.
    """
    w = torch.ones(y3.shape[0], dtype=torch.float32, device=y3.device)
    for c in range(n_classes):
        m = y3 == c
        n_c = int(m.sum())
        if n_c:
            w[m] = 1.0 / n_c
    return w * (w.numel() / w.sum().clamp_min(1e-12))


def weighted_ce(logits: torch.Tensor, target: torch.Tensor,
                w: torch.Tensor | None = None,
                label_smoothing: float = 0.0) -> torch.Tensor:
    """CE with optional per-row weights. At w=None this is F.cross_entropy."""
    if w is None:
        return F.cross_entropy(logits, target, label_smoothing=label_smoothing)
    per = F.cross_entropy(logits, target, label_smoothing=label_smoothing,
                          reduction="none")
    return (per * w).sum() / w.sum().clamp_min(1e-12)


@torch.no_grad()
def collect_margins(model, loader, acc) -> dict:
    was = model.training
    model.eval()
    ds, d2s, ys, y3s, rid = [], [], [], [], []
    for batch in loader:
        z = model(batch["x"].to(acc.device, non_blocking=True))
        d = binary_margin(z).float()
        d2 = type_margin(z).float()
        ds.append(acc.gather_for_metrics(d).cpu())
        d2s.append(acc.gather_for_metrics(d2).cpu())
        ys.append(acc.gather_for_metrics(batch["y"].to(acc.device)).cpu())
        y3s.append(acc.gather_for_metrics(batch["y3"].to(acc.device)).cpu())
        rid.append(acc.gather_for_metrics(
            batch["row_id"].to(acc.device)).cpu())
    model.train(was)
    return {"d": torch.cat(ds).numpy(), "d2": torch.cat(d2s).numpy(),
            "y": torch.cat(ys).numpy(), "y3": torch.cat(y3s).numpy(),
            "row_id": torch.cat(rid).numpy()}


def cap_eval_frame(sub: pd.DataFrame, cap: int, seed: int) -> pd.DataFrame:
    """Down-sample ONE eval split to ~`cap` rows, stratified on dataset.

    Dense evals (eval_every: 500) over the full val splits cost more wall
    clock than the training between them once the corpus grows past ~100k
    rows. This trades eval precision for eval FREQUENCY: a fixed
    representative subsample measured often beats the whole split measured
    two or three times.

    Three properties the caller depends on:

      * DETERMINISTIC, and stable as the manifest grows. Rows are ranked by
        blake2b(image_id | seed) -- the same hash build_splits.py ranks
        blocks with -- not by an RNG draw, so the same rows survive on every
        eval, across restarts, AND across a corpus rebuild: a row's rank
        depends only on its own id, so growing the manifest shrinks the
        subsample without RESAMPLING it. That is what makes two runs (or two
        rungs of a budget ladder) a PAIRED comparison, which is the only
        reason a ~0.002 sn34 difference is readable at n=8000. `df.sample()`
        has none of this. `seed` is cfg["seed"], the global eval yardstick.
      * EVERY (dataset, label) cell survives: the quota is floored at 1.
        val_xgen's selection score is cross-fitted by folding on `dataset`
        and falls back to a pooled temperature below min_groups=4
        (calibrate.crossfit_temperature), so silently dropping small
        datasets would change WHICH estimator picks best.pt. The floor makes
        `cap` a target, not a hard ceiling.
      * Row ORDER is preserved (boolean mask, no reorder). The deploy and
        robust loaders are built from this one frame and evaluate_and_report
        pairs their margins elementwise -- see the call site.

    Label balance comes free: build_manifest.py writes `label` and `kind`
    from the per-dataset registry entry, so both are constant within a
    dataset and preserving dataset proportions preserves them exactly.
    `label` is in the strata as insurance against a future mixed-label
    dataset, not as a no-op today.
    """
    if cap <= 0 or len(sub) <= cap:
        return sub                  # absent key / already small: untouched
    # .astype(str) is load-bearing: groupby drops NaN keys, which would leave
    # those rows with a NaN quota and make the comparison below raise.
    cell = [sub["dataset"].astype(str), sub["label"]]
    g = sub["image_id"].astype(str).map(
        lambda k: _stable_frac(f"evalcap|{k}", seed)).groupby(cell)
    quota = (g.transform("size") * (cap / len(sub))).round().clip(lower=1)
    return sub[g.rank(method="first") <= quota].reset_index(drop=True)


def build_eval_loader(df, view, mode, bs, workers, seed):
    ds = ManifestDataset(df, view, train=False, eval_mode=mode, seed=seed)
    # persistent_workers=False on purpose: 6 eval loaders x 8 workers would
    # otherwise hold 48 resident processes between evals for splits a few
    # thousand rows each; respawn cost per eval is seconds.
    return DataLoader(ds, batch_size=bs, shuffle=False, num_workers=workers,
                      pin_memory=True, persistent_workers=False,
                      worker_init_fn=worker_init_fn)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--branch", required=True,
                    help=f"one of {sorted(BRANCHES)}")
    ap.add_argument("--max-steps", type=int, default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    validate_config(cfg)
    spec = resolve([args.branch])[0]
    bcfg = (cfg.get("branches") or {}).get(args.branch) or {}

    set_seed(cfg["seed"])
    out = Path(cfg["output_dir"]) / args.branch
    out.mkdir(parents=True, exist_ok=True)
    tr, lo = dict(cfg["training"]), cfg["loss"]
    # Branch entries may override the memory/optimisation knobs: the branches
    # span 11M-full-trainable to ViT-L@448, so one global batch size either
    # starves the small ones or OOMs the big ones. Everything else stays
    # global so runs remain comparable.
    for k in ("batch_size", "grad_accum_steps", "lr", "epochs"):
        if k in bcfg:
            tr[k] = bcfg[k]
    acc = Accelerator(gradient_accumulation_steps=tr["grad_accum_steps"],
                      mixed_precision=tr["amp"])

    df = pd.read_parquet(cfg["data"]["manifest"])
    if "split" not in df.columns or (df["split"] == "").all():
        raise SystemExit("manifest has no splits; run build_splits.py --write")
    if "kind" not in df.columns:
        raise SystemExit("manifest has no `kind` column; rebuild with "
                         "build_manifest.py (resolve_kind writes it)")

    d = cfg["data"]
    view = build_view(d)                     # base config: eval loaders + epoch default
    # Fail fast on EVERY epoch's degradation overrides, not just epoch 0's --
    # validated even when the schedule is disabled, so config errors surface
    # before the experiment run rather than during it.
    for i, ov in enumerate(d.get("degradation_schedule") or []):
        build_view(d, ov, tag=f"degradation_schedule[{i}]")
    # The schedule only ACTIVATES via the explicit flag: the baseline run must
    # train under the unmodified base config (control arm of the experiment).
    sched = (d.get("degradation_schedule") or []) \
        if d.get("use_degradation_schedule", False) else []
    if sched and acc.is_main_process:
        print(f"[{spec.name}] degradation schedule ACTIVE: {len(sched)} epoch "
              f"profiles (epoch e uses profile e % {len(sched)})")
    if acc.is_main_process:
        # Always printed: the no-degradation arm is only real if this says
        # "deploy" -- a silently-defaulted robust view_b would fake the arm.
        print(f"[{spec.name}] view_b (consistency view): {view.view_b_mode}"
              + (" -- clean duplicate of the deploy render; no degraded "
                 "pixels enter the loss"
                 if view.view_b_mode == "deploy" else ""), flush=True)

    tr_df = df[df.split == "train"].reset_index(drop=True)
    w, rep = balanced_weights(tr_df, amp_max=cfg["sampler"]["max_replay"],
                              kind_balance=cfg["sampler"]["kind_balance"],
                              gasstation_boost=float(
                                  cfg["sampler"].get("gasstation_boost", 1.0)),
                              verbose=acc.is_main_process)
    pairs = clique_pairs(tr_df)
    sampler = BalancedSampler(w, num_samples=len(tr_df), seed=cfg["seed"],
                              rank=acc.process_index,
                              world_size=acc.num_processes, pairs=pairs,
                              pair_frac=cfg["sampler"]["clique_pair_frac"])
    if acc.is_main_process:
        realized_stream_report(tr_df, w, rep, pairs,
                               cfg["sampler"]["clique_pair_frac"], cfg["seed"])
    train_ds = ManifestDataset(tr_df, view, train=True, seed=cfg["seed"])
    train_loader = DataLoader(
        train_ds, batch_size=tr["batch_size"], sampler=sampler,
        num_workers=d["num_workers"], pin_memory=True, drop_last=True,
        # persistent_workers MUST stay False: workers fork the dataset ONCE,
        # so with persistent workers set_epoch() never reaches them -- epoch 2
        # would render byte-identical augmentations to epoch 1 (verified
        # against torch 2.7: _MultiProcessingDataLoaderIter._reset only
        # rebuilds the sampler iter). Respawn cost is seconds per epoch.
        persistent_workers=False,
        worker_init_fn=worker_init_fn)

    # Deployment semisynthetic share of the FAKE half, used to reweight the
    # multiclass selection metric away from the sampler's ~0.5 P(semi|fake).
    # export.py resolves the same number from the same inputs (--deploy-prior
    # or its registry fallback), so selection and export rank checkpoints alike.
    type_prior, prior_src = resolve_prior(tr, df)
    objective = lo.get("objective", "factorized")
    if acc.is_main_process:
        print(f"[{spec.name}] prior: P(fake)={deploy_p_fake():.4f} "
              f"type_prior={type_prior:.4f} ({prior_src})")
    if acc.is_main_process and tr.get("multiclass_selection"):
        print(f"[{spec.name}] selection = MULTICLASS sn34 "
              f"(Gorodkin + mc Brier), type_prior={type_prior:.4f}")
    if acc.is_main_process:
        # Loss-shaping flags are otherwise invisible: nothing downstream prints
        # them, so a run's log could not be used to tell whether they were on.
        print(f"[{spec.name}] loss: objective={objective} "
              + (f"type_weight={lo['type_weight']} " if objective == "factorized"
                 else "(type_weight unused) ")
              + f"kind_class_weights={bool(lo.get('kind_class_weights'))} "
              f"kl={lo['kl_consistency']} brier={lo['brier']} "
              f"label_smoothing={lo['label_smoothing']}")

    evals = {}
    # `data.eval_max_rows` fixes a stratified subsample of each val split
    # ONCE, here; every eval from then on reads exactly those rows, so a
    # moving val curve is the model moving and not the yardstick. Absent (or
    # a split already under the cap) => the full split, byte-identical to
    # the uncapped pipeline.
    cap = int(d.get("eval_max_rows") or 0)
    for split in ("val_id", "val_xgen", "val_stress"):
        full = df[df.split == split].reset_index(drop=True)
        if len(full):
            sub = cap_eval_frame(full, cap, cfg["seed"])
            if len(sub) != len(full) and acc.is_main_process:
                print(f"[{spec.name}] eval_max_rows={cap}: {split} "
                      f"{len(full):,} -> {len(sub):,} rows, datasets "
                      f"{full['dataset'].nunique()} -> "
                      f"{sub['dataset'].nunique()}, p_fake "
                      f"{full['label'].mean():.4f} -> "
                      f"{sub['label'].mean():.4f}", flush=True)
            # ONE frame for BOTH modes, deliberately. evaluate_and_report
            # indexes the stored frame positionally by row_id, and pairs the
            # robust margins elementwise against the DEPLOY labels with no
            # realignment -- deploy[i] and robust[i] must be the same image.
            # Sharing the object is what guarantees that.
            for mode in ("deploy", "robust"):
                evals[f"{split}:{mode}"] = (sub, build_eval_loader(
                    sub, view, mode, tr["batch_size"], d["num_workers"],
                    cfg["seed"]))

    model = BranchModel(
        spec, lora_r=cfg["lora"]["r"], lora_alpha=cfg["lora"]["alpha"],
        lora_dropout=cfg["lora"]["dropout"],
        gradient_checkpointing=bcfg.get("gradient_checkpointing", True))
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if acc.is_main_process:
        print(f"[{spec.name}] {spec.model_id}  head={spec.head}  "
              f"native={spec.native_size}  trainable={n_train/1e6:.1f}M")

    opt = torch.optim.AdamW(model.param_groups(tr["lr"]),
                            weight_decay=tr["weight_decay"])
    model, opt, train_loader = acc.prepare(model, opt, train_loader)

    steps_per_epoch = len(train_loader)
    total = args.max_steps or steps_per_epoch * tr["epochs"]
    warmup = int(total * tr["warmup_ratio"])
    # Eval cadence: an integer = every N steps; the string "epoch" = at each
    # epoch boundary -- with a degradation schedule this aligns every
    # measurement with exactly one epoch profile (the val_stress dflip rows
    # become a per-profile learning curve).
    eval_interval = (steps_per_epoch if str(tr["eval_every"]) == "epoch"
                     else int(tr["eval_every"]))
    if eval_interval <= 0:
        raise SystemExit(f"eval_every resolves to {eval_interval}; use a "
                         f"positive integer or 'epoch'")
    ema = EMA(acc.unwrap_model(model), tr["ema_decay"])

    best_sel, step = -1.0, 0
    lora_recheck: set[str] = set()
    t0 = time.time()
    for epoch in range(tr["epochs"]):
        sampler.set_epoch(epoch)
        # With a degradation schedule, each epoch trains under its own
        # laundering/jitter/arm profile (a NEW ViewConfig -- the eval loaders
        # keep the untouched base `view` instance).
        epoch_view = (build_view(d, sched[epoch % len(sched)],
                                 tag=f"epoch {epoch}") if sched else None)
        train_ds.set_epoch(epoch, epoch_view)
        if sched and acc.is_main_process:
            ev = epoch_view
            print(f"[{spec.name}] epoch {epoch} degradation profile "
                  f"{epoch % len(sched)}: jpeg_p={ev.prechain_jpeg_p} "
                  f"q={ev.prechain_jpeg_q} double_p={ev.prechain_double_p} "
                  f"jitter_p={ev.resample_jitter_p} rng={ev.resample_jitter_range} "
                  f"arms=({ev.arm_deploy},{ev.arm_ladder},{ev.arm_robust}) "
                  # view_b is half the CE mass and now varies per epoch, so it
                  # has to be in the log -- a clean epoch that silently kept a
                  # robust view_b would look identical to a real one here.
                  f"view_b={ev.view_b_mode}",
                  flush=True)
        run_acc, run_n = 0.0, 0
        run_acc3, run_semi_hit, run_semi_n = 0.0, 0.0, 0
        for batch in train_loader:
            if step >= total:
                break
            progress = step / max(1, total)
            with acc.accumulate(model):
                xa = batch["x_a"].to(acc.device, non_blocking=True)
                xb = batch["x_b"].to(acc.device, non_blocking=True)
                y = batch["y"].to(acc.device)
                y3 = batch["y3"].to(acc.device)
                za, zb = model(xa), model(xb)
                ls = lo["label_smoothing"]
                # Factorized 3-class CE: CE_3(y3) == CE_bin(collapse, y)
                # + CE_cond(fake rows). Keeping the terms separate keeps the
                # binary term byte-identical to the 2-logit pipeline (incl.
                # 2-class label smoothing -- plain 3-class smoothing would
                # leak ls/3 onto the semi logit and drag q toward 0.5) and
                # makes type_weight the single risk dial (0 => binary-only
                # training with a 3-wide head).
                # kl held in a name so the periodic print can report its raw
                # magnitude: with a deploy view_b the pair is byte-identical
                # and this term becomes dropout-consistency (R-Drop) -- the
                # experiment writeup needs its size, not an assumption of 0.
                kl = symmetric_kl(za, zb)
                # Per-row kind weights: real / synthetic / semisynthetic each
                # carry equal total mass in the batch -- a representation-
                # learning balance for the scarce semi class, NOT a match to
                # the scorer's prior (see kind_weights).
                # None => byte-identical to the previous unweighted path, so
                # kind_class_weights: false reproduces earlier runs exactly.
                kw = kind_weights(y3) if lo.get("kind_class_weights") else None
                if objective == "multiclass":
                    # Direct 3-class objective (2026-09-13): CE and Brier on
                    # the full probability vector the scorer reads, so the
                    # synthetic-vs-semisynthetic split is inside BOTH terms --
                    # the factorized Brier below scores only 1 - p_real and is
                    # blind to it. type_weight is unused: CE_3 already contains
                    # the conditional type CE (CE_3 = CE_bin(collapse) + CE_cond
                    # on fake rows). label_smoothing here is 3-class, i.e. ls/3
                    # target mass on the semi logit of every row; set it to 0
                    # to rule that out. export.py needs no change: its
                    # [1-p, p(1-q), pq] composition of logsumexp margins IS
                    # softmax(z) at identity calibration.
                    loss = (weighted_ce(za, y3, kw, ls)
                            + weighted_ce(zb, y3, kw, ls)
                            + lo["kl_consistency"] * kl)
                    r0, r1 = lo["brier_ramp"]
                    ramp = min(1.0, max(0.0, (progress - r0) / max(1e-9, r1 - r0)))
                    loss = loss + lo["brier"] * ramp * 0.5 * (
                        brier3_loss(za, y3, kw) + brier3_loss(zb, y3, kw))
                else:
                    loss = (weighted_ce(collapse_binary(za), y, kw, ls)
                            + weighted_ce(collapse_binary(zb), y, kw, ls)
                            + lo["kl_consistency"] * kl)
                    lam = lo["type_weight"]
                    fake = y3 > 0
                    if lam > 0 and bool(fake.any()):
                        ls_t = lo["type_label_smoothing"]
                        # The type term sees ONLY fake rows, so `kw` (a corpus-wide
                        # 3-kind balance) is the wrong weighting here -- it needs
                        # its own 2-class balance over the fake half, where
                        # semisynthetic is ~19% and plain CE lets synthetic
                        # dominate the one term that separates the two.
                        tw = (kind_weights(y3[fake] - 1, n_classes=2)
                              if kw is not None else None)
                        loss = loss + lam * 0.5 * (
                            weighted_ce(za[fake][:, 1:], y3[fake] - 1, tw, ls_t)
                            + weighted_ce(zb[fake][:, 1:], y3[fake] - 1, tw, ls_t))
                    r0, r1 = lo["brier_ramp"]
                    ramp = min(1.0, max(0.0, (progress - r0) / max(1e-9, r1 - r0)))
                    loss = loss + lo["brier"] * ramp * 0.5 * (
                        brier_loss(za, y, kw) + brier_loss(zb, y, kw))
                acc.backward(loss)
                if step == 0:
                    # Gradient checkpointing silently drops gradients when the
                    # input to a checkpointed block does not require grad --
                    # which is the normal state here, since every backbone is
                    # frozen except LoRA/norms. The HF path is handled in
                    # model.py (use_reentrant=False + enable_input_require_grads);
                    # timm's set_grad_checkpointing has no such switch, so the
                    # convnext/eva branches are checked empirically instead of
                    # trusted. A branch that trains with dead gradients looks
                    # perfectly healthy in the loss curve and wastes the run.
                    #
                    # Dropped-by-checkpointing means grad is None. A grad that
                    # EXISTS but is all-zero is a different animal: LoRALinear
                    # zero-inits B, so dL/dA = scaling * B^T d L/dh x^T is
                    # exactly zero on the first step for every LoRA A weight,
                    # by construction and only on the first step. Those are
                    # deferred and re-checked after the first optimizer update
                    # (B != 0 by then, so a still-zero A is genuinely dead).
                    unwrapped = acc.unwrap_model(model)
                    live, dead, deferred = [], [], []
                    for n, p in unwrapped.named_parameters():
                        if not p.requires_grad:
                            continue
                        if (p.grad is not None and torch.isfinite(p.grad).any()
                                and p.grad.abs().sum() > 0):
                            live.append(n)
                        elif p.grad is not None and n.endswith(".A.weight"):
                            deferred.append(n)
                        else:
                            dead.append(n)
                    if acc.is_main_process:
                        print(f"[{spec.name}] grad check: {len(live)} params "
                              f"receiving gradient, {len(dead)} dead, "
                              f"{len(deferred)} LoRA A zero-by-init "
                              f"(re-checked after first update)", flush=True)
                    if dead:
                        raise SystemExit(
                            f"{len(dead)} trainable parameters received no "
                            f"gradient on the first step, e.g. {dead[:6]}. "
                            f"Most likely gradient checkpointing is dropping "
                            f"them: set branches.{spec.name}."
                            f"gradient_checkpointing: false in config.yaml "
                            f"(lower batch_size if VRAM is tight) and re-run.")
                    lora_recheck = set(deferred)
                if lora_recheck and step == tr["grad_accum_steps"]:
                    # First step after a completed optimizer update: B has
                    # moved off zero, so the step-0 excuse no longer applies.
                    unwrapped = acc.unwrap_model(model)
                    still = [n for n, p in unwrapped.named_parameters()
                             if n in lora_recheck
                             and (p.grad is None or p.grad.abs().sum() == 0)]
                    if still:
                        raise SystemExit(
                            f"{len(still)} LoRA A weights still receive no "
                            f"gradient after the first optimizer update, e.g. "
                            f"{still[:6]}. These are genuinely disconnected -- "
                            f"check gradient checkpointing for "
                            f"branches.{spec.name}.")
                    if acc.is_main_process:
                        print(f"[{spec.name}] grad check: all "
                              f"{len(lora_recheck)} deferred LoRA A weights "
                              f"receive gradient after first update", flush=True)
                    lora_recheck = set()
                if acc.sync_gradients:
                    acc.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad],
                        tr["max_grad_norm"])
                lr = tr["lr"] * cosine_lr(step, total, warmup)
                for g in opt.param_groups:
                    g["lr"] = lr
                opt.step()
                opt.zero_grad(set_to_none=True)
            if acc.sync_gradients:
                ema.update(step)
            # Binary accuracy via the collapse margin, not argmax over 3:
            # the running print stays comparable with 2-logit history and
            # is not distorted by how the fake mass splits across classes.
            run_acc += ((binary_margin(za) > 0).long() == y).float().sum().item()
            run_n += len(y)
            # 3-class view of the same batches: `acc` alone can climb while
            # semisynthetic is being called synthetic. Raw argmax on view_a,
            # uncalibrated and at the sampler's kind mix -- a trend line, not
            # the scored number.
            pred3 = za.argmax(dim=1)
            run_acc3 += (pred3 == y3).float().sum().item()
            semi = y3 == 2
            run_semi_hit += (pred3[semi] == 2).float().sum().item()
            run_semi_n += int(semi.sum().item())
            step += 1

            if step % 50 == 0 and acc.is_main_process:
                semi_str = (f"{run_semi_hit/run_semi_n:.4f}" if run_semi_n
                            else "--")
                print(f"[{spec.name}] ep{epoch} step {step}/{total} "
                      f"loss={loss.item():.4f} kl={kl.item():.4f} "
                      f"acc={run_acc/max(1,run_n):.4f} "
                      f"acc3={run_acc3/max(1,run_n):.4f} "
                      f"semi_recall={semi_str} "
                      f"lr={lr:.2e} {(time.time()-t0)/step:.2f}s/step",
                      flush=True)
                run_acc, run_n = 0.0, 0
                run_acc3, run_semi_hit, run_semi_n = 0.0, 0.0, 0

            if step % eval_interval == 0 or step == total:
                with ema.swapped_in():
                    res = {k: collect_margins(model, ld, acc)
                           for k, (sub, ld) in evals.items()}
                if acc.is_main_process:
                    sel = evaluate_and_report(
                        res, evals, spec.name, step, tr.get("selection"),
                        semi_prior=type_prior,
                        multiclass=bool(tr.get("multiclass_selection")))
                    if sel > best_sel:
                        best_sel = sel
                        m = acc.unwrap_model(model)
                        with ema.swapped_in():
                            torch.save(
                                {"branch": spec.name, "step": step,
                                 "state": m.state_dict(),
                                 "selection": sel, "config": cfg,
                                 # cfg records the raw file; per-branch
                                 # overrides (epochs/batch/lr) live in tr.
                                 # spec records native_size etc.: no tensor
                                 # shape depends on it, so a mismatched
                                 # strict load would otherwise be silent --
                                 # export.py cross-checks it vs the registry.
                                 "spec": asdict(spec),
                                 "training_effective": dict(tr)},
                                out / "best.pt")
                        print(f"[{spec.name}] saved best.pt (sel={sel:.4f})",
                              flush=True)
                acc.wait_for_everyone()
        if step >= total:
            break
    if acc.is_main_process:
        print(f"[{spec.name}] done. best selection {best_sel:.4f} "
              f"-> {out/'best.pt'}")


def multiclass_blended_sn34(r: dict, ra: dict | None, T: float,
                            semi_prior: float,
                            aug_weight: float = 0.2) -> float | None:
    """Deployment-weighted 3-class sn34, blended 0.8 base / 0.2 robust.

    The scorer is 3-class (Gorodkin MCC + multiclass Brier). A checkpoint
    picked on the binary collapse is therefore picked on the wrong quantity:
    `compose_3class` makes the collapse invariant to q by construction, so the
    entire synthetic-vs-semisynthetic axis -- the axis this pipeline actually
    loses on -- is INVISIBLE to the binary score. Two checkpoints with equal
    binary sn34 and opposite typing behaviour are indistinguishable to it.

    Mirrors export.py's composition exactly (same fit_type_posterior, same
    deployment-share weights), so selection and export rank checkpoints the
    same way instead of disagreeing silently.

    Returns None when the split cannot support the fit (fewer than 3 kinds
    present), which leaves the caller on the binary path.
    """
    if "y3" not in r or "d2" not in r:
        return None
    y3 = r["y3"]
    if len(set(y3.tolist())) < 3:
        return None
    pf = 1.0 / (1.0 + np.exp(-r["d"] / T))
    tp = fit_type_posterior(y3, r["d2"], pf, semi_prior=semi_prior)
    w3 = y3_class_weights(y3, semi_prior)
    T2, B2, qlo, qhi = (tp["type_temperature"], tp["type_bias"],
                        tp["q_min"], tp["q_max"])

    def _mc(rr: dict) -> float:
        # y3/w3 come from the DEPLOY frame on purpose: the robust loader is
        # the same rows in the same order (see build_eval_loader call site).
        p = 1.0 / (1.0 + np.exp(-rr["d"] / T))
        q = np.clip(1.0 / (1.0 + np.exp(-(rr["d2"] / T2 + B2))), qlo, qhi)
        return multiclass_metrics(y3, compose_3class(p, q), w3)["mc_sn34"]

    base = _mc(r)
    if ra is None:
        return base
    return (1.0 - aug_weight) * base + aug_weight * _mc(ra)


def _t_fit_weights(r: dict, multiclass: bool, semi_prior: float):
    """Weights for the binary temperature fit.

    Under multiclass selection the type posterior is fitted at the deployment
    real/synthetic/semisynthetic shares, so T is fitted at the SAME shares --
    otherwise T targets the val pool's own fake-half mix (kind_balance
    trains near P(semi|fake)~0.5) while q targets semi_prior. Binary path
    (or a split lacking a kind): None, i.e. fit_temperature's real/fake
    balance, exactly as before.
    """
    if not multiclass or "y3" not in r or len(set(r["y3"].tolist())) < 3:
        return None
    return y3_class_weights(r["y3"], semi_prior)


def evaluate_and_report(res: dict, evals: dict, name: str, step: int,
                        weights: dict | None = None,
                        semi_prior: float = 0.19,
                        multiclass: bool = False) -> float:
    """Blended sn34 per split; selection is a weighted mix of split scores."""
    print(f"  [eval] step {step}")
    print(f"    {'split':22s}{'n':>8}{'acc':>8}{'sn34':>8}{'type':>8}"
          f"{'mc_sn34':>9}")
    for key, r in res.items():
        acc_v = float(((r["d"] > 0).astype(int) == r["y"]).mean())
        # Diagnostic only, never selection: syn-vs-semi accuracy of the raw
        # type margin on fake rows (needs both fake kinds in the split).
        tstr = "--"
        if "y3" in r:
            fk = r["y3"] > 0
            if fk.any() and len(set(r["y3"][fk].tolist())) == 2:
                tacc = float(((r["d2"][fk] > 0).astype(int)
                              == (r["y3"][fk] == 2).astype(int)).mean())
                tstr = f"{tacc:.4f}"
        if len(set(r["y"].tolist())) < 2:
            # Single-label canary split (val_stress is real-only): MCC/sn34
            # are undefined and a temperature fit is degenerate -- accuracy
            # here is the false-positive rate complement, which is the point.
            print(f"    {key:22s}{len(r['y']):>8,}{acc_v:>8.4f}{'--':>8}"
                  f"{tstr:>8}{'--':>9}")
            continue
        wts = class_balance_weights(r["y"])
        cal = fit_temperature(r["y"], r["d"],
                              w=_t_fit_weights(r, multiclass, semi_prior))
        m = sn34_from_probs(r["y"],
                            1 / (1 + np.exp(-r["d"] / cal["temperature"])),
                            wts)
        # The scored quantity, reported next to the binary one so the two can
        # be watched diverging. `type` above is raw-margin accuracy at
        # threshold 0; this is the calibrated 3-class sn34 the chain computes.
        mc = multiclass_blended_sn34(r, None, cal["temperature"], semi_prior)
        mstr = "--" if mc is None else f"{mc:.4f}"
        print(f"    {key:22s}{len(r['y']):>8,}{acc_v:>8.4f}{m['sn34']:>8.4f}"
              f"{tstr:>8}{mstr:>9}")
    # -- selection -----------------------------------------------------------
    # Which split should choose the checkpoint depends on what the model is
    # being built for, and the two goals disagree:
    #
    #   val_xgen  measures generalisation to UNSEEN generators -- the right
    #             signal when holdouts are the target (hidden-holdout score).
    #   val_id    measures unseen IMAGES from SEEN datasets -- which is exactly
    #             what the public benchmark does, now that the ship split folds
    #             ~170 of 183 benchmark datasets into train.
    #
    # Measured on the 2026-08-07 dinov3 run at step 1000: val_id:deploy sn34
    # 0.8561 vs gasbench full-suite 0.8644 (0.008 apart), while val_xgen:deploy
    # read 0.9172 -- a 0.05 overestimate. val_id is the honest predictor of the
    # public number; val_xgen peaked at step 1000 and fell while val_id kept
    # climbing, so selecting on val_xgen freezes best.pt long before the public
    # score stops improving.
    #
    # Default keeps historical behaviour (pure val_xgen). Set
    # training.selection: {val_id: 0.7, val_xgen: 0.3} for a ship run.
    sel_w = dict(weights or {"val_xgen": 1.0})
    parts: dict[str, float] = {}

    if "val_id:deploy" in res and sel_w.get("val_id", 0.0) > 0:
        r = res["val_id:deploy"]
        ra = res.get("val_id:robust")
        w_id = class_balance_weights(r["y"])
        cal = fit_temperature(r["y"], r["d"],
                              ra["d"] if ra is not None else None,
                              w=_t_fit_weights(r, multiclass, semi_prior))
        T = cal["temperature"]
        base = sn34_from_probs(r["y"], 1 / (1 + np.exp(-r["d"] / T)), w_id)
        if ra is not None:
            aug = sn34_from_probs(r["y"], 1 / (1 + np.exp(-ra["d"] / T)), w_id)
            parts["val_id"] = 0.8 * base["sn34"] + 0.2 * aug["sn34"]
        else:
            parts["val_id"] = base["sn34"]
        # The chain scores 3-class. Selecting on the binary collapse cannot
        # see the syn-vs-semi axis at all (compose_3class makes the collapse
        # invariant to q), so best.pt was being chosen on a quantity that is
        # blind to the class this pipeline loses on. Falls back to the binary
        # value when the split lacks all three kinds.
        if multiclass:
            mc = multiclass_blended_sn34(r, ra, T, semi_prior)
            if mc is not None:
                parts["val_id"] = mc

    if "val_xgen:deploy" in res:
        sub = evals["val_xgen:deploy"][0]
        r = res["val_xgen:deploy"]
        # margins are gathered in loader order; row_id maps them back onto the
        # split frame so each sample carries its dataset for fold assignment
        # (deploy and robust loaders share the same order). Fold on `dataset`,
        # not `generator_family`: every real dataset carries the family "real",
        # which would collapse the whole real class into one fold unit and
        # leave the other fold single-label (pooled-T fallback, silently).
        fam = sub["dataset"].astype(str).to_numpy()[r["row_id"]]
        ra = res.get("val_xgen:robust")
        xf = crossfit_temperature(r["y"], r["d"],
                                  ra["d"] if ra is not None else None, fam)
        parts["val_xgen"] = float(xf["blended_crossfit"])
        tag = "" if xf.get("crossfit", True) else \
            f"  [FALLBACK pooled T, {xf['n_groups']} groups]"
        print(f"    val_xgen blend (cross-fit) = "
              f"{parts['val_xgen']:.4f}{tag}")

    used = {k: v for k, v in sel_w.items() if k in parts and v > 0}
    tot = sum(used.values())
    if not tot:
        return 0.0
    sel = sum(parts[k] * w for k, w in used.items()) / tot
    if len(used) > 1:
        detail = "  ".join(f"{k}={parts[k]:.4f}x{used[k]:g}" for k in used)
        print(f"    selection = {sel:.4f}   ({detail})")
    else:
        print(f"    selection = {sel:.4f}")
    return float(sel)


if __name__ == "__main__":
    main()
