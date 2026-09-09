"""Shortcut-learning audit: does the DATA teach label-correlated shortcuts,
and does the MODEL read them?

This file discharges three standing promises in the package:
  * build_manifest.py:  "audit_shortcuts.py quantifies which of these
    [container/geometry differences] actually reach the model"
  * data.py (ViewConfig.prechain): "audit_shortcuts.py --mode tensors prices
    it before it is switched on"
  * overrides.yaml: "so audit_shortcuts.py can measure whether the
    [category] fixes mattered"

Five modes, cheapest first (`--mode all` runs them in this order):

  metadata       manifest-only, CPU, runs on the GPU-less VM. Can file
                 container / geometry / codec columns alone predict the
                 label? (M1-M6)
  tensors        CPU + image cache + gasbench. Can low-level statistics of
                 the RENDERED deploy view predict the label, which metadata
                 shortcuts survive the render, and how many AUC points each
                 laundering knob removes? (T1-T4)
  slices         GPU + checkpoints. Per-slice accuracy/margins joined back
                 onto manifest columns, plus within-class nuisance response:
                 does the score track codec / resolution / category / mode
                 with the label held fixed? Per-branch attribution. (S1-S8)
  pairs          GPU. Within-clique (matched real/fake near-duplicate)
                 margin gaps -- the semantic-confound control. (P1-P3)
  interventions  GPU, heaviest. Causal battery: re-encode / grayscale /
                 resize / crop / blur / noise each image identically for
                 both classes and measure prediction movement. Movement of
                 p(fake) on REAL images = the model reads that nuisance
                 channel. (I1-I11)

Conventions follow eval_probes.py: argparse, an append-only JSONL ledger
(shortcut_history.jsonl) for cycle-over-cycle regression tracking, and
threshold buckets. Verdicts are ok/warn/hard; the exit code is 0 unless
--strict is passed AND a hard fires (measurement first, gating second --
the audit_data.py stance). The first run per (mode, test, split, key) is
stamped baseline and never fails on regression.

Import discipline (the brier_headroom.py pattern): module level needs only
stdlib + numpy + pandas, so `--mode metadata` runs anywhere a manifest copy
exists. torch / sklearn / data / train / eval_probes / PIL / cv2 load lazily
inside the modes that need them. The metric helpers below are deliberate
LOCAL COPIES for the same reason (rank_auc is the tie-aware Mann-Whitney --
metadata features are heavily tied, so measure_margins.py's tie-unaware
variant would mis-rank them).
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:          # let verify_shortcuts import us from anywhere
    sys.path.insert(0, str(HERE))

SUITE_VERSION = 1

# Absolute-prior thresholds. Tunable: review after the first baseline run and
# adjust for this corpus, documenting the change here.
TH = {
    "metadata_auc":         {"warn": 0.70, "hard": 0.85},  # M1 blocked OOF
    "metadata_feature_auc": {"warn": 0.65},                # M2 single feature
    "val_inherit":          {"warn": 0.05},                # M4 val - blocked
    "category_prior":       {"warn": 0.30},                # M5 |P(fake|cat)-.5|
    "tensor_auc":           {"warn": 0.65, "hard": 0.80},  # T1 rendered stats
    "survival_auc":         {"warn": 0.75},                # T2 stats->nuisance
    "nuisance_auc":         {"warn": 0.65, "hard": 0.75},  # S1/S6 within-real
    "margin_rho":           {"warn": 0.30, "hard": 0.50},  # S2 spearman
    "slice_gap":            {"warn": 0.25},                # S3 category gap
    "kind_gap":             {"warn": 0.15},                # S5 semisyn deficit
    "error_pred_auc":       {"warn": 0.70},                # S8 metadata->wrong
    "flip_real":            {"warn": 0.10, "hard": 0.25},  # I* real flip rate
    "dp_real":              {"warn": 0.15},                # I* mean |dp| reals
    "fake_drop":            {"warn": 0.20},                # I1 fake acc drop
    "pairs_drop":           {"warn": 0.15},                # P1 in - out clique
    "pairs_confound_rho":   {"warn": 0.50},                # P2
    "clique_fpr_mult":      {"warn": 2.0},                 # P3
    "regress_acc":          {"warn": 0.05},                # vs last ledger row
    "regress_auc":          {"warn": 0.05},
}

NUM_FEATS = ("width", "height", "min_side", "max_side", "aspect",
             "megapixels", "bytes_per_pixel", "file_bytes", "m_eval")
CAT_FEATS = ("ext", "file_format", "pil_mode", "source_format")


# ---------------------------------------------------------------------------
# numpy-only metric helpers (local copies; see module docstring for why)
# ---------------------------------------------------------------------------

def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)),
                    np.exp(x) / (1.0 + np.exp(x)))


def rank_auc(score, flag) -> float:
    """Tie-aware Mann-Whitney AUC of `score` predicting binary `flag`."""
    s = np.asarray(score, dtype=float)
    y = np.asarray(flag).astype(int)
    ok = np.isfinite(s)
    s, y = s[ok], y[ok]
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    s_sorted = s[order]
    ranks = np.empty(len(s), dtype=float)
    base = np.arange(1, len(s) + 1, dtype=float)
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        ranks[order[i:j + 1]] = base[i:j + 1].mean()
        i = j + 1
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n0 * n1))


def folded_auc(score, flag) -> float:
    """Direction-agnostic AUC in [0.5, 1]: max(a, 1-a)."""
    a = rank_auc(score, flag)
    return a if math.isnan(a) else max(a, 1.0 - a)


def _probit(p: float) -> float:
    """Inverse standard normal CDF by bisection (no scipy)."""
    lo, hi = -8.0, 8.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if 0.5 * (1.0 + math.erf(mid / math.sqrt(2.0))) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def auc_ceiling(auc: float) -> float:
    """Best balanced accuracy any threshold can reach at this AUC
    (equal-variance Gaussian: Phi(Phi^-1(AUC)/sqrt(2)))."""
    if math.isnan(auc):
        return float("nan")
    auc = min(max(auc, 1e-6), 1.0 - 1e-6)
    return float(0.5 * (1.0 + math.erf(_probit(auc) / math.sqrt(2.0)
                                       / math.sqrt(2.0))))


def spearman(a, b) -> float:
    """Spearman rho via rank transform + Pearson. Tie-aware, no scipy."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if len(a) < 3:
        return float("nan")

    def _ranks(v):
        order = np.argsort(v, kind="mergesort")
        r = np.empty(len(v), dtype=float)
        base = np.arange(1, len(v) + 1, dtype=float)
        vs = v[order]
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and vs[j + 1] == vs[i]:
                j += 1
            r[order[i:j + 1]] = base[i:j + 1].mean()
            i = j + 1
        return r

    ra, rb = _ranks(a), _ranks(b)
    sa, sb = ra.std(), rb.std()
    if sa == 0 or sb == 0:
        return float("nan")
    return float(((ra - ra.mean()) * (rb - rb.mean())).mean() / (sa * sb))


# ---------------------------------------------------------------------------
# shared frame machinery
# ---------------------------------------------------------------------------

def block_col(df: pd.DataFrame) -> str:
    """The column that groups same-source rows, best available first.
    Blocking folds on it separates shortcut signal from dataset memorisation."""
    for c in ("split_unit", "group", "dataset"):
        if c in df.columns and df[c].notna().any():
            return c
    raise SystemExit("manifest has none of split_unit/group/dataset")


def blocked_folds(df: pd.DataFrame, n_folds: int = 5,
                  block: str | None = None) -> np.ndarray:
    """Group folds blocked on same-source units, dealt largest-first per
    dominant label so every fold carries both labels (the calibrate.py
    _assign_folds pattern)."""
    block = block or block_col(df)
    key = df[block].astype(str)
    sizes = key.value_counts()
    dom = (df.groupby(key)["label"].mean() >= 0.5)
    fold_of: dict[str, int] = {}
    # Staggered start so the largest real and largest fake blocks do not
    # deterministically co-locate in fold 0.
    counters = {False: 0, True: n_folds // 2}
    for name in sizes.index:            # largest first
        d = bool(dom.get(name, False))
        fold_of[name] = counters[d] % n_folds
        counters[d] += 1
    return key.map(fold_of).to_numpy()


def naive_folds(n: int, n_folds: int = 5, seed: int = 34) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.permutation(n) % n_folds


def stratified_sample(df: pd.DataFrame, n: int | None,
                      seed: int = 34) -> pd.DataFrame:
    """Deterministic proportional sample over (label, dataset), >=1 row per
    cell. Sorted by a stable key first so the draw is independent of the
    manifest's row order."""
    if n is None or len(df) <= n:
        return df.reset_index(drop=True)
    key = "image_id" if "image_id" in df.columns else "path"
    df = df.sort_values(key, kind="mergesort").reset_index(drop=True)
    rng = np.random.default_rng(seed)
    frac = n / len(df)
    parts = []
    for _, g in df.groupby(["label", "dataset"], sort=True):
        k = min(len(g), max(1, int(round(len(g) * frac))))
        idx = rng.choice(len(g), size=k, replace=False)
        parts.append(g.iloc[np.sort(idx)])
    return pd.concat(parts).reset_index(drop=True)


def featurize(df: pd.DataFrame, extra_cat: str | None = None,
              top_k: int = 12) -> tuple[np.ndarray, list[str]]:
    """Metadata feature matrix: log1p numerics + top-k one-hots per
    categorical. No pixel is read."""
    cols, names = [], []
    for c in NUM_FEATS:
        if c not in df.columns:
            continue
        v = pd.to_numeric(df[c], errors="coerce").astype(float).to_numpy()
        cols.append(np.log1p(np.clip(v, 0.0, None)))
        names.append(f"log1p_{c}")
    cats = list(CAT_FEATS) + ([extra_cat] if extra_cat else [])
    for c in cats:
        if c not in df.columns:
            continue
        s = df[c].astype(str)
        for val in s.value_counts().index[:top_k]:
            cols.append((s == val).to_numpy(dtype=float))
            names.append(f"{c}={val}")
    if not cols:
        raise SystemExit("no metadata feature columns found in the manifest")
    return np.column_stack(cols), names


def oof_probe_auc(X: np.ndarray, y: np.ndarray,
                  folds: np.ndarray) -> tuple[float, np.ndarray]:
    """Out-of-fold probability of a small gradient-boosted probe, and its
    AUC. sklearn imported lazily (requirements.txt pins it for this file)."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    y = np.asarray(y).astype(int)
    oof = np.full(len(y), np.nan)
    for f in np.unique(folds):
        tr, te = folds != f, folds == f
        if te.sum() == 0 or len(np.unique(y[tr])) < 2:
            continue
        clf = HistGradientBoostingClassifier(max_iter=200, random_state=34)
        clf.fit(X[tr], y[tr])
        oof[te] = clf.predict_proba(X[te])[:, 1]
    return rank_auc(oof, y), oof


def fit_probe(X: np.ndarray, y: np.ndarray):
    from sklearn.ensemble import HistGradientBoostingClassifier
    clf = HistGradientBoostingClassifier(max_iter=200, random_state=34)
    clf.fit(X, np.asarray(y).astype(int))
    return clf


def probe_verdict(auc: float, oof: np.ndarray, th: dict) -> tuple[str, dict]:
    """Verdict for an OOF-probe AUC that refuses to pass silently when the
    MEASUREMENT failed. A NaN AUC (all positives of the target confined to
    one block -> its fold is skipped as single-class-in-train and the
    scoreable remainder is single-class) or skipped-fold coverage loss means
    cross-block predictability could not be measured -- and a target
    concentrated in one source block is exactly the suspicious case the
    probe exists to catch -- so it surfaces as warn, never ok."""
    cov = float(np.isfinite(oof).mean()) if len(oof) else 0.0
    extra = {"oof_coverage": cov}
    if (auc is None or (isinstance(auc, float) and math.isnan(auc))
            or cov < 0.99):
        extra["unmeasurable"] = True
        return "warn", extra
    return bucket(auc, th), extra


# ---------------------------------------------------------------------------
# ledger (eval_probes.py conventions: append-only JSONL, delta vs last row)
# ---------------------------------------------------------------------------

def _key_str(mode: str, test: str, split: str, key: dict | None) -> str:
    kv = ",".join(f"{k}={key[k]}" for k in sorted(key)) if key else ""
    return "|".join([mode, test, split, kv])


def make_row(mode: str, test: str, split: str, key: dict | None, n: int,
             metrics: dict, verdict: str, args, *, headline: str | None = None,
             higher_is_bad: bool = True, extra: dict | None = None) -> dict:
    row = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "tag": args.tag, "suite_version": SUITE_VERSION,
        "mode": mode, "test": test, "split": split, "key": key or {},
        "key_str": _key_str(mode, test, split, key),
        "n": int(n), "seed": args.seed, "image_size": args.image_size,
        "verdict": verdict,
        "headline": headline, "higher_is_bad": higher_is_bad,
        "metrics": {k: (float(v) if isinstance(v, (int, float, np.floating))
                        and not isinstance(v, bool)
                        else v) for k, v in metrics.items()},
    }
    if extra:
        row.update(extra)
    return row


def append_ledger(path: Path, rows: list[dict]) -> None:
    with path.open("a") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")


def last_history(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                r = json.loads(line)
                if isinstance(r, dict) and "key_str" in r:
                    out[r["key_str"]] = r
            except json.JSONDecodeError:
                continue
    return out


def bucket(value: float, th: dict, higher_is_bad: bool = True) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "ok"
    v = value if higher_is_bad else -value
    if "hard" in th and v > th["hard"]:
        return "hard"
    if "warn" in th and v > th["warn"]:
        return "warn"
    return "ok"


# ---------------------------------------------------------------------------
# mode: metadata (CPU-only)
# ---------------------------------------------------------------------------

def mode_metadata(df: pd.DataFrame, args) -> list[dict]:
    rows: list[dict] = []
    y = df["label"].astype(int).to_numpy()
    X, names = featurize(df)
    blocked = blocked_folds(df)

    # M1: can metadata alone identify the label, across sources?
    auc_b, oof_b = oof_probe_auc(X, y, blocked)
    v1, ex1 = probe_verdict(auc_b, oof_b, TH["metadata_auc"])
    rows.append(make_row(
        "metadata", "M1_metadata_probe", "all", None, len(df),
        {"auc": auc_b, "balanced_acc_ceiling": auc_ceiling(auc_b),
         "block": block_col(df), "n_features": len(names), **ex1},
        v1, args, headline="auc"))

    # M2: which single feature carries it?
    feats = {}
    for c in NUM_FEATS:
        if c in df.columns:
            v = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
            feats[c] = folded_auc(v, y)
    for c in CAT_FEATS:
        if c in df.columns:
            s = df[c].astype(str)
            lift = s.map(df.groupby(s)["label"].mean()).to_numpy(dtype=float)
            feats[c] = folded_auc(lift, y)
    offenders = sorted((c for c, a in feats.items()
                        if not math.isnan(a)
                        and a > TH["metadata_feature_auc"]["warn"]),
                       key=lambda c: -feats[c])
    for c, a in sorted(feats.items(), key=lambda kv: -(kv[1] or 0)):
        rows.append(make_row(
            "metadata", "M2_feature_auc", "all", {"feature": c}, len(df),
            {"auc": a}, bucket(a, TH["metadata_feature_auc"]), args,
            headline="auc"))

    # M3: blocking honesty -- unblocked minus blocked = memorisation share.
    auc_n, _ = oof_probe_auc(X, y, naive_folds(len(df), seed=args.seed))
    rows.append(make_row(
        "metadata", "M3_blocking_gap", "all", None, len(df),
        {"auc_blocked": auc_b, "auc_naive": auc_n,
         "memorisation_gap": auc_n - auc_b},
        "ok", args, headline="memorisation_gap"))

    # M4: does the selection metric itself reward the shortcut?
    if "split" in df.columns and (df["split"] == "train").any():
        tr = df[df["split"] == "train"]
        for split in ("val_id", "val_xgen", "val_stress"):
            va = df[df["split"] == split]
            if len(va) == 0 or va["label"].nunique() < 2:
                continue
            # featurize's one-hot columns depend on the frame, so build the
            # matrix once on train+val and slice -- consistent schema.
            both = pd.concat([tr, va]).reset_index(drop=True)
            Xb, _ = featurize(both)
            clf2 = fit_probe(Xb[:len(tr)], tr["label"].to_numpy())
            p = clf2.predict_proba(Xb[len(tr):])[:, 1]
            a = rank_auc(p, va["label"].to_numpy())
            excess = a - auc_b
            rows.append(make_row(
                "metadata", "M4_val_inheritance", split, None, len(va),
                {"auc_val": a, "auc_blocked_ref": auc_b, "excess": excess},
                bucket(excess, TH["val_inherit"]), args, headline="excess"))

    # M5: category prior per split + MI(label; category).
    cat_col = "category" if "category" in df.columns else None
    if cat_col and "split" in df.columns:
        for split in sorted(df["split"].astype(str).unique()):
            sub = df[df["split"] == split]
            if len(sub) < 100 or sub["label"].nunique() < 2:
                continue
            pj = pd.crosstab(sub[cat_col], sub["label"], normalize="all")
            pc = pj.sum(axis=1)
            pl = pj.sum(axis=0)
            mi = 0.0
            for c in pj.index:
                for l in pj.columns:
                    p = pj.loc[c, l]
                    if p > 0:
                        mi += p * math.log2(p / (pc[c] * pl[l]))
            worst, worst_cat = 0.0, ""
            for c in pc.index:
                if pc[c] < 0.05:
                    continue
                pf = pj.loc[c, 1] / pc[c] if 1 in pj.columns else 0.0
                if abs(pf - 0.5) > worst:
                    worst, worst_cat = abs(pf - 0.5), str(c)
            rows.append(make_row(
                "metadata", "M5_category_prior", split, None, len(sub),
                {"mi_bits": mi, "worst_abs_prior": worst,
                 "worst_category": worst_cat},
                bucket(worst, TH["category_prior"]), args,
                headline="worst_abs_prior"))

    # M6: did the category overrides reduce label correlation?
    if {"content_category", "content_category_fixed"} <= set(df.columns):
        aucs = {}
        for col in ("content_category", "content_category_fixed"):
            Xc, _ = featurize(df, extra_cat=col)
            aucs[col], _ = oof_probe_auc(Xc, y, blocked)
        delta = aucs["content_category_fixed"] - aucs["content_category"]
        rows.append(make_row(
            "metadata", "M6_override_efficacy", "all", None, len(df),
            {"auc_raw_category": aucs["content_category"],
             "auc_fixed_category": aucs["content_category_fixed"],
             "delta": delta}, "ok", args, headline="delta"))

    rows[0]["metrics"]["offenders"] = offenders   # M1 row carries the names
    return rows


# ---------------------------------------------------------------------------
# lazy dataset subclasses (factory keeps module import numpy/pandas-only)
# ---------------------------------------------------------------------------

def _dataset_classes():
    from data import (ManifestDataset, source_prechain,
                      source_resample_jitter)

    class InterventionDataset(ManifestDataset):
        """Deploy-view eval render with a deterministic label-blind `fn`
        applied to the SOURCE image. Overrides only the `_load` seam, so the
        view pipeline itself is untouched (verify_shortcuts proves a no-op fn
        is byte-identical to the base class)."""

        def __init__(self, df, cfg, *, fn, seed: int = 34):
            super().__init__(df, cfg, train=False, eval_mode="deploy",
                             seed=seed)
            self.fn = fn

        def _load(self, i: int):
            return self.fn(super()._load(i), i)

    class PrechainDataset(ManifestDataset):
        """Deploy-view eval render preceded by the TRAIN laundering fns,
        driven by an audit-owned rng stream (never the training stream)."""

        def __init__(self, df, cfg, *, variant: str, seed: int = 34):
            assert variant in ("prechain", "jitter", "both")
            super().__init__(df, cfg, train=False, eval_mode="deploy",
                             seed=seed)
            self.variant = variant

        def _load(self, i: int):
            img = super()._load(i)
            rng = random.Random((self.seed * 1_000_003 + i) * 2 + 1)
            if self.variant in ("prechain", "both"):
                img = source_prechain(img, rng, self.cfg)
            if self.variant in ("jitter", "both"):
                img = source_resample_jitter(img, rng, self.cfg)
            return img

    return InterventionDataset, PrechainDataset


# ---------------------------------------------------------------------------
# mode: tensors (model-free; CPU + image cache + gasbench)
# ---------------------------------------------------------------------------

TENSOR_STAT_NAMES = (
    ["mean_r", "mean_g", "mean_b", "std_r", "std_g", "std_b"]
    + [f"fft_band_{k}" for k in range(8)]
    + ["hf_lf_ratio", "blockiness", "sat_mean", "sat_std",
       "edge_density", "hist_entropy"]
)


def render_stats_batch(x_u8) -> np.ndarray:
    """~20 low-level statistics per image of a uint8 [B,3,S,S] batch:
    channel stats, radial FFT log-band energies, HF/LF ratio, 8x8 blockiness,
    saturation, edge density, histogram entropy."""
    import torch
    x = x_u8.to(torch.float32) / 255.0
    b, _, hh, ww = x.shape
    mean_c = x.mean(dim=(2, 3))
    std_c = x.std(dim=(2, 3))
    luma = (0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2])

    f = torch.fft.rfft2(luma)
    power = (f.real ** 2 + f.imag ** 2)
    fy = torch.fft.fftfreq(hh, device=x.device).abs().view(-1, 1)
    fx = torch.fft.rfftfreq(ww, device=x.device).view(1, -1)
    r = torch.sqrt(fy ** 2 + fx ** 2)          # cycles/pixel, max ~0.707
    edges = torch.tensor(
        np.geomspace(0.01, 0.70, 9), dtype=torch.float32, device=x.device)
    bands = []
    for k in range(8):
        m = (r >= edges[k]) & (r < edges[k + 1])
        e = (power * m).sum(dim=(1, 2)) / m.sum().clamp(min=1)
        bands.append(torch.log1p(e))
    bands = torch.stack(bands, dim=1)
    hf = (power * (r > 0.35)).sum(dim=(1, 2))
    lf = (power * ((r > 0) & (r <= 0.0875))).sum(dim=(1, 2))
    hf_lf = torch.log1p(hf / lf.clamp(min=1e-9)).unsqueeze(1)

    dh = (luma[:, :, 1:] - luma[:, :, :-1]).abs()
    cols = torch.arange(dh.shape[2], device=x.device)
    bnd = ((cols + 1) % 8 == 0)
    blk_h = dh[:, :, bnd].mean(dim=(1, 2)) / dh[:, :, ~bnd].mean(dim=(1, 2)).clamp(min=1e-9)
    dv = (luma[:, 1:, :] - luma[:, :-1, :]).abs()
    rws = torch.arange(dv.shape[1], device=x.device)
    bndv = ((rws + 1) % 8 == 0)
    blk_v = dv[:, bndv, :].mean(dim=(1, 2)) / dv[:, ~bndv, :].mean(dim=(1, 2)).clamp(min=1e-9)
    blockiness = (0.5 * (blk_h + blk_v)).unsqueeze(1)

    sat = x.max(dim=1).values - x.min(dim=1).values
    sat_mean = sat.mean(dim=(1, 2)).unsqueeze(1)
    sat_std = sat.std(dim=(1, 2)).unsqueeze(1)
    edge = (0.5 * (dh.mean(dim=(1, 2)) + dv.mean(dim=(1, 2)))).unsqueeze(1)

    ent = []
    for i in range(b):
        h = torch.histc(luma[i] * 255.0, bins=32, min=0, max=255)
        p = h / h.sum().clamp(min=1)
        ent.append(-(p * torch.log2(p.clamp(min=1e-12))).sum())
    ent = torch.stack(ent).unsqueeze(1)

    out = torch.cat([mean_c, std_c, bands, hf_lf, blockiness,
                     sat_mean, sat_std, edge, ent], dim=1)
    return out.cpu().numpy()


def collect_tensor_stats(sub: pd.DataFrame, view_cfg, args,
                         variant: str | None = None) -> np.ndarray:
    import torch
    from torch.utils.data import DataLoader
    from data import ManifestDataset, worker_init_fn
    if variant is None:
        ds = ManifestDataset(sub, view_cfg, train=False, eval_mode="deploy",
                             seed=args.seed)
    else:
        _, PrechainDataset = _dataset_classes()
        ds = PrechainDataset(sub, view_cfg, variant=variant, seed=args.seed)
    ld = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.workers, worker_init_fn=worker_init_fn)
    out, rid = [], []
    with torch.no_grad():
        for b in ld:
            out.append(render_stats_batch(b["x"]))
            rid.append(b["row_id"])
    r = torch.cat(rid).numpy()
    assert (r == np.arange(len(r))).all(), "tensor-stats row order canary"
    return np.concatenate(out, axis=0)


def _laundering_view(args):
    """ViewConfig carrying the config file's laundering/jitter parameters,
    built by train.build_view so the mapping cannot drift. Falls back to
    defaults if the config or its heavy imports are unavailable."""
    import yaml
    try:
        from train import build_view
        cfg = yaml.safe_load(Path(args.config).read_text())
        v = build_view(cfg["data"])
        v.image_size = args.image_size
        return v, None
    except Exception as e:                          # noqa: BLE001
        from data import ViewConfig
        return ViewConfig(image_size=args.image_size), \
            f"laundering params unavailable ({e}); using ViewConfig defaults"


def mode_tensors(df: pd.DataFrame, args) -> list[dict]:
    rows: list[dict] = []
    active = df[df["split"].astype(str).isin(
        ["train", "val_id", "val_xgen", "val_stress", "test"])] \
        if "split" in df.columns else df
    if len(active) == 0:
        active = df
    cap = args.rows_cap or 6000
    sample = stratified_sample(active, cap, args.seed)
    y = sample["label"].astype(int).to_numpy()
    folds = blocked_folds(sample)
    vcfg, warn = _laundering_view(args)
    if warn:
        print(f"[tensors] {warn}")

    stats0 = collect_tensor_stats(sample, vcfg, args, variant=None)

    # T1: rendered-tensor probe -- the model's ACTUAL input separates labels?
    auc0, oof0 = oof_probe_auc(stats0, y, folds)
    vt1, ext1 = probe_verdict(auc0, oof0, TH["tensor_auc"])
    rows.append(make_row(
        "tensors", "T1_rendered_probe", "sample", None, len(sample),
        {"auc": auc0, "balanced_acc_ceiling": auc_ceiling(auc0), **ext1},
        vt1, args, headline="auc"))

    # T2: shortcut survival -- can rendered stats predict the nuisance cols?
    nuisances = {}
    if "file_format" in sample.columns:
        nuisances["is_png"] = (sample["file_format"].astype(str)
                               .str.upper() == "PNG").to_numpy(int)
    if "pil_mode" in sample.columns:
        nuisances["pil_non_rgb"] = (sample["pil_mode"].astype(str)
                                    != "RGB").to_numpy(int)
    if "m_eval" in sample.columns:
        nuisances["upsampled"] = (pd.to_numeric(sample["m_eval"],
                                                errors="coerce")
                                  > 1.0).to_numpy(int)
    for name, col in nuisances.items():
        if len(np.unique(col)) < 2:
            continue
        a_surv, oof_s = oof_probe_auc(stats0, col, folds)
        _, ex_s = probe_verdict(a_surv, oof_s, TH["survival_auc"])
        unmeasurable = bool(ex_s.get("unmeasurable"))
        a_label = folded_auc(col.astype(float), y)
        live = (not unmeasurable
                and a_surv > TH["survival_auc"]["warn"]
                and a_label > TH["metadata_feature_auc"]["warn"])
        rows.append(make_row(
            "tensors", "T2_survival", "sample", {"nuisance": name},
            len(sample),
            {"survival_auc": a_surv, "label_auc": a_label,
             "live_shortcut": None if unmeasurable else bool(live), **ex_s},
            "warn" if (live or unmeasurable) else "ok", args,
            headline="survival_auc"))

    # T3: laundering pricing -- delta AUC per knob (the data.py:372 promise).
    for variant in ("prechain", "jitter", "both"):
        stats_v = collect_tensor_stats(sample, vcfg, args, variant=variant)
        auc_v, _ = oof_probe_auc(stats_v, y, folds)
        rows.append(make_row(
            "tensors", "T3_laundering_price", "sample",
            {"variant": variant}, len(sample),
            {"auc": auc_v, "auc_deploy": auc0, "delta": auc_v - auc0},
            "ok", args, headline="auc"))

    # T4: which physical channel carries it -- per-stat marginal AUC.
    per_dim = {TENSOR_STAT_NAMES[j]: folded_auc(stats0[:, j], y)
               for j in range(stats0.shape[1])}
    top = sorted(per_dim.items(), key=lambda kv: -(kv[1] or 0))[:8]
    rows.append(make_row(
        "tensors", "T4_per_stat_auc", "sample", None, len(sample),
        {"top": {k: round(v, 4) for k, v in top}}, "ok", args))
    return rows


# ---------------------------------------------------------------------------
# GPU collection shared by slices / interventions / pairs
# ---------------------------------------------------------------------------

_GPU: dict = {}


def gpu_ctx(args):
    if not _GPU:
        import torch
        from branches import fusion_weights, resolve
        from eval_probes import load_models
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        names = [n.strip() for n in args.branches.split(",") if n.strip()]
        specs, models, steps = load_models(Path(args.runs_dir), names, dev)
        _GPU.update(specs=specs, models=models, steps=steps, dev=dev,
                    names=names,
                    w=np.asarray(fusion_weights(specs), dtype=float))
    return _GPU


def collect_rows(models, sub: pd.DataFrame, view, device, bs, workers,
                 eval_mode: str = "deploy", dataset_cls=None, seed: int = 34,
                 **ds_kw):
    """(D [N, n_branches], y [N], row_id [N]) with the order canary that
    collect_branch_margins relies on implicitly, made explicit."""
    import torch
    from torch.utils.data import DataLoader
    from data import ManifestDataset, worker_init_fn
    from model import binary_margin
    cls = dataset_cls or ManifestDataset
    if cls is ManifestDataset:
        ds = cls(sub, view, train=False, eval_mode=eval_mode, seed=seed)
    else:
        ds = cls(sub, view, seed=seed, **ds_kw)
    ld = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=workers,
                    worker_init_fn=worker_init_fn)
    dlist, ylist, rlist = [], [], []
    with torch.no_grad():
        for b in ld:
            x = b["x"].to(device)
            zs = [m(x).to(torch.float32) for m in models]
            dlist.append(torch.stack([binary_margin(z) for z in zs],
                                     dim=1).cpu())
            ylist.append(b["y"])
            rlist.append(b["row_id"])
    D = torch.cat(dlist).numpy()
    y = torch.cat(ylist).numpy()
    r = torch.cat(rlist).numpy()
    assert (r == np.arange(len(r))).all(), "row order canary tripped"
    return D, y, r


def _view(args):
    from data import ViewConfig
    return ViewConfig(image_size=args.image_size)


# ---------------------------------------------------------------------------
# mode: slices
# ---------------------------------------------------------------------------

def _acc(d: np.ndarray, y: np.ndarray) -> float:
    return float(((d > 0).astype(int) == y).mean()) if len(y) else float("nan")


def mode_slices(df: pd.DataFrame, args) -> list[dict]:
    rows: list[dict] = []
    ctx = gpu_ctx(args)
    view = _view(args)
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    for split in splits:
        sub = df[df["split"].astype(str) == split].reset_index(drop=True)
        if len(sub) == 0:
            print(f"[slices] split {split!r} empty; skipped")
            continue
        D, y, _ = collect_rows(ctx["models"], sub, view, ctx["dev"],
                               args.batch_size, args.workers,
                               eval_mode=args.eval_view, seed=args.seed)
        d = D @ ctx["w"]
        p = _sigmoid(d)
        pred = (d > 0).astype(int)
        reals = y == 0

        # S1: codec nuisance response, label held fixed at real.
        if "file_format" in sub.columns and reals.sum() > 50:
            is_png = (sub["file_format"].astype(str).str.upper()
                      == "PNG").to_numpy(int)
            if len(np.unique(is_png[reals])) == 2:
                a = folded_auc(p[reals], is_png[reals])
                per_branch = {n: round(folded_auc(_sigmoid(D[reals, j]),
                                                  is_png[reals]), 4)
                              for j, n in enumerate(ctx["names"])}
                rows.append(make_row(
                    "slices", "S1_codec_nuisance", split, None,
                    int(reals.sum()),
                    {"auc_real_pfake_vs_png": a, "per_branch": per_branch},
                    bucket(a, TH["nuisance_auc"]), args,
                    headline="auc_real_pfake_vs_png",
                    extra={"branches": ctx["names"],
                           "steps": ctx["steps"]}))

        # S2: resolution nuisance -- margin tracks the resample factor?
        if "m_eval" in sub.columns:
            logm = np.log(np.clip(pd.to_numeric(sub["m_eval"],
                                                errors="coerce")
                                  .to_numpy(dtype=float), 1e-6, None))
            for lab, mask in (("real", reals), ("fake", ~reals)):
                if mask.sum() < 50:
                    continue
                rho = spearman(d[mask], logm[mask])
                per_branch = {n: round(spearman(D[mask, j], logm[mask]), 4)
                              for j, n in enumerate(ctx["names"])}
                rows.append(make_row(
                    "slices", "S2_resolution_rho", split, {"class": lab},
                    int(mask.sum()),
                    {"spearman_d_logm": rho, "abs_rho": abs(rho),
                     "per_branch": per_branch},
                    bucket(abs(rho), TH["margin_rho"]) if lab == "real"
                    else "ok",
                    args, headline="abs_rho"))

        # S3: per-category real/fake accuracy gap.
        if "category" in sub.columns:
            worst_gap, worst_cat, table = 0.0, "", {}
            for c, g in sub.groupby(sub["category"].astype(str)):
                m = sub.index.isin(g.index)
                ar = _acc(d[m & reals], y[m & reals])
                af = _acc(d[m & ~reals], y[m & ~reals])
                if math.isnan(ar) or math.isnan(af):
                    continue
                table[c] = {"acc_real": round(ar, 4), "acc_fake": round(af, 4),
                            "n": int(m.sum())}
                if abs(ar - af) > worst_gap:
                    worst_gap, worst_cat = abs(ar - af), c
            rows.append(make_row(
                "slices", "S3_category_gap", split, None, len(sub),
                {"worst_gap": worst_gap, "worst_category": worst_cat,
                 "table": table},
                bucket(worst_gap, TH["slice_gap"]), args,
                headline="worst_gap"))

        # S4: per-dataset accuracy, ledgered for regression tracking.
        per_ds = []
        for ds_name, g in sub.groupby(sub["dataset"].astype(str)):
            m = sub.index.isin(g.index)
            per_ds.append((ds_name, _acc(d[m], y[m]), int(m.sum())))
        per_ds.sort(key=lambda t: t[1])
        for ds_name, a, n_ds in per_ds:
            rows.append(make_row(
                "slices", "S4_dataset_acc", split, {"dataset": ds_name},
                n_ds, {"acc": a}, "ok", args, headline="acc",
                higher_is_bad=False))
        rows.append(make_row(
            "slices", "S4_worst_datasets", split, None, len(sub),
            {"bottom15": [{"dataset": t[0], "acc": round(t[1], 4),
                           "n": t[2]} for t in per_ds[:15]]},
            "ok", args))

        # S5: kind slice -- the semisynthetic deficit, regression-tracked.
        if "kind" in sub.columns:
            kacc = {k: _acc(d[(sub["kind"].astype(str) == k).to_numpy()],
                            y[(sub["kind"].astype(str) == k).to_numpy()])
                    for k in ("real", "synthetic", "semisynthetic")}
            gap = ((kacc.get("synthetic") or 0)
                   - (kacc.get("semisynthetic") or 0)) \
                if kacc.get("semisynthetic") is not None else float("nan")
            rows.append(make_row(
                "slices", "S5_kind", split, None, len(sub),
                {**{f"acc_{k}": v for k, v in kacc.items()},
                 "semisyn_deficit": gap},
                bucket(gap, TH["kind_gap"]), args,
                headline="semisyn_deficit"))

        # S6: pil_mode nuisance on reals.
        if "pil_mode" in sub.columns and reals.sum() > 50:
            non_rgb = (sub["pil_mode"].astype(str) != "RGB").to_numpy(int)
            if len(np.unique(non_rgb[reals])) == 2:
                a = folded_auc(p[reals], non_rgb[reals])
                rows.append(make_row(
                    "slices", "S6_pilmode_nuisance", split, None,
                    int(reals.sum()), {"auc_real_pfake_vs_nonrgb": a},
                    bucket(a, TH["nuisance_auc"]), args,
                    headline="auc_real_pfake_vs_nonrgb"))

        # S8: are the fused model's errors metadata-predictable?
        wrong = (pred != y).astype(int)
        if wrong.sum() >= 30 and len(sub) - wrong.sum() >= 30:
            Xs, _ = featurize(sub)
            a, oof_w = oof_probe_auc(Xs, wrong, blocked_folds(sub))
            vw, exw = probe_verdict(a, oof_w, TH["error_pred_auc"])
            rows.append(make_row(
                "slices", "S8_error_predictability", split, None, len(sub),
                {"auc_metadata_vs_wrong": a, "error_rate": wrong.mean(),
                 **exw},
                vw, args, headline="auc_metadata_vs_wrong"))
    return rows


# ---------------------------------------------------------------------------
# mode: interventions
# ---------------------------------------------------------------------------

def _interventions():
    """name -> [(setting_key, fn)] with fn(img: PIL.Image, i: int) -> Image.
    Every fn is deterministic given (image bytes, i) and label-blind by
    construction (verify_shortcuts asserts no fn signature mentions labels)."""
    import io

    import cv2
    from PIL import Image, ImageFilter

    def _jpeg(img, q, subsampling=2):
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=q,
                                subsampling=subsampling)
        buf.seek(0)
        with Image.open(buf) as out:
            return out.convert("RGB")

    def _webp(img, q):
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="WEBP", quality=q)
        buf.seek(0)
        with Image.open(buf) as out:
            return out.convert("RGB")

    def _mk_jpeg(q):
        return lambda img, i: _jpeg(img, q)

    def _mk_webp(q):
        return lambda img, i: _webp(img, q)

    def _double_aligned(img, i):
        return _jpeg(_jpeg(img, 90), 75)

    def _double_shifted(img, i):
        out = _jpeg(img, 90)
        w, h = out.size
        if w > 20 and h > 20:
            out = out.crop((4, 4, w, h))
        return _jpeg(out, 75)

    def _gray(img, i):
        return img.convert("L").convert("RGB")

    def _mk_resize(target):
        def fn(img, i):
            a = np.asarray(img.convert("RGB"))
            h, w = a.shape[:2]
            m = min(h, w)
            if m == target:
                return img
            s = target / m
            interp = cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC
            nw = max(16, int(round(w * s)))
            nh = max(16, int(round(h * s)))
            return Image.fromarray(cv2.resize(a, (nw, nh),
                                              interpolation=interp))
        return fn

    def _mk_crop(frac):
        def fn(img, i):
            w, h = img.size
            cw, ch = max(16, int(w * frac)), max(16, int(h * frac))
            x0, y0 = (w - cw) // 2, (h - ch) // 2
            return img.crop((x0, y0, x0 + cw, y0 + ch))
        return fn

    def _mk_blur(sigma):
        return lambda img, i: img.filter(ImageFilter.GaussianBlur(sigma))

    def _highpass(img, i):
        a = np.asarray(img.convert("RGB"), dtype=np.float32)
        b = np.asarray(img.convert("RGB")
                       .filter(ImageFilter.GaussianBlur(2.0)),
                       dtype=np.float32)
        return Image.fromarray(
            np.clip(a - b + 128.0, 0, 255).astype(np.uint8))

    def _noise(img, i):
        rng = np.random.default_rng(34 * 1_000_003 + i)
        a = np.asarray(img.convert("RGB"), dtype=np.float32)
        a = a + rng.normal(0.0, 2.0, size=a.shape)
        return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))

    return {
        "jpeg_norm": [({"q": 85}, _mk_jpeg(85))],
        "jpeg_sweep": [({"q": q}, _mk_jpeg(q)) for q in (95, 75, 60, 45, 30)],
        "webp": [({"q": 80}, _mk_webp(80))],
        "double_jpeg": [({"kind": "aligned"}, _double_aligned),
                        ({"kind": "shifted"}, _double_shifted)],
        "gray": [({}, _gray)],
        "resize_sweep": [({"min_side": t}, _mk_resize(t))
                         for t in (192, 384, 768)],
        "crop_sweep": [({"frac": f}, _mk_crop(f)) for f in (0.9, 0.75, 0.5)],
        "blur": [({"sigma": s}, _mk_blur(s)) for s in (0.5, 1.0, 2.0)],
        "highpass": [({"sigma": 2.0}, _highpass)],
        "noise": [({"sigma255": 2.0}, _noise)],
    }


def mode_interventions(df: pd.DataFrame, args) -> list[dict]:
    rows: list[dict] = []
    ctx = gpu_ctx(args)
    view = _view(args)
    InterventionDataset, _ = _dataset_classes()
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    pool = df[df["split"].astype(str).isin(splits)]
    if len(pool) == 0:
        print("[interventions] no rows in the requested splits; skipped")
        return rows
    sample = stratified_sample(pool, args.rows_cap or 4000, args.seed)
    D0, y, _ = collect_rows(ctx["models"], sample, view, ctx["dev"],
                            args.batch_size, args.workers, seed=args.seed)
    d0 = D0 @ ctx["w"]
    p0 = _sigmoid(d0)
    pred0 = (d0 > 0).astype(int)
    reals = y == 0

    registry = _interventions()
    wanted = [n.strip() for n in args.interventions.split(",") if n.strip()]
    unknown = sorted(set(wanted) - set(registry))
    if unknown:
        raise SystemExit(f"unknown interventions {unknown}; "
                         f"registry has {sorted(registry)}")

    scored: list[tuple[float, dict, np.ndarray]] = []
    for name in wanted:
        for key, fn in registry[name]:
            Dv, yv, _ = collect_rows(
                ctx["models"], sample, view, ctx["dev"], args.batch_size,
                args.workers, dataset_cls=InterventionDataset, fn=fn,
                seed=args.seed)
            assert (yv == y).all()
            dv = Dv @ ctx["w"]
            pv = _sigmoid(dv)
            predv = (dv > 0).astype(int)

            def _cls(mask):
                return {
                    "acc_before": _acc(d0[mask], y[mask]),
                    "acc_after": _acc(dv[mask], y[mask]),
                    "flip": float((predv[mask] != pred0[mask]).mean()),
                    "dp_mean": float((pv[mask] - p0[mask]).mean()),
                    "adp_mean": float(np.abs(pv[mask] - p0[mask]).mean()),
                }
            mr, mf = _cls(reals), _cls(~reals)
            metrics = {**{f"{k}_real": v for k, v in mr.items()},
                       **{f"{k}_fake": v for k, v in mf.items()}}
            verdict = bucket(mr["flip"], TH["flip_real"])
            if verdict == "ok" and bucket(mr["adp_mean"],
                                          TH["dp_real"]) != "ok":
                verdict = "warn"
            # Fake-evidence fragility is expected under blur/highpass; only
            # codec/geometry interventions treat a fake collapse as a flag.
            if name in ("jpeg_norm", "webp", "double_jpeg", "gray",
                        "crop_sweep") and verdict == "ok":
                drop = mf["acc_before"] - mf["acc_after"]
                if bucket(drop, TH["fake_drop"]) != "ok":
                    verdict = "warn"
            rows.append(make_row(
                "interventions", name, ",".join(splits), key, len(sample),
                metrics, verdict, args, headline="adp_mean_real",
                extra={"branches": ctx["names"], "steps": ctx["steps"]}))
            scored.append((mr["flip"] + mr["adp_mean"],
                           {"name": name, **key}, Dv))

    # I11: per-branch attribution for the 3 most real-moving settings.
    scored.sort(key=lambda t: -t[0])
    for badness, key, Dv in scored[:3]:
        per_branch = {}
        for j, n in enumerate(ctx["names"]):
            f0 = (D0[reals, j] > 0)
            fv = (Dv[reals, j] > 0)
            per_branch[n] = {
                "flip_real": round(float((f0 != fv).mean()), 4),
                "adp_real": round(float(np.abs(
                    _sigmoid(Dv[reals, j]) - _sigmoid(D0[reals, j])).mean()),
                    4)}
        rows.append(make_row(
            "interventions", "I11_branch_attribution", ",".join(splits),
            key, int(reals.sum()), {"per_branch": per_branch}, "ok", args))
    return rows


# ---------------------------------------------------------------------------
# mode: pairs
# ---------------------------------------------------------------------------

def mode_pairs(df: pd.DataFrame, args) -> list[dict]:
    rows: list[dict] = []
    ctx = gpu_ctx(args)
    view = _view(args)
    from data import clique_pairs

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    pool = df[df["split"].astype(str).isin(splits)].reset_index(drop=True)
    if len(pool) == 0:
        print("[pairs] no rows in the requested splits; skipped")
        return rows
    pairs = clique_pairs(pool)
    if not pairs:
        print("[pairs] no mixed-label split_units in the requested splits")
        return rows

    cap = args.pairs_cap
    rng = np.random.default_rng(args.seed)
    keep_idx: list[int] = []
    unit_of: dict[int, str] = {}
    for unit, (neg, pos) in sorted(pairs.items()):
        for side in (neg, pos):
            take = side if len(side) <= cap else \
                np.sort(rng.choice(side, size=cap, replace=False))
            for i in take:
                keep_idx.append(int(i))
                unit_of[int(i)] = unit
    clique_rows = pool.loc[sorted(set(keep_idx))]

    in_clique_units = set(pool.loc[list(unit_of), "split_unit"].astype(str)) \
        if "split_unit" in pool.columns else set()
    out_pool = pool[~pool["split_unit"].astype(str).isin(in_clique_units)] \
        if "split_unit" in pool.columns else pool.drop(index=clique_rows.index)
    out_sample = stratified_sample(out_pool, 2000, args.seed)

    frame = pd.concat([clique_rows.assign(_clique=1),
                       out_sample.assign(_clique=0)]).reset_index(drop=True)
    D, y, _ = collect_rows(ctx["models"], frame, view, ctx["dev"],
                           args.batch_size, args.workers, seed=args.seed)
    d = D @ ctx["w"]
    p = _sigmoid(d)
    in_c = frame["_clique"].to_numpy() == 1
    unit = frame["split_unit"].astype(str).to_numpy() \
        if "split_unit" in frame.columns else np.array([""] * len(frame))

    # P1: within-clique fake accuracy vs non-clique fake accuracy.
    acc_fake_in = _acc(d[in_c & (y == 1)], y[in_c & (y == 1)])
    acc_fake_out = _acc(d[~in_c & (y == 1)], y[~in_c & (y == 1)])
    drop = (acc_fake_out - acc_fake_in) \
        if not (math.isnan(acc_fake_in) or math.isnan(acc_fake_out)) \
        else float("nan")
    rows.append(make_row(
        "pairs", "P1_clique_fake_acc", ",".join(splits), None,
        int(in_c.sum()),
        {"acc_fake_in_clique": acc_fake_in,
         "acc_fake_out_clique": acc_fake_out, "drop": drop},
        bucket(drop, TH["pairs_drop"]), args, headline="drop",
        extra={"branches": ctx["names"], "steps": ctx["steps"]}))

    # per-unit table + P2 confound correlation + P3 real FPR.
    global_fpr = float((p[~in_c & (y == 0)] > 0.5).mean()) \
        if (~in_c & (y == 0)).sum() else float("nan")
    gaps, png_deltas, res_deltas, fpr_flags = [], [], [], []
    for u in sorted(set(unit[in_c])):
        m = in_c & (unit == u)
        mr, mf = m & (y == 0), m & (y == 1)
        if mr.sum() == 0 or mf.sum() == 0:
            continue
        gap = float(d[mf].mean() - d[mr].mean())
        fr = frame[mr]
        ff = frame[mf]
        png_d = res_d = float("nan")
        if "file_format" in frame.columns:
            png_d = float((ff["file_format"].astype(str).str.upper()
                           == "PNG").mean()
                          - (fr["file_format"].astype(str).str.upper()
                             == "PNG").mean())
        if "min_side" in frame.columns:
            res_d = float(np.log(
                max(float(ff["min_side"].median()), 1.0)
                / max(float(fr["min_side"].median()), 1.0)))
        fpr = float((p[mr] > 0.5).mean())
        mult = fpr / global_fpr if global_fpr and global_fpr > 0 \
            else float("inf") if fpr > 0 else 0.0
        flagged = (mr.sum() >= 20 and fpr > 0.05
                   and mult > TH["clique_fpr_mult"]["warn"])
        gaps.append(gap)
        png_deltas.append(png_d)
        res_deltas.append(res_d)
        if flagged:
            fpr_flags.append({"unit": u, "fpr": round(fpr, 4),
                              "n_real": int(mr.sum())})
        rows.append(make_row(
            "pairs", "P_unit", ",".join(splits), {"unit": u},
            int(m.sum()),
            {"margin_gap": gap, "acc_real": _acc(d[mr], y[mr]),
             "acc_fake": _acc(d[mf], y[mf]), "real_fpr": fpr,
             "png_share_delta": png_d, "log_res_ratio": res_d},
            "warn" if flagged else "ok", args, headline="acc_fake",
            higher_is_bad=False))

    if len(gaps) >= 4:
        rho_png = spearman(png_deltas, gaps)
        rho_res = spearman(res_deltas, gaps)
        worst = max((abs(r) for r in (rho_png, rho_res)
                     if not math.isnan(r)), default=float("nan"))
        rows.append(make_row(
            "pairs", "P2_confounded_cliques", ",".join(splits), None,
            len(gaps),
            {"rho_gap_vs_png_delta": rho_png,
             "rho_gap_vs_res_delta": rho_res},
            bucket(worst, TH["pairs_confound_rho"]), args))
    rows.append(make_row(
        "pairs", "P3_clique_real_fpr", ",".join(splits), None,
        int(in_c.sum()),
        {"global_real_fpr": global_fpr, "flagged_units": fpr_flags},
        "warn" if fpr_flags else "ok", args))
    return rows


# ---------------------------------------------------------------------------
# report / main
# ---------------------------------------------------------------------------

def _regress(rows: list[dict], prev: dict[str, dict]) -> None:
    """Stamp baseline flags and escalate ok->warn on regressions vs the last
    ledger row with the same key. Never downgrades, never hards."""
    for r in rows:
        pr = prev.get(r["key_str"])
        r["baseline"] = pr is None
        h = r.get("headline")
        if (pr is None or not h or h not in r["metrics"]
                or h not in (pr.get("metrics") or {})):
            continue
        try:
            delta = float(r["metrics"][h]) - float(pr["metrics"][h])
        except (TypeError, ValueError):
            continue
        r["delta"] = delta
        tol = TH["regress_auc"]["warn"] if "auc" in h or "rho" in h \
            else TH["regress_acc"]["warn"]
        worse = delta > tol if r.get("higher_is_bad", True) else delta < -tol
        if worse and r["verdict"] == "ok":
            r["verdict"] = "warn"
            r["regressed"] = True


def print_report(rows: list[dict]) -> None:
    order = {"hard": 0, "warn": 1, "ok": 2}
    print("=" * 78)
    for mode in ("metadata", "tensors", "slices", "pairs", "interventions"):
        sect = [r for r in rows if r["mode"] == mode]
        if not sect:
            continue
        print(f"-- {mode} " + "-" * (74 - len(mode)))
        for r in sorted(sect, key=lambda r: (order[r["verdict"]],
                                             r["test"], str(r["key"]))):
            if r["test"] == "S4_dataset_acc":     # ledgered, printed via
                continue                          # the bottom-15 summary
            h = r.get("headline")
            val = r["metrics"].get(h) if h else None
            val_s = f"{h}={val:.4f}" if isinstance(val, float) else ""
            delta = r.get("delta")
            d_s = f" d-prev={delta:+.4f}" if isinstance(delta, float) else ""
            b_s = " (baseline)" if r.get("baseline") else ""
            key_s = ",".join(f"{k}={v}" for k, v in r["key"].items())
            print(f"[{r['verdict']:>4}] {r['test']:<28} {r['split']:<18} "
                  f"{key_s:<22} n={r['n']:<7} {val_s}{d_s}{b_s}")
            for k in ("offenders", "worst_category", "bottom15",
                      "flagged_units", "top", "live_shortcut"):
                if k in r["metrics"] and r["metrics"][k]:
                    print(f"       {k}: {r['metrics'][k]}")
    n = {v: sum(1 for r in rows if r["verdict"] == v)
         for v in ("hard", "warn", "ok")}
    print("=" * 78)
    print(f"verdicts: {n['hard']} hard, {n['warn']} warn, {n['ok']} ok "
          f"({sum(1 for r in rows if r.get('baseline'))} baseline rows)")


MODE_ORDER = ("metadata", "tensors", "slices", "pairs", "interventions")
MODE_FNS = {"metadata": mode_metadata, "tensors": mode_tensors,
            "slices": mode_slices, "pairs": mode_pairs,
            "interventions": mode_interventions}


def load_manifest(args) -> pd.DataFrame:
    path = Path(args.manifest)
    if not path.exists():
        raise SystemExit(f"{path} missing -- run build_manifest.py first "
                         f"(or pass --manifest a copied parquet)")
    df = pd.read_parquet(path)
    if args.data_root:
        root = Path(args.data_root)

        def reroot(p: str) -> str:
            parts = Path(p).parts
            if "datasets" in parts:
                i = len(parts) - 1 - parts[::-1].index("datasets")
                return str(root.joinpath(*parts[i:]))
            return p
        df = df.assign(path=df["path"].astype(str).map(reroot))
    return df


def main() -> None:
    ap = argparse.ArgumentParser(
        description="shortcut-learning audit (see module docstring)")
    ap.add_argument("--mode", default="metadata",
                    help="comma list of metadata,tensors,slices,pairs,"
                         "interventions or 'all'")
    ap.add_argument("--manifest", default=str(HERE / "data" /
                                              "manifest.parquet"))
    ap.add_argument("--data-root", default="",
                    help="re-root manifest paths onto this cache dir "
                         "(only needed when the manifest moved machines)")
    ap.add_argument("--runs-dir", default=str(HERE / "runs" / "v7"))
    ap.add_argument("--branches", default="dinov3,clip,convnext,eva,dct")
    ap.add_argument("--overrides", default=str(HERE / "overrides.yaml"))
    ap.add_argument("--config", default=str(HERE / "config.yaml"),
                    help="source of the laundering params for --mode tensors")
    ap.add_argument("--splits", default="val_id,val_xgen,val_stress",
                    help="manifest splits for the model-conditional modes")
    ap.add_argument("--eval-view", default="deploy",
                    choices=["deploy", "robust"])
    ap.add_argument("--image-size", type=int, default=384)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=34)
    ap.add_argument("--rows-cap", type=int, default=0,
                    help="override the per-mode sample caps "
                         "(tensors 6000, interventions 4000)")
    ap.add_argument("--pairs-cap", type=int, default=60,
                    help="max rows per clique side in --mode pairs")
    ap.add_argument("--interventions",
                    default="jpeg_norm,jpeg_sweep,webp,double_jpeg,gray,"
                            "resize_sweep,crop_sweep,blur,highpass,noise")
    ap.add_argument("--tag", default="")
    ap.add_argument("--history", default=str(HERE /
                                             "shortcut_history.jsonl"))
    ap.add_argument("--json-out", default="")
    ap.add_argument("--strict", action="store_true",
                    help="exit 1 on any hard verdict (default: report only)")
    args = ap.parse_args()
    args.rows_cap = args.rows_cap or None

    modes = [m.strip() for m in args.mode.split(",") if m.strip()]
    if "all" in modes:
        modes = list(MODE_ORDER)
    unknown = sorted(set(modes) - set(MODE_FNS))
    if unknown:
        raise SystemExit(f"unknown modes {unknown}; "
                         f"choose from {list(MODE_ORDER)} or 'all'")
    modes = [m for m in MODE_ORDER if m in modes]

    df = load_manifest(args)
    print(f"[audit] manifest {args.manifest}: {len(df):,} rows; "
          f"modes: {', '.join(modes)}")
    prev = last_history(Path(args.history))

    rows: list[dict] = []
    for m in modes:
        print(f"[audit] running mode {m} ...")
        rows.extend(MODE_FNS[m](df, args))

    for r in rows:
        r["manifest_rows"] = len(df)
    _regress(rows, prev)
    print_report(rows)
    append_ledger(Path(args.history), rows)
    print(f"[audit] appended {len(rows)} rows -> {args.history}")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(rows, indent=2,
                                                  default=str))
        print(f"[audit] wrote {args.json_out}")
    n_hard = sum(1 for r in rows if r["verdict"] == "hard")
    if args.strict and n_hard:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
