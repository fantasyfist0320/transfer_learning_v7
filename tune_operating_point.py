#!/usr/bin/env python3
"""Tune the shipped operating point (T multiplier, p_min/p_max clamp) on a
gasbench records.parquet, under assumed hidden-inversion scenarios.

Why: r22 was lost on confident inversions -- hidden holdout pools scored
p_fake~0.03 (fakes) / ~0.97 (reals), each error paying ~(1)^2 Brier at
holdout weight with the score's Brier exponent 1.8. Local pools are
below-floor calibrated (brier_headroom.py verdict), so any insurance
(softening T, clamping p) COSTS local Brier; whether it pays depends on how
much hidden inverted mass you assume. This tool makes that trade explicit:
it re-scores the model's own benchmark records under a grid of
(t_mult, p_min, p_max) x scenario, using the competition's exact weighted
scoring (provenance composition public/holdout/gasstation, per-sample class
weights derived exactly like gasbench recording.derive_provenance_weights,
sn34 = sqrt(mcc_norm^1.2 * brier_norm^1.8), 0.8/0.2 base/aug blend).

Key structural fact (proved in verify.py's factorized-export check): with
0 < p_min < 0.5 < p_max < 1, neither the T multiplier nor the clamp can flip
a 0.5-threshold decision, so MCC is CONSTANT across the whole sweep and the
optimization is a pure weighted-Brier minimization. sn34 is still reported,
via the constant MCC.

Transform exactness: recorded p_fake = sigmoid(dbar / T_eff_ship). Applying
p' = sigmoid(logit(p) / t_mult) is byte-equivalent to shipping
temperature*t_mult (the hinge's T_eff scales proportionally with T0), so a
chosen t_mult multiplies the exported temperature; p_min/p_max ship as the
new buffers: export.py --p-min/--p-max.

Scenarios: hidden pools are SIMULATED (locally, released 34data classify as
"public"; real holdouts are unknowable). The inverted-fake pool's p_fake
shape is sampled from a real measured inversion if you have one --
--inversion-source pointing at the no-degradation experiment's records
(gemini31 rows: the actual shape of a zero-signal generator's scores) --
else a default Beta concentrated near 0.03. Hidden reals mirror it at 1-p.

Usage (training venv, after the ship benchmark):
    python tune_operating_point.py \
        --records ../gasbench/results/image_<ts>/records.parquet \
        --results-json ../gasbench/results/image_<ts>/results.json \
        --inversion-source <clean-run records.parquet>

Reads nothing from the network; writes --out (default tune_report.json).
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

RNG = np.random.default_rng(34)
EPS = 1e-7          # gasbench clips probs to [1e-7, 1-1e-7] before Brier


# ---------------------------------------------------------------------------
# Scoring (mirrors gasbench metrics.py / recording.py; parity-checked)
# ---------------------------------------------------------------------------

def classify_provenance(names: pd.Series) -> np.ndarray:
    """gasbench recording.py:365-377, verbatim semantics."""
    low = names.astype(str).str.lower()
    out = np.full(len(names), "public", dtype=object)
    out[low.str.contains("gasstation").to_numpy()] = "gasstation"
    out[low.str.contains("-holdout-").to_numpy()] = "holdout"
    return out


def provenance_weights(classes: np.ndarray, targets: dict[str, float]) -> np.ndarray:
    """gasbench recording.py:380-411: per-class w = share_c * N / n_c, absent
    classes dropped and remaining target shares renormalized."""
    present = [c for c in ("public", "holdout", "gasstation")
               if (classes == c).sum() > 0 and targets.get(c, 0.0) > 0]
    tot = sum(targets[c] for c in present)
    if tot <= 0:
        return np.ones(len(classes))
    w = np.ones(len(classes))
    n = len(classes)
    for c in present:
        mask = classes == c
        w[mask] = (targets[c] / tot) * n / mask.sum()
    return w


def weighted_mcc(y: np.ndarray, pred: np.ndarray, w: np.ndarray) -> float:
    tp = w[(pred == 1) & (y == 1)].sum(); tn = w[(pred == 0) & (y == 0)].sum()
    fp = w[(pred == 1) & (y == 0)].sum(); fn = w[(pred == 0) & (y == 1)].sum()
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return float((tp * tn - fp * fn) / den) if den > 0 else 0.0


def sn34(mcc: float, brier: float, alpha=1.2, beta=1.8) -> float:
    mn = max(0.0, (mcc + 1.0) / 2.0) ** alpha
    bs = max(0.0, (0.25 - brier) / 0.25) ** beta
    return float(np.sqrt(mn * bs))


def pool_score(y, p, w):
    p = np.clip(p, EPS, 1.0 - EPS)
    brier = float(np.average((p - y) ** 2, weights=w))
    mcc = weighted_mcc(y, (p > 0.5).astype(int), w)
    return mcc, brier


# ---------------------------------------------------------------------------
# Records ingestion + scenario pools
# ---------------------------------------------------------------------------

def load_records(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df = df[df["probs"].notna()].copy()
    df["p_fake"] = df["probs"].apply(lambda pr: 1.0 - float(pr[0]))
    df["y"] = (df["label"].astype(int) > 0).astype(int)   # semi scores as fake
    df["aug"] = df["aug_pass"].fillna(False).astype(bool) \
        if "aug_pass" in df.columns else False
    return df[["dataset_name", "p_fake", "y", "aug"]]


def inversion_shape(source: str | None, substr: str, n: int) -> np.ndarray:
    """p_fake values for n simulated hidden-inverted FAKES."""
    if source:
        src = load_records(source)
        pool = src[src.dataset_name.astype(str).str.contains(substr)
                   & (src.y == 1) & (~src.aug)]["p_fake"].to_numpy()
        if len(pool):
            return RNG.choice(pool, size=n, replace=True)
        print(f"[warn] --inversion-source has no rows matching {substr!r}; "
              f"falling back to the synthetic shape")
    # Confidently-real fakes: mass near 0.03, thin tail toward 0.5
    return np.clip(RNG.beta(1.2, 20.0, size=n), 1e-4, 0.499)


def build_pool(df: pd.DataFrame, n_fake: int, n_real: int, frac_inv: float,
               inv_p: np.ndarray) -> pd.DataFrame:
    """Simulated hidden-holdout class (names carry -holdout- so the
    provenance classifier weights them as holdouts).

    gasbench's class-weight derivation makes the TOTAL holdout mass invariant
    to the class's sample count (weights are share*N/n_c), so pool SIZE is
    not a scenario dimension -- only the pool's score SHAPE is. The realistic
    shape is a mostly-well-predicted class with an inverted minority (r22:
    ~626 inverted of ~6-7k holdout samples ~= 10%), so `frac_inv` is the
    scenario knob: that fraction draws from the inversion shape, the rest
    resamples the model's own in-distribution scores per label.
    """
    base = df[~df["aug"]]
    ok_f = base[base.y == 1]["p_fake"].to_numpy()
    ok_r = base[base.y == 0]["p_fake"].to_numpy()
    ki_f = int(round(n_fake * frac_inv)); ki_r = int(round(n_real * frac_inv))
    p_f = np.concatenate([inv_p[:ki_f],
                          RNG.choice(ok_f, size=n_fake - ki_f, replace=True)])
    p_r = np.concatenate([1.0 - inv_p[ki_f:ki_f + ki_r],
                          RNG.choice(ok_r, size=n_real - ki_r, replace=True)])
    rows = [pd.DataFrame({"dataset_name": "synthetic-image-holdout-simf",
                          "p_fake": p_f, "y": 1, "aug": False}),
            pd.DataFrame({"dataset_name": "real-image-holdout-simr",
                          "p_fake": p_r, "y": 0, "aug": False})]
    return pd.concat([df] + rows, ignore_index=True)


# ---------------------------------------------------------------------------
# The sweep
# ---------------------------------------------------------------------------

def score_setting(base, aug, t_mult, p_min, p_max, aug_weight):
    """(weighted mcc, brier, blended sn34) with the transform applied."""
    out = {}
    for tag, (y, logit, w) in (("base", base), ("aug", aug)):
        if y is None:
            out[tag] = None
            continue
        p = 1.0 / (1.0 + np.exp(-logit / t_mult))
        p = np.clip(p, p_min, p_max)
        out[tag] = pool_score(y, p, w)
    m_b, b_b = out["base"]
    s_base = sn34(m_b, b_b)
    if out["aug"] is None:
        return m_b, b_b, s_base, s_base
    s_aug = sn34(*out["aug"])
    return m_b, b_b, s_base, (1 - aug_weight) * s_base + aug_weight * s_aug


def prep(df: pd.DataFrame, targets: dict[str, float]):
    """Freeze (y, logit(p), weight) triples per pool; weights derived on the
    BASE pool and joined onto aug rows by dataset (gasbench reuses base
    class weights for the aug pass, recording.py:598)."""
    classes = classify_provenance(df["dataset_name"])
    w_all = provenance_weights(classes[~df["aug"].to_numpy()],
                               targets)
    base = df[~df["aug"]].reset_index(drop=True)
    wb = w_all
    p = np.clip(base["p_fake"].to_numpy(), EPS, 1.0 - EPS)
    triple_b = (base["y"].to_numpy(), np.log(p / (1 - p)), wb)
    aug = df[df["aug"]].reset_index(drop=True)
    if not len(aug):
        return triple_b, (None, None, None)
    cw = pd.Series(wb, index=base["dataset_name"]).groupby(level=0).first()
    wa = aug["dataset_name"].map(cw).fillna(1.0).to_numpy()
    pa = np.clip(aug["p_fake"].to_numpy(), EPS, 1.0 - EPS)
    return triple_b, (aug["y"].to_numpy(), np.log(pa / (1 - pa)), wa)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--results-json", default=None,
                    help="the run's results.json: parity-check this script's "
                         "unweighted sn34 against gasbench's own before "
                         "trusting the sweep")
    ap.add_argument("--inversion-source", default=None,
                    help="records.parquet to sample the inverted-fake p_fake "
                         "shape from (e.g. the no-degradation run)")
    ap.add_argument("--inversion-datasets", default="gemini31")
    ap.add_argument("--composition", default="public=0.50,holdout=0.35,gasstation=0.15")
    ap.add_argument("--aug-weight", type=float, default=0.2)
    ap.add_argument("--n-hidden-fake", type=int, default=4000,
                    help="simulated holdout-class size (statistical "
                         "resolution only -- gasbench's class weights make "
                         "total holdout mass size-invariant)")
    ap.add_argument("--n-hidden-real", type=int, default=2000)
    ap.add_argument("--inverted-fracs", default="0,0.05,0.10,0.20",
                    help="scenario grid: fraction of the hidden class that "
                         "is confidently inverted (r22 realized ~0.10)")
    ap.add_argument("--max-local-cost", type=float, default=0.002,
                    help="max blended-sn34 loss tolerated in the inv=0 "
                         "scenario (clean holdout class, no inversions) -- "
                         "the insurance premium cap")
    ap.add_argument("--out", default="tune_report.json")
    args = ap.parse_args()

    targets = {k: float(v) for k, v in
               (kv.split("=") for kv in args.composition.split(","))}
    df = load_records(args.records)
    print(f"[records] {len(df):,} rows ({int(df.aug.sum()):,} aug) from "
          f"{args.records}")

    # -- parity gate: our scorer must reproduce gasbench's own number -------
    base_u = df[~df.aug]
    mcc_u, brier_u = pool_score(base_u.y.to_numpy(), base_u.p_fake.to_numpy(),
                                np.ones(len(base_u)))
    s_u = sn34(mcc_u, brier_u)
    print(f"[parity] unweighted base: mcc={mcc_u:.4f} brier={brier_u:.4f} "
          f"sn34={s_u:.4f}")
    if args.results_json:
        ref = json.load(open(args.results_json))

        def _find_ref(doc):
            """Walk nested dicts/lists; first key wins. base/binary are the
            blend-free scores this script's number is comparable to;
            sn34_score is last-resort (it is the 0.8/0.2 BLEND when an aug
            pass ran, so it only gets an advisory comparison)."""
            for key in ("base_sn34_score", "binary_sn34_score", "sn34_score"):
                stack = [doc]
                while stack:
                    o = stack.pop()
                    if isinstance(o, dict):
                        v = o.get(key)
                        if isinstance(v, (int, float)) and not isinstance(v, bool):
                            return key, float(v)
                        stack.extend(x for x in o.values()
                                     if isinstance(x, (dict, list)))
                    elif isinstance(o, list):
                        stack.extend(x for x in o
                                     if isinstance(x, (dict, list)))
            return None, None

        ref_key, ref_s = _find_ref(ref)
        if ref_s is None:
            print(f"[parity][warn] no sn34 key found anywhere in "
                  f"{args.results_json} (top-level keys: "
                  f"{sorted(ref)[:12]}) -- gasbench cross-check SKIPPED; "
                  f"the internal scorer still ran, eyeball its numbers "
                  f"against summary.txt before trusting the sweep")
        else:
            drift = abs(s_u - ref_s)
            print(f"[parity] gasbench {ref_key}: {ref_s:.4f} "
                  f"(drift {drift:.4f})")
            if ref_key == "sn34_score" and bool(df.aug.any()):
                print("[parity][warn] matched the BLENDED sn34_score (base "
                      "keys absent) -- drift vs this base-only number is "
                      "expected; advisory only")
            elif drift > 0.005:
                raise SystemExit("parity FAILED -- this script's scorer "
                                 "disagrees with gasbench; do not trust "
                                 "the sweep")

    inv = inversion_shape(args.inversion_source, args.inversion_datasets,
                          args.n_hidden_fake + args.n_hidden_real)
    # 1.0 appended explicitly: it is the identity/reference point and a
    # geomspace grid does not otherwise contain it exactly.
    t_grid = np.unique(np.concatenate([[1.0], np.geomspace(0.7, 2.5, 21)]))
    pmin_grid = [1e-6, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12]
    pmax_grid = [1.0 - 1e-6, 0.99, 0.98, 0.97, 0.95, 0.92]

    scen_mults = [float(x) for x in args.inverted_fracs.split(",")]
    tables = {}
    for sm in scen_mults:
        pool = build_pool(df, args.n_hidden_fake, args.n_hidden_real, sm, inv)
        b, a = prep(pool, targets)
        rows = []
        for tm in t_grid:
            for lo in pmin_grid:
                for hi in pmax_grid:
                    m, br, s_b, s_bl = score_setting(b, a, tm, lo, hi,
                                                     args.aug_weight)
                    rows.append((tm, lo, hi, m, br, s_bl))
        tab = pd.DataFrame(rows, columns=["t_mult", "p_min", "p_max",
                                          "mcc", "brier", "sn34_blended"])
        # MCC must be constant across the sweep -- the invariance proof.
        assert tab.mcc.nunique() == 1, "clamp/T flipped a decision?!"
        tables[sm] = tab.set_index(["t_mult", "p_min", "p_max"])
        ident = tables[sm].loc[(1.0, 1e-6, 1.0 - 1e-6), "sn34_blended"]
        best = tab.loc[tab.sn34_blended.idxmax()]
        print(f"[scenario inv={sm:.2f}] identity sn34 {ident:.4f}; best "
              f"{best.sn34_blended:.4f} at t={best.t_mult:.2f} "
              f"p_min={best.p_min:g} p_max={best.p_max:g}")

    # -- robust choice: maximize worst-case GAIN across scenarios, subject to
    #    the 0x premium cap ---------------------------------------------------
    idents = {sm: t.loc[(1.0, 1e-6, 1.0 - 1e-6), "sn34_blended"]
              for sm, t in tables.items()}
    worst = None
    for key in tables[scen_mults[0]].index:
        g = [tables[sm].loc[key, "sn34_blended"] - idents[sm]
             for sm in scen_mults]
        if 0.0 in scen_mults and \
                g[scen_mults.index(0.0)] < -args.max_local_cost:
            continue
        score = min(g)
        if worst is None or score > worst[1]:
            worst = (key, score, g)
    (tm, lo, hi), min_gain, g = worst
    print(f"\n[recommend] t_mult={tm:.3f} p_min={lo:g} p_max={hi:g} "
          f"(min gain {min_gain:+.4f} across scenarios; per-scenario "
          f"{[f'{x:+.4f}' for x in g]})")
    print(f"  ship via: export.py ... --p-min {lo:g} --p-max {hi:g}  and "
          f"multiply the fitted T0 by {tm:.3f} "
          f"(or accept the fitted T if t_mult~1)")

    json.dump({"recommend": {"t_mult": tm, "p_min": lo, "p_max": hi,
                             "min_gain": min_gain},
               "identity_sn34": {str(k): float(v) for k, v in idents.items()},
               "parity_unweighted_sn34": s_u},
              open(args.out, "w"), indent=2)
    print(f"[out] {args.out}")


if __name__ == "__main__":
    main()
