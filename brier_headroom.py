"""Measure the NO-RETRAINING Brier headroom of a deployed model, and find the
best shippable temperature if any headroom exists.

    # Stage 1 (no GPU, instant): analytic bounds from the dashboard numbers
    python brier_headroom.py --dashboard-mcc 0.79 --dashboard-brier 0.098

    # Stage 2 (GPU box): empirical T sweep anchored to the dashboard
    python brier_headroom.py --dashboard-mcc 0.79 --dashboard-brier 0.098 \
        --current-temperature 0.7058 --runs-dir runs/v7 --branches dinov3 \
        --manifest data/manifest.parquet --ood-root ood_pool/

Why the bounds work (the ledger this codebase accumulated):
  * Any monotone recalibration (temperature, Platt-a) preserves argmax, so
    MCC is FIXED. Only Brier moves.
  * A model whose confidence carries no per-sample error information cannot
    beat Brier = a(1-a) at accuracy a ("the floor"). The distance
    (observed - floor) is therefore the CEILING on what any global
    recalibration can recover; measured margin-AUC ~0.5 on OOD data means
    the realistic ceiling IS the floor gap.
  * The deployed T was fitted on local pools (~high accuracy); the cloud mix
    has lower accuracy, so the single-T optimum can differ. Stage 2 finds it
    empirically: local val_id margins proxy the public/gasstation pools, the
    OOD-pool margins proxy the hidden pool, blended 0.7/0.3 and ANCHORED so
    the blend at the current T reproduces the observed dashboard Brier.
  * The verdict never recommends a T whose predicted gain is inside the
    anchor's own error bar, and prints the export command only when it is.

Reuses: branch loading + folder margins from measure_margins.py, the eval
collection from export.py, sigmoid/score helpers from calibrate.py, and the
existing `export.py --override-temperature` path for shipping the result.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


# Local copies of two tiny calibrate.py helpers, so stage 1 (analytic) runs
# with numpy alone -- importing calibrate pulls the whole gasbench bridge
# (cv2/scipy/torch), which the orchestration VM does not have.
def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))


def class_balance_weights(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y).astype(int)
    w = np.ones(len(y), dtype=float)
    for c in (0, 1):
        m = y == c
        if m.any():
            w[m] = 0.5 / m.sum()
    return w * len(y) / max(w.sum(), 1e-12)


def sn34(mcc: float, brier: float) -> float:
    mn = max(0.0, (mcc + 1.0) / 2.0) ** 1.2
    bn = max(0.0, (0.25 - brier) / 0.25) ** 1.8
    return float((mn * bn) ** 0.5)


def analytic(mcc: float, brier: float, mcc_aug: float | None = None,
             brier_aug: float | None = None, aug_weight: float = 0.2) -> dict:
    """Stage 1: what the dashboard numbers alone pin down."""
    acc = (1.0 + mcc) / 2.0            # balanced accuracy implied by MCC
    floor = acc * (1.0 - acc)          # single-confidence calibration floor
    gap = max(0.0, brier - floor)
    now = sn34(mcc, brier)
    at_floor = sn34(mcc, floor)
    out = {
        "implied_accuracy": round(acc, 4),
        "brier_observed": brier,
        "brier_floor": round(floor, 4),
        "recoverable_gap": round(gap, 4),
        "sn34_now": round(now, 4),
        "sn34_at_floor_CEILING": round(at_floor, 4),
        "max_gain_without_training": round(at_floor - now, 4),
    }
    if mcc_aug is not None and brier_aug is not None:
        acc_a = (1.0 + mcc_aug) / 2.0
        floor_a = acc_a * (1.0 - acc_a)
        blend_now = (1 - aug_weight) * now + aug_weight * sn34(mcc_aug, brier_aug)
        blend_ceiling = ((1 - aug_weight) * at_floor
                         + aug_weight * sn34(mcc_aug, floor_a))
        out["aug"] = {
            "brier_observed": brier_aug, "brier_floor": round(floor_a, 4),
            "recoverable_gap": round(max(0.0, brier_aug - floor_a), 4),
        }
        out["blended_sn34_now"] = round(blend_now, 4)
        out["blended_sn34_ceiling"] = round(blend_ceiling, 4)
        out["blended_max_gain"] = round(blend_ceiling - blend_now, 4)
    return out


def pool_brier(d: np.ndarray, y: np.ndarray, T: float) -> float:
    w = class_balance_weights(y)
    return float(np.average((_sigmoid(d / T) - y) ** 2, weights=w))


def empirical(dash: dict, T_cur: float, d_val: np.ndarray, y_val: np.ndarray,
              d_ood: np.ndarray, y_ood: np.ndarray,
              hidden_weight: float = 0.3,
              t_lo: float = 0.2, t_hi: float = 6.0, n_t: int = 121) -> dict:
    """Stage 2: anchored single-T sweep over the modeled cloud mix.

    blend(T) = (1-hw) * Brier_val(T) + hw * Brier_ood(T), anchored additively
    so blend(T_cur) equals the observed dashboard Brier. The anchor residual
    is also the honesty bar: gains smaller than |residual|/2 are noise.
    """
    Ts = np.geomspace(t_lo, t_hi, n_t)
    blend = np.array([(1 - hidden_weight) * pool_brier(d_val, y_val, T)
                      + hidden_weight * pool_brier(d_ood, y_ood, T)
                      for T in Ts])
    b_cur = ((1 - hidden_weight) * pool_brier(d_val, y_val, T_cur)
             + hidden_weight * pool_brier(d_ood, y_ood, T_cur))
    anchor = dash["brier_observed"] - b_cur       # model-vs-dashboard residual
    pred = blend + anchor
    k = int(np.argmin(pred))
    T_star = float(Ts[k])
    gain_brier = float(pred[np.argmin(np.abs(Ts - T_cur))] - pred[k])
    mcc = dash["_mcc"]
    sn_cur = sn34(mcc, dash["brier_observed"])
    sn_star = sn34(mcc, max(0.0, float(pred[k])))
    noise_bar = abs(anchor) / 2.0
    return {
        "T_current": T_cur, "T_star": round(T_star, 4),
        "anchor_residual": round(anchor, 4),
        "predicted_brier_at_T_star": round(float(pred[k]), 4),
        "predicted_brier_gain": round(gain_brier, 4),
        "predicted_sn34_gain": round(sn_star - sn_cur, 4),
        "noise_bar_brier": round(noise_bar, 4),
        "confident": bool(gain_brier > max(noise_bar, 0.002)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dashboard-mcc", type=float, required=True)
    ap.add_argument("--dashboard-brier", type=float, required=True)
    ap.add_argument("--dashboard-mcc-aug", type=float, default=None)
    ap.add_argument("--dashboard-brier-aug", type=float, default=None)
    # stage 2 (all optional; omit to run the analytic stage only)
    ap.add_argument("--current-temperature", type=float, default=None,
                    help="T0 baked into the submission (model_config.yaml "
                         "metadata.temperature)")
    ap.add_argument("--runs-dir", default=None)
    ap.add_argument("--branches", default="dinov3")
    ap.add_argument("--manifest", default="data/manifest.parquet")
    ap.add_argument("--ood-root", default=None)
    ap.add_argument("--hidden-weight", type=float, default=0.3)
    ap.add_argument("--calib-limit", type=int, default=8000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="brier_headroom_report.json")
    args = ap.parse_args()

    rep = {"analytic": analytic(args.dashboard_mcc, args.dashboard_brier,
                                args.dashboard_mcc_aug,
                                args.dashboard_brier_aug)}
    a = rep["analytic"]
    print(json.dumps(a, indent=2))
    print(f"\n[stage 1] ceiling on ANY no-training calibration gain: "
          f"sn34 +{a['max_gain_without_training']:.4f} "
          f"(brier {a['brier_observed']:.4f} -> floor {a['brier_floor']:.4f})")

    if args.runs_dir and args.current_temperature is not None:
        import torch
        from branches import resolve, fusion_weights
        from model import BranchModel
        from measure_margins import folder_margins

        dev = "cuda" if torch.cuda.is_available() else "cpu"
        names = [n.strip() for n in args.branches.split(",") if n.strip()]
        specs = resolve(names)
        weights = np.asarray(fusion_weights(specs))
        models = []
        for spec in specs:
            ck = torch.load(Path(args.runs_dir) / spec.name / "best.pt",
                            map_location="cpu", weights_only=False)
            m = BranchModel(spec, lora_r=ck["config"]["lora"]["r"],
                            lora_alpha=ck["config"]["lora"]["alpha"],
                            lora_dropout=0.0, gradient_checkpointing=False)
            m.load_state_dict(ck["state"], strict=True)
            m.eval(); m.merge_and_strip()
            models.append(m.to(dev))
            print(f"[load] {spec.name} @ step {ck.get('step')}")

        import pandas as pd
        from data import ManifestDataset, ViewConfig
        from export import Ensemble, collect_branch_margins
        df = pd.read_parquet(args.manifest)
        sub = df[df.split == "val_id"].reset_index(drop=True)
        if args.calib_limit and len(sub) > args.calib_limit:
            sub = sub.sample(n=args.calib_limit, random_state=34) \
                     .reset_index(drop=True)
        ens = Ensemble(models, list(weights)).to(dev).eval()
        D_val, _, y_val, _ = collect_branch_margins(
            ens, sub, ViewConfig(image_size=384), dev,
            args.batch_size, args.workers, "deploy")
        d_val = (D_val * weights).sum(axis=1)

        if not args.ood_root:
            raise SystemExit("--ood-root required for stage 2 (the hidden-"
                             "pool proxy); rebuild the pool's REAL side if "
                             "stale -- fakes now in-train are excluded by "
                             "using the real side only when the fake side "
                             "predates the current manifest")
        D_ood, y_ood, _ = folder_margins(Path(args.ood_root), models, 384,
                                         dev, args.batch_size)
        d_ood = (D_ood * weights).sum(axis=1)

        dash = {"brier_observed": args.dashboard_brier,
                "_mcc": args.dashboard_mcc}
        emp = empirical(dash, args.current_temperature,
                        d_val, np.asarray(y_val), d_ood, np.asarray(y_ood),
                        hidden_weight=args.hidden_weight)
        rep["empirical"] = emp
        print(json.dumps(emp, indent=2))
        print("\n=== VERDICT ===")
        if emp["confident"]:
            print(f"headroom IS measurable: re-export with\n"
                  f"  python export.py --branches {args.branches} "
                  f"--override-temperature {emp['T_star']:.4f}\n"
                  f"predicted: brier -{emp['predicted_brier_gain']:.4f}, "
                  f"sn34 +{emp['predicted_sn34_gain']:.4f} "
                  f"(noise bar {emp['noise_bar_brier']:.4f})")
        else:
            print(f"NO shippable headroom: predicted gain "
                  f"{emp['predicted_brier_gain']:.4f} brier is inside the "
                  f"noise bar ({emp['noise_bar_brier']:.4f}) or below 0.002. "
                  f"The gap to the sn34 ceiling "
                  f"(+{a['max_gain_without_training']:.4f}) is accuracy-"
                  f"shaped, not calibration-shaped -- it needs training data.")
    Path(args.out).write_text(json.dumps(rep, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
