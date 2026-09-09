"""Measure whether the ensemble's margins rank its own errors on OOD data,
then fit the temperature on a local+OOD mixture.

    python measure_margins.py --ood-root ood_pool/ --runs-dir runs/v7 \
        --branches dinov3,clip,convnext,eva,dct --manifest data/manifest.parquet

`--ood-root` holds the out-of-distribution probe pool:

    ood_pool/real/*.jpg|png|webp     never-trained real images (aesthetic,
    ood_pool/fake/*.jpg|png|webp     stylized...) and frontier-generator fakes

Why this exists (2026-08-12): the dashboard showed Brier sitting exactly on
the a(1-a) constant-confidence floor -- the shipped probabilities carry no
information about which predictions are wrong. The friend's single-DINOv3
model beats its own floor by 26% because its errors hug the boundary. The
question that decides whether calibration is worth +0.00 or up to +0.07:
does OUR fused margin |d| rank correctness on data we never trained on?

  AUC(|d|, correct) >> 0.5  ->  errors have smaller margins; a softer,
                                mixture-fitted T converts that into Brier
                                credit. Ship the refit T via
                                `export.py --override-temperature`.
  AUC(|d|, correct) ~= 0.5  ->  consensus errors are saturated; calibration
                                is a dead end and accuracy is the only lever.

Also reports AUC(spread, error) -- expected ~0.5 after the hinge null result,
recorded for completeness -- and label-accuracy on the pool, which doubles as
a preview of how the next retrain's data helps.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from branches import resolve, fusion_weights
from model import BranchModel, binary_margin
from calibrate import (_sigmoid, class_balance_weights, fit_temperature,
                       sn34_from_probs)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def rank_auc(score: np.ndarray, flag: np.ndarray) -> float:
    """Mann-Whitney AUC of `score` for predicting boolean `flag`."""
    flag = np.asarray(flag).astype(bool)
    n1, n0 = int(flag.sum()), int((~flag).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=float)
    ranks[order] = np.arange(1, len(score) + 1)
    return float((ranks[flag].sum() - n1 * (n1 + 1) / 2) / (n0 * n1))


def analyse(D: np.ndarray, y: np.ndarray, w: np.ndarray,
            D_local: np.ndarray | None = None,
            y_local: np.ndarray | None = None) -> dict:
    """Pure-numpy analysis: margin/spread ranking + mixture temperature fit.

    Separated from the GPU collection so it is unit-testable without
    checkpoints. D is [N, n_branches] per-branch margins on the OOD pool;
    D_local/y_local optionally add the local calibration rows so the fitted
    temperature serves BOTH pools (fitting on OOD alone would over-soften the
    ~99%-accurate public traffic).
    """
    w = np.asarray(w, dtype=float).reshape(1, -1)
    d = (D * w).sum(axis=1)
    spread = np.sqrt(((D - d[:, None]) ** 2 * w).sum(axis=1))
    pred = d > 0
    correct = pred == (np.asarray(y).astype(int) == 1)

    out = {
        "n_ood": int(len(y)),
        "ood_accuracy": float(correct.mean()),
        "ood_acc_real": float(correct[np.asarray(y) == 0].mean())
        if (np.asarray(y) == 0).any() else float("nan"),
        "ood_acc_fake": float(correct[np.asarray(y) == 1].mean())
        if (np.asarray(y) == 1).any() else float("nan"),
        # THE decision number: do small margins mark errors?
        "auc_margin_ranks_correct": rank_auc(np.abs(d), correct),
        # post-mortem for the hinge: does spread mark errors? (expect ~0.5)
        "auc_spread_ranks_error": rank_auc(spread, ~correct),
        "median_abs_margin_correct": float(np.median(np.abs(d)[correct])),
        "median_abs_margin_error": float(np.median(np.abs(d)[~correct]))
        if (~correct).any() else float("nan"),
    }

    # Temperature refit on the mixture (class-balanced within each pool, OOD
    # pool weighted to ~30% of total mass to mirror the benchmark's hidden
    # share).
    if D_local is not None and len(D_local):
        d_loc = (D_local * w).sum(axis=1)
        y_all = np.concatenate([np.asarray(y_local), np.asarray(y)])
        d_all = np.concatenate([d_loc, d])
        w_loc = class_balance_weights(np.asarray(y_local))
        w_ood = class_balance_weights(np.asarray(y))
        w_loc *= 0.7 / max(w_loc.sum(), 1e-9)
        w_ood *= 0.3 / max(w_ood.sum(), 1e-9)
        w_all = np.concatenate([w_loc, w_ood]) * len(y_all)
        cal = fit_temperature(y_all, d_all, w=w_all, balance_classes=False)
        out["mixture_temperature"] = cal["temperature"]
        # per-pool Brier at the refit T, for the ship/no-ship judgement
        T = cal["temperature"]
        out["local_brier_at_T"] = float(np.average(
            (_sigmoid(d_loc / T) - np.asarray(y_local)) ** 2,
            weights=class_balance_weights(np.asarray(y_local))))
        out["ood_brier_at_T"] = float(np.average(
            (_sigmoid(d / T) - np.asarray(y)) ** 2,
            weights=class_balance_weights(np.asarray(y))))
        out["ood_brier_at_1"] = float(np.average(
            (_sigmoid(d) - np.asarray(y)) ** 2,
            weights=class_balance_weights(np.asarray(y))))
        out["ood_metrics_at_T"] = sn34_from_probs(
            np.asarray(y).astype(float), _sigmoid(d / T))
    return out


@torch.no_grad()
def folder_margins(root: Path, models, image_size: int, device,
                   batch_size: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Per-branch margins over ood_root/{real,fake}/*. Deploy-path transform:
    gasbench level-0 (center-crop to square, INTER_LINEAR resize)."""
    import cv2
    from gasbench_bridge import apply_random_augmentations

    paths, labels = [], []
    for lab, sub in ((0, "real"), (1, "fake")):
        for p in sorted((root / sub).rglob("*")):
            if p.suffix.lower() in IMG_EXTS:
                paths.append(p)
                labels.append(lab)
    if not paths:
        raise SystemExit(f"no images under {root}/real or {root}/fake")

    D, kept, failed = [], [], 0
    buf = []
    for i, p in enumerate(paths):
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            failed += 1
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        aug, _, _, _ = apply_random_augmentations(
            img, image_size, level=0, crop_prob=0.0, seed=34)
        buf.append(torch.from_numpy(np.ascontiguousarray(aug))
                   .permute(2, 0, 1))
        kept.append(labels[i])
        if len(buf) == batch_size or i == len(paths) - 1:
            if buf:
                x = torch.stack(buf).to(device)
                zs = [m(x).to(torch.float32) for m in models]
                D.append(torch.stack(
                    [binary_margin(z) for z in zs], dim=1).cpu())
                buf = []
    if failed:
        print(f"[warn] {failed} unreadable images skipped")
    return torch.cat(D).numpy(), np.asarray(kept), failed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ood-root", required=True)
    ap.add_argument("--runs-dir", default="runs/v7")
    ap.add_argument("--branches", default="dinov3,clip,convnext,eva,dct")
    ap.add_argument("--manifest", default="data/manifest.parquet",
                    help="adds the local calibration rows to the mixture fit; "
                         "pass '' to skip")
    ap.add_argument("--image-size", type=int, default=384)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--calib-limit", type=int, default=6000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="margin_report.json")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    names = [n.strip() for n in args.branches.split(",") if n.strip()]
    specs = resolve(names)
    weights = fusion_weights(specs)

    models = []
    for spec in specs:
        ck = torch.load(Path(args.runs_dir) / spec.name / "best.pt",
                        map_location="cpu", weights_only=False)
        m = BranchModel(spec, lora_r=ck["config"]["lora"]["r"],
                        lora_alpha=ck["config"]["lora"]["alpha"],
                        lora_dropout=0.0, gradient_checkpointing=False)
        m.load_state_dict(ck["state"], strict=True)
        m.eval()
        m.merge_and_strip()
        models.append(m.to(dev))
        print(f"[load] {spec.name} @ step {ck.get('step')}")

    D_ood, y_ood, _ = folder_margins(Path(args.ood_root), models,
                                     args.image_size, dev, args.batch_size)
    print(f"[ood] {len(y_ood)} images "
          f"({int((y_ood == 0).sum())} real / {int((y_ood == 1).sum())} fake)")

    D_loc = y_loc = None
    if args.manifest:
        import pandas as pd
        from data import ManifestDataset, ViewConfig, worker_init_fn
        from export import Ensemble, collect_branch_margins
        df = pd.read_parquet(args.manifest)
        sub = df[df.split.isin(["val_id", "val_xgen", "val_stress"])] \
            .reset_index(drop=True)
        if args.calib_limit and len(sub) > args.calib_limit:
            sub = sub.sample(n=args.calib_limit, random_state=34) \
                     .reset_index(drop=True)
        ens = Ensemble(models, weights).to(dev).eval()
        D_loc, _, y_loc, _ = collect_branch_margins(
            ens, sub, ViewConfig(image_size=args.image_size), dev,
            args.batch_size, args.workers, "deploy")

    rep = analyse(D_ood, y_ood, weights, D_loc, y_loc)
    print(json.dumps(rep, indent=2))
    Path(args.out).write_text(json.dumps(rep, indent=2))

    a = rep["auc_margin_ranks_correct"]
    print("\n=== VERDICT ===")
    if a >= 0.70:
        print(f"margin AUC {a:.3f}: errors ARE hedged in the margins. Ship "
              f"the mixture temperature via\n  python export.py ... "
              f"--override-temperature {rep.get('mixture_temperature', 1):.4f}")
    elif a >= 0.60:
        print(f"margin AUC {a:.3f}: weak ranking; the refit T buys a little. "
              f"Compare per-pool briers above before shipping.")
    else:
        print(f"margin AUC {a:.3f}: consensus errors are saturated -- "
              f"calibration is a dead end; every point must come from "
              f"training data.")


if __name__ == "__main__":
    main()
