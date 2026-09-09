"""Temperature scaling — fit, apply, and bake.

No version of this codebase has ever fitted a temperature: v2 takes a
`temperature` argument and never passes anything but 1.0, and v4 has no
calibration at all. Given `compute_sn34_score` uses beta=1.8 on the Brier term
against alpha=1.2 on MCC, calibration carries more of the score than
discrimination does, which makes this the highest-value few lines in the
package.

Two facts make it unusually safe:

  * **MCC is invariant to pure temperature scaling.** gasbench takes the argmax
    of a 2-class softmax, which is `p1 > 0.5`, and dividing both logits by
    T > 0 cannot change which is larger. So T moves Brier and only Brier: it
    can improve sn34_score and cannot degrade MCC. There is no trade to manage.

  * **The bake is exact.** softmax over 2 classes is
    `softmax([z0, z1])[1] = sigmoid(z1 - z0)`, so the whole problem is 1-D in
    `d = z1 - z0`. Rewriting the final Linear as
        W' = [[0...0], (W1 - W0)/T],   c' = [0, (c1 - c0)/T]
    makes `softmax(W'f + c')[1] == sigmoid(d/T)` identically, with no runtime
    temperature parameter to forget or double-apply.

Fit under export numerics. `PyTorchInferenceSession` does
`model.to(device, dtype).eval()` (pytorch_session.py:78) — the *whole* module,
LayerNorms included — whereas `torch.autocast(bf16)` keeps LayerNorm in fp32.
Those are different functions, so a temperature fitted under autocast is not
the temperature for the deployed cast.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from gasbench_bridge import Metrics


def sn34_from_probs(y: np.ndarray, p: np.ndarray, w: np.ndarray | None = None,
                    alpha: float = 1.2, beta: float = 1.8) -> dict:
    """Score through gasbench's own Metrics so we can never drift from the scorer."""
    m = Metrics()
    pred = (p > 0.5).astype(int)
    if w is None:
        w = np.ones_like(p, dtype=float)
    for yi, pi, pr, wi in zip(y.astype(int), p, pred, w):
        m.update(label=int(yi), pred=int(pr),
                 pred_probs=np.array([1.0 - float(pi), float(pi)]),
                 weight=float(wi))
    return {
        "mcc": float(m.calculate_binary_mcc()),
        "brier": float(m.calculate_brier()),
        "sn34": float(m.compute_sn34_score(alpha=alpha, beta=beta)),
        "accuracy": float(((pred == y.astype(int)) * w).sum() / max(w.sum(), 1e-12)),
    }


def class_balance_weights(y: np.ndarray) -> np.ndarray:
    """Per-sample weights giving each class equal total mass.

    Both MCC and Brier depend on the class prior, and the benchmark's prior is
    near 0.5: `calculate_weighted_dataset_sampling` draws a roughly equal cap
    from each of ~89 real and ~91 synthetic datasets. A validation split rarely
    matches that -- with the current holdout lists val_id lands near P(fake)=0.64
    and val_xgen near 0.33 -- so fitting a temperature on the split's raw prior
    calibrates for a distribution that will not be deployed against. Reweighting
    to 50/50 removes that mismatch without touching the model.
    """
    y = np.asarray(y).astype(int)
    w = np.ones(len(y), dtype=float)
    for c in (0, 1):
        m = y == c
        if m.any():
            w[m] = 0.5 / m.sum()
    return w * len(y) / max(w.sum(), 1e-12)


# ---------------------------------------------------------------------------
# Multiclass (3-class image taxonomy) -- local mirror of gasbench's new
# multiclass sn34 (Gorodkin MCC + multiclass Brier, PR #136). Same Metrics
# class as the benchmark, so the numbers cannot drift from the scorer.
# ---------------------------------------------------------------------------

def compose_3class(p_fake: np.ndarray, q: np.ndarray) -> np.ndarray:
    """[N] calibrated p_notreal + [N] p(semi|fake) -> [N,3] probabilities.

    Exactly the exported forward's composition: the binary collapse
    1 - P[:,0] equals p_fake for ANY q, so q can never move the binary score.
    """
    return np.stack([1.0 - p_fake, p_fake * (1.0 - q), p_fake * q], axis=1)


def y3_class_weights(y3: np.ndarray, semi_prior: float) -> np.ndarray:
    """Per-sample weights rebalancing a pool to the deployment class shares.

    Target: real 0.5, synthetic 0.5*(1-semi_prior), semi 0.5*semi_prior --
    the benchmark's ~50/50 real/fake prior with the semisynthetic share of
    the fake half. The local val pool is nowhere near this (kind_balance
    trains/validates at P(semi|fake)~0.5), so scoring it raw would let the
    semi class dominate every multiclass number 10x beyond deployment.
    """
    y3 = np.asarray(y3).astype(int)
    target = {0: 0.5, 1: 0.5 * (1.0 - semi_prior), 2: 0.5 * semi_prior}
    w = np.ones(len(y3), dtype=float)
    for c, tgt in target.items():
        m = y3 == c
        if m.any():
            w[m] = tgt / m.sum()
    return w * len(y3) / max(w.sum(), 1e-12)


def multiclass_metrics(y3: np.ndarray, P: np.ndarray,
                       w: np.ndarray | None = None,
                       alpha: float = 1.2, beta: float = 1.8) -> dict:
    """Gorodkin MCC / multiclass Brier / multiclass sn34 via gasbench Metrics."""
    y3 = np.asarray(y3).astype(int)
    P = np.asarray(P, dtype=float)
    if w is None:
        w = np.ones(len(y3), dtype=float)
    m = Metrics(num_classes=3)
    preds = P.argmax(axis=1)
    for yi, pi, pr, wi in zip(y3, P, preds, w):
        m.update(label=int(yi), pred=int(pr), pred_probs=pi, weight=float(wi))
    conf = np.asarray(m.confusion, dtype=float)
    rows = conf.sum(axis=1)
    recall = np.divide(np.diag(conf), rows, out=np.zeros(3), where=rows > 0)
    return {
        "gorodkin": float(m.calculate_multiclass_mcc()),
        "mc_brier": float(m.calculate_multiclass_brier()),
        "mc_sn34": float(m.compute_sn34_score(alpha=alpha, beta=beta,
                                              multiclass=True)),
        "per_class_recall": recall.tolist(),
    }


def fit_type_posterior(y3: np.ndarray, d2: np.ndarray, p_fake: np.ndarray,
                       semi_prior: float = 0.05, q_min: float = 1e-3) -> dict:
    """Fit q = clamp(sigmoid(d2/T2 + b2), q_min, q_max), the semi-vs-syn
    posterior of the factorized 3-class export.

    Two stages, both prior-corrected (the sampler trains the type margin at
    P(semi|fake)~0.5; deployment is ~semi_prior -- an order of magnitude
    apart, so an uncorrected fit would be confidently wrong on the ~48%-share
    synthetic class, which costs more multiclass Brier than a binary head's
    zeroed semi column):

      1. (T2, b2) grid minimising the prior-reweighted conditional Brier
         E_w[(q - 1{semi})^2] on fake rows only.
      2. q_max grid minimising nothing locally EXCEPT through the composed
         full-pool multiclass sn34 (deployment-share weights) -- the cap on
         how wrong a confident semi call can be.

    Returns the buffers plus mc_sn34_before (q pinned at q_min == a 2-logit
    head's multiclass behaviour) and mc_sn34_after (fitted q). after <= before
    means the discriminator does not pay for itself: ship pinned. Neither
    choice can move the binary score (see compose_3class).
    """
    y3 = np.asarray(y3).astype(int)
    d2 = np.asarray(d2, dtype=float)
    p_fake = np.asarray(p_fake, dtype=float)
    fake = y3 > 0
    t_semi = (y3[fake] == 2).astype(float)

    w3 = y3_class_weights(y3, semi_prior)
    pinned = compose_3class(p_fake, np.full(len(y3), q_min))
    mc_before = multiclass_metrics(y3, pinned, w3)["mc_sn34"]
    out = {"type_temperature": 1.0, "type_bias": -3.0,
           "q_min": q_min, "q_max": q_min,
           "cond_brier_before": float("nan"), "cond_brier_after": float("nan"),
           "mc_sn34_before": mc_before, "mc_sn34_after": mc_before}
    if not fake.any() or t_semi.sum() == 0 or t_semi.sum() == fake.sum():
        return out  # degenerate pool: nothing to fit, pin q

    d2f = d2[fake]
    share = float(t_semi.mean())
    wc = np.where(t_semi == 1.0, semi_prior / share,
                  (1.0 - semi_prior) / (1.0 - share))
    wc = wc / wc.sum()
    out["cond_brier_before"] = float((wc * (q_min - t_semi) ** 2).sum())

    b_grid = np.linspace(-6.0, 1.0, 57)
    best = (np.inf, 1.0, -3.0)
    for T2 in np.geomspace(0.25, 8.0, 60):
        Q = _sigmoid(d2f[None, :] / T2 + b_grid[:, None])       # [57, Nf]
        briers = ((Q - t_semi[None, :]) ** 2 * wc[None, :]).sum(axis=1)
        i = int(np.argmin(briers))
        if briers[i] < best[0]:
            best = (float(briers[i]), float(T2), float(b_grid[i]))
    out["cond_brier_after"], T2, B2 = best
    out["type_temperature"], out["type_bias"] = T2, B2

    q_all = _sigmoid(d2 / T2 + B2)
    best_q = (mc_before, q_min)
    for qm in np.linspace(0.2, 0.95, 16):
        P = compose_3class(p_fake, np.clip(q_all, q_min, qm))
        mc = multiclass_metrics(y3, P, w3)["mc_sn34"]
        if mc > best_q[0]:
            best_q = (mc, float(qm))
    out["mc_sn34_after"], out["q_max"] = best_q
    return out


def blended_sn34(y, d_base, d_aug, T, w=None, aug_weight: float = 0.2,
                 balance_classes: bool = True) -> float:
    """The benchmark's own blend: 0.8 * base + 0.2 * robustness (common.py:280)."""
    if w is None and balance_classes:
        w = class_balance_weights(y)
    b = sn34_from_probs(y, _sigmoid(d_base / T), w)["sn34"]
    if d_aug is None:
        return b
    a = sn34_from_probs(y, _sigmoid(d_aug / T), w)["sn34"]
    return (1.0 - aug_weight) * b + aug_weight * a


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))


def fit_temperature(y: np.ndarray, d_base: np.ndarray,
                    d_aug: np.ndarray | None = None,
                    w: np.ndarray | None = None,
                    aug_weight: float = 0.2, balance_classes: bool = True,
                    lo: float = 0.08, hi: float = 8.0, n: int = 240) -> dict:
    """Grid search over log T minimising the blended Brier score.

    A grid rather than an optimiser because the objective is a cheap O(N) numpy
    expression and 240 evaluations cost milliseconds.

    The lower bound is 0.08, not 0.25: an early run hit the 0.25 floor with a
    badly under-confident model, which silently clips the fit on exactly the
    term the score is most sensitive to.

    `balance_classes` reweights to a 50/50 prior before fitting -- see
    `class_balance_weights` for why that matters here.
    """
    if w is None and balance_classes:
        w = class_balance_weights(y)
    grid = np.geomspace(lo, hi, n)

    # Minimise BRIER, not maximise sn34.
    #
    # sn34's brier_norm term is max(0, (0.25 - brier)/0.25), which clips at
    # zero. Whenever the model is weak, brier sits at or above 0.25 for most
    # temperatures, sn34 is identically 0 across a wide region, and argmax over
    # a flat-zero objective returns noise. Observed at step 250 of run_2: two
    # halves of val_xgen picked T=4.9989 and T=0.6403, a 7.8x disagreement, and
    # the T=5 half collapsed every probability to ~0.5 and drove the pooled
    # score to exactly 0.0000.
    #
    # Brier is smooth and strictly informative over that same region, is a
    # proper scoring rule, and is the quantity sn34 is a monotone function of
    # once it is below 0.25 -- so this maximises sn34 wherever sn34 is
    # meaningful and stays well-posed where it is not.
    def brier_at(T):
        b = float(np.average((_sigmoid(d_base / T) - y) ** 2, weights=w))
        if d_aug is None:
            return b
        a = float(np.average((_sigmoid(d_aug / T) - y) ** 2, weights=w))
        return (1.0 - aug_weight) * b + aug_weight * a

    briers = [brier_at(T) for T in grid]
    k = int(np.argmin(briers))
    T = float(grid[k])
    if k in (0, n - 1):
        print(f"[calibrate] WARNING fitted T={T:.4f} sits at the edge of the "
              f"[{lo}, {hi}] grid; the logits may be degenerate.")
    if briers[k] >= 0.245:
        # No temperature can help: the model carries essentially no usable
        # probability signal on this data yet. Say so and stay neutral.
        print(f"[calibrate] WARNING best achievable brier is {briers[k]:.4f} "
              f"(>=0.245, i.e. no better than a constant 0.5). The model has no "
              f"calibratable signal here yet; falling back to T=1.")
        T = 1.0
    before = sn34_from_probs(y, _sigmoid(d_base), w)
    after = sn34_from_probs(y, _sigmoid(d_base / T), w)
    return {
        "temperature": T,
        "blended_before": blended_sn34(y, d_base, d_aug, 1.0, w, aug_weight,
                                       balance_classes=False),
        "blended_after": blended_sn34(y, d_base, d_aug, T, w, aug_weight,
                                      balance_classes=False),
        "brier_grid_min": float(briers[k]),
        "brier_before": before["brier"], "brier_after": after["brier"],
        "mcc_before": before["mcc"], "mcc_after": after["mcc"],
        "n": int(len(y)), "p_fake_raw": float(np.mean(y)),
        "class_balanced": bool(balance_classes),
    }


def fit_spread_temperature(y: np.ndarray, D_base: np.ndarray,
                           fusion_w: np.ndarray,
                           D_aug: np.ndarray | None = None,
                           w: np.ndarray | None = None,
                           aug_weight: float = 0.2,
                           balance_classes: bool = True,
                           t_lo: float = 0.08, t_hi: float = 8.0,
                           n_t: int = 80,
                           k_max: float = 2.0, n_k: int = 41) -> dict:
    """Fit (T0, k) for the disagreement-aware temperature T0 * (1 + k * s).

    D_base is the [N, n_branches] matrix of per-branch margins
    d_i = binary_margin(z_i) (logsumexp collapse for 3-logit heads);
    `fusion_w` the renormalised fusion weights.
    The fused margin is d = D @ w and the per-sample spread
    s = sqrt(sum_i w_i (d_i - d)^2) -- the model's own OOD signal: branches
    agree on distributions they trained on and disagree on foreign ones.

    Why this exists: a single T has one optimum, c* = pooled accuracy, and one
    confidence level cannot serve a ~99%-accurate in-distribution pool and a
    ~75%-accurate hidden pool at once (measured 2026-08-12: refitting scalar T
    against the dashboard mix was worth +0.002 sn34; pool-conditional
    calibration is worth ~+0.05). k conditions the temperature on the
    disagreement signal, approximating the pool-conditional oracle without
    knowing the pools.

    Same safety properties as scalar T: k >= 0 and s >= 0 give
    T_eff >= T0 > 0, so argmax/MCC never move; Brier-only, like
    fit_temperature, and minimised on the same blended objective. The k = 0
    column of the grid IS the scalar fit, so this can never do worse than
    fit_temperature on the fitting data.
    """
    y = np.asarray(y).astype(float)
    if w is None and balance_classes:
        w = class_balance_weights(y)
    fw = np.asarray(fusion_w, dtype=float).reshape(1, -1)

    def _margins(D):
        d = (np.asarray(D, dtype=float) * fw).sum(axis=1)
        s = np.sqrt(((np.asarray(D, dtype=float) - d[:, None]) ** 2 * fw)
                    .sum(axis=1))
        return d, s

    db, sb = _margins(D_base)
    da, sa = _margins(D_aug) if D_aug is not None else (None, None)

    Ts = np.geomspace(t_lo, t_hi, n_t)
    ks = np.concatenate([[0.0], np.geomspace(1e-3, k_max, n_k - 1)])

    def brier_at(T, k):
        b = float(np.average(
            (_sigmoid(db / (T * (1.0 + k * sb))) - y) ** 2, weights=w))
        if da is None:
            return b
        a = float(np.average(
            (_sigmoid(da / (T * (1.0 + k * sa))) - y) ** 2, weights=w))
        return (1.0 - aug_weight) * b + aug_weight * a

    best_T, best_k, best_b = 1.0, 0.0, np.inf
    scalar_b, scalar_T = np.inf, 1.0
    for k in ks:
        for T in Ts:
            b = brier_at(T, k)
            if b < best_b:
                best_T, best_k, best_b = float(T), float(k), b
            if k == 0.0 and b < scalar_b:
                scalar_b, scalar_T = b, float(T)

    if best_T in (Ts[0], Ts[-1]) or best_k == ks[-1]:
        print(f"[calibrate] WARNING spread fit at grid edge "
              f"(T0={best_T:.4f}, k={best_k:.4f}); widen the grid.")
    if best_b >= 0.245:
        print("[calibrate] WARNING no calibratable signal; falling back to "
              "T0=1, k=0.")
        best_T, best_k = 1.0, 0.0

    p0 = _sigmoid(db)
    p1 = _sigmoid(db / (best_T * (1.0 + best_k * sb)))
    before = sn34_from_probs(y, p0, w)
    after = sn34_from_probs(y, p1, w)
    return {
        "temperature": best_T, "temp_slope": best_k,
        "brier_blend_min": best_b,
        "scalar_temperature": scalar_T, "scalar_brier_blend": scalar_b,
        "spread_gain_vs_scalar": scalar_b - best_b,
        "brier_before": before["brier"], "brier_after": after["brier"],
        "mcc_before": before["mcc"], "mcc_after": after["mcc"],
        "sn34_before": before["sn34"], "sn34_after": after["sn34"],
        "spread_p50": float(np.median(sb)),
        "spread_p95": float(np.quantile(sb, 0.95)),
        "n": int(len(y)),
    }


def fit_platt(d: np.ndarray, y: np.ndarray, iters: int = 100) -> tuple:
    """Two-parameter Platt fit, p = sigmoid(a*d + b), by LBFGS on NLL.

    Ported from v3. Two parameters rather than one temperature, and the
    difference is not cosmetic: the bias `b` shifts the decision threshold, so
    unlike pure temperature scaling it can move MCC and accuracy, not only
    Brier. On a model with asymmetric error rates -- run_3 measured real
    accuracy 0.893 against synthetic 0.874 -- that is exactly the degree of
    freedom a scale-only fit cannot use.
    """
    import torch as _t

    z = _t.as_tensor(d, dtype=_t.float64)
    t = _t.as_tensor(y, dtype=_t.float64)
    a = _t.ones(1, dtype=_t.float64, requires_grad=True)
    b = _t.zeros(1, dtype=_t.float64, requires_grad=True)
    opt = _t.optim.LBFGS([a, b], max_iter=iters, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = _t.nn.functional.binary_cross_entropy_with_logits(a * z + b, t)
        loss.backward()
        return loss

    with _t.enable_grad():
        opt.step(closure)
    return float(a.item()), float(b.item())


def platt_metrics(y: np.ndarray, d: np.ndarray, a: float, b: float,
                  w: np.ndarray | None = None) -> dict:
    """Score under a fitted Platt map."""
    if w is None:
        w = class_balance_weights(y)
    return sn34_from_probs(y, _sigmoid(a * d + b), w)


def _assign_folds(groups: np.ndarray, y: np.ndarray, n_folds: int = 2) -> np.ndarray:
    """Split GROUPS (not rows) into folds, keeping both labels in each fold.

    Real and fake groups are dealt out alternately so no fold can end up
    single-label, which would make MCC undefined there.
    """
    groups = np.asarray(groups)
    order = {}
    for lab in (0, 1):
        g = sorted({str(x) for x in groups[y == lab]})
        for i, name in enumerate(g):
            order.setdefault(name, i % n_folds)
    return np.array([order.get(str(g), 0) for g in groups])


def crossfit_temperature(y: np.ndarray, d_base: np.ndarray,
                         d_aug: np.ndarray | None, groups: np.ndarray,
                         n_folds: int = 2, aug_weight: float = 0.2,
                         min_groups: int = 4,
                         max_fold_spread: float = 2.0) -> dict:
    """Cross-fitted temperature: fit on one set of generators, score the others.

    Why this exists. Fitting T on `val_id` and scoring `val_xgen` was actively
    harmful in run_1: as the model overfitted, val_id accuracy climbed to 0.93
    and the fit chose ever-harder sharpening (T 0.736 -> 0.587), but val_xgen
    carried ~2.4x the error rate and sharpening penalises a wrong prediction
    quadratically. 95% of the observed val_xgen decline from step 2500 to 4000
    was calibration, not discrimination.

    Fitting T on val_xgen directly would fix the distribution mismatch and
    introduce a worse problem: the selection metric would then be scored with a
    temperature tuned on the very data it scores, flattering every checkpoint
    and unevenly. Cross-fitting gets both properties. Each sample is calibrated
    with a T fitted on generators it does not belong to, so the resulting
    score is an honest estimate of "calibrated on some generators, applied to
    unseen ones" -- which is exactly the deployment situation.

    Returns the cross-fitted metrics plus `temperature_full` (fitted on
    everything) for reporting and export.
    """
    y = np.asarray(y).astype(int)
    n_groups = len({str(g) for g in np.asarray(groups)})
    full = fit_temperature(y, d_base, d_aug, aug_weight=aug_weight)
    if n_groups < min_groups:
        # Too few generators to split; fall back and say so rather than
        # silently reporting a number that is not cross-fitted.
        return {**full, "crossfit": False, "n_groups": n_groups,
                "temperature_full": full["temperature"],
                "blended_crossfit": full["blended_after"], "fold_temperatures": []}

    fold = _assign_folds(groups, y, n_folds)
    temps = []
    for f in range(n_folds):
        tr = fold != f
        if not (fold == f).any() or len(set(y[tr].tolist())) < 2:
            temps.append(full["temperature"])
        else:
            temps.append(fit_temperature(
                y[tr], d_base[tr], d_aug[tr] if d_aug is not None else None,
                aug_weight=aug_weight)["temperature"])

    # Platt, cross-fitted on the same folds. v3 fits (a, b) on val_xgen and
    # scores those same rows -- its own comment calls the leak "negligible" --
    # which flatters every checkpoint and does so unevenly, since a checkpoint
    # whose miscalibration a 2-parameter map happens to fit well gains most.
    # Same expressiveness, honest fit.
    p_platt = np.empty(len(y), dtype=float)
    platt_ab = []
    for f in range(n_folds):
        te, tr = fold == f, fold != f
        if not te.any() or len(set(y[tr].tolist())) < 2:
            pa, pb = 1.0, 0.0
        else:
            pa, pb = fit_platt(d_base[tr], y[tr])
        platt_ab.append((round(pa, 4), round(pb, 4)))
        p_platt[te] = _sigmoid(pa * d_base[te] + pb)

    # If the folds disagree wildly, the per-fold fits are not measuring a
    # property of the model -- they are fitting noise, and applying a wild T to
    # half the samples destroys the pooled metric rather than reporting it.
    # Fall back to the pooled fit and say so.
    spread = max(temps) / max(min(temps), 1e-9)
    unstable = spread > max_fold_spread
    if unstable:
        temps = [full["temperature"]] * n_folds

    p_base = np.empty(len(y), dtype=float)
    p_aug = np.empty(len(y), dtype=float) if d_aug is not None else None
    for f in range(n_folds):
        te = fold == f
        p_base[te] = _sigmoid(d_base[te] / temps[f])
        if p_aug is not None:
            p_aug[te] = _sigmoid(d_aug[te] / temps[f])

    w = class_balance_weights(y)
    base = sn34_from_probs(y, p_base, w)
    out = {"crossfit": True, "n_groups": n_groups,
           "fold_spread": round(float(spread), 3), "unstable_folds": bool(unstable),
           "fold_temperatures": [round(t, 4) for t in temps],
           "temperature_full": full["temperature"],
           "temperature": full["temperature"],
           "mcc": base["mcc"], "brier": base["brier"],
           "accuracy": base["accuracy"], "sn34": base["sn34"], "n": int(len(y))}
    if p_aug is not None:
        aug = sn34_from_probs(y, p_aug, w)
        out["blended_crossfit"] = ((1 - aug_weight) * base["sn34"]
                                   + aug_weight * aug["sn34"])
        out["aug_sn34"] = aug["sn34"]
    else:
        out["blended_crossfit"] = base["sn34"]

    pm = sn34_from_probs(y, p_platt, w)
    out["platt_folds"] = platt_ab
    out["platt_sn34"] = pm["sn34"]
    out["platt_mcc"] = pm["mcc"]
    out["platt_brier"] = pm["brier"]
    out["platt_full"] = fit_platt(d_base, y)
    return out


@torch.no_grad()
def bake_temperature(classifier: nn.Linear, T: float) -> None:
    """Fold T into the final Linear so the export needs no runtime parameter.

    Row 0 becomes identically zero and row 1 becomes (W1-W0)/T, which makes
    softmax(.)[1] == sigmoid(d/T) exactly and leaves argmax unchanged. Computed
    in float64 and cast back once.
    """
    if classifier.out_features != 2:
        raise ValueError("temperature baking assumes a 2-class head")
    W = classifier.weight.detach().double()
    c = classifier.bias.detach().double()
    Wn = torch.zeros_like(W)
    cn = torch.zeros_like(c)
    Wn[1] = (W[1] - W[0]) / T
    cn[1] = (c[1] - c[0]) / T
    classifier.weight.copy_(Wn.to(classifier.weight.dtype))
    classifier.bias.copy_(cn.to(classifier.bias.dtype))
