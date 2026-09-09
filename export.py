"""Fuse trained branches into one submission.

    python export.py --branches dinov3,clip --out submission/
    python export.py --branches dinov3,clip,convnext,eva --out submission/ \
        --manifest data/manifest.parquet

Steps:
  1. load each branch's best.pt, rebuild its BranchModel, merge LoRA, and
     probe-check that merged == unmerged on a real uint8 batch;
  2. assemble the ensemble (fixed logit averaging, weights renormalised over
     the shipped subset);
  3. collect ENSEMBLE margins on val_id+val_xgen under export numerics
     (bf16, module-wide cast -- exactly what pytorch_session does) and fit one
     temperature;
  4. store T as a BUFFER on the ensemble; forward divides the fused logits by
     it once. Exact for every head type (a weight-bake cannot pass through the
     bounded-cosine head's tanh), argmax-invariant, and it ships inside the
     safetensors like any other tensor -- no runtime kwarg, nothing for
     load_model to receive;
  5. write model.safetensors (the ensemble's own state_dict, so the template
     loads it with one strict load_state_dict), config.json (DINOv3's HF
     config -- the one branch needing a file), model.py rendered from
     templates/inference_model.py with all constants as literals, and
     model_config.yaml;
  6. re-load the submission through the rendered model.py and assert the
     state dicts match by SHAPE and the forward matches the training-side
     ensemble on a probe batch.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from branches import resolve, fusion_weights
from calibrate import fit_temperature, fit_type_posterior
from data import ManifestDataset, ViewConfig, worker_init_fn
from model import BranchModel, binary_margin, type_margin
from allowlist import render_template, scan_allowlist


class Ensemble(torch.nn.Module):
    """Must stay state-dict-identical to the template's Ensemble class.

    Factorized 3-class output (2026-08-25). The binary axis is unchanged in
    substance from the 2-logit design: per-branch binary margins
    d_i = logsumexp(z_i[:,1:]) - z_i[:,0] (sigmoid(d_i) = branch p_notreal),
    fused linearly, passed through the disagreement-aware temperature
    T_eff = T0 * (1 + k * max(0, spread - pivot)) where spread is the
    fusion-weighted std of the d_i -- a per-sample OOD signal. On top, a
    conditional syn-vs-semi posterior q = clamp(sigmoid(d2/T2 + b2), qmin,
    qmax) from the fused type margins d2_i = z_i[:,2] - z_i[:,1]. The output
    is log([1-p_fake, p_fake*(1-q), p_fake*q]): softmax recovers the
    probabilities exactly, so the benchmark's binary collapse
    (1 - p[0]) equals sigmoid(d/T_eff) -- the calibrated binary quantity --
    BY CONSTRUCTION, for any q. q moves only the multiclass score.

    Two invariants (verify.py proves both):
      * margins are fused per-branch then combined -- never "fuse logits,
        then logsumexp" (not equivalent for 3 logits);
      * the raw logits are never divided by T (softmax(z/T)[0] !=
        1 - sigmoid(d/T) for K=3); temperature lives inside p_fake only.
    """

    def __init__(self, models: list[BranchModel], weights: list[float],
                 temperature: float = 1.0, temp_slope: float = 0.0,
                 spread_pivot: float = 0.0, type_temperature: float = 1.0,
                 type_bias: float = -3.0, q_min: float = 1e-3,
                 q_max: float = 1.0, p_min: float = 1e-6,
                 p_max: float = 1.0 - 1e-6):
        super().__init__()
        self.branches = torch.nn.ModuleList(models)
        self.register_buffer("w", torch.tensor(weights, dtype=torch.float32))
        self.register_buffer("temperature",
                             torch.tensor(float(temperature)))
        self.register_buffer("temp_slope",
                             torch.tensor(float(temp_slope)))
        # The slope only engages ABOVE this pivot (set to the local p99 of
        # spread at export). Below it T_eff == T0 exactly, so the calibrated
        # local/public behaviour is untouched by construction; k cannot be
        # FITTED locally (the 2026-08-12 export measured spread p95=2.26 with
        # ~99% local accuracy -- no local errors correlate with disagreement,
        # so the local optimum is always k=0) and is instead a reasoned prior
        # aimed at the hidden pools where branches disagree at multiples of
        # the local range.
        self.register_buffer("spread_pivot",
                             torch.tensor(float(spread_pivot)))
        # Conditional semi-vs-syn posterior parameters (fit_type_posterior).
        # Defaults reproduce the prior-shift init (q ~ 0.05); q_min > 0 keeps
        # log(p) finite; q_min == q_max pins q -- the --binary-equivalent
        # export, whose multiclass score equals a 2-logit head's to ~1e-6
        # while the binary path is untouched.
        self.register_buffer("type_temperature",
                             torch.tensor(float(type_temperature)))
        self.register_buffer("type_bias", torch.tensor(float(type_bias)))
        self.register_buffer("q_min", torch.tensor(float(q_min)))
        self.register_buffer("q_max", torch.tensor(float(q_max)))
        # Probability clamp: the confident-inversion insurance (r22 lesson --
        # hidden pools scored p_fake~0.03 at holdout weight cost Brier^1.8;
        # the hinge cannot catch UNANIMOUS inversions). Defaults reproduce the
        # historical hardcoded numerical clamp exactly. With p_min < 0.5 <
        # p_max the benchmark's 0.5 threshold means the clamp provably cannot
        # move MCC (gasbench metrics.py thresholds 1-p[0] > 0.5) -- it is a
        # pure Brier dial, tuned by tune_operating_point.py on records.
        self.register_buffer("p_min", torch.tensor(float(p_min)))
        self.register_buffer("p_max", torch.tensor(float(p_max)))

    def forward(self, x_uint8: torch.Tensor) -> torch.Tensor:
        zs = [m(x_uint8).to(torch.float32) for m in self.branches]
        d = torch.stack([torch.logsumexp(zi[:, 1:], dim=1) - zi[:, 0]
                         for zi in zs], dim=1)
        w = self.w.to(torch.float32).view(1, -1)
        dbar = (w * d).sum(dim=1)
        spread = ((w * (d - dbar.unsqueeze(1)) ** 2).sum(dim=1)).sqrt()
        excess = torch.clamp(spread - self.spread_pivot.to(torch.float32),
                             min=0.0)
        t = self.temperature.to(torch.float32) * (
            1.0 + self.temp_slope.to(torch.float32) * excess)
        p_fake = torch.clamp(torch.sigmoid(dbar / t),
                             self.p_min.to(torch.float32),
                             self.p_max.to(torch.float32))
        d2 = torch.stack([zi[:, 2] - zi[:, 1] for zi in zs], dim=1)
        d2bar = (w * d2).sum(dim=1)
        q = torch.clamp(
            torch.sigmoid(d2bar / self.type_temperature.to(torch.float32)
                          + self.type_bias.to(torch.float32)),
            self.q_min.to(torch.float32), self.q_max.to(torch.float32))
        p = torch.stack([1.0 - p_fake, p_fake * (1.0 - q), p_fake * q],
                        dim=1)
        return torch.log(p)


@torch.no_grad()
def collect_branch_margins(ens, sub, view, device, bs, workers, eval_mode):
    """Per-branch binary and type margins [N, n_branches] under one eval view.

    Per-branch (not fused) because the spread-temperature fit needs the
    branch disagreement per sample, and the calibration subset includes
    val_stress -- the gated zero-shot pools -- so the fit sees genuinely
    out-of-distribution spread, not just in-distribution agreement.
    """
    from torch.utils.data import DataLoader
    ds = ManifestDataset(sub, view, train=False, eval_mode=eval_mode, seed=34)
    ld = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=workers,
                    worker_init_fn=worker_init_fn)
    dlist, d2list, ylist, y3list = [], [], [], []
    for b in ld:
        x = b["x"].to(device)
        zs = [m(x).to(torch.float32) for m in ens.branches]
        dlist.append(torch.stack([binary_margin(zi) for zi in zs],
                                 dim=1).cpu())
        d2list.append(torch.stack([type_margin(zi) for zi in zs],
                                  dim=1).cpu())
        ylist.append(b["y"])
        y3list.append(b["y3"])
    return (torch.cat(dlist).numpy(), torch.cat(d2list).numpy(),
            torch.cat(ylist).numpy(), torch.cat(y3list).numpy())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--branches", required=True,
                    help="comma-separated, e.g. dinov3,clip")
    ap.add_argument("--runs-dir", default="runs/v7")
    ap.add_argument("--out", default="submission")
    ap.add_argument("--manifest", default="data/manifest.parquet")
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float32"])
    ap.add_argument("--calib-limit", type=int, default=8000)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--no-refit-temperature", action="store_true")
    ap.add_argument("--calib-splits", default="val_id,val_xgen,val_stress",
                    help="comma-joined splits the T0/type fit runs on. The "
                         "default keeps the historical behaviour (all three "
                         "val pools). Pass val_id,val_xgen for the "
                         "no-degradation experiment arms: their val_stress IS "
                         "the zero-shot judgment set (gemini31), and fitting "
                         "temperature on the judged pool leaks it into the "
                         "benchmarked export.")
    ap.add_argument("--override-temperature", type=float, default=None,
                    help="ship this T0 instead of fitting locally -- for a "
                         "temperature fitted on a local+OOD mixture by "
                         "measure_margins.py, which a local-only fit cannot "
                         "reproduce (local pools have no errors)")
    ap.add_argument("--no-spread", action="store_true",
                    help="ship k = 0 (pure scalar temperature); the A/B "
                         "control for the disagreement hinge")
    ap.add_argument("--p-min", type=float, default=1e-6,
                    help="probability floor on p_fake -- confident-inversion "
                         "insurance (a hidden fake scored confidently real "
                         "pays Brier^1.8 at holdout weight; the hinge cannot "
                         "catch UNANIMOUS inversions). Default = the "
                         "historical numerical clamp, i.e. no insurance. "
                         "Tune with tune_operating_point.py; must stay < 0.5 "
                         "so the 0.5 decision threshold (and MCC) is "
                         "provably untouched.")
    ap.add_argument("--p-max", type=float, default=1.0 - 1e-6,
                    help="probability ceiling on p_fake -- the mirror "
                         "insurance for hidden REALS scored confidently fake "
                         "(r22: real-image-holdout-e3fee2e2 at 51.8%%). Must "
                         "stay > 0.5.")
    ap.add_argument("--spread-slope", type=float, default=1.0,
                    help="k in T_eff = T0*(1 + k*max(0, spread - pivot)). A "
                         "REASONED PRIOR, not a fit: local pools carry no "
                         "errors correlated with disagreement, so any local "
                         "fit returns k=0. With T0~0.7 and pivot~2.3, k=1 "
                         "maps hidden-scale disagreement (spread ~8-12) to "
                         "T_eff ~4-7, i.e. confidence ~0.8-0.9 instead of "
                         "~0.99 exactly where hidden errors concentrate. "
                         "Local/public behaviour is unchanged by the hinge.")
    ap.add_argument("--type-prior", type=float, default=None,
                    help="deployment P(semisynthetic | fake) used to "
                         "reweight the (T2, b2) fit. Default: the "
                         "semisynthetic share of the manifest's fake "
                         "DATASETS (dataset-level, matching the benchmark's "
                         "fixed per-dataset sampling), ~0.05.")
    ap.add_argument("--binary-equivalent", action="store_true",
                    help="pin q at q_min (no type fit): the 3-logit export "
                         "whose multiclass score equals a 2-logit head's "
                         "while the binary path is byte-identical to the "
                         "fitted-q export. The fallback submission.")
    args = ap.parse_args()

    # The scoring harness rebuilds the template with AutoModel.from_config
    # under gasbench's pinned transformers (gasbench/pyproject.toml), so the
    # safetensors key layout is dictated by THAT version. 5.2.0 builds
    # DINOv3 flat (embeddings/layer/norm); newer releases nest the encoder
    # (model.layer.*) and produce an artifact the harness cannot load.
    HARNESS_TRANSFORMERS = "5.2.0"
    import transformers as _tfm
    if _tfm.__version__ != HARNESS_TRANSFORMERS:
        raise SystemExit(
            f"export must run under transformers=={HARNESS_TRANSFORMERS} "
            f"(gasbench's pin), found {_tfm.__version__}: the exported "
            f"state-dict keys would not match the scoring harness. "
            f"pip install transformers=={HARNESS_TRANSFORMERS}")

    names = [n.strip() for n in args.branches.split(",") if n.strip()]
    specs = resolve(names)
    weights = fusion_weights(specs)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    models, image_size, cfg0 = [], None, None
    for spec in specs:
        ck_path = Path(args.runs_dir) / spec.name / "best.pt"
        if not ck_path.exists():
            raise SystemExit(f"{ck_path} missing -- train that branch first")
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        if ck.get("branch") != spec.name:
            raise SystemExit(f"{ck_path} was trained as branch "
                             f"{ck.get('branch')!r}, not {spec.name!r}")
        # native_size provenance: dinov3-family shapes are size-independent,
        # so a checkpoint trained at another resolution would strict-load
        # SILENTLY wrong. Old checkpoints lack "spec" and skip the check.
        ck_native = (ck.get("spec") or {}).get("native_size",
                                               spec.native_size)
        if ck_native != spec.native_size:
            raise SystemExit(
                f"{spec.name}: checkpoint trained at native_size={ck_native} "
                f"but the registry now says {spec.native_size}. Retrain or "
                f"fix branches.py.")
        cfg = ck["config"]
        cfg0 = cfg0 or cfg
        sz = int(cfg["data"]["image_size"])
        if image_size is None:
            image_size = sz
        elif sz != image_size:
            raise SystemExit(
                f"branch {spec.name} trained at image_size={sz} but an "
                f"earlier branch at {image_size}; the submission declares ONE "
                f"resolution. Retrain or exclude.")
        m = BranchModel(spec, lora_r=cfg["lora"]["r"],
                        lora_alpha=cfg["lora"]["alpha"],
                        lora_dropout=0.0, gradient_checkpointing=False)
        # transformers 5.x moved the HF backbone's encoder layers one level
        # deeper (backbone.model.layer.*) vs the 4.5x flat layout the
        # checkpoint may carry -- and 5.2.0 keeps embeddings/norm FLAT, so
        # the nesting is per-key, not global. Remap each key against what
        # this env's rebuilt model actually expects; pure renames only,
        # anything unresolved is left for the strict load to flag.
        want = set(m.state_dict().keys())

        def _remap(k):
            if k in want:
                return k
            if k.startswith("backbone.model."):
                alt = "backbone." + k[len("backbone.model."):]
            elif k.startswith("backbone."):
                alt = "backbone.model." + k[len("backbone."):]
            else:
                return k
            return alt if alt in want else k

        state = {_remap(k): v for k, v in ck["state"].items()}
        m.load_state_dict(state, strict=True)
        m.eval()
        # merge check on real pixels, before/after must agree
        probe = torch.randint(0, 256, (2, 3, image_size, image_size),
                              dtype=torch.uint8)
        with torch.no_grad():
            z0 = m(probe)
            n_merged = m.merge_and_strip()
            z1 = m(probe)
        err = (z0 - z1).abs().max().item()
        if err > 1e-3:
            raise SystemExit(f"{spec.name}: LoRA merge drift {err:.2e}")
        print(f"[export] {spec.name}: merged {n_merged} LoRA modules, "
              f"drift {err:.2e}, selection {ck.get('selection'):.4f} "
              f"@ step {ck.get('step')}")
        models.append(m)

    ens = Ensemble(models, weights).to(device=dev, dtype=dtype).eval()
    # uint8 buffers survive a module-wide cast; float buffers (pixel stats,
    # fusion weights) intentionally ride along exactly as the harness will.

    # -- temperature under export numerics ---------------------------------
    # T0 is FITTED (scalar grid, blended deploy 0.8 / robust 0.2, on
    # val_id + val_xgen + val_stress). The disagreement hinge on top --
    # T_eff = T0 * (1 + k * max(0, spread - pivot)) -- takes its pivot from
    # the local spread p99 (both views), so no local sample crosses it and
    # the fitted local calibration is preserved exactly; k is the
    # --spread-slope prior (a local fit provably returns 0: local accuracy is
    # ~99% everywhere, so no errors correlate with disagreement here).
    T, K, PIVOT = 1.0, 0.0, 0.0
    PMIN, PMAX = float(args.p_min), float(args.p_max)
    if not (0.0 < PMIN < 0.5 < PMAX < 1.0):
        # The MCC-invariance guarantee (benchmark thresholds 1-p[0] > 0.5)
        # holds ONLY inside this range; outside it the clamp changes
        # decisions and every parity proof below is void.
        raise SystemExit(f"--p-min/--p-max must satisfy 0 < p_min < 0.5 < "
                         f"p_max < 1, got ({PMIN}, {PMAX})")
    # Type-posterior defaults = binary-equivalent pinning; overwritten only
    # by a successful fit. q_min == q_max => q pinned, multiclass behaves
    # like a 2-logit head, binary path unaffected either way.
    T2, B2, QMIN, QMAX = 1.0, -3.0, 1e-3, 1e-3
    type_fitted = False
    if args.override_temperature is not None:
        T = float(args.override_temperature)
        print(f"[export] T0={T:.4f} (OVERRIDE -- mixture fit from "
              f"measure_margins.py; local fit skipped, hinge k=0)")
    elif not args.no_refit_temperature:
        df = pd.read_parquet(args.manifest)
        view = ViewConfig(image_size=image_size)
        calib_splits = [s.strip() for s in args.calib_splits.split(",")
                        if s.strip()]
        bad = set(calib_splits) - {"val_id", "val_xgen", "val_stress"}
        if bad or not calib_splits:
            raise SystemExit(f"--calib-splits: unknown split(s) {sorted(bad)}; "
                             f"choose from val_id,val_xgen,val_stress")
        sub = df[df.split.isin(calib_splits)].reset_index(drop=True)
        # The tee'd log must prove what the fit saw: a val_stress that holds
        # a zero-shot judgment set must never silently enter the calibration.
        print(f"[export] calibration fit splits: {','.join(calib_splits)} "
              f"(n={len(sub):,})")
        if args.calib_limit and len(sub) > args.calib_limit:
            sub = sub.sample(n=args.calib_limit, random_state=34) \
                     .reset_index(drop=True)
        D_dep, D2_dep, y, y3 = collect_branch_margins(
            ens, sub, view, dev, args.batch_size, args.workers, "deploy")
        D_rob, _, _, _ = collect_branch_margins(
            ens, sub, view, dev, args.batch_size, args.workers, "robust")
        fw = np.asarray(weights)
        d_dep = (D_dep * fw).sum(axis=1)
        d_rob = (D_rob * fw).sum(axis=1)
        cal = fit_temperature(y, d_dep, d_rob)
        T = float(cal["temperature"])
        s_dep = np.sqrt(((D_dep - d_dep[:, None]) ** 2 * fw).sum(axis=1))
        s_rob = np.sqrt(((D_rob - d_rob[:, None]) ** 2 * fw).sum(axis=1))
        PIVOT = float(max(np.quantile(s_dep, 0.99), np.quantile(s_rob, 0.99)))
        K = 0.0 if args.no_spread else float(args.spread_slope)
        n_above = int((s_dep > PIVOT).sum() + (s_rob > PIVOT).sum())
        print(f"[export] T0={T:.4f} (brier {cal['brier_before']:.4f} -> "
              f"{cal['brier_after']:.4f} on n={len(sub):,}); hinge k={K:.2f} "
              f"pivot={PIVOT:.2f} (local spread p50={np.median(s_dep):.2f} "
              f"p99={np.quantile(s_dep, 0.99):.2f}; {n_above}/{2*len(sub)} "
              f"local views above pivot)")

        if not args.binary_equivalent:
            # Conditional posterior on the deploy view with the SCALAR T:
            # >=99% of local rows sit below the hinge pivot by construction,
            # so t_eff == T for the fit population.
            prior = args.type_prior
            if prior is None:
                fake_ds = df[df.label == 1].drop_duplicates("dataset")
                prior = float((fake_ds["kind"] == "semisynthetic").mean())
            d2_dep = (D2_dep * fw).sum(axis=1)
            p_fake = np.clip(1.0 / (1.0 + np.exp(-d_dep / T)),
                             1e-6, 1.0 - 1e-6)
            tp = fit_type_posterior(y3, d2_dep, p_fake, semi_prior=prior)
            print(f"[export] type posterior: T2={tp['type_temperature']:.3f} "
                  f"b2={tp['type_bias']:.3f} q_max={tp['q_max']:.2f} "
                  f"(prior={prior:.4f}, cond-brier "
                  f"{tp['cond_brier_before']:.4f} -> "
                  f"{tp['cond_brier_after']:.4f}); local multiclass sn34 "
                  f"{tp['mc_sn34_before']:.4f} (q pinned) -> "
                  f"{tp['mc_sn34_after']:.4f} (fitted)")
            if tp["mc_sn34_after"] > tp["mc_sn34_before"]:
                T2, B2 = float(tp["type_temperature"]), float(tp["type_bias"])
                QMIN, QMAX = float(tp["q_min"]), float(tp["q_max"])
                type_fitted = True
            else:
                print("[export] fitted q does NOT beat pinned q locally -- "
                      "shipping binary-equivalent (q pinned at q_min)")
    if args.binary_equivalent:
        print("[export] --binary-equivalent: q pinned at q_min, no type fit")

    with torch.no_grad():
        ens.temperature.fill_(T)
        ens.temp_slope.fill_(K)
        ens.spread_pivot.fill_(PIVOT)
        ens.type_temperature.fill_(T2)
        ens.type_bias.fill_(B2)
        ens.q_min.fill_(QMIN)
        ens.q_max.fill_(QMAX)
        ens.p_min.fill_(PMIN)
        ens.p_max.fill_(PMAX)

    # -- write artifacts ----------------------------------------------------
    # The ensemble's own state_dict, cast once; keys match the template's
    # Ensemble exactly, so load is one strict load_state_dict.
    state = {k: (v.to(dtype) if v.is_floating_point() else v)
             for k, v in ens.state_dict().items()}
    # fusion weights and temperature stay fp32 regardless of export dtype --
    # the harness will cast the module again, and these two carry the score.
    state["w"] = ens.w.detach().float().cpu()
    state["temperature"] = ens.temperature.detach().float().cpu()
    state["temp_slope"] = ens.temp_slope.detach().float().cpu()
    state["spread_pivot"] = ens.spread_pivot.detach().float().cpu()
    state["type_temperature"] = ens.type_temperature.detach().float().cpu()
    state["type_bias"] = ens.type_bias.detach().float().cpu()
    state["q_min"] = ens.q_min.detach().float().cpu()
    state["q_max"] = ens.q_max.detach().float().cpu()
    state["p_min"] = ens.p_min.detach().float().cpu()
    state["p_max"] = ens.p_max.detach().float().cpu()

    dino = next((m for s, m in zip(specs, ens.branches)
                 if s.family == "hf_dinov3"), None)
    if dino is not None:
        (out / "config.json").write_text(
            json.dumps(dino.backbone.config.to_dict(), indent=2, default=str))
    clip_cfg = next((m.backbone.config.to_dict()
                     for s, m in zip(specs, ens.branches)
                     if s.family == "hf_clip"), None)

    src = render_template(specs, weights, image_size, clip_cfg=clip_cfg)
    bad = scan_allowlist(src)
    if bad:
        raise SystemExit(f"rendered model.py violates the allowlist: {bad}")
    (out / "model.py").write_text(src)

    from safetensors.torch import save_file
    save_file({k: v.contiguous() for k, v in state.items()},
              str(out / "model.safetensors"),
              metadata={"branches": ",".join(names),
                        "temperature": str(T),
                        "temp_slope": str(K),
                        "spread_pivot": str(PIVOT),
                        "type_temperature": str(T2),
                        "type_bias": str(B2),
                        "q_min": str(QMIN), "q_max": str(QMAX),
                        "p_min": str(PMIN), "p_max": str(PMAX),
                        "type_fitted": str(type_fitted),
                        "image_size": str(image_size)})

    (out / "model_config.yaml").write_text(yaml.safe_dump({
        "name": "v7-ensemble", "version": "7.1.0", "modality": "image",
        "dtype": args.dtype,
        "preprocessing": {"resize": [image_size, image_size]},
        # 3-class contract (gasbench image taxonomy 0=real/1=syn/2=semi);
        # pytorch_session validates the forward's last dim against this.
        "model": {"num_classes": 3, "weights_file": "model.safetensors"},
        "metadata": {"branches": ",".join(names), "temperature": T,
                     "temp_slope": K, "spread_pivot": PIVOT,
                     "type_temperature": T2, "type_bias": B2,
                     "q_min": QMIN, "q_max": QMAX,
                     "p_min": PMIN, "p_max": PMAX,
                     "type_fitted": type_fitted},
    }, sort_keys=False))

    # -- round-trip through the rendered model.py ---------------------------
    import importlib.util as il
    spec_m = il.spec_from_file_location("_exported", out / "model.py")
    mod = il.module_from_spec(spec_m)
    spec_m.loader.exec_module(mod)
    re_model = mod.load_model(str(out / "model.safetensors"))
    ref = {k: tuple(v.shape) for k, v in re_model.state_dict().items()}
    got = {k: tuple(v.shape) for k, v in state.items()}
    missing, extra = set(ref) - set(got), set(got) - set(ref)
    shape_bad = {k: (ref[k], got[k]) for k in set(ref) & set(got)
                 if ref[k] != got[k]}
    if missing or extra or shape_bad:
        raise SystemExit("rendered model.py != exported weights\n"
                         f"  only in model.py : {sorted(missing)[:6]}\n"
                         f"  only in weights  : {sorted(extra)[:6]}\n"
                         f"  shape mismatches : {list(shape_bad.items())[:6]}")
    probe = torch.randint(0, 256, (16, 3, image_size, image_size),
                          dtype=torch.uint8)
    with torch.no_grad():
        za = ens.float().cpu()(probe)
        zb = re_model.float()(probe)
    # Random-noise probes are maximally OOD: their spread sits far above the
    # hinge pivot, so the per-sample T_eff amplifies bf16-level module
    # discrepancies multiplicatively (measured 2026-08-12: 6.8e-3 logit /
    # 1.4e-3 prob on the probe vs the benign ~e-4 regime on real images,
    # where local spread p99 < pivot keeps the hinge inert). Gate on what the
    # score actually consumes: argmax must match EXACTLY (MCC), and
    # probability drift below 5e-3 (Brier moves by ~drift^2 -- invisible).
    logit_drift = (za - zb).abs().max().item()
    p_drift = (torch.softmax(za, -1) - torch.softmax(zb, -1)) \
        .abs().max().item()
    argmax_ok = bool((za.argmax(-1) == zb.argmax(-1)).all())
    if not argmax_ok or p_drift > 5e-3:
        raise SystemExit(f"round-trip mismatch: argmax_equal={argmax_ok}, "
                         f"probability drift {p_drift:.2e} "
                         f"(logit drift {logit_drift:.2e})")
    # The forward returns log-probabilities: exp must sum to 1 (=> softmax of
    # the output IS the composed p, so the binary collapse 1 - p[0] equals
    # sigmoid(dbar/t_eff) by construction -- verify.py proves the identity).
    psum_err = (za.exp().sum(-1) - 1.0).abs().max().item()
    if psum_err > 1e-4:
        raise SystemExit(f"output is not a log-prob vector: "
                         f"|exp(out).sum - 1| = {psum_err:.2e}")
    print(f"[export] round-trip OK (argmax exact on {probe.shape[0]} probes; "
          f"prob drift {p_drift:.2e}, logit drift {logit_drift:.2e}). "
          f"Wrote {out}/ -- zip and run gasbench --small before submitting.")


if __name__ == "__main__":
    main()
