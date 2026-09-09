"""v7 pre-flight. Run before every training cycle and before every export.

Checks are ordered cheapest-first and skip (not fail) on missing optional
dependencies, so the file is useful both on a laptop and on the training box.
"""
from __future__ import annotations

import traceback

CHECKS = []


def check(name):
    def deco(fn):
        CHECKS.append((name, fn))
        return fn
    return deco


@check("branch registry is internally consistent")
def _registry():
    from branches import BRANCHES, fusion_weights, resolve
    specs = resolve(sorted(BRANCHES))
    assert all(s.native_size > 0 and s.feat_dim > 0 for s in specs)
    assert all(len(s.norm_mean) == 3 and len(s.norm_std) == 3 for s in specs)
    assert all(s.fusion_weight > 0 for s in specs)
    # The BMF 4-branch core keeps its designed ratios (Table 1) and sums to 1
    # on its own. Auxiliary branches (dct) sit OUTSIDE that identity: the raw
    # registry sum exceeds 1 by their weight, and fusion_weights() renormalises
    # whatever subset ships -- which is the invariant that guards the baked
    # temperature, asserted below for every subset shape we ship.
    core = sum(BRANCHES[n].fusion_weight
               for n in ("dinov3", "clip", "convnext", "eva"))
    assert abs(core - 1.0) < 1e-9, f"BMF core weights sum to {core}"
    for subset in (["dinov3", "clip"],
                   ["dinov3", "clip", "convnext", "eva"],
                   sorted(BRANCHES)):
        w = fusion_weights(resolve(subset))
        assert abs(sum(w) - 1.0) < 1e-9, f"renormalised weights sum {sum(w)}"
    w5 = fusion_weights(resolve(["dinov3", "clip", "convnext", "eva", "dct"]))
    assert max(abs(a - b) for a, b in
               zip(sorted(w5), sorted([2/7, 2/7, 1/7, 1/7, 1/7]))) < 1e-9
    # Resolution variants must be exact twins of dinov3 except native_size,
    # and a swap must reproduce the shipped 5-way vector exactly (they take
    # dinov3's slot, never ship alongside it).
    base = BRANCHES["dinov3"]
    for vname, vsize in (("dinov3_336", 336), ("dinov3_384", 384)):
        v = BRANCHES[vname]
        assert v.native_size == vsize, f"{vname} native_size {v.native_size}"
        assert (v.model_id, v.feat_dim, v.head, v.lora_targets,
                v.norm_mean, v.norm_std, v.fusion_weight) == \
               (base.model_id, base.feat_dim, base.head, base.lora_targets,
                base.norm_mean, base.norm_std, base.fusion_weight), \
            f"{vname} drifted from dinov3 on a non-resolution field"
        wv = fusion_weights(resolve([vname, "clip", "convnext", "eva", "dct"]))
        assert max(abs(a - b) for a, b in
                   zip(sorted(wv), sorted([2/7, 2/7, 1/7, 1/7, 1/7]))) < 1e-9
    heads = {s.head for s in specs}
    assert heads <= {"mac_lite", "clip_linear", "bounded_cosine"}
    return (f"{len(specs)} branches; BMF core sums 1.0; renormalised subsets "
            f"sum 1.0 (5-way = 2/7,2/7,1/7,1/7,1/7); resolution variants are "
            f"exact twins and swap-exact")


@check("rendered template passes the import allowlist (all subsets)")
def _template():
    from allowlist import render_template, scan_allowlist
    from branches import fusion_weights, resolve
    clip_cfg = {"hidden_size": 1024, "intermediate_size": 4096,
                "num_hidden_layers": 24, "num_attention_heads": 16,
                "image_size": 224, "patch_size": 14,
                "hidden_act": "quick_gelu", "layer_norm_eps": 1e-05,
                "attention_dropout": 0.0, "projection_dim": 768}
    for subset in (["dinov3"], ["dinov3", "clip"],
                   ["dinov3", "clip", "convnext", "eva"],
                   ["dinov3_336"], ["dinov3_384"],
                   ["dinov3_336", "clip", "convnext", "eva"]):
        specs = resolve(subset)
        src = render_template(specs, fusion_weights(specs), 384,
                              clip_cfg=clip_cfg if "clip" in subset else None)
        assert not scan_allowlist(src)
        for bad in ("getattr(", "setattr(", "import os", "import json",
                    "peft", "__BRANCH_DEFS__", "__CLIP_CFG__"):
            assert bad not in src, f"{bad!r} present in rendered model.py"
        assert "def load_model" in src
    return "renders clean for 1-, 2- and 4-branch subsets incl. 336/384 swaps"


@check("sampler identities survive the kind level")
def _sampler():
    import pandas as pd
    import build_manifest as bm
    from data import balanced_weights
    reg = bm.load_registry()
    ov = bm.load_overrides()
    fixed = bm.apply_category_overrides(reg, ov)
    fold = bm.compute_category_folding(reg, fixed)
    kinds = bm.resolve_kind(reg, ov)
    rows = [{"dataset": n, "group": n,
             "label": bm.media_type_to_label(e["media_type"]),
             "category": fold[fixed[n]], "kind": kinds[n],
             "_n": 10000 if "gasstation" in n else 500}
            for n, e in reg.items()]
    base = pd.DataFrame(rows)
    df = base.loc[base.index.repeat(base["_n"])].reset_index(drop=True)
    kb = {"real": 0.50, "synthetic": 0.25, "semisynthetic": 0.25}
    for kind_balance in (None, kb):
        w, rep = balanced_weights(df, kind_balance=kind_balance,
                                  verbose=False)
        assert abs(w.sum() - 1.0) < 1e-9
        assert abs(rep["p_fake"] - 0.5) < 1e-6
        off = {c: v for c, v in rep["p_fake_given_cat"].items()
               if abs(v - 0.5) > 1e-6}
        assert not off, f"P(fake|category) off 0.5: {off}"
    _, r0 = balanced_weights(df, verbose=False)
    _, r1 = balanced_weights(df, kind_balance=kb, verbose=False)
    lift = r1["p_kind"]["semisynthetic"] / max(r0["p_kind"]["semisynthetic"],
                                               1e-9)
    assert lift > 1.5, f"kind level lifted semisynthetic only {lift:.2f}x"
    # Absolute guardrails (added 2026-08-25, after the v23 audit shrank the
    # semi pool to 5 faces-only datasets). The relative lift above passes
    # even for a 1-dataset pool; these catch the failure modes it cannot:
    # concentration (a tiny pool hoarding mass at high replay) and silent
    # water-fill clipping (which breaks the P(fake|category)=0.5 identity).
    import yaml as _yaml
    from pathlib import Path as _Path
    max_replay = float(_yaml.safe_load(
        (_Path(__file__).parent / "config.yaml").read_text()
    )["sampler"]["max_replay"])
    p_semi = r1["p_kind"]["semisynthetic"]
    assert 0.05 <= p_semi <= 0.30, (
        f"P(semisynthetic)={p_semi:.4f} outside [0.05, 0.30]. The semi "
        f"pool's category geometry changed -- retune sampler.kind_balance "
        f"(shares are cell-local: realized share tracks WHICH categories "
        f"hold semis, see data.py:_kind_shares).")
    amp_semi = r1["amp_kind"].get("semisynthetic", 0.0)
    assert amp_semi < max_replay / 2, (
        f"semisynthetic peak replay {amp_semi:.1f}x >= max_replay/2 "
        f"({max_replay / 2:.1f}) -- the semi pool is too small for its "
        f"kind_balance share; lower it or grow the pool.")
    for r, tag in ((r0, "unbalanced"), (r1, "balanced")):
        assert not r["replay_cap_bound"], (
            f"water-fill replay cap BOUND on the {tag} config -- the "
            f"P(fake|category)=0.5 identity no longer holds; rebalance.")

    # The [0.05, 0.30] band above is wide enough to pass a share that misses
    # its target 2.6x, which is exactly what shipped on 2026-09-08 (0.25 asked,
    # 0.0952 delivered, printed "[balanced]"). Assert the IDENTITY instead:
    # _kind_shares is cell-local, so when every cell holding semis also holds
    # synthetics the realized share is exactly
    #     ceiling * kb_semi / (kb_semi + kb_syn),   ceiling = label_w * sum(q_c)
    # This catches a real regression in the share logic without demanding a
    # global target the registry's category geometry cannot supply.
    tgt_semi = r1["kind_target"]["semisynthetic"]
    ceil_semi = r1["kind_ceiling"]["semisynthetic"]
    pred_semi = ceil_semi * kb["semisynthetic"] / (kb["semisynthetic"]
                                                   + kb["synthetic"])
    assert abs(p_semi - pred_semi) < 1e-6, (
        f"P(semisynthetic)={p_semi:.4f} != the cell-local prediction "
        f"{pred_semi:.4f} (ceiling {ceil_semi:.4f} x cell split). "
        f"_kind_shares no longer behaves as its docstring describes.")
    unreachable = tgt_semi > ceil_semi + 1e-9
    return (f"P(fake)=0.5 and per-category 0.5 with and without kind level; "
            f"semisynthetic x{lift:.2f}, P(semi)={p_semi:.3f} "
            f"(target {tgt_semi:.3f}, ceiling {ceil_semi:.3f}"
            f"{' -- UNREACHABLE, corrected in the loss by '
               'kind_class_weights' if unreachable else ''}), "
            f"peak replay {amp_semi:.1f}x < {max_replay / 2:.1f}, cap unbound")


@check("transformers version matches gasbench's pin")
def _transformers_pin():
    import re
    import transformers
    import paths
    src = paths.find_gasbench_src()
    assert src is not None, f"gasbench checkout not found. {paths.GASBENCH_HINT}"
    py = (src.parent / "pyproject.toml").read_text()
    m = re.search(r'"transformers==([^"]+)"', py)
    assert m, "gasbench pyproject.toml no longer pins transformers== exactly"
    want, got = m.group(1), transformers.__version__
    # The submission is reconstructed inside the VALIDATOR's environment,
    # which gasbench's pin mirrors. A different training/export transformers
    # can change the HF module tree itself: 2026-08-25 a fresh venv built
    # DINOv3 with an extra `.model.` nesting level, export round-tripped
    # against its own naming, and gasbench (flat naming under the pin)
    # refused every backbone key -- a submission that would have scored 0.
    assert got == want, (
        f"transformers {got} != gasbench pin {want}. Backbone state_dict "
        f"naming differs across versions; run "
        f"pip install 'transformers=={want}' in THIS venv, then retrain/"
        f"re-export -- checkpoints saved under the other version will not "
        f"strict-load.")
    return f"transformers {got} == gasbench pin"


@check("eval contract unchanged in gasbench")
def _contract():
    import paths
    src = paths.find_gasbench_src()
    assert src is not None, f"gasbench checkout not found. {paths.GASBENCH_HINT}"
    root = src / "gasbench"
    ib = (root / "benchmarks" / "image_bench.py").read_text()
    assert "augment_level: Optional[int] = 0" in ib
    assert "crop_prob: float = 0.0" in ib
    tf = (root / "processing" / "transforms.py").read_text()
    assert "cv2.INTER_LINEAR" in tf
    cm = (root / "benchmarks" / "common.py").read_text()
    assert "aug_weight: float = 0.2" in cm
    return "level=0, crop_prob=0, INTER_LINEAR, 0.8/0.2 blend"


@check("factorized 3-class export: binary parity for any q, hinge, clamps")
def _factorized():
    import torch
    from model import binary_margin, collapse_binary, type_margin
    torch.manual_seed(34)
    n = 512
    zs = [torch.randn(n, 3) * 3.0 for _ in range(3)]
    w = [1 / 6, 1 / 3, 1 / 2]
    wt = torch.tensor(w).view(1, -1)
    T, k, pivot = 0.6321, 0.85, 1.2

    # helper identities on one branch
    z = zs[0]
    p = torch.softmax(z, -1)
    assert (torch.sigmoid(binary_margin(z)) - (1 - p[:, 0])).abs().max() < 1e-6
    assert (torch.softmax(collapse_binary(z), -1)[:, 0]
            - p[:, 0]).abs().max() < 1e-6
    assert (torch.sigmoid(type_margin(z))
            - p[:, 2] / (p[:, 1] + p[:, 2])).abs().max() < 1e-5

    # fused binary margin: per-branch margins fused linearly -- THE convention.
    dm = torch.stack([binary_margin(zi) for zi in zs], dim=1)
    dbar = (wt * dm).sum(dim=1)
    spread = ((wt * (dm - dbar.unsqueeze(1)) ** 2).sum(dim=1)).sqrt()
    assert (spread >= 0).all()
    # Guard against the silent-break variant: margin of the FUSED logits is a
    # genuinely different quantity for 3-logit heads (logsumexp is nonlinear).
    d_of_fused = binary_margin(sum(wi * zi for wi, zi in zip(w, zs)))
    assert (d_of_fused - dbar).abs().max() > 1e-3

    t_eff = T * (1.0 + k * torch.clamp(spread - pivot, min=0.0))
    p_fake = torch.clamp(torch.sigmoid(dbar / t_eff), 1e-6, 1.0 - 1e-6)

    for qmode, q in (("random", torch.rand(n).clamp(1e-3, 1.0)),
                     ("pinned", torch.full((n,), 1e-3))):
        out = torch.log(torch.stack(
            [1.0 - p_fake, p_fake * (1.0 - q), p_fake * q], dim=1))
        pr = torch.softmax(out, -1)
        assert (out.exp().sum(-1) - 1.0).abs().max() < 1e-5
        # BINARY PARITY: the benchmark's collapse (1 - p[0]) equals the
        # calibrated sigmoid(dbar/t_eff) for ANY q -- q cannot move the
        # binary score, only the multiclass one.
        assert ((1.0 - pr[:, 0]) - p_fake).abs().max() < 1e-6
        # decisions equal a 2-logit export built from the same (dbar, t_eff)
        two = torch.stack([-dbar / t_eff, dbar / t_eff], dim=1) * 0.5
        assert (((1.0 - pr[:, 0]) > 0.5)
                == (torch.softmax(two, -1)[:, 1] > 0.5)).all(), qmode
        if qmode == "pinned":
            # binary-equivalent mode: the semi column carries <= q_min mass
            assert pr[:, 2].max() <= 1e-3 + 1e-6

    # below-pivot rows reproduce the scalar-T calibration exactly
    below = spread <= pivot
    assert below.any() and (~below).any()
    p_scalar = torch.clamp(torch.sigmoid(dbar / T), 1e-6, 1.0 - 1e-6)
    assert (p_fake[below] - p_scalar[below]).abs().max() < 1e-6
    assert (p_fake[~below] - p_scalar[~below]).abs().max() > 1e-4

    # probability clamp (p_min/p_max buffers): pure-Brier insurance. Inside
    # 0 < p_min < 0.5 < p_max < 1 it must (a) never flip a decision -- the
    # benchmark thresholds 1-p[0] > 0.5, so MCC is invariant by construction,
    # (b) bind exactly at the bounds, (c) leave interior probabilities
    # byte-identical, and (d) preserve the binary-parity identity.
    pmin, pmax = 0.08, 0.92
    assert (p_fake < pmin).any() and (p_fake > pmax).any()  # test has power
    p_cl = torch.clamp(torch.sigmoid(dbar / t_eff), pmin, pmax)
    assert ((p_cl > 0.5) == (p_fake > 0.5)).all()
    assert p_cl.min() >= pmin - 1e-7 and p_cl.max() <= pmax + 1e-7
    interior = (p_fake > pmin) & (p_fake < pmax)
    assert (p_cl[interior] - p_fake[interior]).abs().max() < 1e-7
    qc = torch.rand(n).clamp(1e-3, 1.0)
    prc = torch.softmax(torch.log(torch.stack(
        [1.0 - p_cl, p_cl * (1.0 - qc), p_cl * qc], dim=1)), -1)
    assert ((1.0 - prc[:, 0]) - p_cl).abs().max() < 1e-6

    # multiclass argmax structure: fake argmax iff p_fake > 1/(1+max(q,1-q));
    # a default-confident-synthetic q keeps this near the binary 0.5.
    q = torch.full((n,), 0.15)
    pm = torch.stack([1.0 - p_fake, p_fake * (1.0 - q), p_fake * q], dim=1)
    thr = 1.0 / (1.0 + 0.85)
    assert ((pm.argmax(-1) > 0) == (p_fake > thr)).all()
    return ("binary collapse == sigmoid(dbar/t_eff) for any q; margin-fusion "
            "convention guarded; below-pivot == scalar exactly; pinned q ==> "
            "semi mass <= q_min; fake-argmax threshold 1/(1+max(q,1-q)); "
            "p-clamp decision-invariant, bound-exact, parity-preserving")


@check("gasbench 3-class image contract unchanged")
def _multiclass_contract():
    import paths
    src = paths.find_gasbench_src()
    assert src is not None, f"gasbench checkout not found. {paths.GASBENCH_HINT}"
    root = src / "gasbench"
    con = (root / "constants.py").read_text()
    # image taxonomy the y3 derivation and the head ordering rely on
    assert "IMAGE_MEDIA_TYPE_TO_LABEL" in con
    assert '"real": 0' in con and '"synthetic": 1' in con
    assert '"semisynthetic": 2' in con
    mt = (root / "benchmarks" / "utils" / "metrics.py").read_text()
    # the binary collapse the parity proof leans on, and the multiclass leg
    assert "1.0 - pred_probs[0]" in mt
    assert "def calculate_multiclass_mcc" in mt
    assert "def calculate_multiclass_brier" in mt
    return "image = {real:0, synthetic:1, semisynthetic:2}; binary collapse " \
           "is 1 - p[0]; multiclass MCC/Brier present"


@check("dct branch: orthonormal basis, uint8 forward, no-resample at 384")
def _dct_branch():
    # Needs torch + timm but NO network: the backbone is scratch-built, so
    # this is the one branch that can be fully smoke-tested offline.
    import torch
    from branches import BRANCHES
    from model import BranchModel, dct_basis
    spec = BRANCHES["dct"]
    assert spec.native_size == 384, \
        "dct native_size must equal data.image_size (no resample in front)"
    d = dct_basis(spec.native_size)
    eye = d @ d.transpose(0, 1)
    err = (eye - torch.eye(d.shape[0])).abs().max().item()
    assert err < 1e-4, f"DCT basis not orthonormal (max |D@D.T - I| = {err:.2e})"
    m = BranchModel(spec, gradient_checkpointing=False)
    n_train = sum(p.numel() for p in m.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in m.parameters())
    assert n_train == n_all, "dct branch must be fully trainable (scratch)"
    m.eval()
    x = torch.randint(0, 256, (2, 3, 384, 384), dtype=torch.uint8)
    with torch.no_grad():
        z = m(x)
    assert z.shape == (2, 3)
    assert m.merge_and_strip() == 0   # no LoRA on the scratch net
    return (f"basis orthonormal (err {err:.1e}); {n_all/1e6:.1f}M params all "
            f"trainable; uint8 [2,3,384,384] -> [2,3]")


@check("branch models build, forward uint8, and round-trip the template")
def _end_to_end():
    # heavy: needs torch + transformers + timm + network for pretrained
    # weights. On the training box this is the real test; elsewhere it skips.
    import torch
    from branches import resolve, fusion_weights
    from model import BranchModel
    spec = resolve(["clip"])[0]
    m = BranchModel(spec, gradient_checkpointing=False)
    m.eval()
    x = torch.randint(0, 256, (2, 3, 384, 384), dtype=torch.uint8)
    with torch.no_grad():
        z = m(x)
    assert z.shape == (2, 3)
    m.merge_and_strip()
    with torch.no_grad():
        z2 = m(x)
    assert (z - z2).abs().max() < 1e-3
    return "clip branch: uint8 [2,3,384,384] -> [2,3]; merge exact"


def main() -> None:
    print("v7 pre-flight")
    print("-" * 72)
    passed = failed = skipped = 0
    for name, fn in CHECKS:
        try:
            msg = fn()
            print(f"[  ok  ] {name}\n         {msg}")
            passed += 1
        except ImportError as e:
            print(f"[ skip ] {name}\n         missing dependency: {e}")
            skipped += 1
        except Exception:
            print(f"[ FAIL ] {name}")
            traceback.print_exc(limit=3)
            failed += 1
    print("-" * 72)
    print(f"{passed} passed, {failed} failed, {skipped} skipped")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
