"""Self-tests for audit_shortcuts.py. Run before trusting any audit number.

Every detection path is exercised against a PLANTED positive and a NULL
control (no false alarm), the eval-time intervention hook is proven
bit-transparent when idle, blocking is proven to separate cross-source
shortcut signal from dataset memorisation, and the scoring helpers are
pinned to documented values.

Conventions follow verify.py: module CHECKS list + @check decorator,
[ ok ]/[ FAIL ]/[ skip ] lines, exit 1 on failure, ImportError -> skip so
the CPU-only checks run on the GPU-less VM (checks needing torch/gasbench
skip there and run on the training box).
"""
from __future__ import annotations

import inspect
import json
import math
import tempfile
import traceback
import types
from pathlib import Path

CHECKS = []


def check(name):
    def deco(fn):
        CHECKS.append((name, fn))
        return fn
    return deco


def _args(**kw) -> types.SimpleNamespace:
    base = dict(tag="verify", seed=34, image_size=384, batch_size=16,
                workers=0)
    base.update(kw)
    return types.SimpleNamespace(**base)


def _toy_planted(n_datasets=24, rows=60, planted=True, seed=7):
    """Toy manifest: half the pseudo-datasets real, half fake. planted=True
    ties ext/file_format and min_side to the LABEL consistently across
    datasets (a cross-source shortcut); planted=False draws every feature
    iid (null control)."""
    import numpy as np
    import pandas as pd
    rng = np.random.default_rng(seed)
    recs = []
    for di in range(n_datasets):
        label = di % 2
        for ri in range(rows):
            if planted:
                png = rng.random() < (0.9 if label else 0.2)
                ms = float(rng.lognormal(math.log(512 if label else 800),
                                         0.15))
            else:
                png = rng.random() < 0.5
                ms = float(rng.lognormal(math.log(640), 0.3))
            recs.append({
                "image_id": f"ds{di:02d}/img_{ri:06d}",
                "dataset": f"ds{di:02d}", "label": label,
                "min_side": ms, "max_side": ms * 1.3, "aspect": 1.3,
                "megapixels": ms * ms * 1.3 / 1e6,
                "file_bytes": float(rng.integers(20_000, 400_000)),
                "bytes_per_pixel": float(rng.uniform(0.1, 3.0)),
                "m_eval": 384.0 / ms,
                "width": ms * 1.3, "height": ms,
                "ext": "png" if png else "jpg",
                "file_format": "PNG" if png else "JPEG",
                "pil_mode": "RGB", "source_format": "parquet",
            })
    return pd.DataFrame(recs)


@check("planted metadata shortcut is detected; null control stays quiet")
def _planted_metadata():
    import audit_shortcuts as ash
    df = _toy_planted(planted=True)
    X, _ = ash.featurize(df)
    y = df["label"].to_numpy()
    auc_p, _ = ash.oof_probe_auc(X, y, ash.blocked_folds(df))
    assert auc_p > 0.80, f"planted shortcut missed: blocked AUC {auc_p:.3f}"
    a_ext = ash.folded_auc(
        df["ext"].map(df.groupby("ext")["label"].mean()).to_numpy(), y)
    assert a_ext > 0.65, f"M2 failed to name ext: {a_ext:.3f}"

    null = _toy_planted(planted=False)
    Xn, _ = ash.featurize(null)
    auc_n, _ = ash.oof_probe_auc(Xn, null["label"].to_numpy(),
                                 ash.blocked_folds(null))
    assert 0.40 < auc_n < 0.60, f"false alarm on null: AUC {auc_n:.3f}"
    return f"planted AUC {auc_p:.3f} > 0.80; null AUC {auc_n:.3f} in (.4,.6)"


@check("blocked folds separate shortcut signal from dataset memorisation")
def _blocking_honesty():
    import numpy as np
    import pandas as pd
    import audit_shortcuts as ash
    rng = np.random.default_rng(11)
    recs = []
    # 60 datasets; each has a UNIQUE metadata constant, independent of label
    # across datasets -> memorisable row-level, useless cross-dataset.
    for di in range(60):
        label = di % 2
        ms = float(rng.lognormal(math.log(640), 0.4))    # per-dataset const
        png = bool(rng.random() < 0.5)
        for ri in range(30):
            recs.append({
                "image_id": f"ds{di:02d}/img_{ri:06d}",
                "dataset": f"ds{di:02d}", "label": label,
                "min_side": ms, "max_side": ms, "aspect": 1.0,
                "megapixels": ms * ms / 1e6,
                "file_bytes": ms * 100.0, "bytes_per_pixel": 1.0,
                "m_eval": 384.0 / ms, "width": ms, "height": ms,
                "ext": "png" if png else "jpg",
                "file_format": "PNG" if png else "JPEG",
                "pil_mode": "RGB", "source_format": "parquet",
            })
    df = pd.DataFrame(recs)
    X, _ = ash.featurize(df)
    y = df["label"].to_numpy()
    auc_naive, _ = ash.oof_probe_auc(X, y, ash.naive_folds(len(df), seed=34))
    auc_blocked, _ = ash.oof_probe_auc(X, y, ash.blocked_folds(df))
    assert auc_naive > 0.85, f"naive folds should memorise: {auc_naive:.3f}"
    assert 0.30 < auc_blocked < 0.70, \
        f"blocked folds leaked dataset identity: {auc_blocked:.3f}"
    return (f"naive {auc_naive:.3f} (memorised) vs blocked "
            f"{auc_blocked:.3f} (honest)")


def _write_toy_images(tmp: Path, n_per_class=6, size=(384, 288), seed=5,
                      fake_jpeg_q=None):
    """Noise images on disk: reals as PNG; fakes as PNG too unless
    fake_jpeg_q is set (then fakes carry a JPEG-q artifact -- the planted
    pixel shortcut). Returns a manifest-like frame."""
    import numpy as np
    import pandas as pd
    from PIL import Image
    rng = np.random.default_rng(seed)
    recs = []
    for label in (0, 1):
        for i in range(n_per_class):
            a = rng.integers(0, 256, size=(size[1], size[0], 3),
                             dtype=np.uint8)
            img = Image.fromarray(a)
            name = f"l{label}_{i:03d}"
            if label == 1 and fake_jpeg_q:
                p = tmp / f"{name}.jpg"
                img.save(p, format="JPEG", quality=fake_jpeg_q)
            else:
                p = tmp / f"{name}.png"
                img.save(p, format="PNG")
            recs.append({"image_id": name, "path": str(p), "label": label,
                         "dataset": f"toy-{'fake' if label else 'real'}-"
                                    f"{i % 4}"})
    return pd.DataFrame(recs)


@check("InterventionDataset: no-op fn is byte-identical; active fn is "
       "deterministic")
def _intervention_hook():
    import torch
    import audit_shortcuts as ash
    from data import ManifestDataset, ViewConfig
    InterventionDataset, PrechainDataset = ash._dataset_classes()
    with tempfile.TemporaryDirectory() as td:
        df = _write_toy_images(Path(td), n_per_class=4)
        cfg = ViewConfig(image_size=128)
        base = ManifestDataset(df, cfg, train=False, eval_mode="deploy",
                               seed=7)
        noop = InterventionDataset(df, cfg, fn=lambda img, i: img, seed=7)
        for i in range(len(df)):
            xb, xn = base[i]["x"], noop[i]["x"]
            assert xb.dtype == torch.uint8 and xb.shape == (3, 128, 128)
            assert torch.equal(xb, xn), f"no-op hook perturbed row {i}"

        reg = ash._interventions()
        jpeg60 = reg["jpeg_sweep"][2][1]        # q=60
        a1 = InterventionDataset(df, cfg, fn=jpeg60, seed=7)
        a2 = InterventionDataset(df, cfg, fn=jpeg60, seed=7)
        changed = 0
        for i in range(len(df)):
            x1, x2 = a1[i]["x"], a2[i]["x"]
            assert torch.equal(x1, x2), "active fn nondeterministic"
            assert x1.dtype == torch.uint8
            if not torch.equal(x1, base[i]["x"]):
                changed += 1
        assert changed >= len(df) - 1, "jpeg fn changed almost nothing"

        # PrechainDataset renders without error and stays deterministic.
        pc1 = PrechainDataset(df, ViewConfig(image_size=128, prechain=True),
                              variant="prechain", seed=7)
        pc2 = PrechainDataset(df, ViewConfig(image_size=128, prechain=True),
                              variant="prechain", seed=7)
        assert torch.equal(pc1[0]["x"], pc2[0]["x"])
    return "no-op byte-identical; jpeg-q60 deterministic and non-trivial"


@check("planted pixel shortcut survives the render and is detected (T1)")
def _planted_pixels():
    import audit_shortcuts as ash
    from data import ViewConfig
    with tempfile.TemporaryDirectory() as td:
        df = _write_toy_images(Path(td), n_per_class=60, size=(256, 256),
                               fake_jpeg_q=60)
        stats = ash.collect_tensor_stats(df, ViewConfig(image_size=128),
                                         _args(), variant=None)
        y = df["label"].to_numpy()
        auc, _ = ash.oof_probe_auc(stats, y, ash.blocked_folds(df))
        assert auc > 0.80, f"jpeg-vs-png pixel shortcut missed: {auc:.3f}"
    with tempfile.TemporaryDirectory() as td:
        # null: identical processing for both classes.
        dfn = _write_toy_images(Path(td), n_per_class=60, size=(256, 256),
                                fake_jpeg_q=None)
        stats_n = ash.collect_tensor_stats(dfn, ViewConfig(image_size=128),
                                           _args(), variant=None)
        auc_n, _ = ash.oof_probe_auc(stats_n, dfn["label"].to_numpy(),
                                     ash.blocked_folds(dfn))
        assert 0.25 < auc_n < 0.75, f"false alarm on null: {auc_n:.3f}"
    return f"planted AUC {auc:.3f} > 0.80; identical-processing {auc_n:.3f}"


@check("metric helpers pinned: rank_auc ties, auc_ceiling, spearman")
def _metric_pins():
    import numpy as np
    import audit_shortcuts as ash
    y = np.array([0, 0, 0, 1, 1, 1])
    assert ash.rank_auc(np.array([1, 2, 3, 4, 5, 6]), y) == 1.0
    assert ash.rank_auc(np.array([6, 5, 4, 3, 2, 1]), y) == 0.0
    assert ash.rank_auc(np.ones(6), y) == 0.5           # all-tied
    assert abs(ash.auc_ceiling(0.5) - 0.5) < 1e-6
    ceil95 = ash.auc_ceiling(0.95)
    assert abs(ceil95 - 0.878) < 0.01, ceil95           # documented ~0.88
    assert abs(ash.spearman([1, 2, 3, 4], [10, 20, 30, 40]) - 1.0) < 1e-9
    assert abs(ash.spearman([1, 2, 3, 4], [4, 3, 2, 1]) + 1.0) < 1e-9
    a = ash.folded_auc(np.array([6, 5, 4, 3, 2, 1]), y)
    assert a == 1.0                                     # direction-agnostic
    return "separable 1.0/0.0, ties 0.5, ceiling(0.95)~0.878, spearman +/-1"


@check("pairs machinery: clique_pairs contract and confound correlation")
def _pairs_math():
    import numpy as np
    import pandas as pd
    import audit_shortcuts as ash
    from data import clique_pairs
    rng = np.random.default_rng(3)
    recs = []
    for u in range(8):
        for label in (0, 1):
            for i in range(10):
                recs.append({"split_unit": f"unit{u}", "label": label,
                             "dataset": f"d{u}_{label}"})
    pool = pd.DataFrame(recs, index=np.arange(1000, 1000 + len(recs)))
    # clique_pairs uses the PASSED frame's index -- the reset_index contract.
    pool_r = pool.reset_index(drop=True)
    pairs = clique_pairs(pool_r)
    assert len(pairs) == 8
    for unit, (neg, pos) in pairs.items():
        assert (pool_r.loc[neg, "label"] == 0).all()
        assert (pool_r.loc[pos, "label"] == 1).all()
        assert neg.max() < len(pool_r) and pos.max() < len(pool_r)

    # P2 math: a planted linear confound must cross the 0.5 rho threshold.
    gaps = np.arange(8, dtype=float)
    png_delta_confounded = gaps * 0.1 + rng.normal(0, 0.01, 8)
    png_delta_clean = rng.permutation(8).astype(float)
    assert abs(ash.spearman(png_delta_confounded, gaps)) > 0.9
    assert abs(ash.spearman(png_delta_clean, gaps)) < 0.85
    return "clique index contract holds; planted confound rho > 0.9"


@check("ledger round-trip, baseline stamping, regression escalation")
def _ledger():
    import audit_shortcuts as ash
    args = _args(tag="t1")
    with tempfile.TemporaryDirectory() as td:
        hist = Path(td) / "h.jsonl"
        r1 = ash.make_row("slices", "S4_dataset_acc", "val_id",
                          {"dataset": "toy"}, 100, {"acc": 0.95}, "ok",
                          args, headline="acc", higher_is_bad=False)
        ash._regress([r1], ash.last_history(hist))
        assert r1["baseline"] is True
        ash.append_ledger(hist, [r1])
        prev = ash.last_history(hist)
        assert r1["key_str"] in prev

        r2 = ash.make_row("slices", "S4_dataset_acc", "val_id",
                          {"dataset": "toy"}, 100, {"acc": 0.85}, "ok",
                          args, headline="acc", higher_is_bad=False)
        ash._regress([r2], prev)
        assert r2["baseline"] is False
        assert abs(r2["delta"] + 0.10) < 1e-9
        assert r2["verdict"] == "warn" and r2.get("regressed")

        r3 = ash.make_row("metadata", "M1_metadata_probe", "all", None,
                          100, {"auc": 0.60}, "ok", args, headline="auc")
        prev3 = {r3["key_str"]: {"metrics": {"auc": 0.52}}}
        ash._regress([r3], prev3)
        assert r3["verdict"] == "warn"      # auc UP is bad (higher_is_bad)
        # corrupt line tolerated
        hist.write_text(hist.read_text() + "not json\n")
        assert r1["key_str"] in ash.last_history(hist)
    return "round-trip ok; acc-drop and auc-rise both escalate to warn"


@check("stratified_sample is deterministic and proportional")
def _sampling():
    import numpy as np
    import pandas as pd
    import audit_shortcuts as ash
    rng = np.random.default_rng(9)
    df = pd.DataFrame({
        "image_id": [f"i{i:05d}" for i in range(3000)],
        "dataset": rng.choice(["a", "b", "c"], 3000, p=[0.6, 0.3, 0.1]),
        "label": rng.integers(0, 2, 3000),
    })
    s1 = ash.stratified_sample(df, 300, seed=34)
    s2 = ash.stratified_sample(df, 300, seed=34)
    assert list(s1["image_id"]) == list(s2["image_id"])
    assert abs(len(s1) - 300) <= len(df["dataset"].unique()) * 2
    share = (s1["dataset"] == "a").mean()
    assert 0.45 < share < 0.75, f"proportionality broken: {share:.2f}"
    for _, g in df.groupby(["label", "dataset"]):
        assert len(s1[(s1["label"] == g["label"].iloc[0])
                      & (s1["dataset"] == g["dataset"].iloc[0])]) >= 1
    return f"deterministic; n={len(s1)}; dataset-a share {share:.2f}"


@check("every intervention fn is label-blind by signature")
def _label_blind():
    import audit_shortcuts as ash
    reg = ash._interventions()
    n = 0
    for name, settings in reg.items():
        for key, fn in settings:
            params = set(inspect.signature(fn).parameters)
            assert not params & {"label", "y", "target", "kind"}, \
                f"{name}{key} signature mentions the label: {params}"
            n += 1
    return f"{n} fns across {len(reg)} interventions, none see a label"


@check("single-block nuisance surfaces as unmeasurable warn, never ok")
def _concentrated_nuisance():
    # Review finding 2026-08-19: a probe target confined to one source block
    # makes its fold single-class-in-train -> skipped -> pooled OOF AUC NaN
    # -> bucket(NaN)=='ok'. probe_verdict must convert that to warn.
    import numpy as np
    import pandas as pd
    import audit_shortcuts as ash
    rng = np.random.default_rng(21)
    recs = []
    for di in range(10):
        for ri in range(100):
            ms = float(rng.lognormal(math.log(640), 0.3))
            recs.append({"image_id": f"d{di:02d}/i{ri:04d}",
                         "dataset": f"d{di:02d}", "label": di % 2,
                         "min_side": ms, "max_side": ms, "aspect": 1.0,
                         "megapixels": ms * ms / 1e6,
                         "file_bytes": float(rng.integers(1e4, 1e6)),
                         "bytes_per_pixel": float(rng.uniform(0.1, 3)),
                         "m_eval": 384.0 / ms, "width": ms, "height": ms,
                         "ext": "jpg", "file_format": "JPEG",
                         "pil_mode": "RGB", "source_format": "parquet"})
    df = pd.DataFrame(recs)
    X, _ = ash.featurize(df)
    folds = ash.blocked_folds(df)
    target = (df["dataset"] == "d00").to_numpy(int)   # one block only
    auc, oof = ash.oof_probe_auc(X, target, folds)
    v, ex = ash.probe_verdict(auc, oof, ash.TH["survival_auc"])
    assert v == "warn" and ex.get("unmeasurable"), (v, ex, auc)
    # a well-spread target stays measurable with full coverage
    spread = rng.integers(0, 2, len(df))
    auc2, oof2 = ash.oof_probe_auc(X, spread, folds)
    v2, ex2 = ash.probe_verdict(auc2, oof2, ash.TH["survival_auc"])
    assert not ex2.get("unmeasurable") and ex2["oof_coverage"] == 1.0
    assert not math.isnan(auc2)
    # blocked_folds still yields both labels in every fold on this frame
    for f in np.unique(folds):
        assert df.loc[folds == f, "label"].nunique() == 2
    return (f"concentrated -> warn/unmeasurable (auc={auc}); "
            f"spread -> {v2} auc {auc2:.3f} cov 1.0")


@check("thresholds table is well-formed and bucket() respects it")
def _thresholds():
    import audit_shortcuts as ash
    for name, th in ash.TH.items():
        assert "warn" in th, f"TH[{name!r}] missing warn"
        if "hard" in th:
            assert th["hard"] > th["warn"], f"TH[{name!r}] hard <= warn"
    assert ash.bucket(0.9, ash.TH["metadata_auc"]) == "hard"
    assert ash.bucket(0.75, ash.TH["metadata_auc"]) == "warn"
    assert ash.bucket(0.6, ash.TH["metadata_auc"]) == "ok"
    assert ash.bucket(float("nan"), ash.TH["metadata_auc"]) == "ok"
    assert ash.bucket(-0.2, ash.TH["regress_acc"],
                      higher_is_bad=False) == "warn"
    return f"{len(ash.TH)} thresholds; hard>warn everywhere; nan-safe"


def main() -> None:
    print("audit_shortcuts self-tests")
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
