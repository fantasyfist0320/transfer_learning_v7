"""Structural checks for the 2026-08-07 view-pipeline deltas.

Companion to verify.py (same check/deco convention, cheapest-first, CPU-only,
no manifest needed -- test images are synthesised into a tempdir). Each check
pins one mechanical property that EXPECTED_OUTCOMES.md relies on; none of them
measures model quality, that is the A/B's job.
"""
from __future__ import annotations

import inspect
import random
import tempfile
import traceback
from pathlib import Path

CHECKS = []


def check(name):
    def deco(fn):
        CHECKS.append((name, fn))
        return fn
    return deco


def _test_images(tmp: Path):
    """Three PNGs with gradient+noise content at awkward sizes, plus a df."""
    import numpy as np
    import pandas as pd
    from PIL import Image

    rng = np.random.RandomState(34)
    rows = []
    for k, (w, h, kind, label) in enumerate([(500, 400, "real", 0),
                                             (640, 640, "synthetic", 1),
                                             (300, 520, "semisynthetic", 1)]):
        gx = np.linspace(0, 255, w, dtype=np.float32)[None, :, None]
        gy = np.linspace(0, 255, h, dtype=np.float32)[:, None, None]
        a = (0.5 * gx + 0.5 * gy + rng.randint(0, 64, (h, w, 3))).clip(0, 255)
        p = tmp / f"img_{k}.png"
        Image.fromarray(a.astype(np.uint8)).save(p)
        rows.append({"path": str(p), "label": label, "kind": kind,
                     "dataset": f"ds{k}", "category": "c", "group": f"g{k}"})
    return pd.DataFrame(rows)


@check("prechain arm frequencies match config and the arm helper is label-blind")
def _prechain_arms():
    from data import ViewConfig, _prechain_arm, source_prechain

    assert "label" not in inspect.signature(source_prechain).parameters
    assert "label" not in inspect.signature(_prechain_arm).parameters

    cfg = ViewConfig(prechain_jpeg_p=0.35, prechain_double_p=0.10)
    n = 10_000
    counts = {"none": 0, "jpeg": 0, "double": 0, "webp": 0}
    for k in range(n):  # deterministic uniform grid: counts exact within 1
        counts[_prechain_arm((k + 0.5) / n, cfg)] += 1
    want = {"none": 0.30, "jpeg": 0.35, "double": 0.10, "webp": 0.25}
    for arm, p in want.items():
        assert abs(counts[arm] / n - p) < 1e-3, (arm, counts)

    zero = ViewConfig()  # dataclass defaults: double arm off, old three arms
    counts0 = {"none": 0, "jpeg": 0, "double": 0, "webp": 0}
    for k in range(n):
        counts0[_prechain_arm((k + 0.5) / n, zero)] += 1
    assert counts0["double"] == 0
    return f"none/jpeg/double/webp = {tuple(counts[a] / n for a in want)}"


@check("double-JPEG arm shifts the lattice: dims shrink by 1-7px per axis")
def _double_jpeg():
    import numpy as np
    from PIL import Image
    from data import ViewConfig, source_prechain

    cfg = ViewConfig(prechain_none_p=0.0, prechain_jpeg_p=0.0,
                     prechain_double_p=1.0)
    img = Image.fromarray(
        np.random.RandomState(7).randint(0, 255, (400, 500, 3), dtype=np.uint8))
    for seed in range(20):
        out = source_prechain(img, random.Random(seed), cfg)
        dw, dh = img.size[0] - out.size[0], img.size[1] - out.size[1]
        assert 1 <= dw <= 7 and 1 <= dh <= 7, (dw, dh)
    # Deterministic given the rng.
    a = source_prechain(img, random.Random(3), cfg)
    b = source_prechain(img, random.Random(3), cfg)
    assert np.array_equal(np.asarray(a), np.asarray(b))
    return "shift in [1,7]^2, deterministic per rng seed"


@check("robustness view: skip-webp replays gasbench's JPEG-only chain exactly")
def _skip_webp():
    import numpy as np
    from data import robustness_view
    from gasbench_bridge import apply_robustness_augmentations

    u8 = np.random.RandomState(11).randint(0, 255, (480, 600, 3), dtype=np.uint8)

    # skip_webp_p=1.0: replay the rng stream (skip, scale, jpeg) and compare
    # against a direct call with webp_quality=None.
    out = robustness_view(u8, 384, seed=5, rng=random.Random(123),
                          skip_webp_p=1.0)
    r = random.Random(123)
    assert r.random() < 1.0  # the skip draw
    scale, jq = r.uniform(0.35, 1.0), r.randint(40, 75)
    direct, _, _, _ = apply_robustness_augmentations(
        u8, (384, 384), seed=5, scale_factor=scale, jpeg_quality=jq,
        webp_quality=None)
    assert np.array_equal(out, direct)

    # skip_webp_p=0.0 consumes NO extra draw: the stream is the pre-delta one.
    out0 = robustness_view(u8, 384, seed=5, rng=random.Random(9),
                           skip_webp_p=0.0)
    r = random.Random(9)
    scale, jq, wq = r.uniform(0.35, 1.0), r.randint(40, 75), r.randint(60, 90)
    direct0, _, _, _ = apply_robustness_augmentations(
        u8, (384, 384), seed=5, scale_factor=scale, jpeg_quality=jq,
        webp_quality=wq)
    assert np.array_equal(out0, direct0)

    # rng=None (the eval path) always keeps the WebP hop: deployed constants.
    oute = robustness_view(u8, 384, seed=5)
    directe, _, _, _ = apply_robustness_augmentations(u8, (384, 384), seed=5)
    assert np.array_equal(oute, directe)
    return "p=1.0 -> webp=None chain; p=0.0 -> pre-delta stream; eval exact"


@check("ladder crop guard: semisynthetic renders full-frame, others crop")
def _crop_guard():
    import numpy as np
    from data import ManifestDataset, ViewConfig
    from gasbench_bridge import apply_random_augmentations

    with tempfile.TemporaryDirectory() as td:
        df = _test_images(Path(td))
        ds = ManifestDataset(df, ViewConfig(image_size=384), train=True)
        probs = [ds._ladder_crop_prob(i) for i in range(3)]
        assert probs == [0.5, 0.5, 0.0], probs  # real, synthetic, semisynthetic
        off = ManifestDataset(df, ViewConfig(image_size=384,
                                             ladder_crop_guard=False),
                              train=True)
        assert [off._ladder_crop_prob(i) for i in range(3)] == [0.5] * 3
        # 3-class migration (2026-08-25): a TRAIN dataset without `kind`
        # must refuse to build (y3 would silently lose every semi example);
        # eval frames without kind degrade gracefully (y3 = binary label).
        try:
            ManifestDataset(df.drop(columns=["kind"]),
                            ViewConfig(image_size=384), train=True)
            raise AssertionError("train=True without kind must raise")
        except ValueError:
            pass
        nk = ManifestDataset(df.drop(columns=["kind"]),
                             ViewConfig(image_size=384), train=False)
        assert (nk.y3 == df["label"].to_numpy()).all()
        assert [nk._ladder_crop_prob(i) for i in range(3)] == [0.5] * 3
        # y3 derivation on the full frame: real->0, synthetic->1, semi->2
        assert ds.y3.tolist() == [0, 1, 2]

    # crop_prob=0.0 never emits crop params; 0.5 emits them about half the time.
    u8 = np.random.RandomState(2).randint(0, 255, (500, 400, 3), dtype=np.uint8)
    hit0, hit5 = 0, 0
    for seed in range(200):
        _, _, _, p0 = apply_random_augmentations(u8, (384, 384), seed=seed,
                                                 level=None, crop_prob=0.0)
        _, _, _, p5 = apply_random_augmentations(u8, (384, 384), seed=seed,
                                                 level=None, crop_prob=0.5)
        hit0 += "RandomCropWithParams" in p0
        hit5 += "RandomCropWithParams" in p5
    assert hit0 == 0, hit0
    assert 60 <= hit5 <= 140, hit5
    return f"guard 0.0 for semisynthetic; crop emitted 0/200 vs {hit5}/200"


@check("deploy eval path is byte-identical to the scored transform")
def _deploy_identity():
    import numpy as np
    from PIL import Image
    from data import ManifestDataset, ViewConfig
    from gasbench_bridge import apply_random_augmentations

    with tempfile.TemporaryDirectory() as td:
        df = _test_images(Path(td))
        seed = 34
        ds = ManifestDataset(df, ViewConfig(image_size=384), train=False,
                             eval_mode="deploy", seed=seed)
        for i in range(3):
            got = ds[i]["x"].permute(1, 2, 0).numpy()
            with Image.open(df.at[i, "path"]) as im:
                u8 = np.asarray(im.convert("RGB"), dtype=np.uint8)
            row_seed = (seed * 1_000_003 + 0 * 9_176 + i) % (2 ** 31 - 1)
            want, _, _, _ = apply_random_augmentations(
                u8, (384, 384), seed=row_seed, level=0, crop_prob=0.0)
            assert np.array_equal(got, want)
    return "3/3 rows byte-equal to apply_random_augmentations(level=0, crop=0)"


@check("train views are deterministic per (seed, epoch, row)")
def _train_determinism():
    import numpy as np
    from data import ManifestDataset, ViewConfig

    cfg = ViewConfig(image_size=384, prechain=True, prechain_jpeg_p=0.35,
                     prechain_double_p=0.10, resample_jitter=True,
                     robust_skip_webp_p=0.15)
    with tempfile.TemporaryDirectory() as td:
        df = _test_images(Path(td))
        a = ManifestDataset(df, cfg, train=True, seed=34)
        b = ManifestDataset(df, cfg, train=True, seed=34)
        for i in range(3):
            xa, xb = a[i], b[i]
            assert np.array_equal(xa["x_a"].numpy(), xb["x_a"].numpy())
            assert np.array_equal(xa["x_b"].numpy(), xb["x_b"].numpy())
        a.set_epoch(1)
        assert not all(np.array_equal(a[i]["x_a"].numpy(), b[i]["x_a"].numpy())
                       for i in range(3))
    return "same seed -> identical x_a/x_b bytes; epoch changes the stream"


@check("resample jitter: single kernel preserves the pre-dial rng stream")
def _jitter_stream():
    import cv2
    import numpy as np
    from PIL import Image
    from data import ViewConfig, source_resample_jitter, _to_u8_hwc

    img = Image.fromarray(
        np.random.RandomState(5).randint(0, 255, (300, 260, 3), dtype=np.uint8))
    cfg = ViewConfig(resample_jitter_p=1.0)  # default kernels: ("area_linear",)
    out = source_resample_jitter(img, random.Random(21), cfg)
    # Replay of the ORIGINAL algorithm: p draw, s draw, AREA down, LINEAR up.
    r = random.Random(21)
    assert r.random() < 1.0
    s = r.uniform(*cfg.resample_jitter_range)
    a = _to_u8_hwc(img)
    h, w = a.shape[:2]
    nh, nw = max(16, int(round(h * s))), max(16, int(round(w * s)))
    small = cv2.resize(a, (nw, nh), interpolation=cv2.INTER_AREA)
    want = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    assert np.array_equal(np.asarray(out), want)

    multi = ViewConfig(resample_jitter_p=1.0,
                       resample_jitter_kernels=("area_linear", "area_cubic",
                                                "cubic_linear"))
    out2 = source_resample_jitter(img, random.Random(21), multi)
    r = random.Random(21)
    r.random(); s = r.uniform(*multi.resample_jitter_range)
    k = r.randrange(3)
    pairs = [(cv2.INTER_AREA, cv2.INTER_LINEAR),
             (cv2.INTER_AREA, cv2.INTER_CUBIC),
             (cv2.INTER_CUBIC, cv2.INTER_LINEAR)]
    down, up = pairs[k]
    nh, nw = max(16, int(round(h * s))), max(16, int(round(w * s)))
    want2 = cv2.resize(cv2.resize(a, (nw, nh), interpolation=down), (w, h),
                       interpolation=up)
    assert np.array_equal(np.asarray(out2), want2)
    return "single-kernel replay exact (no extra draw); multi-kernel replay exact"


@check("view_b: deploy mode renders a clean scored-transform duplicate, stream-safe")
def _view_b_mode():
    import numpy as np
    from data import ManifestDataset, ViewConfig, deploy_view

    with tempfile.TemporaryDirectory() as td:
        df = _test_images(Path(td))
        seed = 34
        # Clean-arm shape: every degradation source off, view_b: deploy.
        # x_b must equal x_a byte-for-byte -- the render is what the scorer
        # computes and NOTHING degraded enters the loss.
        clean = ViewConfig(image_size=384, view_b_mode="deploy",
                           arm_deploy=1.0, arm_ladder=0.0, arm_robust=0.0)
        ds = ManifestDataset(df, clean, train=True, seed=seed)
        for i in range(3):
            it = ds[i]
            assert np.array_equal(it["x_a"].numpy(), it["x_b"].numpy()), i
        # The duplicate is exact only because the deploy render is
        # seed-invariant (view_a renders at row_seed, view_b at row_seed+1):
        # prove the invariance instead of assuming it.
        u8 = np.random.RandomState(3).randint(0, 255, (400, 500, 3),
                                              dtype=np.uint8)
        assert np.array_equal(deploy_view(u8, 384, 7), deploy_view(u8, 384, 8))
        # The mode must only touch view_b: with heavy upstream augs on, x_a is
        # byte-identical between modes (robust view_b's rng draws are the
        # LAST consumers in __getitem__, so removing them shifts nothing).
        aug = dict(image_size=384, prechain=True, prechain_jpeg_p=0.35,
                   prechain_double_p=0.10, resample_jitter=True,
                   robust_skip_webp_p=0.15)
        rb = ManifestDataset(df, ViewConfig(**aug), train=True, seed=seed)
        dp = ManifestDataset(df, ViewConfig(view_b_mode="deploy", **aug),
                             train=True, seed=seed)
        for i in range(3):
            assert np.array_equal(rb[i]["x_a"].numpy(), dp[i]["x_a"].numpy()), i
            assert not np.array_equal(dp[i]["x_a"].numpy(),
                                      rb[i]["x_b"].numpy()), i
    return "x_b == x_a bytes in clean shape; deploy seed-invariant; x_a unmoved"


@check("config-clean.yaml differs from the control arm ONLY in the treatment")
def _clean_config():
    import yaml

    here = Path(__file__).parent
    clean = yaml.safe_load((here / "config-clean.yaml").read_text())
    exp = yaml.safe_load((here / "config-experiment.yaml").read_text())
    d = clean["data"]
    assert d["clean_view_recompress"] is False
    assert d["resample_jitter"]["enabled"] is False
    assert d["view_arms"] == {"deploy": 1.0, "ladder": 0.0, "robust": 0.0}
    assert d["use_degradation_schedule"] is False
    assert d["view_b"] == "deploy"
    assert clean["output_dir"] != exp["output_dir"], \
        "arms would overwrite each other's best.pt"
    # Outside the treatment the arms must be IDENTICAL -- one variable, or
    # the comparison is not an experiment.
    for sect in ("branches", "sampler", "lora", "training", "loss"):
        assert clean[sect] == exp[sect], f"{sect} differs between arms"
    assert clean["seed"] == exp["seed"]
    treatment = {"clean_view_recompress", "resample_jitter", "view_arms",
                 "use_degradation_schedule", "view_b"}
    for k in set(exp["data"]) | set(d):
        if k not in treatment:
            assert d.get(k) == exp["data"].get(k), f"data.{k} differs"
    # And the knob must actually be wired, or the clean arm silently trains
    # a robust view_b while looking healthy.
    tp = (here / "train.py").read_text()
    opt = tp.split("OPTIONAL_DATA = ")[1].split(")")[0]
    assert '"view_b"' in opt, "validate_config would reject data.view_b"
    assert 'd.get("view_b", "robust")' in tp, "build_view does not read view_b"
    assert 'view_b_mode=view_b' in tp, "build_view does not pass view_b_mode"
    dpy = (here / "data.py").read_text()
    assert 'if cfg.view_b_mode == "deploy":' in dpy, \
        "data.py does not branch on view_b_mode"
    return "treatment knobs set; all other sections byte-equal; wiring present"


@check("config.yaml wires the deltas and train.py reads them")
def _config_wiring():
    # Source-text asserts (verify.py's _contract style) rather than importing
    # train.py, which would pull model.py -> transformers/timm/peft; this
    # check must run on a box with only the light deps.
    import yaml

    here = Path(__file__).parent
    cfg = yaml.safe_load((here / "config.yaml").read_text())
    la = cfg["data"]["laundering"]
    assert la["double_jpeg_p"] == 0.10
    assert la["prechain_jpeg_p"] == 0.35
    assert la["robust_skip_webp_p"] == 0.15
    assert la["ladder_crop_guard"] is True
    named = 0.30 + la["prechain_jpeg_p"] + la["double_jpeg_p"]
    assert named <= 1.0 + 1e-9
    assert "kernels" not in (cfg["data"].get("resample_jitter") or {}), \
        "kernel experiment must stay off by default"

    tp = (here / "train.py").read_text()
    assert '"laundering"' in tp.split("OPTIONAL_DATA = ")[1].split("\n")[0], \
        "validate_config would reject the laundering block"
    for needle in ('la.get("double_jpeg_p", 0.0)',
                   'la.get("prechain_jpeg_p", 0.45)',
                   'la.get("robust_skip_webp_p", 0.0)',
                   'la.get("ladder_crop_guard", True)',
                   'rj.get("kernels"'):
        assert needle in tp, f"train.py does not wire {needle}"
    return "laundering block valid, arms leave webp remainder 0.25, wiring present"


def main() -> None:
    print("v7 view-delta checks")
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
