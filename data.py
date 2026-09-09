"""Dataset, views and balanced sampler.

Three things here differ from every previous version, and each is a response to
something verified in gasbench rather than a preference:

1. **Everything emits uint8.** `PyTorchInferenceSession` hands the model
   `uint8 [0,255]` and the model normalises internally
   (pytorch_session.py:152, Safetensors.md section 4). v2 normalises in its
   collate and v4 in `spatial_tensor`; both put normalisation on the wrong
   side of the export boundary, so a direct export of either receives uint8
   where it expects normalised floats. Nothing in v5 produces a normalised
   float tensor outside `model.forward`.

2. **The final resize is gasbench's own.** The deployed base pass is
   `apply_random_augmentations(level=0, crop_prob=0.0)`
   (image_bench.py:245-246, not overridable from the CLI), i.e.
   `ResizeShortestEdge` = centre-crop to the target aspect then
   `cv2.resize(..., INTER_LINEAR)`. cv2's INTER_LINEAR does **not** antialias
   on downscale, so downscaling a 1024px generator output aliases its
   high-frequency fingerprint into the visible band in a deterministic,
   learnable way. v2 resizes with torchvision and v4 with `antialias=True`;
   both destroy exactly the components the model is scored on. We call
   gasbench's function rather than reproducing it.

3. **The sampler balances four levels, not three.** See `balanced_weights`.
"""
from __future__ import annotations

import io
import random
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler

from gasbench_bridge import (apply_random_augmentations,
                             apply_robustness_augmentations)

Image.MAX_IMAGE_PIXELS = 512_000_000


# ---------------------------------------------------------------------------
# Balanced sampling
# ---------------------------------------------------------------------------

KINDS = ("real", "synthetic", "semisynthetic")


def _kind_shares(df: pd.DataFrame, kb: dict[str, float]) -> np.ndarray:
    """Per-row share of its (label, category) cell that its `kind` should carry.

    Placed BELOW category rather than above it on purpose. Semisynthetic is
    concentrated in a few categories (v23 registry: 5 datasets, ALL in
    `faces` -- it used to span faces/diverse/documents), so splitting mass by
    kind at the top would hand those categories extra fake mass and leave the
    rest short -- reintroducing the content->label shortcut that the folding
    and the q_c construction exist to remove. Applying it inside each cell
    keeps P(fake)=0.5 and P(fake|category)=0.5 exactly, and upweights
    semisynthetic only where it is actually present. Side effect worth
    knowing: the REALIZED global semi share is 0.5 * sum(q_c over categories
    with semis) * kb_semi/(kb_semi+kb_syn) -- it tracks the categories the
    pool occupies, not the pool's size, which is what self-moderated the
    share from 43.4% to ~19.4% of the fake half when the pool shrank to
    faces-only.

    A cell holding one kind gives it the whole cell, so categories with no
    semisynthetic are untouched rather than starved.
    """
    kind = df["kind"].astype(str)
    # Vectorised, and deliberately free of any index assumption: an earlier
    # version used groupby().groups + get_indexer, which silently requires a
    # unique index and materialises one Index object per cell.
    pairs = pd.DataFrame({"l": df["label"].to_numpy(),
                          "c": df["category"].to_numpy(),
                          "k": kind.to_numpy()}).drop_duplicates()
    pairs["v"] = [kb.get(k, 0.0) for k in pairs["k"]]
    agg = pairs.groupby(["l", "c"]).agg(tot=("v", "sum"), n=("k", "size"))
    key = pd.MultiIndex.from_arrays([df["label"], df["category"]])
    tot = key.map(agg["tot"]).to_numpy(dtype=float)
    nk = key.map(agg["n"]).to_numpy(dtype=float)
    kv = np.array([kb.get(k, 0.0) for k in kind], dtype=float)
    # A cell holding one kind gives it the whole cell. A cell whose kinds all
    # carry zero mass would divide by zero; leave those at 1.0 so a degenerate
    # kind_balance cannot silently zero out an entire category.
    return np.where((nk <= 1) | (tot <= 0), 1.0, kv / np.where(tot > 0, tot, 1.0))


def balanced_weights(df: pd.DataFrame, *, gamma: float = 0.5,
                     amp_max: float = 10.0,
                     kind_balance: dict[str, float] | None = None,
                     gasstation_boost: float = 1.0,
                     verbose: bool = True) -> tuple[np.ndarray, dict]:
    """Per-row sampling weights over label -> category -> group -> row.

    `gasstation_boost` multiplies the gasstation group's share WITHIN its
    (label, category, kind) cell -- the training-side mirror of gasbench's
    own GASSTATION_WEIGHT_MULTIPLIER = 5.0 (dataset/config.py), which gives
    gasstation 5x the eval samples of a regular dataset (and the benchmark's
    score_composition weights it further). Applied at the group-share level,
    so P(fake)=0.5 and P(fake|category)=0.5 are untouched: the boost only
    shifts fake-cell mass toward the adversarial stream. Membership matches
    gasbench's own rule ("gasstation" in the dataset name). Same mechanism
    as video_v1; ported 2026-08-10 when the cache grew to 8 gasstation weeks.

    This is v2's cascade (data.py:188) with the defects fixed. The formula is

        w_i = (1/2) * q_c * (n_g^gamma / sum_g' n_g'^gamma) * (1 / n_g)

    where c is the folded category, g the group, and q_c the category mass.

    Normalisation, level by level: summing 1/n_g over a group's rows gives 1;
    summing the sqrt share over a cell's groups gives 1; so each (label,
    category) cell carries exactly (1/2) q_c, each label exactly 1/2, and the
    whole thing exactly 1. Two properties follow that v2 does not achieve:

      * P(fake) = 1/2 exactly and unconditionally. v2 splits each category's
        mass across the labels *present*, so a category with only one label
        donates all its mass to one side. On this registry that put P(fake) at
        0.45. build_manifest.py folds single-label categories away, so every
        category here carries both labels and the identity holds.
      * P(category | label) = q_c for both labels, hence P(fake | category) =
        1/2 for every category and the category variable carries exactly zero
        label information.

    `q_c` is the *geometric* mean of the two labels' dataset counts, normalised.
    Geometric rather than arithmetic because it automatically de-weights
    categories that are thin on one side: `animals` at 9 real / 2 fake would
    otherwise pull its two fake datasets hard enough to memorise them.

    The grouping key is `group`, which build_manifest.py sets to the dataset
    name. This matters: `generator_family` is the string `real` for all 89 real
    datasets, so v2's `grp = df["generator"]` collapses the entire real class
    into one group. Inside a category there is then exactly one real group, the
    sqrt share is 1, and every real row gets weight proportional to 1/n -- i.e.
    uniform over rows, so the largest real corpus in each category absorbs
    nearly all of that category's real mass while the fake side is properly
    flattened. That asymmetry is a plausible mechanical cause of the collapse
    v2 documents (or-real-id 0.084, fairface 0.645, birds 0.994).

    Replay is bounded by water-filling rather than v2's `min_group_rows`
    pooling. Pooling is discontinuous -- a 49-row cell is merged while a
    51-row cell keeps a full group share and gets replayed ~60x -- and it
    controls group size when the thing you care about is draws per image. The
    exact solution to "minimise KL to the target subject to w_i <= A/M" is
    w_i = min(lambda * w*_i, A/M) for a single scalar lambda, found by
    bisection. It normalises exactly, is continuous in the data, and keeps
    group identity.
    """
    n = len(df)
    if n == 0:
        raise ValueError("empty dataframe passed to balanced_weights")
    lab = df["label"].astype(int).to_numpy()
    cat = df["category"].astype(str).to_numpy()
    grp = df["group"].astype(str).to_numpy()

    # Kind level. Disabled -> a constant series, which makes the group cell
    # (label, category, kind, group) identical to (label, category, group) and
    # every share 1.0, so the formula below reduces exactly to the old one.
    if kind_balance:
        if "kind" not in df.columns:
            raise ValueError(
                "sampler.kind_balance is set but the manifest has no `kind` "
                "column. Rebuild it with build_manifest.py -- resolve_kind() "
                "writes the column from overrides.yaml:semisynthetic_datasets.")
        bad = sorted(set(df["kind"].astype(str)) - set(KINDS))
        if bad:
            raise ValueError(f"manifest `kind` has unexpected values: {bad}")
        kb = {k: float(kind_balance.get(k, 0.0)) for k in KINDS}
        if min(kb.values()) < 0 or sum(kb.values()) <= 0:
            raise ValueError(f"sampler.kind_balance must be non-negative and "
                             f"sum above zero, got {kind_balance}")
        fake_mass = kb["synthetic"] + kb["semisynthetic"]
        tot_mass = kb["real"] + fake_mass
        label_w = {0: kb["real"] / tot_mass, 1: fake_mass / tot_mass}
        kshare = _kind_shares(df, kb)
        kind_key = df["kind"].astype(str)
    else:
        label_w = {0: 0.5, 1: 0.5}
        kshare = np.ones(len(df), dtype=float)
        kind_key = pd.Series("_all_", index=df.index)

    # q_c from dataset counts, over categories carrying both labels here.
    ds = df.drop_duplicates("dataset")
    counts: dict[str, list[int]] = {}
    for c, l in zip(ds["category"].astype(str), ds["label"].astype(int)):
        counts.setdefault(c, [0, 0])[l] += 1
    usable = {c: (r * f) ** 0.5 for c, (r, f) in counts.items() if r and f}
    dropped = sorted(set(counts) - set(usable))
    if not usable:
        raise ValueError("no category carries both labels; check build_splits")
    tot_q = sum(usable.values())
    q = {c: v / tot_q for c, v in usable.items()}

    keep = np.array([c in q for c in cat])
    if dropped and verbose:
        n_drop = int((~keep).sum())
        print(f"[sampler] categories with a single label in this split are "
              f"excluded from training: {dropped} ({n_drop:,} rows). They "
              f"cannot be balanced, so serving them would reintroduce the "
              f"content->label shortcut.")

    # sqrt-weighted group share within each (label, category, kind) cell.
    n_in_group = df.groupby([df["label"], df["category"], kind_key,
                             df["group"]])["group"].transform("size")
    gw = n_in_group.astype(float) ** gamma
    if gasstation_boost != 1.0:
        is_gs = df["dataset"].astype(str).str.lower() \
                             .str.contains("gasstation").to_numpy()
        # Per-row multiplier is constant within a group (gasstation is its
        # own group), so group-share and cell-total stay coherent.
        gw = gw * np.where(is_gs, float(gasstation_boost), 1.0)
    cell_tot = gw.groupby([df["label"], df["category"], kind_key, df["group"]]) \
                 .first().groupby(level=[0, 1, 2]).sum()
    key = pd.MultiIndex.from_arrays([df["label"], df["category"], kind_key])
    cell = key.map(cell_tot).to_numpy(dtype=float)

    qv = np.array([q.get(c, 0.0) for c in cat])
    lw = np.array([label_w[int(l)] for l in lab])
    w = (lw * qv * kshare
         * (gw.to_numpy() / np.where(cell > 0, cell, 1.0)) / n_in_group.to_numpy())
    w = np.where(keep, w, 0.0)
    s = w.sum()
    if s <= 0:
        raise ValueError("all sampling weights are zero")
    w = w / s

    # Water-filling cap on expected draws per image per epoch.
    cap = amp_max / n
    replay_cap_bound = bool(w.max() > cap)
    if replay_cap_bound:
        lo, hi = 1.0, max(1.0, float(cap / w[w > 0].min()))
        for _ in range(60):
            mid = (lo + hi) / 2
            if np.minimum(mid * w, cap).sum() < 1.0:
                lo = mid
            else:
                hi = mid
        w = np.minimum(hi * w, cap)
        w = w / w.sum()

    amp = w * n
    # Per-dataset mass, for the report: concentration is otherwise invisible
    # (e.g. v23: the 5 semi datasets carry ~9.7% of ALL training mass at
    # ~3.4x replay each -- fine, but only if someone can see it).
    ds_mass = (pd.Series(w, index=df["dataset"].astype(str).to_numpy())
               .groupby(level=0).sum().sort_values(ascending=False))
    ds_rows = df["dataset"].astype(str).value_counts()
    top_mass = [(name, float(m), float(m * n / max(int(ds_rows[name]), 1)))
                for name, m in ds_mass.head(5).items()]
    report = {
        "replay_cap_bound": replay_cap_bound,
        "top_mass": top_mass,   # [(dataset, mass_share, mean_amp_x), ...]
        "p_fake": float(w[lab == 1].sum()),
        "p_fake_given_cat": {c: float(w[(cat == c) & (lab == 1)].sum() /
                                      max(w[cat == c].sum(), 1e-12))
                             for c in sorted(q)},
        "q": q,
        "p_kind": ({k: float(w[(df["kind"].astype(str) == k).to_numpy()].sum())
                    for k in KINDS} if "kind" in df.columns else {}),
        "amp_kind": ({k: float((w * n)[(df["kind"].astype(str) == k).to_numpy()
                                       & (w > 0)].max())
                      for k in KINDS
                      if ((df["kind"].astype(str) == k).to_numpy() & (w > 0)).any()}
                     if "kind" in df.columns else {}),
        "kind_balance": dict(kind_balance) if kind_balance else None,
        "gasstation_boost": float(gasstation_boost),
        "p_gasstation": float(w[df["dataset"].astype(str).str.lower()
                               .str.contains("gasstation").to_numpy()].sum()),
        "dropped_categories": dropped,
        "amp_min": float(amp[w > 0].min()) if (w > 0).any() else 0.0,
        "amp_max": float(amp.max()),
        "n_rows": n,
        "n_servable": int((w > 0).sum()),
    }
    if verbose:
        print(f"[sampler] P(fake)={report['p_fake']:.4f}  "
              f"replay {report['amp_min']:.2f}x..{report['amp_max']:.2f}x  "
              f"servable {report['n_servable']:,}/{n:,}")
        if report["p_kind"]:
            print("[sampler] P(kind): " + "  ".join(
                f"{k}={v:.4f}" for k, v in report["p_kind"].items() if v > 0)
                + ("  [balanced]" if kind_balance else "  [unbalanced]"))
            if report["amp_kind"]:
                print("[sampler] peak replay by kind: " + "  ".join(
                    f"{k}={v:.1f}x" for k, v in report["amp_kind"].items()))
        print("[sampler] P(fake|category): " + "  ".join(
            f"{c}={v:.4f}" for c, v in report["p_fake_given_cat"].items()))
        off = [c for c, v in report["p_fake_given_cat"].items() if abs(v - 0.5) > 1e-6]
        if off:
            print(f"[sampler] WARNING P(fake|category) != 0.5 for {off}")
        print("[sampler] top dataset mass: " + "  ".join(
            f"{name}={m:.3%}({a:.1f}x)" for name, m, a in report["top_mass"]))
        if report["replay_cap_bound"]:
            print("[sampler] WARNING replay cap BOUND: water-fill clipped "
                  "the heaviest rows and redistributed their mass corpus-"
                  "wide -- P(fake)=0.5, P(fake|category)=0.5 and the kind "
                  "shares are no longer exact. Rebalance (kind_balance / "
                  "gasstation_boost) or raise sampler.max_replay.")
    return w, report


def clique_pairs(df: pd.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """For each mixed-label split_unit, the row indices of each side.

    A `split_unit` groups datasets that share a source: celeb-a-hq with the
    GANs that edit its images, fairface with the FairFaceGen models trained on
    it, every `-real`/`-fake` pair. When a unit carries both labels its two
    halves are near-duplicate images with opposite labels -- the hardest and
    most informative examples in the corpus.
    """
    if "split_unit" not in df.columns:
        return {}
    out = {}
    for unit, sub in df.groupby("split_unit"):
        if sub["label"].nunique() < 2:
            continue
        pos = sub.index[sub.label == 1].to_numpy()
        neg = sub.index[sub.label == 0].to_numpy()
        if len(pos) and len(neg):
            out[str(unit)] = (neg, pos)
    return out


class BalancedSampler(Sampler[int]):
    """Distributed-aware weighted sampler with a reproducible per-epoch stream.

    v2 uses `WeightedRandomSampler`, which is single-process; under accelerate
    each rank would draw the same indices. Here every rank draws the identical
    global sequence from a `(seed, epoch)`-seeded generator and then slices
    `[rank::world_size]`, so the union across ranks is exactly one epoch and
    the result is reproducible without depending on how accelerate happens to
    shard a batch sampler.
    """

    def __init__(self, weights: np.ndarray, num_samples: int, *, seed: int = 0,
                 rank: int = 0, world_size: int = 1,
                 pairs: dict | None = None, pair_frac: float = 0.0):
        self.w = torch.as_tensor(weights, dtype=torch.double)
        self.num_samples = int(num_samples)
        self.seed, self.rank, self.world_size = seed, rank, world_size
        self.pairs = pairs or {}
        self.pair_frac = float(pair_frac) if self.pairs else 0.0
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.num_samples // self.world_size

    def __iter__(self):
        g = torch.Generator().manual_seed(self.seed * 1_000_003 + self.epoch)
        idx = torch.multinomial(self.w, self.num_samples, replacement=True,
                                generator=g)

        # Replace a fraction of the stream with MATCHED CLIQUE PAIRS: a real
        # image and a near-duplicate fake derived from it, adjacent so they land
        # in the same batch.
        #
        # Without this the two halves of a clique are drawn independently and
        # essentially never co-occur, so cross-entropy sees "aligned face crop,
        # label real" and "aligned face crop, label fake" hundreds of steps
        # apart and settles on whichever carries more gradient mass. Measured
        # cost on run_2: fake datasets the model TRAINED ON scored 0.631 inside
        # mixed cliques against 0.941 outside them, with STARGAN at 0.138 and
        # fakeclue-fake-ffpp at 0.000. Putting the pair in one batch makes the
        # contradiction explicit and un-resolvable by majority vote.
        if self.pair_frac > 0:
            n_pairs = int(self.num_samples * self.pair_frac) // 2
            units = list(self.pairs)
            u = torch.randint(len(units), (n_pairs,), generator=g)
            flat = idx.tolist()
            for j in range(n_pairs):
                neg, pos = self.pairs[units[int(u[j])]]
                a = int(neg[torch.randint(len(neg), (1,), generator=g).item()])
                b = int(pos[torch.randint(len(pos), (1,), generator=g).item()])
                flat[2 * j], flat[2 * j + 1] = a, b
            # Shuffle in blocks of 2 so a pair stays contiguous and therefore
            # stays in the same batch, but pairs are spread across the epoch.
            order = torch.randperm(len(flat) // 2, generator=g).tolist()
            idx = torch.tensor([flat[2 * k + o] for k in order for o in (0, 1)])

        return iter(idx[self.rank::self.world_size].tolist())


# ---------------------------------------------------------------------------
# Views
# ---------------------------------------------------------------------------

@dataclass
class ViewConfig:
    image_size: int = 512
    # Label-independent codec laundering, applied to BOTH classes with the same
    # distribution. Off by default: audit_shortcuts.py --mode tensors prices it
    # before it is switched on.
    prechain: bool = False
    prechain_none_p: float = 0.30
    prechain_jpeg_p: float = 0.45
    prechain_jpeg_q: tuple[int, int] = (55, 98)
    prechain_webp_q: tuple[int, int] = (60, 95)
    # Shifted double-JPEG arm: JPEG(q1) -> crop (dx, dy ~ U{1..7}) -> JPEG(q2).
    # Re-shared content recompresses after crops/shifts, leaving misaligned 8x8
    # DCT lattices; no other view produces them (the prechain is single-pass
    # and the robustness chain's JPEG passes land on an aligned grid). Mass is
    # carved from the single-JPEG arm, so config.yaml lowers prechain_jpeg_p
    # when it turns this on. 0.0 here keeps configless code paths on the old
    # three-arm behavior. See EXPECTED_OUTCOMES.md for the acceptance criteria.
    prechain_double_p: float = 0.0
    prechain_double_shift: tuple[int, int] = (1, 7)
    # Label-independent resample jitter: a down-up cycle before the view, to
    # break the correlation between source resolution and the resample factor.
    resample_jitter: bool = False
    resample_jitter_p: float = 0.4
    resample_jitter_range: tuple[float, float] = (0.60, 1.0)
    # Down/up kernel pairs the jitter draws from. The default single entry is
    # the eval chain's own AREA/LINEAR and is byte-identical to the previous
    # behavior (a lone entry consumes no rng draw, preserving the stream).
    # Kernel diversity is a gated EXPERIMENT, not a default: it may blur the
    # deterministic INTER_LINEAR aliasing signature that is learnable signal
    # (run_3 failure family). EXPECTED_OUTCOMES.md defines the adopt criteria.
    resample_jitter_kernels: tuple[str, ...] = ("area_linear",)
    # Fraction of rng-driven robustness views rendered with webp_quality=None
    # -- gasbench's own JPEG-only chain. The eval path (rng=None) never skips.
    robust_skip_webp_p: float = 0.0
    # Never random-crop semisynthetic rows in the ladder arm. Eval crops are
    # mask-aware (RandomCropWithParams keeps the edit in frame); training has
    # no masks, so a crop can exclude the edit and produce a genuinely-real
    # view labeled fake. Default ON: correctness fix, not a dial.
    ladder_crop_guard: bool = True
    # Level mixture inside the ladder arm, as (L0, L1, L2, L3). None keeps
    # apply_random_augmentations' own 25/25/25/25 default.
    #
    # L1 is DEAD MASS at the default: it adds only h/v flips, and __getitem__
    # already flips at SOURCE with hflip_p = vflip_p = 0.5 before the view
    # split. Composing two independent Bernoulli(0.5) flips is Bernoulli(0.5),
    # so L1 renders the same distribution as L0 -- a quarter of the ladder
    # doing nothing. Handing its share to L3 is the only free way to raise
    # blur/noise exposure: it costs no clean-pixel mass and adds no severity
    # (levels stay 0-2, so blur is kernel 7/9 and noise sigma 0.001-0.002),
    # which keeps it clear of the run_3 failure where erasing high-frequency
    # evidence took face-swap to 0.138 and fakeclue-fake-ffpp to 0.000.
    ladder_level_probs: tuple[float, ...] | None = None
    hflip_p: float = 0.5
    vflip_p: float = 0.5
    # Training view mixture. `deploy` is the exact scored transform; `robust`
    # is the exact chain behind aug_binary_*; `ladder` hedges against a
    # validator enabling apply_random_augmentations' own defaults.
    #
    # config.yaml is the source of truth for the mix (currently
    # 0.70/0.20/0.10); these defaults only serve code paths that build a
    # ViewConfig without the config file. The floor under `deploy` is
    # measured, not aesthetic: run_3 at 0.55/0.25/0.20 scored
    # fakeclue-fake-ffpp 0.000 and face-swap 0.138 -- both IN the training
    # set. Old GAN and face-manipulation fakes are identified by
    # high-frequency artifacts at 112-256px, and heavy JPEG / blur / down-up
    # erases them before the model sees the image. Remember view_b is always
    # the robustness chain, so total degraded CE mass is far above the
    # ladder+robust share alone.
    arm_deploy: float = 0.80
    arm_ladder: float = 0.10
    arm_robust: float = 0.10
    # What the second (consistency) view renders. "robust" is the historical
    # behavior: view_b is always the robustness chain, so the symKL term
    # teaches degradation invariance. "deploy" renders view_b as the exact
    # scored transform instead -- the no-degradation arm needs this, because
    # a robust view_b would keep training CE+brier on degraded pixels every
    # step and destroy the treatment. With both views clean the pair is
    # byte-identical (deploy_view is seed-invariant) and symKL degrades to
    # dropout-consistency (R-Drop): log its magnitude, do not assume zero.
    # Run-constant by design -- the degradation schedule cannot override it.
    view_b_mode: str = "robust"


def _to_u8_hwc(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("RGB"), dtype=np.uint8)


def _prechain_arm(u: float, cfg: ViewConfig) -> str:
    """Pure arm selection for one uniform draw; split out so tests can assert
    the arm frequencies without decoding a single image."""
    if u < cfg.prechain_none_p:
        return "none"
    if u < cfg.prechain_none_p + cfg.prechain_jpeg_p:
        return "jpeg"
    if u < cfg.prechain_none_p + cfg.prechain_jpeg_p + cfg.prechain_double_p:
        return "double"
    return "webp"


def _jpeg_roundtrip(img: Image.Image, rng: random.Random,
                    q_range: tuple[int, int]) -> Image.Image:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG",
                            quality=rng.randint(*q_range),
                            subsampling=rng.choice([0, 1, 2]))
    buf.seek(0)
    with Image.open(buf) as out:
        return out.convert("RGB")


def source_prechain(img: Image.Image, rng: random.Random,
                    cfg: ViewConfig) -> Image.Image:
    """Re-encode through a randomly chosen codec, identically for both labels.

    The cache writer re-encodes everything through PIL at default quality, so
    original quantization tables are already homogenised -- v2's stated reason
    for `p_jpeg=1.0` is not the real mechanism. What survives is *grid
    presence*: a source PIL detected as PNG is re-saved losslessly and carries
    no 8x8 DCT lattice at all, while a JPEG source carries a fresh q75 one. On
    this registry 20/91 synthetic datasets declare `source_format: png` against
    4/89 real, so "clean lattice" leans fake. Drawing the codec from the same
    distribution for both labels removes that lean; the `none` arm keeps a
    pristine-input mode in the mixture so the model still sees one.

    The `double` arm recompresses after a 1-7px crop shift, so the two JPEG
    lattices are misaligned -- the signature of re-shared content that no
    aligned roundtrip (this function's `jpeg` arm, the robustness chain)
    produces. Still label-blind: this function never sees the label.
    """
    arm = _prechain_arm(rng.random(), cfg)
    if arm == "none":
        return img
    if arm == "jpeg":
        return _jpeg_roundtrip(img, rng, cfg.prechain_jpeg_q)
    if arm == "double":
        out = _jpeg_roundtrip(img, rng, cfg.prechain_jpeg_q)
        lo, hi = cfg.prechain_double_shift
        dx, dy = rng.randint(lo, hi), rng.randint(lo, hi)
        w, h = out.size
        # Too small to shift (post-view floors make this rare): stay single.
        if w > dx + 16 and h > dy + 16:
            out = out.crop((dx, dy, w, h))
        return _jpeg_roundtrip(out, rng, cfg.prechain_jpeg_q)
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="WEBP",
                            quality=rng.randint(*cfg.prechain_webp_q))
    buf.seek(0)
    with Image.open(buf) as out:
        return out.convert("RGB")


def source_resample_jitter(img: Image.Image, rng: random.Random,
                           cfg: ViewConfig) -> Image.Image:
    """Down-up cycle with a label-independent factor, on a fraction of samples.

    At eval the resample factor is exactly m = S / min_side, deterministic and
    translation-invariant in S, so it cannot be tuned away by choosing a
    different input size. All training can do is stop the model relying on it,
    by making the factor it observes independent of the source. INTER_AREA down
    / INTER_LINEAR up mirrors the robustness chain's own kernel choice.

    This is a DIAL, not a switch, because both extremes were measured and both
    cost score. Turned on for every sample at factor U(0.35, 1.0) it lifted
    val_stress from 0.19 to 0.84 -- the largest single effect of any config
    change here -- while holding fakeclue-fake-ffpp at 0.000 across two runs.
    Turned off entirely, val_stress halved to 0.37 within 500 steps while
    fakeclue-fake-ffpp's DRO loss finally began to fall. A gentler factor
    applied to a fraction of samples keeps the protection for low-resolution
    reals without erasing a locally-confined manipulation artifact.
    """
    import cv2

    if rng.random() >= cfg.resample_jitter_p:
        return img
    lo, hi = cfg.resample_jitter_range
    s = rng.uniform(lo, hi)
    if s >= 0.999:
        return img
    kernels = {"area_linear": (cv2.INTER_AREA, cv2.INTER_LINEAR),
               "area_cubic": (cv2.INTER_AREA, cv2.INTER_CUBIC),
               "cubic_linear": (cv2.INTER_CUBIC, cv2.INTER_LINEAR),
               "lanczos_linear": (cv2.INTER_LANCZOS4, cv2.INTER_LINEAR)}
    names = cfg.resample_jitter_kernels
    # A lone entry consumes no rng draw, so the default ("area_linear",) is
    # byte-identical to the pre-kernel-dial behavior for a given seed.
    name = names[0] if len(names) == 1 else names[rng.randrange(len(names))]
    down, up = kernels[name]
    a = _to_u8_hwc(img)
    h, w = a.shape[:2]
    nh, nw = max(16, int(round(h * s))), max(16, int(round(w * s)))
    small = cv2.resize(a, (nw, nh), interpolation=down)
    return Image.fromarray(cv2.resize(small, (w, h), interpolation=up))


def deploy_view(u8_hwc: np.ndarray, size: int, seed: int) -> np.ndarray:
    """The exact scored transform: level 0, no random crop."""
    out, _, _, _ = apply_random_augmentations(u8_hwc, (size, size), seed=seed,
                                              level=0, crop_prob=0.0)
    return out


def ladder_view(u8_hwc: np.ndarray, size: int, seed: int,
                crop_prob: float = 0.5,
                level_probs: tuple[float, ...] | None = None) -> np.ndarray:
    """apply_random_augmentations' own defaults: crop_prob 0.5, level 0-3 at 25% each.

    Not the deployed configuration -- `run_image_benchmark` pins level=0 and
    crop_prob=0.0 and the CLI cannot change it -- so this is a hedge, not a
    match. Included at modest probability because a validator could flip those
    defaults, and because the geometric/photometric invariances are cheap.
    Note the ladder never reaches its own severe settings: level is drawn
    inclusively from [0, 2] and `get_distortion_parameter` indexes level-1, so
    indices 2-4 of every severity table are unreachable.

    `crop_prob` exists for the semisynthetic guard: eval's crop is mask-aware
    and keeps the edit in frame; training has no masks, so semisynthetic rows
    pass 0.0 and render full-frame instead of risking a crop that excludes the
    edit (a real view trained with label fake).
    """
    # `level_probs` reweights the level draw; a 0.0 entry is safe because
    # apply_random_augmentations picks by cumulative probability and simply
    # never lands on it (its sum-to-1.0 assertion still holds).
    lp = (None if level_probs is None
          else {i: float(v) for i, v in enumerate(level_probs)})
    out, _, _, _ = apply_random_augmentations(u8_hwc, (size, size), seed=seed,
                                              level=None, level_probs=lp,
                                              crop_prob=crop_prob)
    return out


def robustness_view(u8_hwc: np.ndarray, size: int, seed: int,
                    rng: random.Random | None = None,
                    skip_webp_p: float = 0.0) -> np.ndarray:
    """The chain behind aug_binary_*: down-up, JPEG q55, WebP q75, JPEG q80.

    `rng=None` reproduces the deployed constants exactly -- including the WebP
    hop, always. Passing an rng randomises around them for training coverage;
    with probability `skip_webp_p` the WebP hop is dropped (webp_quality=None,
    gasbench's own JPEG-only chain), because real laundering paths are often
    JPEG-only and 100% cross-codec exposure was the previous behavior.

    Draws are sequential and explicit (skip, scale, jpeg, then webp only when
    kept) so verify_views.py can replay the stream against a direct gasbench
    call.
    """
    if rng is None:
        out, _, _, _ = apply_robustness_augmentations(u8_hwc, (size, size), seed=seed)
        return out
    skip = skip_webp_p > 0 and rng.random() < skip_webp_p
    scale = rng.uniform(0.35, 1.0)
    jpeg_q = rng.randint(40, 75)
    webp_q = None if skip else rng.randint(60, 90)
    out, _, _, _ = apply_robustness_augmentations(
        u8_hwc, (size, size), seed=seed,
        scale_factor=scale, jpeg_quality=jpeg_q, webp_quality=webp_q)
    return out


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ManifestDataset(Dataset):
    """Yields uint8 CHW tensors and nothing else that crosses a worker boundary.

    Only tensors and int64 are returned; dataset/generator/category strings are
    joined back on the main process via `row_id`. v4 ships `media_type` and
    `dataset` strings in every batch yet still gathers only logits and labels
    at eval, so it structurally cannot produce a per-generator breakdown.
    """

    def __init__(self, df: pd.DataFrame, cfg: ViewConfig, *, train: bool,
                 eval_mode: str = "deploy", seed: int = 0):
        self.df = df.reset_index(drop=True)
        self.cfg, self.train, self.eval_mode, self.seed = cfg, train, eval_mode, seed
        self.epoch = 0
        self.row_id = self.df.index.to_numpy()
        # numpy array, not per-item .at lookups: __getitem__ runs in workers.
        self._kind = (self.df["kind"].astype(str).to_numpy()
                      if "kind" in self.df.columns else None)
        # 3-class label (gasbench image taxonomy 0=real/1=synthetic/2=semi),
        # derived here from the existing binary `label` + `kind` columns so
        # build_manifest.py stays the single binary collapse point and old
        # manifests keep working. kind is dataset-granular registry truth.
        #
        # No kind column: hard error for TRAINING (a manifest without kind
        # would silently train the type head with zero semi examples), but
        # ad-hoc eval frames (probe_frame, toy test frames) legitimately
        # carry no kind -- their fakes are all treated as synthetic, which
        # is also what eval_probes' binary LABELS map already assumes.
        lab = self.df["label"].to_numpy()
        if self._kind is None:
            if train:
                raise ValueError(
                    "training manifest has no `kind` column -- rebuild it "
                    "with the current build_manifest.py; the 3-class head "
                    "needs per-row kind")
            self.y3 = lab.astype(int)
        else:
            self.y3 = np.where(lab == 0, 0,
                               np.where(self._kind == "semisynthetic", 2, 1))

    def _ladder_crop_prob(self, i: int) -> float:
        """0.0 for semisynthetic rows when the guard is on, else the ladder
        default. Split out so verify_views.py can test the guard without
        rendering a view."""
        if (self.cfg.ladder_crop_guard and self._kind is not None
                and self._kind[i] == "semisynthetic"):
            return 0.0
        return 0.5

    def set_epoch(self, epoch: int, cfg: "ViewConfig | None" = None) -> None:
        """Advance the epoch (fresh per-row RNG streams) and optionally swap
        the view config for a per-epoch degradation profile.

        `cfg` must be a NEW ViewConfig instance -- the eval loaders hold a
        reference to the base config and must never see epoch profiles.
        NOTE: this only reaches dataloader workers when the train loader runs
        with persistent_workers=False (workers fork the dataset per epoch);
        train.py enforces that.
        """
        self.epoch = int(epoch)
        if cfg is not None:
            self.cfg = cfg

    def __len__(self) -> int:
        return len(self.df)

    def _load(self, i: int) -> Image.Image:
        with Image.open(self.df.at[i, "path"]) as im:
            return im.convert("RGB")

    def __getitem__(self, i: int) -> dict:
        cfg = self.cfg
        row_seed = (self.seed * 1_000_003 + self.epoch * 9_176 + i) % (2 ** 31 - 1)
        img = self._load(i)
        y = int(self.df.at[i, "label"])

        if not self.train:
            # Deterministic, and byte-identical to what the scorer computes.
            u8 = _to_u8_hwc(img)
            fn = {"deploy": deploy_view, "ladder": ladder_view,
                  "robust": robustness_view}[self.eval_mode]
            x = fn(u8, cfg.image_size, row_seed)
            return {"x": torch.from_numpy(np.ascontiguousarray(x)).permute(2, 0, 1),
                    "y": y, "y3": int(self.y3[i]), "row_id": int(self.row_id[i])}

        rng = random.Random(row_seed)
        if cfg.prechain:
            img = source_prechain(img, rng, cfg)
        if cfg.resample_jitter:
            img = source_resample_jitter(img, rng, cfg)
        # Flips on the source region, so both views share geometry exactly.
        if rng.random() < cfg.hflip_p:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        if rng.random() < cfg.vflip_p:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
        u8 = _to_u8_hwc(img)

        u = rng.random()
        if u < cfg.arm_deploy:
            xa = deploy_view(u8, cfg.image_size, row_seed)
        elif u < cfg.arm_deploy + cfg.arm_ladder:
            xa = ladder_view(u8, cfg.image_size, row_seed,
                             crop_prob=self._ladder_crop_prob(i),
                             level_probs=cfg.ladder_level_probs)
        else:
            xa = robustness_view(u8, cfg.image_size, row_seed, rng,
                                 cfg.robust_skip_webp_p)
        # view_b draws are the LAST rng consumers in this function, so the
        # deploy mode (which consumes none) leaves every upstream decision --
        # prechain, jitter, flips, arm selection -- on an identical stream.
        if cfg.view_b_mode == "deploy":
            xb = deploy_view(u8, cfg.image_size, row_seed + 1)
        else:
            xb = robustness_view(u8, cfg.image_size, row_seed + 1, rng,
                                 cfg.robust_skip_webp_p)

        to_t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).permute(2, 0, 1)
        return {"x_a": to_t(xa), "x_b": to_t(xb), "y": y,
                "y3": int(self.y3[i]), "row_id": int(self.row_id[i])}


def worker_init_fn(worker_id: int) -> None:
    """cv2's own thread pool plus N forked workers oversubscribes and can deadlock."""
    import cv2

    cv2.setNumThreads(0)
    torch.set_num_threads(1)
