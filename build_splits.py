"""Assign each manifest row to train / val_id / val_xgen / val_stress / test.

The split is the part of this pipeline that decides whether any later number
means anything, so the rules are explicit and the assertions are hard.

  * **Nothing derived from the same source straddles a split.** Datasets are
    grouped by shared HuggingFace `path` (bitmind/FakeClue backs 9 separate
    datasets across both labels; SyntheticFacesHQ backs 4; OpenFake 2) unioned
    with the explicit cliques in overrides.yaml, which cover the cases a path
    join cannot see: real/fake pairs built from the same images, fakes
    generated *from* a real dataset in this same registry, and the CelebA- and
    FFHQ-trained GANs whose outputs share identities with the real corpora.

  * **val_xgen holds out whole generator families and whole real datasets.**
    Held-out fakes are chosen for paradigm and vendor diversity rather than
    "another diffusion checkpoint". `mixed-generation` covers 48 datasets and
    is a catch-all, not a family, so it is never a holdout axis.

  * **val_id is carved by source block, not by row.** Several datasets here are
    video-derived (inst-it-dataset-videos-*, vtuav, bdd100k, FDDB,
    fakeclue-*-ffpp), so consecutive rows are near-duplicate frames of one
    scene. A row-level split -- which is what v4's make_manifest.py does --
    puts frame t in train and frame t+1 in val, and the resulting number reads
    near 1.0 while measuring nothing.

  * **val_stress is reported, never selected on.** It holds the sub-256px real
    corpora, which are the resolution-shortcut canaries. If they were in the
    selection signal, checkpoint selection would partly be selecting for "got
    better at exploiting resolution", which is the one thing that must not be
    optimised.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd
import yaml

_HERE = Path(__file__).resolve().parent


class _Union:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def build_split_units(df: pd.DataFrame, pair_groups: list[list[str]]) -> pd.Series:
    """Map every dataset to a split_unit that must move as one."""
    uf = _Union()
    ds = df.drop_duplicates("dataset")[["dataset", "source_path"]]
    for name in ds["dataset"]:
        uf.find(name)
    # (a) shared HuggingFace repo
    for _, grp in ds.groupby("source_path"):
        names = list(grp["dataset"])
        for other in names[1:]:
            uf.union(names[0], other)
    # (b) explicit cliques
    known = set(ds["dataset"])
    for clique in pair_groups or []:
        present = [n for n in clique if n in known]
        for other in present[1:]:
            uf.union(present[0], other)
    mapping = {n: uf.find(n) for n in known}
    return df["dataset"].map(mapping)


def _stable_frac(key: str, seed: int) -> float:
    h = hashlib.blake2b(f"{key}|{seed}".encode(), digest_size=8).digest()
    return int.from_bytes(h, "big") / float(1 << 64)


def block_key(sub: pd.DataFrame, target_blocks: int = 16) -> pd.Series:
    """A grouping key for near-duplicate rows within one dataset.

    `source_file` names the parquet shard or archive member and is the best
    available proxy for scene identity. When a dataset came from too few files
    to be useful, fall back to contiguous index blocks: the cache writer assigns
    img_{index:06d} in archive order, so adjacent indices are adjacent source
    material.

    The block size is derived from the dataset size rather than fixed. A fixed
    256 was wrong here: `CACHE_MAX_SAMPLES` caps almost every dataset at 500,
    which gives exactly two blocks, so carving val_id at the block level became
    "take half this dataset or none of it" and covered 22 of 136 datasets. An
    in-distribution split that omits five sixths of the training sources is not
    in-distribution.

    Targeting ~`target_blocks` blocks per dataset trades some near-duplicate
    protection for coverage: at 500 images a block is ~31 consecutive frames,
    so a scene longer than that can straddle. Where `source_file` is available
    it is preferred precisely because it has no such limit.
    """
    if sub["source_file"].nunique() >= 8:
        return sub["source_file"].astype(str)
    idx = sub["file_index"]
    if (idx < 0).mean() > 0.5:      # unparseable filenames: fall back to position
        idx = pd.Series(range(len(sub)), index=sub.index)
    size = max(8, len(sub) // max(1, target_blocks))
    return (idx.clip(lower=0) // size).astype(str)


def assign_splits(df: pd.DataFrame, overrides: dict, *, seed: int = 34,
                  val_id_frac: float = 0.06,
                  per_dataset_cap: int | None = None,
                  strict_provenance: bool = False) -> pd.DataFrame:
    df = df.copy()
    cliques = list(overrides.get("pair_groups") or [])
    if strict_provenance:
        cliques += list(overrides.get("distribution_groups") or [])
    df["split_unit"] = build_split_units(df, cliques)

    holds = overrides.get("holdouts") or {}

    # Training gates (2026-08-11). Each named gate lists datasets EXCLUDED
    # from train by default and parked in `park_in` (val_xgen: measured AND
    # steers checkpoint selection; val_stress: measured every eval, never
    # selected on). Flip a gate's include_in_train to true to fold its
    # datasets into train instead. Rationale per gate lives in overrides.yaml.
    gates = overrides.get("train_gates") or {}
    if gates:
        import copy
        holds = copy.deepcopy(holds)
        for gname, gate in gates.items():
            if gate.get("include_in_train", False):
                continue
            dest = gate.get("park_in", "val_xgen")
            if dest not in ("val_xgen", "val_stress", "test"):
                raise ValueError(f"train_gates.{gname}.park_in must be "
                                 f"val_xgen, val_stress or test, got {dest!r}")
            slot = holds.setdefault(dest, {})
            added = []
            for side in ("fake", "real"):
                cur = list(slot.get(side) or [])
                new = [n for n in (gate.get(side) or []) if n not in cur]
                slot[side] = cur + new
                added += new
            if added:
                print(f"[gate] {gname}: excluded from train -> {dest} "
                      f"({len(added)} datasets: {', '.join(added)}). Set "
                      f"train_gates.{gname}.include_in_train: true to train "
                      f"on them.")

    unit_of = df.drop_duplicates("dataset").set_index("dataset")["split_unit"].to_dict()
    known = set(unit_of)

    # A named dataset pulls its whole split_unit with it.
    designated: dict[str, str] = {}
    for split in ("val_xgen", "test", "val_stress"):
        names: list[str] = []
        for side in ("fake", "real"):
            names += (holds.get(split) or {}).get(side) or []
        missing = [n for n in names if n not in known]
        if missing:
            print(f"\n[warn] {split}: {len(missing)} of {len(names)} designated "
                  f"datasets are not in the manifest and were silently dropped "
                  f"from the holdout:\n         {missing}\n"
                  f"       This shrinks {split} without failing, which is how a "
                  f"holdout quietly stops testing what it was designed to test.\n"
                  f"       Either download them (`gasbench download --datasets "
                  f"{' '.join(missing[:3])} ...`)\n"
                  f"       or replace them in overrides.yaml with datasets you "
                  f"actually have.")
        for n in names:
            if n not in known:
                continue
            u = unit_of[n]
            if u in designated and designated[u] != split:
                raise ValueError(
                    f"split_unit {u!r} is designated for both "
                    f"{designated[u]!r} and {split!r}. Two holdout lists name "
                    f"datasets that share a source; pick one.")
            designated[u] = split

    df["split"] = df["split_unit"].map(designated).fillna("train")

    # Carve val_id out of train, by block, WITHIN each dataset.
    #
    # Per-dataset rather than a global hash threshold: val_id's job is to be an
    # in-distribution estimate of the training mix, for early stopping and for
    # fitting the temperature. A global threshold over blocks makes coverage a
    # lottery -- with ~2 blocks per dataset it selected 22 of 136 datasets and
    # skewed P(fake) from 0.57 to 0.64. Taking roughly val_id_frac of every
    # dataset guarantees the split actually looks like train.
    is_train = df["split"] == "train"
    val_id_idx: list = []
    for name, sub in df[is_train].groupby("dataset", sort=False):
        bk = block_key(sub)
        blocks = sorted(bk.unique(), key=lambda b: _stable_frac(f"{name}|{b}", seed))
        if len(blocks) < 2:
            continue          # one block means all-or-nothing; take nothing
        target = max(1, int(round(len(sub) * val_id_frac)))
        chosen, taken = [], 0
        for b in blocks:
            if taken >= target:
                break
            chosen.append(b)
            taken += int((bk == b).sum())
        val_id_idx.extend(sub.index[bk.isin(chosen)].tolist())
    df.loc[val_id_idx, "split"] = "val_id"

    # Optional per-dataset cap on train. The benchmark draws a roughly equal
    # number of samples from every dataset (calculate_weighted_dataset_sampling
    # gives each regular dataset the same cap, gasstation 5x), so a constant
    # per-dataset cap is what makes train's dataset marginal match eval's.
    if per_dataset_cap:
        keep = []
        for name, sub in df[df["split"] == "train"].groupby("dataset", sort=False):
            if len(sub) <= per_dataset_cap:
                keep.extend(sub.index.tolist())
                continue
            fr = sub["image_id"].map(lambda k: _stable_frac(k, seed))
            keep.extend(sub.index[fr.rank(method="first") <= per_dataset_cap].tolist())
        drop = set(df.index[df["split"] == "train"]) - set(keep)
        df.loc[list(drop), "split"] = "unused"

    return df


# Splits that are allowed to share a source. `val_id` is carved out of train
# sources on purpose -- it is the in-distribution estimate used for early
# stopping and temperature fitting, and both want the train distribution. What
# must never be shared is a source between the train side and a held-out side.
SPLIT_FAMILY = {"train": "train", "val_id": "train", "unused": "train",
                "val_xgen": "val_xgen", "val_stress": "val_stress",
                "test": "test"}


def validate(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Returns (errors, warnings). Errors invalidate every downstream number."""
    errors, warnings = [], []

    fam = df["split"].map(SPLIT_FAMILY).fillna(df["split"])
    spans = fam.groupby(df["split_unit"]).nunique()
    bad = spans[spans > 1]
    if len(bad):
        detail = []
        for unit in list(bad.index[:5]):
            fams = sorted(fam[df.split_unit == unit].unique())
            names = sorted(df.loc[df.split_unit == unit, "dataset"].unique())
            detail.append(f"{unit} -> {fams} ({len(names)} datasets)")
        errors.append(f"{len(bad)} split_unit(s) straddle split families:\n      "
                      + "\n      ".join(detail))

    # train / val_id are load-bearing: train.py trains on the first and fits
    # temperature + (under selection: {val_id: 1.0}) selects checkpoints on
    # the second. val_xgen MAY be deliberately empty (the train-everything
    # 80/20 experiment dissolves all dataset-level holdouts): train.py
    # renormalises training.selection over the splits that exist, so this is
    # only unsafe under the DEFAULT weights ({val_xgen: 1.0}), where best.pt
    # would freeze at the first eval -- hence the warning below. `test` is
    # read by no code -- a config that declares `test: {}` is legitimate.
    for s in ("train", "val_id"):
        sub = df[df.split == s]
        if sub.empty:
            errors.append(f"split {s!r} is empty")
        elif sub.label.nunique() < 2:
            errors.append(f"split {s!r} has only one label")

    xg = df[df.split == "val_xgen"]
    if xg.empty:
        warnings.append(
            "split 'val_xgen' is empty -- no held-out generator panel exists. "
            "Safe ONLY when training.selection gives val_id nonzero weight "
            "(the 80/20 experiment configs set selection: {val_id: 1.0}); "
            "under the default selection weights best.pt would freeze at the "
            "first eval.")
    elif xg.label.nunique() < 2:
        errors.append("split 'val_xgen' has only one label")

    test = df[df.split == "test"]
    if test.empty:
        warnings.append(
            "split 'test' is empty -- its datasets were folded into train. "
            "No untouched final-measurement split exists; the gasbench run on "
            "the exported submission is now the only end gate.")
    elif test.label.nunique() < 2:
        errors.append("split 'test' has only one label")

    tr = df[df.split == "train"]
    if not tr.empty:
        for cat, sub in tr.groupby("category"):
            if sub.label.nunique() < 2:
                errors.append(
                    f"train category {cat!r} has a single label; the sampler "
                    f"will exclude it entirely")

    # Held-out splits with single-label categories are usable but their
    # per-category MCC is undefined and their per-category accuracy is really
    # recall. Worth surfacing, not worth blocking on.
    for s in ("val_xgen", "test"):
        sub = df[df.split == s]
        if sub.empty:
            continue
        single = [c for c, g in sub.groupby("category") if g.label.nunique() < 2]
        if single:
            warnings.append(
                f"{s}: categories with one label: {single}. Overall metrics are "
                f"fine; per-category MCC for these is undefined and their "
                f"accuracy is recall for the single label present.")
        p = sub.label.mean()
        if not 0.35 <= p <= 0.65:
            warnings.append(
                f"{s}: P(fake)={p:.4f} is far from the benchmark's ~0.5. MCC and "
                f"Brier both depend on the class prior, so this split will not "
                f"track the leaderboard closely. Rebalance the holdout lists in "
                f"overrides.yaml, or accept it as a ranking signal only.")
    return errors, warnings


def report(df: pd.DataFrame) -> None:
    print(f"\n{'split':12}{'images':>10}{'datasets':>10}{'P(fake)':>10}"
          f"{'categories':>12}")
    for s in ("train", "val_id", "val_xgen", "val_stress", "test", "unused"):
        sub = df[df.split == s]
        if sub.empty:
            continue
        print(f"{s:12}{len(sub):10,}{sub.dataset.nunique():10}"
              f"{sub.label.mean():10.4f}{sub.category.nunique():12}")

    print("\n-- label x category per split --")
    for s in ("train", "val_xgen", "test"):
        sub = df[df.split == s]
        if sub.empty:
            continue
        print(f"  [{s}]")
        for cat, g in sub.groupby("category"):
            r, f = int((g.label == 0).sum()), int((g.label == 1).sum())
            flag = "" if (r and f) else "   <-- SINGLE LABEL"
            print(f"    {cat:12} real={r:7,} fake={f:7,}{flag}")

    xg = df[df.split == "val_xgen"]
    if not xg.empty:
        fams = sorted(xg[xg.label == 1].generator_family.unique())
        print(f"\n  val_xgen held-out generator families: {fams}")
        print(f"  val_xgen held-out real datasets: "
              f"{sorted(xg[xg.label == 0].dataset.unique())}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--manifest", default="data/manifest.parquet")
    ap.add_argument("--overrides", default=str(_HERE / "overrides.yaml"))
    ap.add_argument("--out", default=None, help="defaults to overwriting --manifest")
    ap.add_argument("--seed", type=int, default=34)
    ap.add_argument("--val-id-frac", type=float, default=0.06)
    ap.add_argument("--per-dataset-cap", type=int, default=None)
    ap.add_argument("--strict-provenance", action="store_true",
                    help="also union distribution_groups (generator trained on "
                         "a real corpus, no per-image derivation). Conservative; "
                         "costs training datasets.")
    ap.add_argument("--write", action="store_true",
                    help="without this the split is computed and reported only")
    args = ap.parse_args()

    df = pd.read_parquet(args.manifest)
    overrides = yaml.safe_load(Path(args.overrides).read_text()) or {}
    df = assign_splits(df, overrides, seed=args.seed,
                       val_id_frac=args.val_id_frac,
                       per_dataset_cap=args.per_dataset_cap,
                       strict_provenance=args.strict_provenance)
    report(df)

    errors, warnings = validate(df)
    for w in warnings:
        print(f"\n[warn] {w}")
    if errors:
        print("\n[FAIL] split validation:")
        for p in errors:
            print(f"  - {p}")
    else:
        print("\n[ok] split validation passed"
              + (f" ({len(warnings)} warning(s) above)" if warnings else ""))

    if args.write:
        if errors:
            raise SystemExit("refusing to write a split that fails validation")
        out = Path(args.out or args.manifest)
        df.to_parquet(out, index=False)
        print(f"wrote {len(df):,} rows -> {out}")
    else:
        print("\n(dry run; pass --write to persist)")


if __name__ == "__main__":
    main()
