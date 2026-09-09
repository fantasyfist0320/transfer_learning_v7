"""Pre-training gate over the downloaded corpus + built manifest.

    python audit_data.py --data-root /path/to/datasets --manifest data/manifest.parquet

Run AFTER build_manifest.py (and ideally after build_splits.py --write; the
split section is skipped when splits are absent). Exit code 1 on any HARD
failure, 0 otherwise -- so it can gate a training launch in a shell chain:

    python audit_data.py --data-root $DATA --manifest data/manifest.parquet \
        && accelerate launch train.py --config config.yaml --branch dinov3

Checks, in order of how expensive their failure is downstream:

  1. Coverage (HARD on split-critical): every dataset named by
     overrides.yaml:holdouts must be on disk and in the manifest. A missing
     val_xgen dataset silently degrades the selection metric the whole run
     optimizes; a missing test dataset silently degrades the one honest
     final number. Also reports registry datasets not downloaded (WARN --
     each is a public eval source the model will face untrained) and on-disk
     directories the registry does not know (WARN -- they carry no label and
     build_manifest skipped them).
  2. Counts (WARN): per-dataset row deficits against the expected download
     (500/dataset, gasstation larger). A dataset that lost most of its rows
     to probe failures is a download problem to fix now, not after training.
  3. Format bias (WARN): per-class file_format / pil_mode / file_bytes.
     Detectors trained on class-skewed formats learn the codec, not the
     generator ("Fake or JPEG?", arXiv:2403.17608). The training views
     already recompress (deploy applies the eval's JPEG q75 base chain), so
     this is a report, not a gate -- but a >30pp per-format class gap is
     flagged as the signal to enable clean_view_recompress in config.yaml.
  4. Resolution (WARN): m_eval = S/min_side per class -- how much of each
     class the benchmark will upsample (m>1) or alias-downsample (m<1).
     build_manifest.report() prints the full m-bin coverage table; here only
     the per-class quartiles and the worst per-dataset offenders repeat.
  5. Splits (WARN, skipped if build_splits has not run): per-split size and
     P(fake). val_xgen far from 0.5 stops tracking the leaderboard prior.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from build_manifest import (find_dataset_dirs, load_overrides, load_registry)

GASSTATION = "gasstation-generated-images"


def collect_holdout_names(overrides: dict) -> dict[str, list[str]]:
    """{split_name: [dataset, ...]} from overrides.yaml:holdouts.

    Actual shapes in that section: {fake: [...], real: [...]} for
    val_xgen/test and {real: [...]} for val_stress -- all dicts of lists; the
    plain-list branch is future-tolerance only.
    """
    out: dict[str, list[str]] = {}
    for split, spec in (overrides.get("holdouts") or {}).items():
        names: list[str] = []
        if isinstance(spec, dict):
            for side in spec.values():
                names.extend(side or [])
        elif isinstance(spec, list):
            names.extend(spec)
        out[split] = names
    return out


def pair_clique_of(overrides: dict, name: str) -> list[str]:
    for clique in overrides.get("pair_groups") or []:
        members = clique if isinstance(clique, list) else \
            (clique.get("datasets") if isinstance(clique, dict) else None)
        if members and name in members:
            return [m for m in members if m != name]
    return []


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True,
                    help="directory containing the named dataset folders "
                         "(the build_manifest.py --cache-dir path)")
    ap.add_argument("--manifest", required=True,
                    help="parquet written by build_manifest.py")
    ap.add_argument("--config-dir", default=None,
                    help="gasbench dataset-config dir (default: sibling "
                         "gasbench checkout, same as build_manifest.py)")
    ap.add_argument("--overrides", default=None,
                    help="overrides.yaml path (default: alongside this file)")
    ap.add_argument("--expected-per-dataset", type=int, default=500)
    ap.add_argument("--expected-gasstation", type=int, default=5000)
    ap.add_argument("--deficit-tolerance", type=float, default=0.2,
                    help="warn when a dataset has fewer than "
                         "(1 - tol) * expected rows")
    args = ap.parse_args()

    hard_failures: list[str] = []
    warnings: list[str] = []

    registry = load_registry(Path(args.config_dir) if args.config_dir else None)
    overrides = load_overrides(Path(args.overrides) if args.overrides else None)
    df = pd.read_parquet(args.manifest)
    on_disk = {name for name, _ in find_dataset_dirs(Path(args.data_root))}
    in_manifest = set(df["dataset"].unique())

    # -- 1. coverage --------------------------------------------------------
    print("=" * 72)
    print("1. COVERAGE")
    holdouts = collect_holdout_names(overrides)
    for split, names in holdouts.items():
        for name in names:
            missing_where = [w for w, s in
                             (("disk", on_disk), ("manifest", in_manifest))
                             if name not in s]
            if missing_where:
                hard_failures.append(
                    f"[{split}] {name} missing from {'/'.join(missing_where)}"
                    f" -- this split silently degrades without it")
            else:
                # Pair-group mates ARE part of the designed holdout (e.g.
                # flickr-sd3-real joins val_xgen via its clique; the split's
                # category matching counts it) -- a missing mate degrades the
                # split exactly like a missing listed member. HARD.
                absent_mates = [m for m in pair_clique_of(overrides, name)
                                if m not in in_manifest]
                if absent_mates:
                    hard_failures.append(
                        f"[{split}] {name} present but its pair-group mates "
                        f"{absent_mates} are not -- the matched real/fake "
                        f"pairing this holdout was designed around vanishes")
    n_hold = sum(len(v) for v in holdouts.values())
    print(f"  holdout datasets required: {n_hold} across {list(holdouts)}")
    print(f"  hard failures so far: {len(hard_failures)}")

    not_downloaded = sorted(set(registry) - on_disk)
    if not_downloaded:
        warnings.append(
            f"{len(not_downloaded)} registry datasets not on disk (each is a "
            f"public eval source the model meets untrained): "
            f"{', '.join(not_downloaded[:8])}"
            f"{' ...' if len(not_downloaded) > 8 else ''}")
    # extra_datasets entries are locally added training data -- known by
    # design, not dead weight. probe-* dirs are the probe suite (eval-only by
    # design, see probes.yaml): deliberately unknown to the registry so
    # build_manifest cannot train on them.
    extra_names = {e.get("name")
                   for e in overrides.get("extra_datasets") or []}
    known = set(registry) | extra_names
    probe_dirs = sorted(d for d in on_disk - known if d.startswith("probe-"))
    if probe_dirs:
        print(f"  probe suite (eval-only, by design): {len(probe_dirs)} "
              f"dirs: {', '.join(probe_dirs[:8])}"
              f"{' ...' if len(probe_dirs) > 8 else ''}")
    unknown = sorted(d for d in on_disk - known
                     if not d.startswith("probe-"))
    if unknown:
        warnings.append(
            f"{len(unknown)} on-disk dirs unknown to the registry (skipped by "
            f"build_manifest, dead weight): {', '.join(unknown[:8])}"
            f"{' ...' if len(unknown) > 8 else ''}")
    print(f"  registry {len(registry)} | on disk {len(on_disk)} | "
          f"in manifest {len(in_manifest)}")

    # -- 2. counts ----------------------------------------------------------
    print("=" * 72)
    print("2. COUNTS")
    counts = df.groupby("dataset").size().sort_values()
    floor = int(round(args.expected_per_dataset * (1 - args.deficit_tolerance)))
    gs_floor = int(round(args.expected_gasstation * (1 - args.deficit_tolerance)))
    deficient = counts[(counts < floor) & (counts.index != GASSTATION)]
    print(f"  rows {len(df):,} | datasets {len(counts)} | "
          f"median rows/dataset {int(counts.median())}")
    if GASSTATION in counts.index:
        print(f"  {GASSTATION}: {counts[GASSTATION]:,} rows")
        if counts[GASSTATION] < gs_floor:
            warnings.append(
                f"{GASSTATION} has {counts[GASSTATION]:,} rows, under the "
                f"{gs_floor:,} floor -- the highest-weight single eval source "
                f"is under-represented")
    else:
        warnings.append(f"{GASSTATION} absent from manifest -- the highest-"
                        f"weight single eval source is untrained")
    if len(deficient):
        warnings.append(
            f"{len(deficient)} datasets under {floor} rows (expected "
            f"~{args.expected_per_dataset}): " +
            ", ".join(f"{n}={c}" for n, c in deficient.head(10).items()) +
            (" ..." if len(deficient) > 10 else ""))

    # -- 3. format bias -----------------------------------------------------
    print("=" * 72)
    print("3. FORMAT BIAS (real vs fake; the codec-shortcut ticket)")
    for col in ("file_format", "pil_mode"):
        tab = (df.groupby("label")[col].value_counts(normalize=True)
                 .unstack(fill_value=0.0))
        print(f"  -- {col} share by class (0=real, 1=fake) --")
        print(tab.to_string(float_format=lambda v: f"{v:8.1%}"))
        gaps = (tab.loc[0] - tab.loc[1]).abs() if set(tab.index) >= {0, 1} \
            else pd.Series(dtype=float)
        big = gaps[gaps > 0.30]
        for fmt, gap in big.items():
            warnings.append(
                f"{col}={fmt}: {gap:.0%} class share gap -- codec/mode is a "
                f"label shortcut; enable clean_view_recompress in config.yaml "
                f"and re-check")
    bpp = df.groupby("label")["bytes_per_pixel"].describe(
        percentiles=[0.25, 0.5, 0.75])
    print("  -- bytes/pixel by class --")
    print(bpp[["25%", "50%", "75%"]].to_string(
        float_format=lambda v: f"{v:8.3f}"))

    # -- 4. resolution ------------------------------------------------------
    print("=" * 72)
    print("4. RESOLUTION (m_eval = S/min_side; >1 upsampled, <1 aliased)")
    q = df.groupby("label")["m_eval"].describe(percentiles=[0.25, 0.5, 0.75])
    print(q[["25%", "50%", "75%"]].to_string(
        float_format=lambda v: f"{v:8.2f}"))
    per_ds = df.groupby("dataset")["m_eval"].median()
    extreme = per_ds[(per_ds > 2.0) | (per_ds < 0.25)]
    if len(extreme):
        print(f"  {len(extreme)} datasets with extreme median m_eval "
              f"(benchmark heavily resamples them; worst first, both ends):")
        import numpy as np
        by_dist = extreme.iloc[np.argsort(-np.abs(np.log(extreme.to_numpy())))]
        for n, m in by_dist.head(12).items():
            print(f"    {m:6.2f}  {n}")
    print("  (full m-bin both-label coverage: see build_manifest.report())")

    # -- 5. splits ----------------------------------------------------------
    print("=" * 72)
    print("5. SPLITS")
    if "split" not in df.columns or (df["split"] == "").all():
        warnings.append("no splits in manifest yet -- run build_splits.py "
                        "--write, then re-run this audit")
    else:
        for split, sub in df[df["split"] != ""].groupby("split"):
            pf = float((sub["label"] == 1).mean())
            note = ""
            # Prior check on val_xgen only: it is the selection signal that
            # must track the leaderboard's ~0.5 prior. val_stress is real-only
            # by design and val_id inherits the train mix.
            if split == "val_xgen" and abs(pf - 0.5) > 0.15:
                note = "  <-- far from the benchmark's ~0.5 prior"
                warnings.append(f"split {split} has P(fake)={pf:.2f}{note}")
            print(f"  {split:12s} n={len(sub):7,}  P(fake)={pf:.2f}{note}")

    # -- verdict ------------------------------------------------------------
    print("=" * 72)
    for w in warnings:
        print(f"WARN  {w}")
    for h in hard_failures:
        print(f"HARD  {h}")
    print(f"{len(hard_failures)} hard failure(s), {len(warnings)} warning(s)")
    if hard_failures:
        print("Fix the hard failures before training: a targeted re-download "
              "of only the named datasets (~500 images each) is enough.")
        sys.exit(1)
    print("Audit passed -- clear to train.")


if __name__ == "__main__":
    main()
