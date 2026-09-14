"""Check whether a LOCAL evaluation panel predicts what the CHAIN measured.

    # 1. pull chain per-dataset results (any machine with network)
    python panel_check.py fetch --version 24 --out-dir chain_refs \
        --hotkey sep7=5D7Wk4RHQJ7pzzCdGufvisNUHiEbhw5UfvYuJoDdDinyGY4G \
        --hotkey log7=5HpXa6tmkZ6g2bcKm4CTUb1s6T6uXGpim2P8aBugX1dSSwRp \
        --hotkey oc1=5HKgDDF8QAgPH6o5meKUEYRqDCvomTNDxGssX3b8sdpmoiR5 \
        --hotkey armF=5GMzQZU9HarB1HGaenvaPdVjeSwc44NLLjgeoJxsuqyLNCbk

    # 2. score the SAME submission locally with gasbench (GPU box)
    gasbench run --image-model <submission_dir> --full --multiclass-scoring \
        --seed 34 --results-dir panel_results --run-name sep7-v23 \
        --datasets $(cat panel_v23.txt)

    # deploy prior for training.deploy_prior / export --deploy-prior
    python panel_check.py prior --chain chain_refs/armF.json

    # 3a. faithfulness: does local reproduce chain dataset by dataset?
    python panel_check.py compare --chain chain_refs/sep7.json \
        --local panel_results/sep7-v23/results.json

    # 3b. predictiveness: does a local dataset ORDER the models the way a
    #     chain holdout did?
    python panel_check.py rank \
        --model sep7:chain_refs/sep7.json:panel_results/sep7-reals/results.json \
        --model log7:chain_refs/log7.json:panel_results/log7-reals/results.json ...

Why this exists: local gasbench is in-sample (every public set is in the
training manifest), so it cannot see the failure that decides rounds -- a
whole unseen real domain called fake. In v24 two hidden real holdouts
(8dca83a2, d766cda9; 630 of ~82k images) carry 67-69% of ALL weighted error
for log7 and oc-1, and 22% for arm F. A local panel is only worth building
decisions on once it has been shown to reproduce such outcomes on models
whose chain results are already known. This tool is that check; it does not
train, export or download anything.

Both inputs share one shape: the API's `per_source_accuracy` and gasbench's
results.json `per_source_accuracy` are {truth: {dataset: {pred_label: n}}}.
Accuracy here is EXACT-MATCH (multiclass), the quantity v23+ scores; the
binary column (anything-but-real counts as fake) is printed alongside.

Caveats the numbers cannot remove:
  * the chain draws ~315 images per set, local draws up to the 500-image
    cache -- different images, so differences are judged with a two-
    proportion z, and slice-heterogeneous sets (see multishard notes) can
    disagree beyond it without either side being wrong;
  * `rank` over 4 models has only 24 orderings: with many candidate datasets
    some will match by chance. Treat a match as a hypothesis to confirm on
    the next scored model, never as proof.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

API = "https://gas.bitmind.ai/api/v1/analytics/discriminator-benchmark-results"
# The API answers Forbidden without browser-like headers.
HEADERS = {
    "Origin": "https://competition.bitmind.ai",
    "Referer": "https://competition.bitmind.ai/",
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) Gecko/20100101 Firefox/130.0",
}
RUN_FIELDS = ("run_id", "run_date", "benchmark_version", "sn34_score",
              "base_sn34_score", "aug_sn34_score", "gorodkin_mcc",
              "multiclass_brier", "binary_mcc", "n_items")
Z_FLAG = 3.0


# --- loading ------------------------------------------------------------------

def load_psa(path: str) -> dict[str, tuple[str, dict[str, int]]]:
    """{dataset: (truth, {pred_label: count})} from a chain ref or results.json."""
    d = json.loads(Path(path).read_text())
    psa = d.get("per_source_accuracy")
    if not psa:
        sys.exit(f"{path}: no per_source_accuracy (gasbench run incomplete?)")
    return {name: (truth, counts)
            for truth, sets in psa.items() for name, counts in sets.items()}


def tally(truth: str, counts: dict[str, int]) -> tuple[int, int, int]:
    """(n, exact-match correct, binary correct)."""
    n = sum(counts.values())
    exact = counts.get(truth, 0)
    real = counts.get("real", 0)
    binary = real if truth == "real" else n - real
    return n, exact, binary


def provenance(name: str) -> str:
    if "-holdout-" in name:
        return "holdout"
    return "gasstation" if "gasstation" in name else "public"


def z_two_prop(k1: int, n1: int, k2: int, n2: int) -> float:
    if n1 == 0 or n2 == 0:
        return float("nan")
    p = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    return 0.0 if se == 0 else (k1 / n1 - k2 / n2) / se


def read_names(arg: list[str] | None) -> set[str] | None:
    """--datasets accepts names and/or @file (one name per line)."""
    if not arg:
        return None
    out: set[str] = set()
    for a in arg:
        if a.startswith("@"):
            out.update(l.strip() for l in Path(a[1:]).read_text().split()
                       if l.strip())
        else:
            out.add(a)
    return out


# --- fetch --------------------------------------------------------------------

def cmd_fetch(args) -> None:
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for spec in args.hotkey:
        label, _, hotkey = spec.partition("=")
        if not hotkey:
            sys.exit(f"--hotkey wants label=HOTKEY, got {spec!r}")
        url = (f"{API}?ss58_address={hotkey}&benchmark_version={args.version}"
               f"&runs_limit=200")
        req = urllib.request.Request(url, headers=HEADERS)
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                d = json.load(r)
        except urllib.error.HTTPError as e:
            # e.g. "Results from this benchmark version are not public." --
            # older versions close after the round; fetch while they are live.
            sys.exit(f"{label}: HTTP {e.code} {e.read().decode()[:200]}")
        runs = [r for r in d.get("detailed_runs", [])
                if r.get("modality") == "image"]
        if not runs:
            sys.exit(f"{label}: no image runs for v{args.version}: {str(d)[:200]}")
        runs.sort(key=lambda r: r["run_date"], reverse=True)
        if len(runs) > 1:
            print(f"[fetch] {label}: {len(runs)} runs, keeping latest "
                  f"{runs[0]['run_id']} ({runs[0]['run_date']})")
        run = runs[0]
        ref = {"label": label, "hotkey": hotkey,
               "discriminator_id": d.get("discriminator_id"),
               **{k: run.get(k) for k in RUN_FIELDS},
               "per_source_accuracy": run["per_source_accuracy"]}
        path = out / f"{label}.json"
        path.write_text(json.dumps(ref, indent=1))
        n_sets = sum(len(v) for v in run["per_source_accuracy"].values())
        print(f"[fetch] {label}: sn34 {run['sn34_score']:.4f}, {n_sets} datasets "
              f"-> {path}")


# --- compare ------------------------------------------------------------------

def cmd_compare(args) -> None:
    chain, local = load_psa(args.chain), load_psa(args.local)
    names = read_names(args.datasets)
    common = sorted(set(chain) & set(local))
    if names is not None:
        missing = sorted(names - set(common))
        if missing:
            print(f"[compare] WARNING not in both files: {missing}")
        common = [n for n in common if n in names]
    if not common:
        sys.exit("no datasets in common")

    rows = []
    for name in common:
        tc, cc = chain[name]
        tl, cl = local[name]
        if tc != tl:
            print(f"[compare] WARNING truth differs for {name}: chain {tc} "
                  f"local {tl} (registry taxonomy changed?)")
        nc, ec, bc = tally(tc, cc)
        nl, el, bl = tally(tl, cl)
        rows.append(dict(dataset=name, truth=tc, prov=provenance(name),
                         n_chain=nc, n_local=nl,
                         chain=100 * ec / nc if nc else np.nan,
                         local=100 * el / nl if nl else np.nan,
                         bin_chain=100 * bc / nc if nc else np.nan,
                         bin_local=100 * bl / nl if nl else np.nan,
                         z=z_two_prop(el, nl, ec, nc)))
    df = pd.DataFrame(rows)
    df["delta"] = df["local"] - df["chain"]
    df = df.sort_values("chain")

    pd.set_option("display.width", 200)
    print(df.to_string(index=False, float_format=lambda v: f"{v:6.1f}"))

    flagged = df[df["z"].abs() >= Z_FLAG]
    spread = df["chain"].std()
    r = (np.corrcoef(df["chain"], df["local"])[0, 1]
         if len(df) > 2 and spread > 0 and df["local"].std() > 0 else float("nan"))
    print(f"\n[compare] {len(df)} datasets | median |delta| "
          f"{df['delta'].abs().median():.1f} pts | max |delta| "
          f"{df['delta'].abs().max():.1f} | pearson r {r:.3f} | "
          f"|z|>={Z_FLAG:g}: {len(flagged)}")
    for _, f in flagged.iterrows():
        print(f"  DISAGREES {f['dataset']}: chain {f['chain']:.1f} local "
              f"{f['local']:.1f} (z {f['z']:+.1f})")
    # The verdict is about the datasets that CAN fail. A panel where every set
    # sits at 100% on both sides agrees trivially and proves nothing.
    informative = df[(df["chain"] < 95) | (df["local"] < 95)]
    print(f"[compare] informative (either side < 95%): {len(informative)} "
          f"datasets, {int((informative['z'].abs() >= Z_FLAG).sum())} disagree")
    if args.out:
        df.to_csv(args.out, index=False)
        print(f"[compare] -> {args.out}")


# --- prior --------------------------------------------------------------------

def cmd_prior(args) -> None:
    """Score-weighted class shares of one chain run -> training.deploy_prior.

    gasbench weights each sample so provenance classes carry fixed shares of
    the score (recording.derive_provenance_weights; public 0.50 / holdout
    0.35 / gasstation 0.15 reproduced v22-v23 runs to 5 decimals). The label
    prior the model is graded on is therefore the provenance-weighted count,
    not the registry's dataset count -- and it moves with each benchmark's
    hidden holdouts (v23 P(semi|fake) ~0.23, v24 ~0.10).
    """
    psa = load_psa(args.chain)
    comp = dict(zip(("public", "holdout", "gasstation"), args.composition))
    n_prov: dict[str, int] = {}
    for name, (truth, counts) in psa.items():
        n_prov[provenance(name)] = n_prov.get(provenance(name), 0) + sum(counts.values())
    present = {p: comp[p] for p in n_prov}
    total, share_sum = sum(n_prov.values()), sum(present.values())
    w = {p: present[p] / share_sum * total / n_prov[p] for p in n_prov}

    mass: dict[str, float] = {}
    raw: dict[str, int] = {}
    for name, (truth, counts) in psa.items():
        n = sum(counts.values())
        mass[truth] = mass.get(truth, 0.0) + n * w[provenance(name)]
        raw[truth] = raw.get(truth, 0) + n
    other = sorted(set(mass) - {"real", "synthetic", "semisynthetic"})
    if other:
        print(f"[prior] WARNING truth groups outside the image taxonomy "
              f"ignored: {other}")
    classes = ("real", "synthetic", "semisynthetic")
    for label, src in (("unweighted", raw), ("score-weighted", mass)):
        tot = sum(src.get(c, 0) for c in classes)
        sh = [src.get(c, 0) / tot for c in classes]
        print(f"[prior] {label:15s} real {sh[0]:.4f}  synthetic {sh[1]:.4f}  "
              f"semisynthetic {sh[2]:.4f}  P(semi|fake) "
              f"{sh[2] / (sh[1] + sh[2]):.4f}")
    print(f"[prior] provenance counts {n_prov}, per-sample weights "
          f"{ {p: round(v, 3) for p, v in w.items()} }")
    print(f"\n  config:  training.deploy_prior: "
          f"[{sh[0]:.4f}, {sh[1]:.4f}, {sh[2]:.4f}]")
    print(f"  export:  --deploy-prior {sh[0]:.4f},{sh[1]:.4f},{sh[2]:.4f}")


# --- rank ---------------------------------------------------------------------

def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = pd.Series(a).rank().to_numpy(), pd.Series(b).rank().to_numpy()
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def pooled_acc(psa: dict, names: list[str]) -> float:
    n = k = 0
    for name in names:
        dn, dk, _ = tally(*psa[name])
        n, k = n + dn, k + dk
    return 100 * k / n if n else float("nan")


def cmd_rank(args) -> None:
    models = []
    for spec in args.model:
        parts = spec.split(":")
        if len(parts) != 3:
            sys.exit(f"--model wants label:chain.json:local.json, got {spec!r}")
        models.append((parts[0], load_psa(parts[1]), load_psa(parts[2])))
    if len(models) < 3:
        sys.exit("rank needs >= 3 models; an ordering of 2 is a coin flip")

    # Chain targets: explicit names, else every hidden REAL holdout all models share.
    shared_chain = set.intersection(*(set(c) for _, c, _ in models))
    if args.target:
        targets = sorted(read_names(args.target))
        missing = [t for t in targets if t not in shared_chain]
        if missing:
            sys.exit(f"targets not in every chain ref: {missing}")
    else:
        targets = sorted(n for n in shared_chain
                         if provenance(n) == "holdout"
                         and models[0][1][n][0] == "real")
    labels = [m[0] for m in models]
    chain_vecs = {t: np.array([pooled_acc(c, [t]) for _, c, _ in models])
                  for t in targets}
    chain_vecs["POOLED"] = np.array([pooled_acc(c, targets) for _, c, _ in models])

    print("[rank] chain target accuracy (exact-match):")
    tdf = pd.DataFrame(chain_vecs, index=labels).T
    print(tdf.to_string(float_format=lambda v: f"{v:6.1f}"))

    shared_local = set.intersection(*(set(l) for _, _, l in models))
    names = read_names(args.datasets)
    if names is not None:
        shared_local &= names
    if args.truth:
        shared_local = {n for n in shared_local if models[0][2][n][0] == args.truth}
    if not shared_local:
        sys.exit("no local datasets shared by every model")

    rows = []
    for name in sorted(shared_local):
        vec = np.array([pooled_acc(l, [name]) for _, _, l in models])
        row = {"local_dataset": name, "truth": models[0][2][name][0],
               **{lab: v for lab, v in zip(labels, vec)},
               "range": vec.max() - vec.min()}
        for t, cv in chain_vecs.items():
            row[f"rho:{t.split('-holdout-')[-1]}"] = spearman(vec, cv)
        rows.append(row)
    df = pd.DataFrame(rows).sort_values(["rho:POOLED", "range"],
                                        ascending=[False, False])
    print(f"\n[rank] {len(df)} local datasets vs chain targets "
          f"({len(models)} models):")
    print(df.to_string(index=False, float_format=lambda v: f"{v:6.2f}"))

    # A dataset every model aces carries no ordering, whatever its rho says.
    useful = df[(df["range"] >= args.min_range) & (df["rho:POOLED"] >= 0.8)]
    print(f"\n[rank] candidates (rho >= 0.8 vs POOLED, range >= "
          f"{args.min_range:g} pts): {len(useful)}")
    for _, u in useful.iterrows():
        print(f"  {u['local_dataset']}")
    print(f"[rank] {len(models)} models -> {math.factorial(len(models))} "
          f"possible orderings; confirm any candidate on the NEXT scored model.")
    if args.out:
        df.to_csv(args.out, index=False)
        print(f"[rank] -> {args.out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fetch", help="pull chain per-dataset results")
    f.add_argument("--hotkey", action="append", required=True, metavar="LABEL=HK")
    f.add_argument("--version", type=int, required=True)
    f.add_argument("--out-dir", default="chain_refs")
    f.set_defaults(fn=cmd_fetch)

    c = sub.add_parser("compare", help="local vs chain, dataset by dataset")
    c.add_argument("--chain", required=True)
    c.add_argument("--local", required=True)
    c.add_argument("--datasets", nargs="+", metavar="NAME|@FILE")
    c.add_argument("--out")
    c.set_defaults(fn=cmd_compare)

    p = sub.add_parser("prior", help="score-weighted class shares of a chain run")
    p.add_argument("--chain", required=True)
    p.add_argument("--composition", nargs=3, type=float, default=[0.50, 0.35, 0.15],
                   metavar=("PUBLIC", "HOLDOUT", "GASSTATION"))
    p.set_defaults(fn=cmd_prior)

    r = sub.add_parser("rank", help="does a local set order models like the chain?")
    r.add_argument("--model", action="append", required=True,
                   metavar="LABEL:CHAIN.json:LOCAL.json")
    r.add_argument("--target", nargs="+", metavar="NAME|@FILE",
                   help="chain datasets to predict (default: all real holdouts)")
    r.add_argument("--datasets", nargs="+", metavar="NAME|@FILE",
                   help="restrict local candidates")
    r.add_argument("--truth", choices=["real", "synthetic", "semisynthetic"])
    r.add_argument("--min-range", type=float, default=10.0)
    r.add_argument("--out")
    r.set_defaults(fn=cmd_rank)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
