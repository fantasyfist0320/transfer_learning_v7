#!/usr/bin/env python3
"""Name the datasets that bind sampler.max_replay.

The run log warns `replay cap BOUND` and prints `peak replay by kind`, but never
says WHICH dataset is being drawn 25x/epoch -- and a few-hundred-image dataset
replayed 25x per epoch is an overfitting risk worth knowing about. This reruns
the exact sampler weighting on the real manifest and reports per-dataset
expected draws (`amp = w * n_rows`) next to each dataset's row count.

  python diagnose_replay.py --manifest data/manifest.parquet \
      --config config-experiment-2.yaml [--top 25]
"""
import argparse

import pandas as pd
import yaml

from data import balanced_weights


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/manifest.parquet")
    ap.add_argument("--config", default="config-experiment-2.yaml")
    ap.add_argument("--top", type=int, default=25)
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config))
    s = cfg["sampler"]
    df = pd.read_parquet(a.manifest)
    tr = df[df.split == "train"].reset_index(drop=True)
    print(f"train rows {len(tr):,}  datasets {tr.dataset.nunique()}\n")

    w, rep = balanced_weights(
        tr, amp_max=s["max_replay"], kind_balance=s["kind_balance"],
        gasstation_boost=float(s.get("gasstation_boost", 1.0)), verbose=True)

    d = tr.assign(amp=w * len(tr))
    g = d.groupby("dataset").agg(rows=("amp", "size"), amp=("amp", "max"),
                                 kind=("kind", "first"),
                                 category=("category", "first"))
    cap = float(s["max_replay"])
    g = g.sort_values("amp", ascending=False)
    print(f"\nTop {a.top} by expected draws per image per epoch (cap = {cap:g}x):")
    print(f"  {'amp':>8}  {'rows':>8}  {'kind':<15}{'category':<12}dataset")
    for name, r in g.head(a.top).iterrows():
        # A row AT the cap was clipped: its true target share was higher.
        flag = "  <-- AT CAP" if r.amp >= cap - 1e-6 else ""
        print(f"  {r.amp:7.1f}x  {int(r.rows):>8,}  {str(r.kind):<15}"
              f"{str(r.category):<12}{name}{flag}")

    at_cap = g[g.amp >= cap - 1e-6]
    if len(at_cap):
        print(f"\n{len(at_cap)} dataset(s) clipped by the cap, holding "
              f"{int(at_cap.rows.sum()):,} rows. Each of those images is served "
              f"~{cap:g}x per epoch.\nRaise sampler.max_replay for exact shares "
              f"(more repetition), or leave it: the log's measured deviation "
              f"was <=0.0008 on P(fake) and P(fake|category).")
    else:
        print("\nCap not bound: every dataset is below the replay ceiling.")


if __name__ == "__main__":
    main()
