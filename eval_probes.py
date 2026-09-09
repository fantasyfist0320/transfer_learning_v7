"""Score the trained ensemble on every probe dataset; build the type-coverage
map and append to the cycle-over-cycle regression ledger.

    python eval_probes.py --probes probes.yaml --data-root <CACHE> \
        --runs-dir runs/v7 --branches dinov3,clip,convnext,eva,dct \
        [--modes deploy,robust] [--tag cycle-2026-08-13]

Why this exists: hidden benchmark holdouts are ordinary online datasets with
distinctive STYLES (the v21 release proved it: dflip3k, streetview, palm
images, art photos -- all public sets). A model that has never seen a style
calls the whole domain fake (dflip3k reals measured 15%). This tool measures
that per type BEFORE the benchmark does: each probes.yaml type holds a small
external probe dataset the model never trains on, and the per-type accuracy
table is the coverage map that decides where the next training data goes.

Probe datasets live in the ordinary cache under `probe-*` names but are NEVER
listed in extra_datasets, so build_manifest.py skips them and they cannot
leak into the manifest or the sampler. This tool reads them straight off the
disk. Isolation is checked here at load time: a probe name that appears in
overrides extra_datasets is a hard error.

Per-branch accuracies are reported alongside the fused number because they
say WHICH branch fails a type -- e.g. whether the dct spectral branch rescues
stylized reals that the semantic branches call fake.

History: one JSON line per (dataset, mode) appended to probe_history.jsonl;
the report's delta column compares against the most recent prior entry.
Determinism: the deploy view is row-seeded, so two runs on the same
checkpoints must produce byte-identical accuracies.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from branches import resolve, fusion_weights
from build_manifest import read_sample_metadata
from calibrate import (_sigmoid, class_balance_weights, fit_temperature,
                       sn34_from_probs)
from data import ViewConfig
from export import Ensemble, collect_branch_margins
from model import BranchModel

WEAK_BELOW, STRONG_ABOVE = 0.80, 0.92
LABELS = {"real": 0, "fake": 1, "synthetic": 1, "semisynthetic": 1}


def load_probes(path: Path) -> dict:
    cfg = yaml.safe_load(path.read_text()) or {}
    types = cfg.get("types") or {}
    if not types:
        raise SystemExit(f"{path} has no `types` section")
    for tname, spec in types.items():
        for p in spec.get("probes") or []:
            if not str(p.get("name", "")).startswith("probe-"):
                raise SystemExit(
                    f"probes.yaml type {tname!r}: probe {p.get('name')!r} "
                    f"must start with 'probe-' -- the prefix is the "
                    f"isolation mechanism")
            if p.get("label") not in LABELS:
                raise SystemExit(
                    f"probes.yaml type {tname!r}: probe {p.get('name')!r} "
                    f"needs label real|fake|semisynthetic")
    return types


def check_isolation(types: dict, overrides_path: Path) -> None:
    """A probe name in extra_datasets means probe data is being TRAINED ON."""
    if not overrides_path.exists():
        return
    ov = yaml.safe_load(overrides_path.read_text()) or {}
    extra = {e.get("name") for e in ov.get("extra_datasets") or []}
    for tname, spec in types.items():
        for p in spec.get("probes") or []:
            if p["name"] in extra:
                raise SystemExit(
                    f"ISOLATION VIOLATION: probe {p['name']!r} (type "
                    f"{tname!r}) is listed in {overrides_path.name}:"
                    f"extra_datasets -- the probe would be trained on and "
                    f"its number would measure memorisation, not coverage.")
        # Same-origin train data degrades the probe more quietly: warn only.
        # Compare full normalized URLs (prefix match), not hosts -- half the
        # candidate sources live on huggingface.co and a host-level check
        # would warn on every one of them.
        def _norm(u) -> str:
            u = str(u or "").lower().rstrip("/")
            return u.split("://", 1)[-1]
        probe_urls = {_norm(p.get("source_url"))
                      for p in spec.get("probes") or []
                      if p.get("source_url")}
        for t in spec.get("train") or []:
            tu = _norm(t.get("source_url"))
            if tu and any(tu == pu or tu.startswith(pu + "/")
                          or pu.startswith(tu + "/") for pu in probe_urls):
                print(f"[warn] type {tname!r}: train set {t.get('name')!r} "
                      f"shares its source ({tu}) with a probe -- the probe "
                      f"no longer measures cross-origin generalisation for "
                      f"this type")


def probe_frame(data_root: Path, name: str, label: int) -> pd.DataFrame | None:
    for base in (data_root / "datasets" / name, data_root / name):
        if (base / "sample_metadata.json").exists():
            meta = read_sample_metadata(base)
            rows = [{"path": str(base / "samples" / f), "label": label,
                     "dataset": name}
                    for f in sorted(meta)
                    if (base / "samples" / f).exists()]
            return pd.DataFrame(rows) if rows else None
    return None


def load_models(runs_dir: Path, names: list[str], device: str):
    specs = resolve(names)
    models, steps = [], {}
    for spec in specs:
        ck_path = runs_dir / spec.name / "best.pt"
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        m = BranchModel(spec, lora_r=ck["config"]["lora"]["r"],
                        lora_alpha=ck["config"]["lora"]["alpha"],
                        lora_dropout=0.0, gradient_checkpointing=False)
        m.load_state_dict(ck["state"], strict=True)
        m.eval()
        m.merge_and_strip()
        models.append(m.to(device))
        steps[spec.name] = int(ck.get("step", -1))
        print(f"[load] {spec.name} @ step {ck.get('step')} "
              f"(sel {ck.get('selection', float('nan')):.4f})")
    return specs, models, steps


def last_history(history_path: Path) -> dict[tuple[str, str], dict]:
    """Most recent prior ledger row per (dataset, mode)."""
    out: dict[tuple[str, str], dict] = {}
    if history_path.exists():
        for line in history_path.read_text().splitlines():
            try:
                r = json.loads(line)
                if isinstance(r, dict):
                    out[(r["dataset"], r["mode"])] = r
            except (json.JSONDecodeError, KeyError):
                continue
    return out


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--probes", default=str(here / "probes.yaml"))
    ap.add_argument("--data-root", required=True,
                    help="gasbench cache root holding the probe-* dirs")
    ap.add_argument("--runs-dir", default="runs/v7")
    ap.add_argument("--branches", default="dinov3,clip,convnext,eva,dct")
    ap.add_argument("--overrides", default=str(here / "overrides.yaml"),
                    help="checked for probe-name isolation violations")
    ap.add_argument("--modes", default="deploy",
                    help="comma list of deploy,robust")
    ap.add_argument("--image-size", type=int, default=384)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--tag", default="")
    ap.add_argument("--history", default=str(here / "probe_history.jsonl"))
    args = ap.parse_args()

    types = load_probes(Path(args.probes))
    # Check EVERY overrides variant next to the named one (-ship,
    # -experiment, future arms), not just the two spelled out before -- a
    # probe leaking into any arm's extra_datasets would train on it.
    # check_isolation returns early on missing files.
    ov = Path(args.overrides)
    for p in sorted({ov, *ov.parent.glob("overrides*.yaml")}):
        check_isolation(types, p)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    names = [n.strip() for n in args.branches.split(",") if n.strip()]
    specs, models, steps = load_models(Path(args.runs_dir), names, dev)
    w = np.asarray(fusion_weights(specs), dtype=float)
    ens = Ensemble(models, w.tolist()).to(dev).eval()
    view = ViewConfig(image_size=args.image_size)
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    prev = last_history(Path(args.history))
    data_root = Path(args.data_root)

    ts = datetime.now(timezone.utc).isoformat()
    ledger: list[dict] = []
    results: dict[str, list[dict]] = {}
    missing: list[str] = []

    for tname, spec in types.items():
        for probe in spec.get("probes") or []:
            name, label = probe["name"], LABELS[probe["label"]]
            sub = probe_frame(data_root, name, label)
            if sub is None or not len(sub):
                missing.append(name)
                continue
            for mode in modes:
                D, _, y, _ = collect_branch_margins(
                    ens, sub, view, dev, args.batch_size, args.workers, mode)
                d = (D * w[None, :]).sum(axis=1)
                correct = (d > 0).astype(int) == y
                row = {
                    "ts": ts, "tag": args.tag, "branches": names,
                    "steps": steps, "dataset": name, "type": tname,
                    "label": probe["label"], "n": int(len(y)), "mode": mode,
                    "acc": float(correct.mean()),
                    "acc_per_branch": {
                        s.name: float(((D[:, j] > 0).astype(int) == y).mean())
                        for j, s in enumerate(specs)},
                    "mean_p_fake": float(_sigmoid(d).mean()),
                }
                if len(set(y.tolist())) == 2:
                    cal = fit_temperature(y, d)
                    m = sn34_from_probs(
                        y.astype(float),
                        _sigmoid(d / cal["temperature"]),
                        class_balance_weights(y))
                    row["sn34"] = float(m["sn34"])
                ledger.append(row)
                results.setdefault(tname, []).append(row)

    if missing:
        print(f"\n[warn] {len(missing)} probe(s) not found under "
              f"{data_root}/datasets: {', '.join(missing)} -- ingest them "
              f"with ingest_external.py --role probe")
    if not ledger:
        raise SystemExit("no probes scored; nothing to report")

    # -- report -------------------------------------------------------------
    bn = [s.name for s in specs]
    print(f"\n{'type':18s}{'probe':30s}{'mode':8s}{'n':>6}{'acc':>8}"
          f"{'d-prev':>8}{'p_fake':>8}  " + " ".join(f"{b:>8s}" for b in bn))
    for tname in results:
        for r in results[tname]:
            p = prev.get((r["dataset"], r["mode"]))
            delta = f"{r['acc'] - p['acc']:+8.3f}" if p else f"{'--':>8s}"
            per_b = " ".join(f"{r['acc_per_branch'][b]:8.3f}" for b in bn)
            print(f"{tname:18s}{r['dataset']:30s}{r['mode']:8s}"
                  f"{r['n']:6,}{r['acc']:8.3f}{delta}"
                  f"{r['mean_p_fake']:8.3f}  {per_b}")

    print(f"\n-- suggested status (deploy acc: weak <{WEAK_BELOW}, "
          f"strong >={STRONG_ABOVE}) --")
    for tname, spec in types.items():
        rows = [r for r in results.get(tname, []) if r["mode"] == "deploy"]
        if not rows:
            print(f"  {tname:18s} untested (no probe data on disk)")
            continue
        worst = min(r["acc"] for r in rows)
        status = ("weak" if worst < WEAK_BELOW
                  else "strong" if worst >= STRONG_ABOVE else "ok")
        cur = spec.get("status", "untested")
        flag = "" if cur == status else f"   (probes.yaml says {cur!r})"
        print(f"  {tname:18s} {status}  worst acc {worst:.3f}{flag}")

    with Path(args.history).open("a") as f:
        for r in ledger:
            f.write(json.dumps(r) + "\n")
    print(f"\n[ledger] appended {len(ledger)} rows -> {args.history}")


if __name__ == "__main__":
    main()
