"""
Wipe judge verdicts from saved collections so the next run re-judges them.

    python examples/reset_labels.py ./pisco_llama_router_ckpt [more run dirs ...]
    python examples/reset_labels.py ./pisco_router_ckpt_combined --dry-run

For each run dir, handles <run>/collection.pt (train) and <run>/eval/collection.pt.
Sets comp_correct / full_correct back to None on every row and saves. The
original is kept as collection.pt.bak. Generations and features are untouched,
so the rerun only re-judges — nothing is regenerated.

Why every row: a failed judge call is written as False, indistinguishable from
a real "wrong" verdict, so there is no way to reset only the failed ones.

Caveat it checks for: a run that COMPLETED saved its train collection after
skip_full_wrong filtering, so rows whose (possibly failed) full verdict was
False are already gone from the file. Re-judging cannot bring them back; the
script reports how many are missing.
"""
import argparse
import shutil
from collections import Counter
from pathlib import Path

import torch


def summarize(results):
    c = Counter((r.get("comp_correct"), r.get("full_correct")) for r in results)
    fmt = lambda v: {None: "None", True: "T", False: "F"}[v]
    return ", ".join(f"comp={fmt(k[0])}/full={fmt(k[1])}: {n}" for k, n in sorted(c.items(), key=str))


def reset(path: Path, dry_run: bool):
    if not path.exists():
        print(f"  {path}: not found, skipping")
        return
    data = torch.load(path, map_location="cpu", weights_only=False)
    results = data["results"]
    n_feat = data["features"].shape[0] if data["features"].dim() > 0 else 0
    print(f"  {path}")
    print(f"    rows={len(results)} features={n_feat} next_idx={data.get('next_idx')}")
    print(f"    before: {summarize(results)}")

    if n_feat != len(results):
        print("    !! features and results differ in length — file is inconsistent, not touching it")
        return

    next_idx = data.get("next_idx", len(results))
    missing = sorted(set(range(next_idx)) - {r["id"] for r in results})
    if missing:
        print(f"    !! {len(missing)} of {next_idx} rows were dropped at save time "
              f"(skip_full_wrong). Re-judging cannot restore them — regenerate "
              f"to get them back. First missing ids: {missing[:10]}")

    if dry_run:
        print("    dry run: nothing written")
        return

    shutil.copy2(path, path.with_suffix(".pt.bak"))
    for r in results:
        r["comp_correct"] = None
        r["full_correct"] = None
    torch.save(data, path)
    print(f"    after:  {summarize(results)}   (backup: {path.with_suffix('.pt.bak').name})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    args = ap.parse_args()
    for d in args.run_dirs:
        d = Path(d)
        print(d)
        reset(d / "collection.pt", args.dry_run)
        reset(d / "eval" / "collection.pt", args.dry_run)


if __name__ == "__main__":
    main()
