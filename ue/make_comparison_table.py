#!/usr/bin/env python3
"""
Generate a LaTeX Combined-split comparison table containing BOTH:
  - threshold-free test AUC from eval_ue_auc.py
  - routed accuracy metrics from eval_ue_acc.py
  - token savings S_tok from eval_ue_acc.py

Manifest format
---------------
One JSON object per baseline result:

{
  "auc_path": "./ue/results/pisco_combined_solar_focus_auc.json",
  "acc_path": "./ue/results/pisco_combined_solar_focus_acc.json"
}

Optional metadata overrides are supported:
{
  "auc_path": "...",
  "acc_path": "...",
  "compressor": "PISCO",
  "model": "solar-7b",
  "method": "Focus"
}

The script verifies that AUC and ACC files refer to the same
compressor/model/method whenever metadata are available.

OverflowGuard reference CSV format
----------------------------------
compressor,model,auc,acc_c,acc_r,delta_acc,r_c,s_tok

All values in the reference CSV are percentages, e.g. auc=72.0.

UE evaluator JSON values are expected as fractions, e.g. auc=0.598.
They are converted automatically to percentages.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import OrderedDict, defaultdict
from pathlib import Path


METHOD_ORDER = {
    "OverflowGuard": 0,
    "MetaUE": 1,
    "Perplexity": 2,
    "Focus": 3,
    "RAUQ": 4,
}


def esc(x):
    s = str(x)
    for a, b in [
        ("&", r"\&"),
        ("%", r"\%"),
        ("_", r"\_"),
        ("#", r"\#"),
    ]:
        s = s.replace(a, b)
    return s


def pct(v):
    """Accept either unit fractions or already-percent values."""
    if v is None:
        return None
    v = float(v)
    return 100.0 * v if abs(v) <= 1.5 else v


def fmt(v):
    return "--" if v is None else f"{float(v):.1f}"


def norm_method(x):
    if x is None:
        return None
    s = str(x)
    aliases = {
        "perplexity": "Perplexity",
        "focus": "Focus",
        "rauq": "RAUQ",
        "metaue": "MetaUE",
        "overflowguard": "OverflowGuard",
    }
    return aliases.get(s.lower(), s)


def load_reference(path):
    rows = OrderedDict()
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            key = (r["compressor"], r["model"])
            rows[key] = {
                "compressor": r["compressor"],
                "model": r["model"],
                "method": "OverflowGuard",
                "auc": float(r["auc"]),
                "acc_c": float(r["acc_c"]),
                "acc_r": float(r["acc_r"]),
                "delta_acc": float(r["delta_acc"]),
                "r_c": float(r["r_c"]),
                "s_tok": float(r["s_tok"]),
                "coverage_auc": 1.0,
                "coverage_acc": 1.0,
            }
    return rows


def auc_metadata(d):
    return {
        "dataset": d.get("dataset"),
        "compressor": d.get("compressor"),
        "model": d.get("model"),
        "method": norm_method(d.get("method") or d.get("field")),
    }


def acc_metadata(d):
    md = d.get("metadata", {})
    return {
        "dataset": md.get("dataset") or d.get("dataset"),
        "compressor": md.get("compressor") or d.get("compressor"),
        "model": md.get("model") or d.get("model"),
        "method": norm_method(
            md.get("router")
            or d.get("method")
            or md.get("field")
            or d.get("field")
        ),
    }


def check_pair(a_meta, c_meta, auc_path, acc_path):
    """Fail loudly on mismatched known metadata."""
    for key in ("dataset", "compressor", "model", "method"):
        av = a_meta.get(key)
        cv = c_meta.get(key)
        if av is not None and cv is not None and str(av).lower() != str(cv).lower():
            raise ValueError(
                f"AUC/ACC metadata mismatch for {key}: "
                f"{auc_path} says {av!r}, {acc_path} says {cv!r}"
            )


def parse_pair(item):
    auc_path = Path(item["auc_path"])
    acc_path = Path(item["acc_path"])

    a = json.loads(auc_path.read_text(encoding="utf-8"))
    c = json.loads(acc_path.read_text(encoding="utf-8"))

    am = auc_metadata(a)
    cm = acc_metadata(c)
    check_pair(am, cm, auc_path, acc_path)

    compressor = (
        item.get("compressor")
        or am["compressor"]
        or cm["compressor"]
    )
    model = item.get("model") or am["model"] or cm["model"]
    method = norm_method(
        item.get("method") or am["method"] or cm["method"]
    )
    dataset = item.get("dataset") or am["dataset"] or cm["dataset"]

    if not compressor or not model or not method:
        raise ValueError(
            f"Could not infer compressor/model/method from pair:\n"
            f"  auc={auc_path}\n  acc={acc_path}\n"
            "Add explicit metadata to the manifest row."
        )

    ev = c.get("eval", {})

    # Use the dedicated AUC evaluation file for the AUC column.
    auc = a.get("auc")
    if auc is None:
        auc = ev.get("auc")

    acc_c = ev.get("acc_always_compressed")
    acc_r = ev.get("accuracy")
    delta = ev.get("delta_accuracy")
    rc = ev.get("pct_compressed")
    stok = ev.get("token_savings")

    # Optional overrides from manifest.
    auc = item.get("auc", auc)
    acc_c = item.get("acc_c", acc_c)
    acc_r = item.get("acc_r", acc_r)
    delta = item.get("delta_acc", delta)
    rc = item.get("r_c", rc)
    stok = item.get("s_tok", stok)

    auc = pct(auc)
    acc_c = pct(acc_c)
    acc_r = pct(acc_r)
    delta = pct(delta)
    rc = pct(rc)
    stok = pct(stok)

    if delta is None and acc_c is not None and acc_r is not None:
        delta = acc_r - acc_c

    return {
        "dataset": dataset,
        "compressor": compressor,
        "model": model,
        "method": method,
        "auc": auc,
        "acc_c": acc_c,
        "acc_r": acc_r,
        "delta_acc": delta,
        "r_c": rc,
        "s_tok": stok,
        "coverage_auc": a.get("coverage_of_collection"),
        "coverage_acc": ev.get("coverage"),
        "auc_path": str(auc_path),
        "acc_path": str(acc_path),
    }


def load_manifest(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            item = json.loads(line)

            if "auc_path" not in item or "acc_path" not in item:
                raise ValueError(
                    f"{path}:{ln}: each row needs both 'auc_path' and 'acc_path'"
                )

            out.append(parse_pair(item))
    return out


def acc_cell(row):
    if row["acc_c"] is None or row["acc_r"] is None:
        return "--"
    return (
        rf"{row['acc_c']:.1f} $\rightarrow$ "
        rf"\textbf{{{row['acc_r']:.1f}}}"
    )


def delta_cell(row):
    if row["delta_acc"] is None:
        return "--"
    return rf"${row['delta_acc']:+.1f}$"


def coverage_cell(row):
    ca = row.get("coverage_auc")
    cc = row.get("coverage_acc")
    vals = [x for x in (ca, cc) if x is not None]
    if not vals:
        return "--"
    # Show the more conservative coverage if they differ.
    return f"{100.0 * min(vals):.1f}"


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--reference",
        required=True,
        help="CSV with recorded OverflowGuard Combined results.",
    )
    ap.add_argument(
        "--ue-manifest",
        required=True,
        help="JSONL with auc_path + acc_path for each UE baseline.",
    )
    ap.add_argument("--output", required=True)

    ap.add_argument(
        "--expected-dataset",
        default="Combined",
    )
    ap.add_argument(
        "--only-with-ue",
        action="store_true",
        help="Only show compressor/model configs having at least one UE result.",
    )
    ap.add_argument(
        "--include-routing-costs",
        action="store_true",
        help="Also add the R_c (percent compressed) column. S_tok is always shown.",
    )
    ap.add_argument(
        "--include-coverage",
        action="store_true",
        help="Add minimum AUC/ACC coverage column.",
    )

    args = ap.parse_args()

    refs = load_reference(args.reference)
    ue_rows = load_manifest(args.ue_manifest)


    ue_by_key = defaultdict(list)

    for r in ue_rows:
        key = (r["compressor"], r["model"])

        if key not in refs:
            print(
                f"WARNING: no OverflowGuard reference for {key}; ignoring "
                f"{r['method']}."
            )
            continue

        if r["dataset"] and args.expected_dataset:
            if str(r["dataset"]).lower() != str(args.expected_dataset).lower():
                print(
                    f"WARNING: {r['method']} / {key} declares "
                    f"dataset={r['dataset']!r}, expected "
                    f"{args.expected_dataset!r}."
                )

        ue_by_key[key].append(r)

    for key, rs in ue_by_key.items():
        rs.sort(
            key=lambda r: (
                METHOD_ORDER.get(r["method"], 99),
                r["method"],
            )
        )

    keys = [
        k for k in refs
        if not args.only_with_ue or k in ue_by_key
    ]

    # total physical rows under each compressor, needed for multirow
    compressor_row_counts = OrderedDict()
    for key in keys:
        n = 1 + len(ue_by_key.get(key, []))
        compressor_row_counts[key[0]] = (
            compressor_row_counts.get(key[0], 0) + n
        )

    # S_tok is always shown. R_c remains optional.
    colspec = "lllcccc"
    if args.include_routing_costs:
        colspec += "c"
    if args.include_coverage:
        colspec += "c"

    header = [
        r"\textbf{Compressor}",
        r"\textbf{Model}",
        r"\textbf{Method}",
        r"\textbf{AUC}",
        r"$\mathbf{Acc_c \rightarrow Acc_r}$",
        r"$\mathbf{\Delta Acc}$",
    ]
    if args.include_routing_costs:
        header += [r"$\mathbf{R_c}$ (\%)"]
    header += [r"$\mathbf{S_{\mathrm{tok}}}$ (\%)"]
    if args.include_coverage:
        header += [r"\textbf{Coverage}"]

    lines = [
        r"\begin{table}[htb!]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\resizebox{\columnwidth}{!}{%",
        rf"\begin{{tabular}}{{{colspec}}}",
        r"\toprule",
        " & ".join(header) + r" \\",
        r"\midrule",
    ]

    seen_compressor = set()
    previous_compressor = None

    for ki, key in enumerate(keys):
        compressor, model = key
        block = [refs[key]] + ue_by_key.get(key, [])

        if (
            previous_compressor is not None
            and compressor != previous_compressor
        ):
            lines.append(r"\midrule")

        for ri, row in enumerate(block):
            if compressor not in seen_compressor:
                ccell = (
                    rf"\multirow{{{compressor_row_counts[compressor]}}}"
                    rf"{{*}}{{{esc(compressor)}}}"
                )
                seen_compressor.add(compressor)
            else:
                ccell = ""

            if ri == 0:
                if len(block) > 1:
                    mcell = (
                        rf"\multirow{{{len(block)}}}{{*}}{{{esc(model)}}}"
                    )
                else:
                    mcell = esc(model)
            else:
                mcell = ""

            method = row["method"]
            if method == "OverflowGuard":
                method_cell = r"\textbf{OverflowGuard}"
            else:
                method_cell = esc(method)

            cells = [
                ccell,
                mcell,
                method_cell,
                fmt(row["auc"]),
                acc_cell(row),
                delta_cell(row),
            ]

            if args.include_routing_costs:
                cells += [fmt(row.get("r_c"))]

            # token_savings is stored by eval_ue_acc.py as a fraction; parse_pair()
            # converts it to percent via pct(), so 0.6935 -> 69.4 here.
            cells += [fmt(row.get("s_tok"))]

            if args.include_coverage:
                cells += [coverage_cell(row)]

            lines.append(" & ".join(cells) + r" \\")

        next_key = keys[ki + 1] if ki + 1 < len(keys) else None
        if next_key is not None and next_key[0] == compressor:
            end_col = len(header)
            lines.append(rf"\cmidrule(lr){{2-{end_col}}}")

        previous_compressor = compressor

    lines += [
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\caption{",
        r"Comparison of OverflowGuard with uncertainty-based overflow detection",
        r"baselines on the Combined split, matched by compressor--generator",
        r"configuration. AUC is threshold-free test AUROC. $Acc_c$ is the",
        r"accuracy of always using compressed context and $Acc_r$ is accuracy",
        r"after routing with the uncertainty baseline. $\Delta Acc$ is the",
        r"absolute routed-accuracy gain in percentage points. $S_{\mathrm{tok}}$",
        r"is the overall token saving relative to always using full context,",
        r"reported as a percentage.",
        r"}",
        r"\label{tab:combined-uncertainty-comparison}",
        r"\end{table}",
    ]

    tex = "\n".join(lines) + "\n"
    Path(args.output).write_text(tex, encoding="utf-8")
    print(tex)


if __name__ == "__main__":
    main()
