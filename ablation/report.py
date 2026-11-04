"""Combine per-run results.json files into summary.csv and ablation_table.tex."""
import json
import os
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

METHOD_NAMES = {
    "bc": "BC", "dagger": "DAgger", "gail": "GAIL", "bco": "BCO", "sqil": "SQIL", "iupe": "IUPE",
}
ENCODER_NAMES = {"cnn": "CNN", "resnet18": "ResNet-18"}
TABLE_SPLITS = (("train", "Train"), ("eval", "Valid."), ("test", "Test"))
SUMMARY_SPLITS = ("train", "eval", "test", "new_s0", "new_s0_and_g")


def run_dir(out: str, method: str, encoder: str, seed: int) -> str:
    return os.path.join(out, method, encoder, f"seed_{seed}")


def load_results(out: str, method: str, encoder: str, seed: int) -> Optional[Dict[str, Any]]:
    path = os.path.join(run_dir(out, method, encoder, seed), "results.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as _file:
        return json.load(_file)


def write_summary(out: str, methods: Sequence[str], encoders: Sequence[str], seeds: Sequence[int]) -> str:
    rows = []
    for method in methods:
        for encoder in encoders:
            for seed in seeds:
                results = load_results(out, method, encoder, seed)
                row = {"method": method, "encoder": encoder, "seed": seed,
                       "status": "missing" if results is None else "done"}
                if results is not None:
                    for split in SUMMARY_SPLITS:
                        values = results["splits"].get(split)
                        if values is None:
                            continue
                        for key in ("aer", "aer_std", "aer_raw", "aer_raw_std", "sr", "n_labyrinths"):
                            row[f"{split}_{key}"] = values[key]
                    for split, summary in results.get("encoding_analysis", {}).items():
                        row[f"{split}_n_states"] = summary["n_states"]
                        row[f"{split}_policy_action_optimal"] = summary["policy_action_optimal"]
                        for metric in ("cosine", "l2"):
                            for flag, value in summary[metric].items():
                                row[f"{split}_{metric}_{flag}"] = value
                    row["wall_time_s"] = results.get("timing", {}).get("total_s")
                rows.append(row)
    path = os.path.join(out, "summary.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _pooled(out: str, method: str, encoder: str, seeds: Sequence[int], split: str):
    """Pool per-labyrinth normalised returns and successes over seeds."""
    returns, successes = [], []
    for seed in seeds:
        results = load_results(out, method, encoder, seed)
        if results is None or split not in results["splits"]:
            continue
        returns += results["splits"][split]["returns_normalized"]
        successes += results["splits"][split]["successes"]
    if not returns:
        return None
    return float(np.mean(returns)), float(np.std(returns)), 100.0 * float(np.mean(successes))


def _cell(text: str, bold: bool) -> str:
    return f"$\\mathbf{{{text}}}$" if bold else f"${text}$"


def write_table(out: str, methods: Sequence[str], encoders: Sequence[str], seeds: Sequence[int]) -> str:
    lines: List[str] = [
        "\\begin{table}[ht]",
        "\\centering",
        "\\scriptsize",
        "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}ll" + "cc" * len(TABLE_SPLITS) + "}",
        "\\toprule",
        "Method & Encoder & " + " & ".join(
            f"{label} AER & {label} SR" for _, label in TABLE_SPLITS
        ) + " \\\\",
        "\\midrule",
    ]
    for m_idx, method in enumerate(methods):
        values = {
            encoder: {split: _pooled(out, method, encoder, seeds, split) for split, _ in TABLE_SPLITS}
            for encoder in encoders
        }
        best = {}
        for split, _ in TABLE_SPLITS:
            present = [values[e][split] for e in encoders if values[e][split] is not None]
            if len(present) > 1:
                best[(split, "aer")] = max(round(v[0], 2) for v in present)
                best[(split, "sr")] = max(round(v[2], 2) for v in present)
        for e_idx, encoder in enumerate(encoders):
            cells = [METHOD_NAMES.get(method, method) if e_idx == 0 else "",
                     ENCODER_NAMES.get(encoder, encoder)]
            for split, _ in TABLE_SPLITS:
                value = values[encoder][split]
                if value is None:
                    cells += ["--", "--"]
                    continue
                mean, std, sr = value
                cells.append(_cell(f"{mean:.2f} \\pm {std:.2f}", round(mean, 2) == best.get((split, "aer"))))
                cells.append(_cell(f"{sr:.2f}\\%", round(sr, 2) == best.get((split, "sr"))))
            lines.append(" & ".join(cells) + " \\\\")
        if m_idx < len(methods) - 1:
            lines.append("\\midrule")
    lines += ["\\bottomrule", "\\end{tabular*}", "\\end{table}", ""]
    path = os.path.join(out, "ablation_table.tex")
    with open(path, "w", encoding="utf-8") as _file:
        _file.write("\n".join(lines))
    return path
