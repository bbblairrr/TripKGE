from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any


MODULE_LABELS = {
    "use_graph_expansion": "Graph Expansion",
    "use_community_retrieval": "Community Retrieval",
    "use_summary_layer": "Summary Layer",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export ablation charts/tables in LaTeX format.")
    parser.add_argument(
        "--ablation-csv",
        default="outputs/ablation_test_all703_llm_qwen/ablation_runs.csv",
        help="Path to ablation_runs.csv",
    )
    parser.add_argument(
        "--summary-json",
        default="outputs/ablation_test_all703_llm_qwen/ablation_summary.json",
        help="Optional ablation_summary.json used to recover the official best-run ordering",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/ablation_test_all703_llm_qwen/latex",
        help="Directory to write LaTeX outputs",
    )
    return parser.parse_args()


def _parse_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _parse_value(key: str, value: str) -> Any:
    if key.startswith("use_") or key == "llm_fallback_to_heuristic":
        return _parse_bool(value)
    if key in {"run_id", "llm_backend", "llm_model", "output_dir"}:
        return value
    if key in {"run_index", "run_total", "hops", "topk_vector", "topk_final", "n"}:
        return int(float(value))
    try:
        return float(value)
    except ValueError:
        return value


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        raw_rows = list(csv.DictReader(f))
    return [{key: _parse_value(key, value) for key, value in raw.items()} for raw in raw_rows]


def load_summary(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def avg(rows: list[dict[str, Any]], key: str) -> float:
    return mean(float(row[key]) for row in rows)


def fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def build_module_stats(rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, float]]]:
    metrics = [
        "constraint_satisfaction_rate",
        "average_route_distance_ratio",
        "ndcg_at_5",
        "personalization_proxy",
    ]
    stats: dict[str, dict[str, dict[str, float]]] = {}
    for module in MODULE_LABELS:
        on_rows = [row for row in rows if row[module]]
        off_rows = [row for row in rows if not row[module]]
        stats[module] = {
            "on": {metric: avg(on_rows, metric) for metric in metrics},
            "off": {metric: avg(off_rows, metric) for metric in metrics},
        }
    return stats


def build_module_extended_stats(rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, float]]]:
    metrics = [
        "answer_relevancy",
        "max_single_day_route_km",
        "recall_at_5",
        "recall_at_10",
    ]
    stats: dict[str, dict[str, dict[str, float]]] = {}
    for module in MODULE_LABELS:
        on_rows = [row for row in rows if row[module]]
        off_rows = [row for row in rows if not row[module]]
        stats[module] = {
            "on": {metric: avg(on_rows, metric) for metric in metrics},
            "off": {metric: avg(off_rows, metric) for metric in metrics},
        }
    return stats


def build_module_deltas(
    module_stats: dict[str, dict[str, dict[str, float]]],
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for module in MODULE_LABELS:
        on_stats = module_stats[module]["on"]
        off_stats = module_stats[module]["off"]
        out[module] = {
            "constraint_satisfaction_rate": on_stats["constraint_satisfaction_rate"] - off_stats["constraint_satisfaction_rate"],
            "average_route_distance_ratio_reduction": off_stats["average_route_distance_ratio"] - on_stats["average_route_distance_ratio"],
            "ndcg_at_5": on_stats["ndcg_at_5"] - off_stats["ndcg_at_5"],
            "personalization_proxy": on_stats["personalization_proxy"] - off_stats["personalization_proxy"],
        }
    return out


def build_topk_stats(rows: list[dict[str, Any]]) -> dict[int, dict[str, float]]:
    metrics = [
        "constraint_satisfaction_rate",
        "average_route_distance_ratio",
        "ndcg_at_5",
    ]
    out: dict[int, dict[str, float]] = {}
    for topk in sorted({int(row["topk_final"]) for row in rows}):
        bucket = [row for row in rows if int(row["topk_final"]) == topk]
        out[topk] = {metric: avg(bucket, metric) for metric in metrics}
    return out


def build_strong_subspace_stats(rows: list[dict[str, Any]]) -> dict[bool, dict[str, float]]:
    strong_rows = [
        row
        for row in rows
        if row["use_graph_expansion"] and row["use_community_retrieval"]
    ]
    metrics = [
        "constraint_satisfaction_rate",
        "average_route_distance_ratio",
        "ndcg_at_5",
    ]
    return {
        summary_on: {
            metric: avg([row for row in strong_rows if row["use_summary_layer"] == summary_on], metric)
            for metric in metrics
        }
        for summary_on in (False, True)
    }


def build_scatter_groups(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    return {
        "community_off_summary_off": [
            row for row in rows if not row["use_community_retrieval"] and not row["use_summary_layer"]
        ],
        "community_off_summary_on": [
            row for row in rows if not row["use_community_retrieval"] and row["use_summary_layer"]
        ],
        "community_on_summary_off": [
            row for row in rows if row["use_community_retrieval"] and not row["use_summary_layer"]
        ],
        "community_on_summary_on": [
            row for row in rows if row["use_community_retrieval"] and row["use_summary_layer"]
        ],
    }


def default_best_run(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return sorted(
        rows,
        key=lambda row: (
            -float(row["constraint_satisfaction_rate"]),
            float(row["average_route_distance_ratio"]),
            -float(row["ndcg_at_5"]),
        ),
    )[0]


def best_run(rows: list[dict[str, Any]], summary: dict[str, Any] | None) -> dict[str, Any]:
    if summary:
        run_id = summary.get("best_run", {}).get("run_id")
        if run_id:
            for row in rows:
                if row["run_id"] == run_id:
                    return row
    return default_best_run(rows)


def route_best_run(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return min(rows, key=lambda row: float(row["average_route_distance_ratio"]))


def top_runs(rows: list[dict[str, Any]], summary: dict[str, Any] | None, limit: int = 5) -> list[dict[str, Any]]:
    if summary and isinstance(summary.get("top5"), list):
        run_ids = [str(item.get("run_id")) for item in summary["top5"][:limit]]
        row_map = {row["run_id"]: row for row in rows}
        resolved = [row_map[run_id] for run_id in run_ids if run_id in row_map]
        if resolved:
            return resolved
    return sorted(
        rows,
        key=lambda row: (
            -float(row["constraint_satisfaction_rate"]),
            float(row["average_route_distance_ratio"]),
            -float(row["ndcg_at_5"]),
        ),
    )[:limit]


def coordinates(items: list[tuple[str, float]], digits: int = 4) -> str:
    return " ".join(f"({label},{fmt(value, digits)})" for label, value in items)


def scatter_coordinates(rows: list[dict[str, Any]]) -> str:
    return " ".join(
        f"({fmt(float(row['average_route_distance_ratio']))},{fmt(float(row['constraint_satisfaction_rate']))})"
        for row in rows
    )


def build_module_table_tex(module_stats: dict[str, dict[str, dict[str, float]]]) -> str:
    lines = [
        "% Requires: \\usepackage{booktabs}",
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{消融实验中各模块开关的平均效果对比}",
        "\\label{tab:ablation_module_effects}",
        "\\begin{tabular}{llcccc}",
        "\\toprule",
        "模块 & 设置 & Constraint Sat.$\\uparrow$ & Route Ratio$\\downarrow$ & nDCG@5$\\uparrow$ & Personalization \\\\",
        "\\midrule",
    ]
    for index, (module, label) in enumerate(MODULE_LABELS.items()):
        on_stats = module_stats[module]["on"]
        off_stats = module_stats[module]["off"]
        lines.append(
            f"{label} & On & {fmt(on_stats['constraint_satisfaction_rate'])} & "
            f"{fmt(on_stats['average_route_distance_ratio'])} & {fmt(on_stats['ndcg_at_5'])} & "
            f"{fmt(on_stats['personalization_proxy'])} \\\\"
        )
        lines.append(
            f"{label} & Off & {fmt(off_stats['constraint_satisfaction_rate'])} & "
            f"{fmt(off_stats['average_route_distance_ratio'])} & {fmt(off_stats['ndcg_at_5'])} & "
            f"{fmt(off_stats['personalization_proxy'])} \\\\"
        )
        lines.append("\\midrule" if index < len(MODULE_LABELS) - 1 else "\\bottomrule")
    lines.extend(["\\end{tabular}", "\\end{table}", ""])
    return "\n".join(lines)


def build_top_runs_table_tex(rows: list[dict[str, Any]]) -> str:
    lines = [
        "% Requires: \\usepackage{booktabs}",
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{消融实验中表现最优的五组配置}",
        "\\label{tab:ablation_top_configs}",
        "\\begin{tabular}{lccccccc}",
        "\\toprule",
        "Run ID & Graph & Community & Summary & TopK$_v$ & TopK$_f$ & Constraint Sat.$\\uparrow$ & Route Ratio$\\downarrow$ \\\\",
        "\\midrule",
    ]
    for index, row in enumerate(rows):
        run_id = str(row["run_id"]).replace("_", "\\_")
        lines.append(
            f"{run_id} & "
            f"{'On' if row['use_graph_expansion'] else 'Off'} & "
            f"{'On' if row['use_community_retrieval'] else 'Off'} & "
            f"{'On' if row['use_summary_layer'] else 'Off'} & "
            f"{int(row['topk_vector'])} & {int(row['topk_final'])} & "
            f"{fmt(float(row['constraint_satisfaction_rate']))} & "
            f"{fmt(float(row['average_route_distance_ratio']))} \\\\"
        )
        if index < len(rows) - 1:
            lines.append("\\midrule")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    return "\n".join(lines)


def build_module_extended_table_tex(module_stats: dict[str, dict[str, dict[str, float]]]) -> str:
    lines = [
        "% Requires: \\usepackage{booktabs}",
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{消融实验中各模块开关的扩展指标对比}",
        "\\label{tab:ablation_module_effects_extended}",
        "\\begin{tabular}{llcccc}",
        "\\toprule",
        "模块 & 设置 & Answer Rel. & Max Day Route$\\downarrow$ & Recall@5$\\uparrow$ & Recall@10$\\uparrow$ \\\\",
        "\\midrule",
    ]
    for index, (module, label) in enumerate(MODULE_LABELS.items()):
        on_stats = module_stats[module]["on"]
        off_stats = module_stats[module]["off"]
        lines.append(
            f"{label} & On & {fmt(on_stats['answer_relevancy'])} & "
            f"{fmt(on_stats['max_single_day_route_km'])} & {fmt(on_stats['recall_at_5'])} & "
            f"{fmt(on_stats['recall_at_10'])} \\\\"
        )
        lines.append(
            f"{label} & Off & {fmt(off_stats['answer_relevancy'])} & "
            f"{fmt(off_stats['max_single_day_route_km'])} & {fmt(off_stats['recall_at_5'])} & "
            f"{fmt(off_stats['recall_at_10'])} \\\\"
        )
        lines.append("\\midrule" if index < len(MODULE_LABELS) - 1 else "\\bottomrule")
    lines.extend(["\\end{tabular}", "\\end{table}", ""])
    return "\n".join(lines)


def build_figures_tex(
    module_stats: dict[str, dict[str, dict[str, float]]],
    module_deltas: dict[str, dict[str, float]],
    topk_stats: dict[int, dict[str, float]],
    strong_stats: dict[bool, dict[str, float]],
    scatter_groups: dict[str, list[dict[str, Any]]],
    best: dict[str, Any],
    route_best: dict[str, Any],
) -> str:
    graph_on = module_stats["use_graph_expansion"]["on"]
    graph_off = module_stats["use_graph_expansion"]["off"]
    community_on = module_stats["use_community_retrieval"]["on"]
    community_off = module_stats["use_community_retrieval"]["off"]
    summary_on = module_stats["use_summary_layer"]["on"]
    summary_off = module_stats["use_summary_layer"]["off"]

    graph_delta = module_deltas["use_graph_expansion"]
    community_delta = module_deltas["use_community_retrieval"]
    summary_delta = module_deltas["use_summary_layer"]

    topk15 = topk_stats[15]
    topk20 = topk_stats[20]
    strong_off = strong_stats[False]
    strong_on = strong_stats[True]

    return "\n".join(
        [
            "% This file is an input snippet for a larger LaTeX document.",
            "% It is not meant to be compiled standalone.",
            "% Requires in preamble:",
            "% \\usepackage{pgfplots}",
            "% \\usepgfplotslibrary{groupplots}",
            "% \\pgfplotsset{compat=1.18}",
            "",
            "\\begin{figure}[htbp]",
            "\\centering",
            "\\begin{tikzpicture}",
            "\\begin{groupplot}[",
            "group style={group size=3 by 1, horizontal sep=1.6cm},",
            "width=0.31\\textwidth,",
            "height=0.24\\textheight,",
            "ybar,",
            "bar width=7pt,",
            "symbolic x coords={Graph,Community,Summary},",
            "xtick=data,",
            "xticklabel style={rotate=20,anchor=east,font=\\small},",
            "ylabel style={font=\\small},",
            "title style={font=\\small},",
            "legend style={font=\\small, draw=none, at={(0.5,1.18)}, anchor=south, legend columns=2},",
            "nodes near coords,",
            "every node near coord/.append style={font=\\scriptsize, rotate=90, anchor=west},",
            "]",
            "\\nextgroupplot[title={Constraint Satisfaction Rate}, ylabel={Score}, ymin=0.78, ymax=0.90]",
            "\\addplot+[fill=blue!55] coordinates {"
            + coordinates(
                [
                    ("Graph", graph_on["constraint_satisfaction_rate"]),
                    ("Community", community_on["constraint_satisfaction_rate"]),
                    ("Summary", summary_on["constraint_satisfaction_rate"]),
                ]
            )
            + "};",
            "\\addplot+[fill=orange!75] coordinates {"
            + coordinates(
                [
                    ("Graph", graph_off["constraint_satisfaction_rate"]),
                    ("Community", community_off["constraint_satisfaction_rate"]),
                    ("Summary", summary_off["constraint_satisfaction_rate"]),
                ]
            )
            + "};",
            "\\legend{On,Off}",
            "\\nextgroupplot[title={Average Route Distance Ratio}, ylabel={Ratio}, ymin=0.0, ymax=1.95]",
            "\\addplot+[fill=blue!55] coordinates {"
            + coordinates(
                [
                    ("Graph", graph_on["average_route_distance_ratio"]),
                    ("Community", community_on["average_route_distance_ratio"]),
                    ("Summary", summary_on["average_route_distance_ratio"]),
                ]
            )
            + "};",
            "\\addplot+[fill=orange!75] coordinates {"
            + coordinates(
                [
                    ("Graph", graph_off["average_route_distance_ratio"]),
                    ("Community", community_off["average_route_distance_ratio"]),
                    ("Summary", summary_off["average_route_distance_ratio"]),
                ]
            )
            + "};",
            "\\nextgroupplot[title={nDCG@5}, ylabel={Score}, ymin=0.72, ymax=0.85]",
            "\\addplot+[fill=blue!55] coordinates {"
            + coordinates(
                [
                    ("Graph", graph_on["ndcg_at_5"]),
                    ("Community", community_on["ndcg_at_5"]),
                    ("Summary", summary_on["ndcg_at_5"]),
                ]
            )
            + "};",
            "\\addplot+[fill=orange!75] coordinates {"
            + coordinates(
                [
                    ("Graph", graph_off["ndcg_at_5"]),
                    ("Community", community_off["ndcg_at_5"]),
                    ("Summary", summary_off["ndcg_at_5"]),
                ]
            )
            + "};",
            "\\end{groupplot}",
            "\\end{tikzpicture}",
            "\\caption{各核心模块开关对系统平均性能的影响。所有结果均基于 24 组消融运行的均值统计。可以看出，Community Retrieval 对约束满足率和路线紧凑性的提升最为显著，而 Summary Layer 的主要收益体现在进一步压缩路线距离。}",
            "\\label{fig:ablation_module_effects}",
            "\\end{figure}",
            "",
            "\\begin{figure}[htbp]",
            "\\centering",
            "\\begin{tikzpicture}",
            "\\begin{groupplot}[",
            "group style={group size=3 by 1, horizontal sep=1.6cm},",
            "width=0.31\\textwidth,",
            "height=0.23\\textheight,",
            "ybar,",
            "bar width=10pt,",
            "symbolic x coords={Graph,Community,Summary},",
            "xtick=data,",
            "xticklabel style={rotate=20,anchor=east,font=\\small},",
            "ylabel style={font=\\small},",
            "title style={font=\\small},",
            "nodes near coords,",
            "every node near coord/.append style={font=\\scriptsize, rotate=90, anchor=west},",
            "]",
            "\\nextgroupplot[title={Constraint Gain}, ylabel={$\\Delta$ Score}, ymin=0.0, ymax=0.09]",
            "\\addplot+[fill=green!60!black] coordinates {"
            + coordinates(
                [
                    ("Graph", graph_delta["constraint_satisfaction_rate"]),
                    ("Community", community_delta["constraint_satisfaction_rate"]),
                    ("Summary", summary_delta["constraint_satisfaction_rate"]),
                ]
            )
            + "};",
            "\\nextgroupplot[title={Route Reduction Gain}, ylabel={$\\Delta$ Ratio}, ymin=0.0, ymax=1.35]",
            "\\addplot+[fill=green!60!black] coordinates {"
            + coordinates(
                [
                    ("Graph", graph_delta["average_route_distance_ratio_reduction"]),
                    ("Community", community_delta["average_route_distance_ratio_reduction"]),
                    ("Summary", summary_delta["average_route_distance_ratio_reduction"]),
                ]
            )
            + "};",
            "\\nextgroupplot[title={nDCG@5 Gain}, ylabel={$\\Delta$ Score}, ymin=-0.08, ymax=0.02]",
            "\\addplot+[fill=green!60!black] coordinates {"
            + coordinates(
                [
                    ("Graph", graph_delta["ndcg_at_5"]),
                    ("Community", community_delta["ndcg_at_5"]),
                    ("Summary", summary_delta["ndcg_at_5"]),
                ]
            )
            + "};",
            "\\end{groupplot}",
            "\\end{tikzpicture}",
            "\\caption{各核心模块的净增益对比。纵轴表示模块开启相对于关闭时的性能变化，其中 Route Reduction Gain 定义为路线比值的下降量，因此数值越大表示路线压缩效果越明显。该图更直接地揭示了 Community Retrieval 的主导贡献，以及 Summary Layer 的二阶增益特征。}",
            "\\label{fig:ablation_module_deltas}",
            "\\end{figure}",
            "",
            "\\begin{figure}[htbp]",
            "\\centering",
            "\\begin{tikzpicture}",
            "\\begin{groupplot}[",
            "group style={group size=3 by 1, horizontal sep=1.6cm},",
            "width=0.31\\textwidth,",
            "height=0.23\\textheight,",
            "ybar,",
            "bar width=10pt,",
            "xtick=data,",
            "xticklabel style={font=\\small},",
            "ylabel style={font=\\small},",
            "title style={font=\\small},",
            "nodes near coords,",
            "every node near coord/.append style={font=\\scriptsize, rotate=90, anchor=west},",
            "]",
            "\\nextgroupplot[title={Constraint Satisfaction Rate}, symbolic x coords={k15,k20}, ymin=0.82, ymax=0.83]",
            "\\addplot+[fill=teal!65] coordinates {"
            + coordinates(
                [
                    ("k15", topk15["constraint_satisfaction_rate"]),
                    ("k20", topk20["constraint_satisfaction_rate"]),
                ]
            )
            + "};",
            "\\nextgroupplot[title={Average Route Distance Ratio}, symbolic x coords={k15,k20}, ymin=1.24, ymax=1.38]",
            "\\addplot+[fill=teal!65] coordinates {"
            + coordinates(
                [
                    ("k15", topk15["average_route_distance_ratio"]),
                    ("k20", topk20["average_route_distance_ratio"]),
                ]
            )
            + "};",
            "\\nextgroupplot[title={nDCG@5}, symbolic x coords={k15,k20}, ymin=0.79, ymax=0.81]",
            "\\addplot+[fill=teal!65] coordinates {"
            + coordinates(
                [
                    ("k15", topk15["ndcg_at_5"]),
                    ("k20", topk20["ndcg_at_5"]),
                ]
            )
            + "};",
            "\\end{groupplot}",
            "\\end{tikzpicture}",
            "\\caption{不同 \\texttt{topk\\_final} 设置的平均效果对比。整体上，较小的最终候选集合更有利于约束满足和路线压缩，但提升幅度明显弱于核心结构模块。}",
            "\\label{fig:ablation_topk_effects}",
            "\\end{figure}",
            "",
            "\\begin{figure}[htbp]",
            "\\centering",
            "\\begin{tikzpicture}",
            "\\begin{groupplot}[",
            "group style={group size=3 by 1, horizontal sep=1.6cm},",
            "width=0.31\\textwidth,",
            "height=0.23\\textheight,",
            "ybar,",
            "bar width=10pt,",
            "xtick=data,",
            "xticklabel style={font=\\small},",
            "ylabel style={font=\\small},",
            "title style={font=\\small},",
            "nodes near coords,",
            "every node near coord/.append style={font=\\scriptsize, rotate=90, anchor=west},",
            "]",
            "\\nextgroupplot[title={Constraint Satisfaction Rate}, symbolic x coords={NoSummary,Summary}, ymin=0.87, ymax=0.88]",
            "\\addplot+[fill=violet!65] coordinates {"
            + coordinates(
                [
                    ("NoSummary", strong_off["constraint_satisfaction_rate"]),
                    ("Summary", strong_on["constraint_satisfaction_rate"]),
                ]
            )
            + "};",
            "\\nextgroupplot[title={Average Route Distance Ratio}, symbolic x coords={NoSummary,Summary}, ymin=0.47, ymax=0.51]",
            "\\addplot+[fill=violet!65] coordinates {"
            + coordinates(
                [
                    ("NoSummary", strong_off["average_route_distance_ratio"]),
                    ("Summary", strong_on["average_route_distance_ratio"]),
                ]
            )
            + "};",
            "\\nextgroupplot[title={nDCG@5}, symbolic x coords={NoSummary,Summary}, ymin=0.745, ymax=0.755]",
            "\\addplot+[fill=violet!65] coordinates {"
            + coordinates(
                [
                    ("NoSummary", strong_off["ndcg_at_5"]),
                    ("Summary", strong_on["ndcg_at_5"]),
                ]
            )
            + "};",
            "\\end{groupplot}",
            "\\end{tikzpicture}",
            "\\caption{在强配置子空间（即同时开启 Graph Expansion 与 Community Retrieval）中，Summary Layer 的边际影响。结果表明，Summary Layer 主要用于进一步降低路线比值，而对约束满足率和 nDCG 的影响较小。}",
            "\\label{fig:ablation_summary_strong_subspace}",
            "\\end{figure}",
            "",
            "\\begin{figure}[htbp]",
            "\\centering",
            "\\begin{tikzpicture}",
            "\\begin{axis}[",
            "width=0.78\\textwidth,",
            "height=0.42\\textheight,",
            "xlabel={Average Route Distance Ratio ($\\downarrow$)},",
            "ylabel={Constraint Satisfaction Rate ($\\uparrow$)},",
            "xmin=0.4, xmax=1.95,",
            "ymin=0.78, ymax=0.89,",
            "grid=both,",
            "grid style={gray!20},",
            "legend style={font=\\small, draw=none, at={(0.98,0.02)}, anchor=south east},",
            "tick label style={font=\\small},",
            "label style={font=\\small},",
            "]",
            "\\addplot+[only marks, mark=o, mark size=2.8pt, blue!70] coordinates {"
            + scatter_coordinates(scatter_groups["community_off_summary_off"])
            + "};",
            "\\addlegendentry{community=off, summary=off}",
            "\\addplot+[only marks, mark=square*, mark size=2.6pt, blue!45] coordinates {"
            + scatter_coordinates(scatter_groups["community_off_summary_on"])
            + "};",
            "\\addlegendentry{community=off, summary=on}",
            "\\addplot+[only marks, mark=triangle*, mark size=3pt, red!70] coordinates {"
            + scatter_coordinates(scatter_groups["community_on_summary_off"])
            + "};",
            "\\addlegendentry{community=on, summary=off}",
            "\\addplot+[only marks, mark=diamond*, mark size=2.8pt, red!45!black] coordinates {"
            + scatter_coordinates(scatter_groups["community_on_summary_on"])
            + "};",
            "\\addlegendentry{community=on, summary=on}",
            "\\addplot+[only marks, mark=star, mark size=4.4pt, black] coordinates {("
            + f"{fmt(float(best['average_route_distance_ratio']))},{fmt(float(best['constraint_satisfaction_rate']))}"
            + ")} node[above right, font=\\scriptsize] {"
            + best["run_id"].replace("_", "\\_")
            + "};",
            "\\addplot+[only marks, mark=star, mark size=4.4pt, orange!90!black] coordinates {("
            + f"{fmt(float(route_best['average_route_distance_ratio']))},{fmt(float(route_best['constraint_satisfaction_rate']))}"
            + ")} node[below left, font=\\scriptsize] {"
            + route_best["run_id"].replace("_", "\\_")
            + "};",
            "\\end{axis}",
            "\\end{tikzpicture}",
            "\\caption{24 组消融配置在“路线紧凑性-约束满足率”平面中的分布。可以看到，开启 Community Retrieval 的配置整体显著向左上方聚集，即同时实现更低的路线比值和更高的约束满足率；其中 "
            + best["run_id"].replace("_", "\\_")
            + " 是按目标指标排序的最佳配置，"
            + route_best["run_id"].replace("_", "\\_")
            + " 则对应最优路线紧凑性。}",
            "\\label{fig:ablation_scatter}",
            "\\end{figure}",
            "",
        ]
    )


def build_figures_standalone_tex() -> str:
    return "\n".join(
        [
            "\\documentclass[UTF8]{ctexart}",
            "\\usepackage[a4paper,margin=1in]{geometry}",
            "\\usepackage{booktabs}",
            "\\usepackage{pgfplots}",
            "\\usepgfplotslibrary{groupplots}",
            "\\pgfplotsset{compat=1.18}",
            "",
            "\\begin{document}",
            "\\input{ablation_figures.tex}",
            "\\end{document}",
            "",
        ]
    )


def build_ablation_report_standalone_tex() -> str:
    return "\n".join(
        [
            "\\documentclass[UTF8]{ctexart}",
            "\\usepackage[a4paper,margin=1in]{geometry}",
            "\\usepackage{booktabs}",
            "\\usepackage{pgfplots}",
            "\\usepgfplotslibrary{groupplots}",
            "\\pgfplotsset{compat=1.18}",
            "",
            "\\title{消融实验图表汇总}",
            "\\date{}",
            "",
            "\\begin{document}",
            "\\maketitle",
            "",
            "\\input{ablation_module_table.tex}",
            "\\input{ablation_module_extended_table.tex}",
            "\\input{ablation_top_runs_table.tex}",
            "\\input{ablation_figures.tex}",
            "\\end{document}",
            "",
        ]
    )


def main() -> None:
    args = parse_args()
    csv_path = Path(args.ablation_csv)
    summary_path = Path(args.summary_json)
    output_dir = Path(args.output_dir)

    rows = load_rows(csv_path)
    summary = load_summary(summary_path)

    module_stats = build_module_stats(rows)
    module_extended_stats = build_module_extended_stats(rows)
    module_deltas = build_module_deltas(module_stats)
    topk_stats = build_topk_stats(rows)
    strong_stats = build_strong_subspace_stats(rows)
    scatter_groups = build_scatter_groups(rows)
    best = best_run(rows, summary)
    route_best = route_best_run(rows)
    top5 = top_runs(rows, summary)

    write_text(output_dir / "ablation_module_table.tex", build_module_table_tex(module_stats))
    write_text(
        output_dir / "ablation_module_extended_table.tex",
        build_module_extended_table_tex(module_extended_stats),
    )
    write_text(output_dir / "ablation_top_runs_table.tex", build_top_runs_table_tex(top5))
    write_text(
        output_dir / "ablation_figures.tex",
        build_figures_tex(
            module_stats=module_stats,
            module_deltas=module_deltas,
            topk_stats=topk_stats,
            strong_stats=strong_stats,
            scatter_groups=scatter_groups,
            best=best,
            route_best=route_best,
        ),
    )
    write_text(output_dir / "ablation_figures_standalone.tex", build_figures_standalone_tex())
    write_text(output_dir / "ablation_report_standalone.tex", build_ablation_report_standalone_tex())


if __name__ == "__main__":
    main()
