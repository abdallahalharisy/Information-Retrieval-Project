"""Evaluation service — qrels-based metrics (requires qrels download on first run)."""

import json
import os
from typing import Optional

from evaluate import (
    EMBEDDING_METHODS,
    compare_baseline_enhanced,
    evaluate as run_evaluation,
    save_json_report,
    write_markdown_report,
)


class EvaluationUnavailableError(Exception):
    """Raised when qrels cannot be loaded (e.g. no internet)."""


def _compact_comparisons(comparisons: list) -> list:
    """Keep the API response small; full per-query details stay in report files."""
    compacted = []
    for comparison in comparisons:
        item = dict(comparison)
        for mode in ("baseline", "enhanced"):
            if mode in item:
                summary = dict(item[mode])
                summary.pop("per_query", None)
                item[mode] = summary
        compacted.append(item)
    return compacted


def _is_network_error(error: Exception) -> bool:
    err = str(error).lower()
    return any(x in err for x in ("timeout", "connection", "network", "download", "resolve"))


def run_eval(dataset: str, method: str, k: int = 10, limit: Optional[int] = None,
             mode: str = "enhanced") -> dict:
    try:
        summary = run_evaluation(dataset, method, k=k, limit=limit, mode=mode)
    except Exception as e:
        if _is_network_error(e):
            raise EvaluationUnavailableError(
                "Qrels evaluation requires internet on first run to download queries/qrels. "
                "Run later when online: python evaluate.py msmarco --method bm25"
            ) from e
        raise

    os.makedirs("reports", exist_ok=True)
    report_path = f"reports/eval_{dataset}_{method}_{mode}_k{k}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return {
        "dataset": summary["dataset"],
        "method": summary["method"],
        "k": summary["k"],
        "num_queries": summary["num_queries"],
        "qrels_query_count": summary["qrels_query_count"],
        "attempted_query_count": summary["attempted_query_count"],
        "avg_precision": summary["avg_P@k"],
        "avg_recall": summary["avg_R@k"],
        "map_score": summary["MAP"],
        "avg_ndcg": summary[f"avg_nDCG@{k}"],
        "report_path": report_path,
        "note": None,
    }


def run_comparison(dataset: str, methods: list, k: int = 10, limit: Optional[int] = None,
                   include_embeddings: bool = False) -> dict:
    selected_methods = list(dict.fromkeys(methods))
    if include_embeddings:
        selected_methods = list(dict.fromkeys([*selected_methods, *EMBEDDING_METHODS]))

    try:
        report = compare_baseline_enhanced(
            dataset,
            selected_methods,
            k=k,
            limit=limit,
            include_embeddings=include_embeddings,
        )
    except Exception as e:
        if _is_network_error(e):
            raise EvaluationUnavailableError(
                "Qrels evaluation requires internet on first run to download queries/qrels. "
                "Run later when online: python evaluate.py msmarco --compare --all-methods"
            ) from e
        raise

    os.makedirs("reports", exist_ok=True)
    json_path = f"reports/eval_compare_{dataset}_k{k}.json"
    markdown_path = f"reports/eval_compare_{dataset}_k{k}.md"
    save_json_report(report, json_path)
    write_markdown_report(report, markdown_path)

    return {
        "dataset": report["dataset"],
        "k": report["k"],
        "limit": report["limit"],
        "qrels_query_count": report["qrels_query_count"],
        "attempted_query_count": report["attempted_query_count"],
        "methods": report["methods"],
        "json_report_path": json_path,
        "markdown_report_path": markdown_path,
        "comparisons": _compact_comparisons(report["comparisons"]),
    }
