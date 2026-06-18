"""Evaluation service — qrels-based metrics (requires qrels download on first run)."""

import json
import os
from typing import Optional

from evaluate import evaluate as run_evaluation


class EvaluationUnavailableError(Exception):
    """Raised when qrels cannot be loaded (e.g. no internet)."""


def run_eval(dataset: str, method: str, k: int = 10, limit: int = 50) -> dict:
    try:
        summary = run_evaluation(dataset, method, k=k, limit=limit)
    except Exception as e:
        err = str(e).lower()
        if any(x in err for x in ("timeout", "connection", "network", "download", "resolve")):
            raise EvaluationUnavailableError(
                "Qrels evaluation requires internet on first run to download queries/qrels. "
                "Run later when online: python evaluate.py msmarco --method bm25"
            ) from e
        raise

    os.makedirs("reports", exist_ok=True)
    report_path = f"reports/eval_{dataset}_{method}_k{k}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return {
        "dataset": summary["dataset"],
        "method": summary["method"],
        "k": summary["k"],
        "num_queries": summary["num_queries"],
        "avg_precision": summary["avg_P@k"],
        "avg_recall": summary["avg_R@k"],
        "map_score": summary["MAP"],
        "avg_ndcg": summary[f"avg_nDCG@{k}"],
        "report_path": report_path,
        "note": None,
    }
