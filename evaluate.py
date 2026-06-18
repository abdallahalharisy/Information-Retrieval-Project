# evaluate.py
import argparse
import contextlib
import json
import logging
import os
from collections import defaultdict
from datetime import datetime
from typing import Dict, Iterable, List, Set

import ir_datasets
import config
import ranking_engine
from ranking_engine import RankingEngine

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DEFAULT_METHODS = ['tfidf', 'bm25', 'index', 'serial', 'parallel', 'rrf']
EMBEDDING_METHODS = ['word2vec', 'bert']

IR_DATASETS = {
    'msmarco': {
        'ir_name': 'msmarco-document/dev',
        'data_file': 'processed_data_msmarco.json',
        'cache_prefix': 'msmarco',
    },
    'fever': {
        'ir_name': 'beir/fever',
        'data_file': 'processed_data_fever.json',
        'cache_prefix': 'fever',
    },
}


def load_engine_for_eval(dataset_key: str) -> RankingEngine:
    cfg = IR_DATASETS[dataset_key]
    data_file = cfg['data_file']
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"{data_file} not found. Run: python main.py {dataset_key} 210000")

    with open(data_file, encoding='utf-8') as f:
        documents = json.load(f)

    engine = RankingEngine()
    if not engine.load_cache(data_file, cache_prefix=cfg['cache_prefix']):
        logger.info("Building engine cache...")
        engine.fit(documents, fit_word2vec=False, fit_bert=False)
        engine.save_cache(data_file, cache_prefix=cfg['cache_prefix'])
    return engine


def load_qrels(dataset_key: str) -> tuple:
    """Load queries and qrels from ir_datasets."""
    ir_name = IR_DATASETS[dataset_key]['ir_name']
    dataset = ir_datasets.load(ir_name)

    queries = {q.query_id: q.text for q in dataset.queries_iter()}
    qrels: Dict[str, Dict[str, int]] = defaultdict(dict)
    for qrel in dataset.qrels_iter():
        qrels[qrel.query_id][qrel.doc_id] = qrel.relevance

    return queries, dict(qrels)


def precision_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    if k == 0:
        return 0.0
    hits = sum(1 for doc_id in retrieved[:k] if doc_id in relevant)
    return hits / k


def recall_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    if not relevant:
        return 0.0
    hits = sum(1 for doc_id in retrieved[:k] if doc_id in relevant)
    return hits / len(relevant)


def average_precision(retrieved: List[str], relevant: Set[str]) -> float:
    if not relevant:
        return 0.0
    score = 0.0
    hits = 0
    for i, doc_id in enumerate(retrieved, 1):
        if doc_id in relevant:
            hits += 1
            score += hits / i
    return score / len(relevant)


def ndcg_at_k(retrieved: List[str], relevant: Dict[str, int], k: int) -> float:
    import math

    def dcg(ranking, rels, n):
        value = 0.0
        for i, doc_id in enumerate(ranking[:n], 1):
            rel = rels.get(doc_id, 0)
            value += (2 ** rel - 1) / math.log2(i + 1)
        return value

    ideal = sorted(relevant.items(), key=lambda x: x[1], reverse=True)
    ideal_ids = [doc_id for doc_id, _ in ideal]
    dcg = dcg(retrieved, relevant, k)
    idcg = dcg(ideal_ids, relevant, k)
    return dcg / idcg if idcg > 0 else 0.0


@contextlib.contextmanager
def refinement_mode(mode: str):
    """Temporarily configure query-refinement features for baseline/enhanced evaluation."""
    original = dict(config.QUERY_REFINEMENT)
    if mode == 'baseline':
        config.QUERY_REFINEMENT.update({
            'expand_synonyms': False,
            'do_spell_check': False,
            'use_search_history': False,
        })
    elif mode == 'enhanced':
        config.QUERY_REFINEMENT.update({
            'expand_synonyms': True,
            'do_spell_check': True,
            'use_search_history': True,
        })
    else:
        raise ValueError(f"Unknown evaluation mode: {mode}")

    try:
        yield
    finally:
        config.QUERY_REFINEMENT.clear()
        config.QUERY_REFINEMENT.update(original)


@contextlib.contextmanager
def hybrid_embedding_scope(include_embeddings: bool):
    """Avoid embedding downloads during standard evaluation unless explicitly requested."""
    if include_embeddings:
        yield
        return

    original_weights = dict(config.FUSION_WEIGHTS)
    original_serial = list(config.SERIAL_FUSION_ORDER)
    original_rrf = list(config.RRF_FUSION_ORDER)

    lexical_weights = {'TF-IDF': 0.5, 'BM25': 0.5}
    lexical_order = ['TF-IDF', 'BM25']
    config.FUSION_WEIGHTS.clear()
    config.FUSION_WEIGHTS.update(lexical_weights)
    config.SERIAL_FUSION_ORDER[:] = lexical_order
    config.RRF_FUSION_ORDER[:] = lexical_order

    # ranking_engine imports these mutable objects; updating the config objects is enough,
    # but keep the module aliases explicit for readability.
    ranking_engine.FUSION_WEIGHTS.clear()
    ranking_engine.FUSION_WEIGHTS.update(lexical_weights)
    ranking_engine.SERIAL_FUSION_ORDER[:] = lexical_order
    ranking_engine.RRF_FUSION_ORDER[:] = lexical_order

    try:
        yield
    finally:
        config.FUSION_WEIGHTS.clear()
        config.FUSION_WEIGHTS.update(original_weights)
        config.SERIAL_FUSION_ORDER[:] = original_serial
        config.RRF_FUSION_ORDER[:] = original_rrf
        ranking_engine.FUSION_WEIGHTS.clear()
        ranking_engine.FUSION_WEIGHTS.update(original_weights)
        ranking_engine.SERIAL_FUSION_ORDER[:] = original_serial
        ranking_engine.RRF_FUSION_ORDER[:] = original_rrf


def evaluate(dataset_key: str, method: str, k: int = 10, limit: int = 50,
             mode: str = 'enhanced', include_embeddings: bool = False) -> dict:
    engine = load_engine_for_eval(dataset_key)
    queries, qrels = load_qrels(dataset_key)

    eval_queries = list(queries.items())[:limit]
    metrics = {'P@k': [], 'R@k': [], 'AP': [], f'nDCG@{k}': []}
    per_query = []

    for query_id, query_text in eval_queries:
        if query_id not in qrels:
            continue

        relevant = {d for d, r in qrels[query_id].items() if r > 0}
        if not relevant:
            continue

        # Only rank documents present in our processed corpus
        available_relevant = relevant & set(engine.doc_ids)
        if not available_relevant:
            continue

        with refinement_mode(mode), hybrid_embedding_scope(include_embeddings):
            results = engine.search(query_text, method=method, top_k=k)
        retrieved = [r['doc_id'] for r in results]

        p = precision_at_k(retrieved, available_relevant, k)
        r = recall_at_k(retrieved, available_relevant, k)
        ap = average_precision(retrieved, available_relevant)
        ndcg = ndcg_at_k(retrieved, qrels[query_id], k)

        metrics['P@k'].append(p)
        metrics['R@k'].append(r)
        metrics['AP'].append(ap)
        metrics[f'nDCG@{k}'].append(ndcg)
        per_query.append({
            'query_id': query_id,
            'query': query_text,
            'P@k': p, 'R@k': r, 'AP': ap, f'nDCG@{k}': ndcg,
            'relevant_in_corpus': len(available_relevant),
        })

    summary = {
        'dataset': dataset_key,
        'method': method,
        'mode': mode,
        'k': k,
        'num_queries': len(per_query),
        'MAP': sum(metrics['AP']) / len(metrics['AP']) if metrics['AP'] else 0.0,
        'avg_P@k': sum(metrics['P@k']) / len(metrics['P@k']) if metrics['P@k'] else 0.0,
        'avg_R@k': sum(metrics['R@k']) / len(metrics['R@k']) if metrics['R@k'] else 0.0,
        f'avg_nDCG@{k}': sum(metrics[f'nDCG@{k}']) / len(metrics[f'nDCG@{k}']) if metrics[f'nDCG@{k}'] else 0.0,
        'per_query': per_query,
    }
    return summary


def evaluate_methods(dataset_key: str, methods: Iterable[str], k: int = 10,
                     limit: int = 50, mode: str = 'enhanced',
                     include_embeddings: bool = False) -> List[dict]:
    results = []
    for method in methods:
        logger.info("Evaluating %s/%s (%s)", dataset_key, method, mode)
        try:
            results.append(evaluate(
                dataset_key,
                method,
                k=k,
                limit=limit,
                mode=mode,
                include_embeddings=include_embeddings,
            ))
        except Exception as e:
            logger.warning("Evaluation failed for %s/%s/%s: %s", dataset_key, method, mode, e)
            results.append({
                'dataset': dataset_key,
                'method': method,
                'mode': mode,
                'k': k,
                'num_queries': 0,
                'MAP': 0.0,
                'avg_P@k': 0.0,
                'avg_R@k': 0.0,
                f'avg_nDCG@{k}': 0.0,
                'error': str(e),
                'per_query': [],
            })
    return results


def compare_baseline_enhanced(dataset_key: str, methods: Iterable[str], k: int = 10,
                              limit: int = 50, include_embeddings: bool = False) -> dict:
    baseline = evaluate_methods(
        dataset_key,
        methods,
        k=k,
        limit=limit,
        mode='baseline',
        include_embeddings=include_embeddings,
    )
    enhanced = evaluate_methods(
        dataset_key,
        methods,
        k=k,
        limit=limit,
        mode='enhanced',
        include_embeddings=include_embeddings,
    )

    enhanced_by_method = {item['method']: item for item in enhanced}
    comparisons = []
    for base in baseline:
        enh = enhanced_by_method.get(base['method'])
        if not enh:
            continue
        comparisons.append({
            'method': base['method'],
            'baseline': base,
            'enhanced': enh,
            'delta_MAP': enh['MAP'] - base['MAP'],
            'delta_P@k': enh['avg_P@k'] - base['avg_P@k'],
            'delta_R@k': enh['avg_R@k'] - base['avg_R@k'],
            f'delta_nDCG@{k}': enh[f'avg_nDCG@{k}'] - base[f'avg_nDCG@{k}'],
        })

    return {
        'dataset': dataset_key,
        'k': k,
        'limit': limit,
        'created_at': datetime.now().isoformat(timespec='seconds'),
        'methods': list(methods),
        'include_embeddings': include_embeddings,
        'comparisons': comparisons,
    }


def save_json_report(report: dict, path: str) -> str:
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return path


def write_markdown_report(report: dict, path: str) -> str:
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    k = report['k']
    lines = [
        '# IR Evaluation Report',
        '',
        f"- Dataset: `{report['dataset']}`",
        f"- Queries limit: `{report['limit']}`",
        f"- K: `{k}`",
        f"- Created at: `{report['created_at']}`",
        '',
        '## Summary Table',
        '',
        f"| Method | Mode | Queries | MAP | Precision@{k} | Recall@{k} | nDCG@{k} |",
        '|---|---:|---:|---:|---:|---:|---:|',
    ]

    for comparison in report['comparisons']:
        for mode_name in ('baseline', 'enhanced'):
            item = comparison[mode_name]
            lines.append(
                f"| {comparison['method']} | {mode_name} | {item['num_queries']} | "
                f"{item['MAP']:.4f} | {item['avg_P@k']:.4f} | "
                f"{item['avg_R@k']:.4f} | {item[f'avg_nDCG@{k}']:.4f} |"
            )

    lines.extend([
        '',
        '## Enhanced vs Baseline Deltas',
        '',
        f"| Method | Delta MAP | Delta Precision@{k} | Delta Recall@{k} | Delta nDCG@{k} |",
        '|---|---:|---:|---:|---:|',
    ])

    for comparison in report['comparisons']:
        lines.append(
            f"| {comparison['method']} | {comparison['delta_MAP']:.4f} | "
            f"{comparison['delta_P@k']:.4f} | {comparison['delta_R@k']:.4f} | "
            f"{comparison[f'delta_nDCG@{k}']:.4f} |"
        )

    lines.extend([
        '',
        '## Interpretation Guide',
        '',
        '- `baseline`: query normalization without extra synonym expansion, spell correction, or history boosting.',
        '- `enhanced`: query refinement enabled using synonyms, spell correction, and configured history boost.',
        '- Positive deltas indicate that the additional features improved the metric.',
        '- Very low scores can happen if qrels reference documents outside the currently processed subset.',
        '',
        '## Notes',
        '',
        '- FEVER evaluation is intentionally deferred until the FEVER 210K preprocessing is completed.',
        '- BERT/Word2Vec are included only when `include_embeddings=true` or `--include-embeddings` is used.',
    ])

    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    return path


def main():
    parser = argparse.ArgumentParser(description='Evaluate IR engine with qrels')
    parser.add_argument('dataset', choices=['msmarco', 'fever'])
    parser.add_argument('--method', default='bm25',
                        choices=['tfidf', 'bm25', 'index', 'word2vec', 'bert', 'serial', 'parallel', 'rrf'])
    parser.add_argument('--methods', nargs='+', default=None,
                        choices=['tfidf', 'bm25', 'index', 'word2vec', 'bert', 'serial', 'parallel', 'rrf'])
    parser.add_argument('--all-methods', action='store_true',
                        help='Evaluate all non-embedding methods by default.')
    parser.add_argument('--include-embeddings', action='store_true',
                        help='Include Word2Vec and BERT methods; may require model downloads/build time.')
    parser.add_argument('--mode', choices=['baseline', 'enhanced'], default='enhanced')
    parser.add_argument('--compare', action='store_true',
                        help='Run baseline and enhanced evaluations and write a comparison report.')
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--limit', type=int, default=50, help='Max queries to evaluate')
    parser.add_argument('--output', default=None, help='Save JSON report to this path')
    args = parser.parse_args()

    methods = args.methods or ([args.method] if not args.all_methods else list(DEFAULT_METHODS))
    if args.include_embeddings:
        methods = list(dict.fromkeys([*methods, *EMBEDDING_METHODS]))

    os.makedirs('reports', exist_ok=True)

    if args.compare:
        logger.info("Comparing baseline vs enhanced for %s: %s", args.dataset, methods)
        report = compare_baseline_enhanced(
            args.dataset,
            methods,
            k=args.k,
            limit=args.limit,
            include_embeddings=args.include_embeddings,
        )
        json_out = args.output or f"reports/eval_compare_{args.dataset}_k{args.k}.json"
        md_out = json_out.rsplit('.', 1)[0] + '.md'
        save_json_report(report, json_out)
        write_markdown_report(report, md_out)
        print("\n=== Evaluation Comparison ===")
        print(f"Dataset: {args.dataset}")
        print(f"Methods: {', '.join(methods)}")
        print(f"JSON report: {json_out}")
        print(f"Markdown report: {md_out}")
        return

    if len(methods) > 1:
        summaries = evaluate_methods(
            args.dataset,
            methods,
            k=args.k,
            limit=args.limit,
            mode=args.mode,
            include_embeddings=args.include_embeddings,
        )
        out = args.output or f"reports/eval_{args.dataset}_{args.mode}_k{args.k}.json"
        save_json_report({
            'dataset': args.dataset,
            'mode': args.mode,
            'k': args.k,
            'limit': args.limit,
            'results': summaries,
        }, out)
        print(f"\nReport saved to {out}")
        return

    method = methods[0]
    logger.info(f"Evaluating {args.dataset} with {method} (mode={args.mode}, k={args.k}, limit={args.limit})")
    summary = evaluate(
        args.dataset,
        method,
        k=args.k,
        limit=args.limit,
        mode=args.mode,
        include_embeddings=args.include_embeddings,
    )

    print("\n=== Evaluation Summary ===")
    print(f"Dataset:     {summary['dataset']}")
    print(f"Method:      {summary['method']}")
    print(f"Mode:        {summary['mode']}")
    print(f"Queries:     {summary['num_queries']}")
    print(f"P@{args.k}:       {summary['avg_P@k']:.4f}")
    print(f"R@{args.k}:       {summary['avg_R@k']:.4f}")
    print(f"MAP:         {summary['MAP']:.4f}")
    print(f"nDCG@{args.k}:    {summary[f'avg_nDCG@{args.k}']:.4f}")

    out = args.output or f"reports/eval_{args.dataset}_{method}_{args.mode}_k{args.k}.json"
    save_json_report(summary, out)
    print(f"\nReport saved to {out}")


if __name__ == '__main__':
    main()
