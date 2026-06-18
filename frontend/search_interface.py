"""
Streamlit frontend — communicates with API Gateway (SOA).
Start gateway first: python run_gateway.py
"""

import logging
import os
import sys

import pandas as pd
import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from frontend.api_client import GatewayClient
from shared.datasets import DATASETS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GATEWAY_URL = os.environ.get("IR_GATEWAY_URL", "http://127.0.0.1:8000")

DISPLAY_DATASETS = {
    "MSMARCO-Document": "msmarco",
    "BEIR/Fever": "fever",
}


def _make_snippet(document_text, max_len=220):
    snippet = document_text.replace("\n", " ").strip()
    if len(snippet) > max_len:
        snippet = snippet[:max_len].rsplit(" ", 1)[0] + "..."
    return snippet


def format_results(results):
    formatted = []
    for i, result in enumerate(results, 1):
        row = {
            "Rank": i,
            "Document ID": result["doc_id"],
            "Score": f"{result['score']:.6f}",
            "Snippet": _make_snippet(result.get("document_text", "")),
        }
        exp = result.get("explanation")
        if exp:
            row["Fusion Type"] = exp.get("fusion_type", "N/A")
            if isinstance(exp.get("methods"), dict):
                row["Method Scores"] = ", ".join(
                    f"{k}:{v:.4f}" for k, v in exp["methods"].items()
                )
        formatted.append(row)
    return formatted


def _render_results_table(df, query_key):
    try:
        event = st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
            key=f"results_table_{query_key}",
        )
        if event.selection and event.selection.rows:
            return event.selection.rows[0]
    except TypeError:
        st.dataframe(df, use_container_width=True, hide_index=True)
        return st.selectbox(
            "Select a result",
            options=list(range(len(df))),
            format_func=lambda i: f"#{i + 1} — {df.iloc[i]['Snippet']}",
            key=f"results_select_{query_key}",
        )
    return st.session_state.get(f"selected_row_{query_key}", 0)


def _render_full_document(result):
    st.subheader("Full Document")
    st.write(f"**Rank:** {result.get('rank', '—')}")
    st.write(f"**Document ID:** {result['doc_id']}")
    st.write(f"**Score:** {result['score']:.6f}")
    st.write(f"**Method:** {result.get('method', 'N/A')}")
    exp = result.get("explanation")
    if exp:
        st.write(f"**Fusion Type:** {exp.get('fusion_type', 'N/A')}")
        if "pipeline" in exp:
            st.write(f"**Pipeline:** {' → '.join(exp['pipeline'])}")
        if isinstance(exp.get("methods"), dict):
            st.write("**Method Scores:**")
            for method, score in exp["methods"].items():
                st.write(f"- {method}: {score:.6f}")
    st.text_area("Document", value=result.get("document_text", ""), height=300, disabled=True)


@st.cache_resource
def get_client():
    return GatewayClient(GATEWAY_URL)


def main():
    st.set_page_config(page_title="IR Search Engine (SOA)", layout="wide")
    st.title("Information Retrieval Search Engine")
    st.caption("Service-Oriented Architecture — Frontend → API Gateway → Services")
    st.markdown("---")

    client = get_client()

    try:
        health = client.health()
        st.sidebar.success(f"Gateway: {health.get('status', 'ok')}")
    except Exception as e:
        st.error(f"Cannot reach API Gateway at {GATEWAY_URL}")
        st.code("python run_gateway.py", language="bash")
        st.stop()

    if "search_history" not in st.session_state:
        st.session_state.search_history = []

    st.sidebar.header("Configuration")

    display_name = st.sidebar.selectbox("Dataset", list(DISPLAY_DATASETS.keys()), index=0)
    dataset_key = DISPLAY_DATASETS[display_name]
    cfg = DATASETS[dataset_key]
    st.sidebar.caption(cfg["description"])

    try:
        status = client.index_status(dataset_key)
        st.sidebar.write(f"**Documents:** {status.get('num_documents', 0):,}")
    except Exception:
        st.sidebar.warning("Index not ready")

    ranking_method = st.sidebar.selectbox(
        "Ranking Method",
        ["tfidf", "bm25", "index", "word2vec", "bert", "serial", "parallel", "rrf"],
        index=6,
    )
    top_k = st.sidebar.slider("Results", 1, 50, 10)

    bm25_k1, bm25_b = 1.5, 0.75
    if ranking_method in ("bm25", "serial", "parallel", "rrf"):
        st.sidebar.subheader("BM25 Parameters")
        bm25_k1 = st.sidebar.slider("k1", 0.5, 3.0, 1.5, 0.1)
        bm25_b = st.sidebar.slider("b", 0.0, 1.0, 0.75, 0.05)

    st.header("Search")
    query = st.text_input("Query", placeholder="e.g., machine learning neural networks")

    if query and len(query.strip()) >= 2:
        try:
            suggestions = client.suggest(query.strip(), dataset_key)
            if suggestions:
                st.caption("Suggestions: " + " · ".join(suggestions))
        except Exception:
            pass

    if st.session_state.search_history:
        st.caption("Recent: " + " | ".join(st.session_state.search_history[-5:]))

    if query:
        try:
            with st.spinner(f"Searching via Gateway ({ranking_method})..."):
                response = client.search(
                    query=query,
                    dataset=dataset_key,
                    method=ranking_method,
                    top_k=top_k,
                    query_history=st.session_state.search_history,
                    bm25_k1=bm25_k1,
                    bm25_b=bm25_b,
                )

            if query.strip() not in st.session_state.search_history:
                st.session_state.search_history.append(query.strip())
                st.session_state.search_history = st.session_state.search_history[-20:]

            results = response.get("results", [])
            st.caption(f"Refined query: `{response.get('refined_query', query)}`")

            if results:
                st.subheader(f"Results ({ranking_method.upper()})")
                df = pd.DataFrame(format_results(results))
                qkey = f"{dataset_key}_{ranking_method}_{query.strip().lower()}"
                idx = max(0, min(_render_results_table(df, qkey), len(results) - 1))
                selected = dict(results[idx])
                selected["rank"] = idx + 1
                selected["method"] = response.get("method", ranking_method)
                _render_full_document(selected)
            else:
                st.warning("No results.")
        except Exception as e:
            st.error(f"Search failed: {e}")
            logger.exception("Search error")

    if st.sidebar.button("Clear history"):
        st.session_state.search_history = []

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"[API Docs]({GATEWAY_URL}/docs)")


if __name__ == "__main__":
    main()
