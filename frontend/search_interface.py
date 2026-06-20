"""
Streamlit frontend — communicates with API Gateway (SOA).
Start gateway first: python run_gateway.py
"""

import base64
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
ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
QUERYFY_WORDMARK = os.path.join(ASSETS_DIR, "queryfy_wordmark.png")
QUERYFY_ICON = os.path.join(ASSETS_DIR, "queryfy_q_icon.png")
QUERYFY_LOGO = os.path.join(ASSETS_DIR, "queryfy_logo_orange.png")

DISPLAY_DATASETS = {
    "MSMARCO-Document": "msmarco",
}


def _image_data_uri(path):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _inject_queryfy_theme():
    st.markdown(
        """
        <style>
        :root {
            --queryfy-orange: #ff8500;
            --queryfy-orange-soft: #fff1dd;
            --queryfy-orange-deep: #f36b00;
            --queryfy-ink: #151922;
            --queryfy-muted: #697386;
            --queryfy-panel: #ffffff;
        }

        .stApp {
            background:
                radial-gradient(circle at top right, rgba(255, 133, 0, 0.22), transparent 26rem),
                linear-gradient(180deg, #fff8ef 0%, #ffffff 38%, #fffaf5 100%);
            color: var(--queryfy-ink);
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #ffffff 0%, #fff5e8 100%);
            border-right: 1px solid rgba(255, 133, 0, 0.18);
        }

        [data-testid="stSidebar"] * {
            color: var(--queryfy-ink);
        }

        .queryfy-sidebar-logo {
            display: block;
            width: min(210px, 88%);
            margin: 0.45rem auto 1.15rem;
            filter: drop-shadow(0 12px 20px rgba(255, 133, 0, 0.18));
        }

        .stApp label,
        .stApp p,
        .stApp span,
        .stApp div[data-testid="stMarkdownContainer"],
        .stApp div[data-testid="stCaptionContainer"],
        .stApp div[data-testid="stWidgetLabel"],
        .stApp div[data-testid="stTickBar"] {
            color: var(--queryfy-ink) !important;
        }

        .stApp div[data-testid="stCaptionContainer"] p,
        .stApp small {
            color: var(--queryfy-muted) !important;
        }

        .stApp button[role="tab"] p {
            color: var(--queryfy-muted) !important;
            font-weight: 800;
        }

        .stApp button[role="tab"][aria-selected="true"] p {
            color: var(--queryfy-orange-deep) !important;
        }

        [data-testid="stSidebar"] .stAlert {
            background-color: rgba(255, 133, 0, 0.10);
            border: 1px solid rgba(255, 133, 0, 0.24);
        }

        .queryfy-hero {
            position: relative;
            padding: clamp(1.7rem, 4vw, 3rem);
            border-radius: 28px;
            overflow: hidden;
            background:
                radial-gradient(circle at 82% 18%, rgba(255, 255, 255, 0.34), transparent 14rem),
                linear-gradient(135deg, #ff8500 0%, #ff9b1f 48%, #ff6b00 100%);
            border: 1px solid rgba(255, 133, 0, 0.42);
            box-shadow: 0 22px 50px rgba(255, 133, 0, 0.24);
            margin-bottom: 1.35rem;
        }

        .queryfy-hero::after {
            content: "";
            position: absolute;
            right: -4rem;
            top: -4rem;
            width: 18rem;
            height: 18rem;
            border-radius: 999px;
            border: 2.8rem solid rgba(255, 255, 255, 0.22);
        }

        .queryfy-hero p {
            color: rgba(21, 25, 34, 0.82);
            font-size: 1.08rem;
            max-width: 46rem;
            margin: 1rem 0 0;
            font-weight: 600;
        }

        .queryfy-pill {
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            color: var(--queryfy-ink) !important;
            background: rgba(255, 255, 255, 0.86);
            padding: 0.45rem 0.85rem;
            border-radius: 999px;
            font-weight: 800;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            box-shadow: 0 10px 24px rgba(243, 107, 0, 0.20);
        }

        .queryfy-wordmark {
            width: min(440px, 76vw);
            display: block;
            margin: 0 0 1.25rem;
            position: relative;
            z-index: 1;
        }

        .queryfy-hero-content {
            position: relative;
            z-index: 1;
        }

        .queryfy-status {
            padding: 0.75rem 0.9rem;
            margin-bottom: 1.15rem;
            border-radius: 16px;
            color: #151922;
            background: linear-gradient(135deg, #fff3e4, #ffffff);
            border: 1px solid rgba(255, 133, 0, 0.30);
            font-weight: 800;
            box-shadow: 0 10px 24px rgba(255, 133, 0, 0.10);
        }

        .queryfy-section-card {
            padding: 1.25rem;
            border-radius: 22px;
            background: rgba(255, 255, 255, 0.82);
            border: 1px solid rgba(255, 133, 0, 0.18);
            box-shadow: 0 16px 40px rgba(17, 24, 39, 0.08);
            margin-bottom: 1rem;
        }

        [data-testid="stVerticalBlockBorderWrapper"] {
            border-color: rgba(255, 133, 0, 0.22);
            border-radius: 22px;
            background: rgba(255, 255, 255, 0.92);
            box-shadow: 0 16px 40px rgba(255, 133, 0, 0.10);
        }

        div[data-testid="stTextInput"] div[data-baseweb="input"] {
            border: 1.5px solid rgba(255, 133, 0, 0.38) !important;
            border-radius: 16px !important;
            background: #ffffff !important;
            background-color: #ffffff !important;
            box-shadow: none !important;
            outline: none !important;
        }

        div[data-testid="stTextInput"] div[data-baseweb="input"] > div {
            background: #ffffff !important;
            background-color: #ffffff !important;
            border-radius: 16px !important;
        }

        div[data-testid="stTextInput"] div[data-baseweb="input"]:focus-within {
            border-color: var(--queryfy-orange) !important;
            box-shadow: 0 0 0 3px rgba(255, 133, 0, 0.14) !important;
            outline: none !important;
        }

        div[data-testid="stTextInput"] input {
            border: 0 !important;
            border-radius: 16px;
            padding: 0.85rem 1rem;
            background: #ffffff !important;
            background-color: #ffffff !important;
            color: var(--queryfy-ink) !important;
            caret-color: var(--queryfy-orange) !important;
            box-shadow: none !important;
            outline: none !important;
        }

        div[data-testid="stTextInput"] input::placeholder {
            color: #8a94a6 !important;
            opacity: 1 !important;
        }

        div[data-testid="stTextInput"] input:focus {
            border: 0 !important;
            background: #ffffff !important;
            background-color: #ffffff !important;
            color: var(--queryfy-ink) !important;
            box-shadow: none !important;
            outline: none !important;
        }

        .stButton > button,
        div[data-testid="stFormSubmitButton"] button {
            border: 0 !important;
            border-radius: 999px;
            color: #ffffff !important;
            background: linear-gradient(135deg, #ff9d2e 0%, #ff8500 55%, #f36b00 100%) !important;
            font-weight: 800 !important;
            min-width: 7rem;
            box-shadow: 0 12px 28px rgba(255, 133, 0, 0.24) !important;
        }

        .stButton > button:hover,
        div[data-testid="stFormSubmitButton"] button:hover {
            color: #ffffff !important;
            transform: translateY(-1px);
            box-shadow: 0 16px 34px rgba(255, 133, 0, 0.32) !important;
        }

        .stButton > button *,
        .stButton > button:hover *,
        div[data-testid="stFormSubmitButton"] button *,
        div[data-testid="stFormSubmitButton"] button:hover *,
        div[data-testid="stFormSubmitButton"] button p {
            color: #ffffff !important;
            font-weight: 800 !important;
            opacity: 1 !important;
        }

        [data-testid="stSidebar"] div[data-baseweb="select"] > div,
        div[data-testid="stSelectbox"] div[data-baseweb="select"] > div,
        [data-testid="stSidebar"] div[data-testid="stNumberInput"] input {
            background-color: #ffffff;
            border-color: rgba(255, 133, 0, 0.28);
        }

        div[data-testid="stSelectbox"] * {
            color: var(--queryfy-ink) !important;
        }

        [data-testid="stSidebar"] [data-testid="stSlider"] div[role="slider"] {
            background-color: var(--queryfy-orange);
            border-color: var(--queryfy-orange);
        }

        div[data-testid="stRadio"] label {
            background: transparent !important;
            border-radius: 999px !important;
            padding: 0.15rem 0.35rem !important;
        }

        div[data-testid="stRadio"] label:hover {
            background: rgba(255, 133, 0, 0.08) !important;
        }

        div[data-testid="stRadio"] label span,
        div[data-testid="stRadio"] label p {
            background: transparent !important;
        }

        div[data-testid="stRadio"] label div:first-child {
            background-color: #ffffff !important;
            border: 2px solid rgba(21, 25, 34, 0.35) !important;
            box-shadow: none !important;
        }

        div[data-testid="stRadio"] label:has(input:checked) div:first-child {
            background-color: var(--queryfy-orange) !important;
            border-color: var(--queryfy-orange) !important;
        }

        div[data-testid="stRadio"] label:has(input:checked) div:first-child::after,
        div[data-testid="stRadio"] label:has(input:checked) div:first-child div {
            background-color: #ffffff !important;
        }

        [data-testid="stDataFrame"] {
            border: 1px solid rgba(255, 133, 0, 0.18);
            border-radius: 18px;
            overflow: hidden;
            box-shadow: 0 14px 34px rgba(17, 24, 39, 0.08);
        }

        h1, h2, h3 {
            color: #111827;
        }

        div[data-testid="stCaptionContainer"] {
            color: var(--queryfy-muted);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_queryfy_header():
    wordmark = _image_data_uri(QUERYFY_WORDMARK)
    wordmark_html = (
        f'<img class="queryfy-wordmark" src="{wordmark}" alt="QUERYFY wordmark" />'
        if wordmark
        else ""
    )
    st.markdown(
        f"""
        <div class="queryfy-hero">
            <div class="queryfy-hero-content">
                {wordmark_html}
                <span class="queryfy-pill">Search smarter with QUERYFY</span>
                <p>
                    A clean information retrieval experience powered by query refinement,
                    ranking services, and fast document exploration.
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_sidebar_logo():
    logo = _image_data_uri(QUERYFY_LOGO)
    if not logo:
        logo = _image_data_uri(QUERYFY_WORDMARK)
    if not logo:
        return
    st.sidebar.markdown(
        f'<img class="queryfy-sidebar-logo" src="{logo}" alt="QUERYFY logo" />',
        unsafe_allow_html=True,
    )


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
    st.subheader("Document Preview")
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


def _render_search_engine_tab(
    client,
    dataset_key,
    search_method,
    ranking_method,
    execution_mode,
    top_k,
    bm25_k1,
    bm25_b,
):
    with st.container(border=True):
        st.header("Find relevant documents")
        st.caption("Type a query and QUERYFY will refine it, rank matching documents, and surface the best evidence.")
        with st.form("search_form"):
            query = st.text_input(
                "Search query",
                placeholder="e.g., machine learning neural networks",
                key="search_query",
            )
            submitted = st.form_submit_button("Search")

        if st.session_state.search_history:
            st.caption("Recent: " + " | ".join(st.session_state.search_history[-5:]))

        search_key = (
            dataset_key,
            search_method,
            ranking_method,
            execution_mode,
            query.strip() if query else "",
            top_k,
            bm25_k1,
            bm25_b,
        )
        response = None

        if submitted:
            if not query or not query.strip():
                st.warning("Enter a search query.")
                return
            try:
                if st.session_state.last_search_key != search_key:
                    with st.spinner(f"QUERYFY is searching with {search_method.upper()}..."):
                        response = client.search(
                            query=query,
                            dataset=dataset_key,
                            search_method=search_method,
                            ranking_method=ranking_method,
                            execution_mode=execution_mode,
                            top_k=top_k,
                            query_history=st.session_state.search_history,
                            bm25_k1=bm25_k1,
                            bm25_b=bm25_b,
                        )
                    st.session_state.last_search_key = search_key
                    st.session_state.last_search_response = response
                else:
                    response = st.session_state.last_search_response

                if query.strip() not in st.session_state.search_history:
                    st.session_state.search_history.append(query.strip())
                    st.session_state.search_history = st.session_state.search_history[-20:]
            except Exception as e:
                st.error(f"Search failed: {e}")
                logger.exception("Search error")
                return
        elif st.session_state.last_search_key == search_key:
            response = st.session_state.last_search_response

        if not response:
            return

        suggestions = response.get("suggestions", [])
        if suggestions:
            st.caption("Suggestions: " + " · ".join(suggestions))

        results = response.get("results", [])
        st.caption(f"Refined query: `{response.get('refined_query', query)}`")

        if results:
            result_method = response.get("method", search_method)
            st.subheader(f"Ranked Results ({result_method.upper()})")
            df = pd.DataFrame(format_results(results))
            qkey = "_".join(
                [
                    dataset_key,
                    search_method,
                    ranking_method,
                    execution_mode,
                    query.strip().lower(),
                ]
            )
            idx = max(0, min(_render_results_table(df, qkey), len(results) - 1))
            selected = dict(results[idx])
            selected["rank"] = idx + 1
            selected["method"] = result_method
            _render_full_document(selected)
        else:
            st.warning("No results.")


def _format_rag_sources(sources):
    rows = []
    for source in sources:
        rows.append(
            {
                "Source": f"[{source.get('label', '')}]",
                "Rank": source.get("rank"),
                "Document ID": source.get("doc_id"),
                "Score": f"{source.get('score', 0):.6f}",
                "Snippet": source.get("snippet", ""),
            }
        )
    return rows


def _render_rag_sources(sources, query_key):
    st.subheader("Sources")
    df = pd.DataFrame(_format_rag_sources(sources))
    idx = max(0, min(_render_results_table(df, f"rag_{query_key}"), len(sources) - 1))
    selected = sources[idx]
    st.write(f"**Selected source:** [{selected.get('label')}] {selected.get('doc_id')}")
    st.text_area(
        "Source Document",
        value=selected.get("document_text", ""),
        height=260,
        disabled=True,
        key=f"rag_source_doc_{query_key}",
    )


def _render_rag_tab(
    client,
    dataset_key,
    search_method,
    ranking_method,
    execution_mode,
    top_k,
    bm25_k1,
    bm25_b,
):
    with st.container(border=True):
        st.header("RAG Chat")
        st.caption("Ask follow-up questions. QUERYFY retrieves evidence for each message, then Gemini answers from those sources.")

        controls_col, action_col = st.columns([3, 1])
        with controls_col:
            rag_mode_label = st.selectbox(
                "RAG retrieval mode",
                options=["BM25 + BERT", "BM25 only"],
                index=0,
                key="rag_retrieval_mode",
                help="BERT loads once per gateway session, then stays in memory. Use BM25 only for the fastest mode.",
            )
            rag_mode = "hybrid" if rag_mode_label == "BM25 + BERT" else "bm25"
            context_k = st.slider(
                "Sources used by Gemini",
                1,
                min(5, top_k),
                min(3, top_k),
                key="rag_context_k",
            )
        with action_col:
            st.write("")
            st.write("")
            if st.button("Clear RAG chat", key="clear_rag_chat"):
                st.session_state.rag_messages = []
                st.session_state.last_rag_response = None
                st.session_state.last_rag_query = None

        gemini_models = [
            "gemini-2.5-flash",
            "gemini-2.5-pro",
            "gemini-2.0-flash",
        ]
        default_model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
        if default_model not in gemini_models:
            gemini_models.insert(0, default_model)
        model = st.selectbox(
            "Gemini model",
            options=gemini_models,
            index=gemini_models.index(default_model),
            key="rag_model",
        )
        temperature = st.slider(
            "LLM temperature",
            min_value=0.0,
            max_value=2.0,
            value=0.2,
            step=0.1,
            key="rag_temperature",
            help="Lower values are more precise. Higher values are more creative.",
        )

        if not st.session_state.rag_messages:
            st.info("Start the RAG chat by asking a question below.")

        for idx, message in enumerate(st.session_state.rag_messages):
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message["role"] == "assistant":
                    refined_query = message.get("refined_query")
                    if refined_query:
                        st.caption(f"Refined query: `{refined_query}`")
                    warning = message.get("warning")
                    if warning:
                        st.warning(warning)
                    sources = message.get("sources", [])
                    if sources:
                        with st.expander("Sources", expanded=False):
                            _render_rag_sources(sources, f"chat_{idx}")

        query = st.chat_input("Ask QUERYFY RAG...")
        if query and query.strip():
            user_query = query.strip()
            st.session_state.rag_messages.append(
                {"role": "user", "content": user_query}
            )
            rag_history = [
                message["content"]
                for message in st.session_state.rag_messages
                if message["role"] == "user"
            ][-3:]
            try:
                with st.spinner("Retrieving sources and generating an answer with Gemini..."):
                    response = client.rag_answer(
                        query=user_query,
                        dataset=dataset_key,
                        search_method=search_method,
                        ranking_method=ranking_method,
                        execution_mode=execution_mode,
                        top_k=top_k,
                        context_k=context_k,
                        query_history=rag_history,
                        bm25_k1=bm25_k1,
                        bm25_b=bm25_b,
                        model=model,
                        rag_mode=rag_mode,
                        temperature=temperature,
                    )
                st.session_state.last_rag_response = response
                st.session_state.last_rag_query = user_query
                st.session_state.rag_messages.append(
                    {
                        "role": "assistant",
                        "content": response.get("generated_answer") or response.get("answer", ""),
                        "refined_query": response.get("refined_query"),
                        "warning": response.get("warning"),
                        "sources": response.get("source_documents") or response.get("sources", []),
                    }
                )
                if user_query not in st.session_state.search_history:
                    st.session_state.search_history.append(user_query)
                    st.session_state.search_history = st.session_state.search_history[-20:]
                st.rerun()
            except Exception as e:
                error_message = f"RAG failed: {e}"
                st.session_state.rag_messages.append(
                    {"role": "assistant", "content": error_message, "warning": str(e)}
                )
                st.error(error_message)
                logger.exception("RAG error")
            st.rerun()


def _format_evaluation_summary(report):
    rows = []
    k = report.get("k", 10)
    for comparison in report.get("comparisons", []):
        method = comparison.get("method", "unknown")
        for mode in ("baseline", "enhanced"):
            item = comparison.get(mode, {})
            rows.append(
                {
                    "Method": method,
                    "Mode": mode,
                    "Qrels Queries": item.get("qrels_query_count", report.get("qrels_query_count", 0)),
                    "Used Queries": item.get("attempted_query_count", report.get("attempted_query_count", 0)),
                    "Evaluated Queries": item.get("num_queries", 0),
                    "MAP": f"{item.get('MAP', 0):.4f}",
                    f"Precision@{k}": f"{item.get('avg_P@k', 0):.4f}",
                    f"Recall@{k}": f"{item.get('avg_R@k', 0):.4f}",
                    f"nDCG@{k}": f"{item.get(f'avg_nDCG@{k}', 0):.4f}",
                }
            )
    return rows


def _format_evaluation_deltas(report):
    rows = []
    k = report.get("k", 10)
    for comparison in report.get("comparisons", []):
        rows.append(
            {
                "Method": comparison.get("method", "unknown"),
                "Delta MAP": f"{comparison.get('delta_MAP', 0):.4f}",
                f"Delta Precision@{k}": f"{comparison.get('delta_P@k', 0):.4f}",
                f"Delta Recall@{k}": f"{comparison.get('delta_R@k', 0):.4f}",
                f"Delta nDCG@{k}": f"{comparison.get(f'delta_nDCG@{k}', 0):.4f}",
            }
        )
    return rows


def _render_evaluation_section(client, dataset_key):
    with st.container(border=True):
        st.header("Evaluation")
        st.caption(
            "Run qrels-based evaluation and compare retrieval quality before and after additional query features."
        )

        col_methods, col_settings = st.columns([2, 1])
        default_methods = ["bm25"]

        with col_methods:
            selected_methods = st.multiselect(
                "Methods to compare",
                options=["tfidf", "bm25", "index", "serial", "parallel", "rrf", "word2vec", "bert"],
                default=default_methods,
                help="Keep BM25 only for a quick UI check. Add more methods for the final comparison.",
                key="eval_methods",
            )
            include_embeddings = st.checkbox(
                "Include embedding methods (Word2Vec/BERT)",
                value=False,
                help="Embedding methods can be slower and may need model files.",
                key="eval_include_embeddings",
            )
        with col_settings:
            k = st.number_input("K", min_value=1, max_value=100, value=10, step=1, key="eval_k")
            limit_value = st.number_input(
                "Query limit (0 = all qrels)",
                min_value=0,
                max_value=1000000,
                value=3,
                step=1,
                key="eval_limit",
            )

        limit = None if limit_value == 0 else int(limit_value)
        confirm_full_eval = True
        if limit is None:
            st.info("Full qrels evaluation is selected. This can take a long time.")
            confirm_full_eval = st.checkbox(
                "I understand full qrels evaluation can take a long time.",
                key="confirm_full_eval",
            )
        else:
            st.caption(
                "Quick UI check: keep BM25 and a small query limit. "
                "For the final report, use all required methods and set Query limit to 0."
            )

        if st.button("Run Baseline vs Enhanced Evaluation", key="run_eval_compare"):
            if not selected_methods:
                st.warning("Choose at least one method to evaluate.")
            elif limit is None and not confirm_full_eval:
                st.warning("Confirm full evaluation first, or set a small Query limit.")
            else:
                try:
                    with st.spinner("Running evaluation. This may take a while on the first run..."):
                        report = client.compare_evaluation(
                            dataset=dataset_key,
                            methods=selected_methods,
                            k=int(k),
                            limit=limit,
                            include_embeddings=include_embeddings,
                        )
                    st.session_state.last_evaluation_report = report
                except Exception as e:
                    st.error(f"Evaluation failed: {e}")
                    logger.exception("Evaluation error")

        report = st.session_state.get("last_evaluation_report")
        if not report:
            st.info("No evaluation report loaded yet. Run the comparison to display metrics here.")
            return

        st.subheader("Evaluation Overview")
        metric_cols = st.columns(4)
        metric_cols[0].metric("Dataset", report.get("dataset", dataset_key))
        metric_cols[1].metric("Qrels Queries", report.get("qrels_query_count", 0))
        metric_cols[2].metric("Used Queries", report.get("attempted_query_count", 0))
        metric_cols[3].metric("K", report.get("k", k))

        summary_rows = _format_evaluation_summary(report)
        if summary_rows:
            st.subheader("Metrics by Method and Mode")
            st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

        delta_rows = _format_evaluation_deltas(report)
        if delta_rows:
            st.subheader("Enhanced vs Baseline")
            st.dataframe(pd.DataFrame(delta_rows), use_container_width=True, hide_index=True)

        st.subheader("Saved Reports")
        st.write(f"**JSON:** `{report.get('json_report_path', 'N/A')}`")
        st.write(f"**Markdown:** `{report.get('markdown_report_path', 'N/A')}`")


@st.cache_resource
def get_client():
    return GatewayClient(GATEWAY_URL)


def main():
    st.set_page_config(
        page_title="QUERYFY Search",
        page_icon=QUERYFY_ICON if os.path.exists(QUERYFY_ICON) else "Q",
        layout="wide",
    )
    _inject_queryfy_theme()
    _render_queryfy_header()

    client = get_client()

    _render_sidebar_logo()

    try:
        health = client.health()
        st.sidebar.markdown(
            f'<div class="queryfy-status">Gateway online: {health.get("status", "ok")}</div>',
            unsafe_allow_html=True,
        )
    except Exception as e:
        st.error(f"Cannot reach API Gateway at {GATEWAY_URL}")
        st.code("python run_gateway.py", language="bash")
        st.stop()

    if "search_history" not in st.session_state:
        st.session_state.search_history = []
    if "last_search_key" not in st.session_state:
        st.session_state.last_search_key = None
    if "last_search_response" not in st.session_state:
        st.session_state.last_search_response = None
    if "last_rag_response" not in st.session_state:
        st.session_state.last_rag_response = None
    if "last_rag_query" not in st.session_state:
        st.session_state.last_rag_query = None
    if "rag_messages" not in st.session_state:
        st.session_state.rag_messages = []
    if "last_evaluation_report" not in st.session_state:
        st.session_state.last_evaluation_report = None

    st.sidebar.header("QUERYFY Controls")

    display_name = st.sidebar.selectbox("Dataset", list(DISPLAY_DATASETS.keys()), index=0)
    dataset_key = DISPLAY_DATASETS[display_name]
    cfg = DATASETS[dataset_key]
    st.sidebar.caption(cfg["description"])

    try:
        status = client.index_status(dataset_key)
        st.sidebar.write(f"**Documents:** {status.get('num_documents', 0):,}")
    except Exception:
        st.sidebar.warning("Index not ready")

    st.sidebar.subheader("Search Settings")
    method_col, ranking_col = st.sidebar.columns(2)
    with method_col:
        search_method = st.selectbox(
            "Search Method",
            ["tfidf", "bm25", "word2vec", "bert"],
            index=1,
            key="search_method",
            help="Primary retrieval model used to score matching documents.",
        )
    with ranking_col:
        ranking_method = st.selectbox(
            "Fusion Method",
            ["none", "rrf"],
            index=0,
            key="ranking_method",
            format_func=lambda value: "None" if value == "none" else value.upper(),
            help="Optional result fusion or re-ranking strategy.",
        )
    execution_mode = "parallel"
    if ranking_method == "rrf":
        execution_mode = st.sidebar.radio(
            "Execution Mode",
            ["serial", "parallel"],
            index=1,
            horizontal=True,
            key="execution_mode",
            help="Processing mode for RRF search-time orchestration.",
        )
    if search_method in ("word2vec", "bert"):
        st.sidebar.warning("This method can be slow on first use because it builds embeddings.")
    top_k = st.sidebar.slider("Results", 1, 50, 10)

    bm25_k1, bm25_b = 1.5, 0.75
    if search_method == "bm25" or ranking_method == "rrf":
        st.sidebar.subheader("BM25 Parameters")
        bm25_k1 = st.sidebar.slider("k1", 0.5, 3.0, 1.5, 0.1)
        bm25_b = st.sidebar.slider("b", 0.0, 1.0, 0.75, 0.05)

    search_tab, rag_tab = st.tabs(["Search Engine", "RAG"])
    with search_tab:
        _render_search_engine_tab(
            client,
            dataset_key,
            search_method,
            ranking_method,
            execution_mode,
            top_k,
            bm25_k1,
            bm25_b,
        )
        _render_evaluation_section(client, dataset_key)
    with rag_tab:
        _render_rag_tab(
            client,
            dataset_key,
            search_method,
            ranking_method,
            execution_mode,
            top_k,
            bm25_k1,
            bm25_b,
        )

    if st.sidebar.button("Clear history"):
        st.session_state.search_history = []
        st.session_state.last_rag_response = None
        st.session_state.last_rag_query = None
        st.session_state.rag_messages = []
        st.session_state.last_evaluation_report = None

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"[API Docs]({GATEWAY_URL}/docs)")


if __name__ == "__main__":
    main()
