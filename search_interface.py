# search_interface.py
"""
Web-based search interface using Streamlit
"""

import streamlit as st
import json
import logging
from ranking_engine import RankingEngine
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@st.cache_resource
def load_engine():
    """Load and cache the ranking engine"""
    logger.info("Loading ranking engine...")
    
    # Load preprocessed documents
    try:
        with open('processed_data.json', 'r', encoding='utf-8') as f:
            processed_documents = json.load(f)
    except FileNotFoundError:
        st.error("processed_data.json not found. Please run main.py first.")
        return None
    
    # Initialize engine
    engine = RankingEngine()
    engine.fit(processed_documents)
    
    return engine


def format_results(results, engine):
    """Format search results for display"""
    formatted = []
    for i, result in enumerate(results, 1):
        doc_id = result['doc_id']
        score = result['score']
        
        row = {
            'Rank': i,
            'Document ID': doc_id,
            'Score': f'{score:.6f}',
        }
        
        if 'explanation' in result:
            exp = result['explanation']
            row['Fusion Type'] = exp.get('fusion_type', 'N/A')
            
            # Add method-specific info
            if 'methods' in exp and isinstance(exp['methods'], dict):
                methods_str = ', '.join([f"{k}:{v:.4f}" for k, v in exp['methods'].items()])
                row['Method Scores'] = methods_str
        
        formatted.append(row)
    
    return formatted


def main():
    st.set_page_config(page_title="IR Search Engine", layout="wide")
    
    st.title("Information Retrieval Search Engine")
    st.markdown("---")
    
    # Sidebar: Configuration
    st.sidebar.header("Configuration")
    
    # Load engine
    engine = load_engine()
    if engine is None:
        return
    
    # Ranking method selection
    ranking_method = st.sidebar.selectbox(
        "Select Ranking Method:",
        options=['tfidf', 'bm25', 'word2vec', 'bert', 'serial', 'parallel', 'rrf'],
        index=5,  # default to parallel
        help="""
        - TF-IDF: Classic vector space model
        - BM25: Probabilistic ranking
        - Word2Vec: Word embedding-based
        - BERT: Deep contextual embeddings
        - Serial: Sequential fusion (TF-IDF → BM25)
        - Parallel: Weighted average fusion
        - RRF: Reciprocal Rank Fusion
        """
    )
    
    top_k = st.sidebar.slider("Number of Results", min_value=1, max_value=50, value=10)
    
    # BM25 parameter tuning
    if ranking_method == 'bm25' or ranking_method in ['serial', 'parallel', 'rrf']:
        st.sidebar.subheader("BM25 Parameters")
        k1 = st.sidebar.slider("k1 (term frequency saturation)", min_value=0.5, max_value=3.0, value=1.5, step=0.1)
        b = st.sidebar.slider("b (document length effect)", min_value=0.0, max_value=1.0, value=0.75, step=0.05)
        
        if k1 != engine.bm25.k1 or b != engine.bm25.b:
            logger.info(f"Updating BM25 parameters: k1={k1}, b={b}")
            engine.bm25.update_parameters(k1=k1, b=b)
            st.sidebar.success("Parameters updated!")
    
    # Main search area
    st.header("Search")
    query = st.text_input("Enter your search query:", placeholder="e.g., information retrieval machine learning")
    
    if query:
        try:
            # Perform search
            with st.spinner(f"Searching using {ranking_method.upper()}..."):
                results = engine.search(query, method=ranking_method, top_k=top_k)
            
            # Display results
            if results:
                st.subheader(f"Results (Method: {ranking_method.upper()})")
                
                # Format and display in table
                formatted_results = format_results(results, engine)
                df = pd.DataFrame(formatted_results)
                
                st.dataframe(df, use_container_width=True)
                
                # Detailed view for top result
                if results:
                    st.subheader("Top Result Details")
                    top_result = results[0]
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Document ID:** {top_result['doc_id']}")
                        st.write(f"**Score:** {top_result['score']:.6f}")
                    
                    with col2:
                        if 'explanation' in top_result:
                            st.write(f"**Fusion Type:** {top_result['explanation'].get('fusion_type', 'N/A')}")
                            if 'methods' in top_result['explanation']:
                                st.write("**Method Scores:**")
                                for method, score in top_result['explanation']['methods'].items():
                                    st.write(f"  - {method}: {score:.6f}")
            else:
                st.warning("No results found.")
        
        except Exception as e:
            st.error(f"Search failed: {str(e)}")
            logger.error(f"Search error: {e}", exc_info=True)
    
    # System info
    st.sidebar.markdown("---")
    st.sidebar.subheader("System Information")
    params = engine.get_parameter_report()
    st.sidebar.write(f"**Documents loaded:** {params['num_documents']}")
    st.sidebar.write(f"**Engine ready:** {'Yes' if params['is_fitted'] else 'No'}")
    st.sidebar.write(f"**BM25 k1:** {params['bm25']['k1']}")
    st.sidebar.write(f"**BM25 b:** {params['bm25']['b']}")


if __name__ == '__main__':
    main()
