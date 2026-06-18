"""
Legacy entry point — use SOA frontend via API Gateway.

1. python run_gateway.py
2. streamlit run frontend/search_interface.py
   (or: streamlit run search_interface.py)
"""

from frontend.search_interface import main

if __name__ == "__main__":
    main()
