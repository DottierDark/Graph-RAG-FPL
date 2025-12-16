"""
FPL Graph-RAG Main Application
===============================
Entry point for the Streamlit application.

Run with: streamlit run app.py
"""

import sys
from pathlib import Path

# Add src directory to path so imports work correctly
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Import the streamlit app to execute it
import streamlit_app
