"""
Gunsmith task loading utilities.
"""

import json

import streamlit as st

from src.config import get_resource_path


@st.cache_data(show_spinner=False)
def load_tasks():
    """Load Gunsmith tasks from JSON file."""
    try:
        with open(get_resource_path("tasks.json"), "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return []
