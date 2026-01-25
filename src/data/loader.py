"""
Data loading functions with Streamlit caching.
"""

import json
from datetime import datetime

import streamlit as st
from loguru import logger
from src.api import fetch_all_data
from src.core import build_item_lookup, build_compatibility_map


@st.cache_data(show_spinner=False)
def load_data():
    """Fetch all guns and mods from API (cached). Saves to debug file."""
    logger.info("Loading game data...")
    guns, mods = fetch_all_data()

    # Save to debug file
    debug_data = {
        "fetched_at": datetime.now().isoformat(),
        "guns_count": len(guns),
        "mods_count": len(mods),
        "guns": guns,
        "mods": mods,
    }
    with open("api_cache_debug.json", "w", encoding="utf-8") as f:
        json.dump(debug_data, f, indent=2, ensure_ascii=False)

    logger.info(f"Loaded {len(guns)} guns and {len(mods)} mods")
    return guns, mods


@st.cache_data(show_spinner=False)
def build_lookup(_guns, _mods):
    """Build item lookup dictionary (cached)."""
    return build_item_lookup(_guns, _mods)


@st.cache_data(show_spinner=False)
def get_compat_map(weapon_id, _item_lookup):
    """Build compatibility map for a weapon (cached per weapon_id)."""
    return build_compatibility_map(weapon_id, _item_lookup)
