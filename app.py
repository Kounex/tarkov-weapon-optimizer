"""
Streamlit Web UI for Tarkov Weapon Mod Optimizer

This is the main entry point for the Streamlit application.
The application is organized into modular components in the `src/` package.
"""

import os
import sys

# Set Streamlit config directory to project directory (must be before streamlit import)
os.environ.setdefault("STREAMLIT_CONFIG_DIR", os.path.dirname(os.path.abspath(__file__)))

import extra_streamlit_components as stx
import streamlit as st

from i18n import t, language_selector
from src.config import setup_logging, PAGE_CONFIG
from src.data import load_data, build_lookup, get_compat_map
from src.components.sidebar import (
    render_weapon_selector,
    render_player_settings,
    sync_cookies,
    render_constraints,
    render_include_exclude,
)
from src.components.sidebar.weapon_selector import render_weapon_details
from src.tabs import render_explore_tab, render_optimize_tab, render_gunsmith_tab


def main():
    """Main application entry point."""
    # Setup logging
    setup_logging()

    # Page configuration
    st.set_page_config(**PAGE_CONFIG)

    # Initialize Cookie Manager for persistent settings
    cookie_manager = stx.CookieManager(key="main_cookie_manager")

    # Language selector at top of sidebar
    with st.sidebar:
        language_selector(label="🌐 Language")
        st.markdown("---")

    # Title
    st.title(f"🔫 {t('app.title')}")
    st.markdown(t("app.subtitle"))

    # Load data with status indicator
    guns, mods, item_lookup = _load_game_data()

    # Sync cookies for persistent settings
    sync_cookies(cookie_manager)

    # Render weapon selector
    selected_gun, weapon_stats, presets = render_weapon_selector(guns, item_lookup)
    weapon_id = selected_gun["id"]

    # Build compatibility map for sidebar filters
    compat_map = get_compat_map(weapon_id, item_lookup)
    reachable_ids = set(compat_map["reachable_items"].keys())

    # Render weapon details (image, stats, presets)
    render_weapon_details(selected_gun, weapon_stats, presets)

    # Render player/trader settings
    player_level, flea_available, trader_levels = render_player_settings(cookie_manager)

    # Render hard constraints
    hard_constraints = render_constraints(weapon_stats)

    # Render include/exclude section
    include_exclude = render_include_exclude(mods, reachable_ids)

    # Create tabs
    tab_explore, tab_optimize, tab_gunsmith = st.tabs(
        [
            f"📊 {t('tabs.explore')}",
            f"🚀 {t('tabs.optimize')}",
            f"🛠️ {t('tabs.gunsmith')}",
        ]
    )

    # Explore Tab
    with tab_explore:
        render_explore_tab(
            selected_gun=selected_gun,
            weapon_id=weapon_id,
            weapon_stats=weapon_stats,
            presets=presets,
            item_lookup=item_lookup,
            hard_constraints=hard_constraints,
            include_exclude=include_exclude,
            player_level=player_level,
            flea_available=flea_available,
            trader_levels=trader_levels,
        )

    # Optimize Tab
    with tab_optimize:
        render_optimize_tab(
            selected_gun=selected_gun,
            weapon_id=weapon_id,
            weapon_stats=weapon_stats,
            presets=presets,
            item_lookup=item_lookup,
            hard_constraints=hard_constraints,
            include_exclude=include_exclude,
            player_level=player_level,
            flea_available=flea_available,
            trader_levels=trader_levels,
        )

    # Gunsmith Tab
    with tab_gunsmith:
        render_gunsmith_tab(
            guns=guns,
            item_lookup=item_lookup,
            player_level=player_level,
            flea_available=flea_available,
            trader_levels=trader_levels,
        )


def _load_game_data():
    """Load game data with status indicator.

    Returns:
        Tuple of (guns, mods, item_lookup)
    """
    with st.status(t("status.loading"), expanded=False) as status:
        try:
            if status:
                status.update(label=t("status.fetching"))
            guns, mods = load_data()
            if status:
                status.update(label=t("status.building_lookup"))
            item_lookup = build_lookup(guns, mods)
            if status:
                status.update(
                    label=t("status.loaded", guns=len(guns), mods=len(mods)),
                    state="complete",
                )
        except Exception as e:
            if status:
                status.update(label=t("status.failed_load"), state="error")
            st.error(f"{t('status.failed_load')}: {e}")
            st.stop()

    return guns, mods, item_lookup


if __name__ == "__main__":
    main()
