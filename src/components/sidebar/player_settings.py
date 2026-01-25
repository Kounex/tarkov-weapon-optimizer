"""
Player and trader settings sidebar component.
"""

from datetime import datetime, timedelta

import streamlit as st
from loguru import logger

from i18n import t
from src.config import TRADERS


def sync_cookies(cookie_manager) -> None:
    """Sync settings from cookies (run once when cookies are available).

    Args:
        cookie_manager: Streamlit cookie manager instance
    """
    if st.session_state.get("cookies_synced", False):
        return

    cookies = cookie_manager.get_all()
    if not cookies:
        return

    found_data = False
    try:
        if "player_level" in cookies:
            st.session_state.player_level = int(cookies["player_level"])
            found_data = True

        if "flea_available" in cookies:
            st.session_state.flea_available = (
                str(cookies["flea_available"]).lower() == "true"
            )
            found_data = True

        for t_key, _ in TRADERS:
            s_key = f"trader_{t_key}"
            if s_key in cookies:
                st.session_state[s_key] = int(cookies[s_key])
                found_data = True

        if found_data:
            st.session_state.cookies_synced = True
            st.rerun()

    except Exception as e:
        logger.error(f"Error syncing cookies: {e}")


def render_player_settings(cookie_manager) -> tuple:
    """Render player level and trader settings in sidebar.

    Args:
        cookie_manager: Streamlit cookie manager instance

    Returns:
        Tuple of (player_level, flea_available, trader_levels)
    """

    def save_cookie(key, value):
        cookie_manager.set(key, value, expires_at=datetime.now() + timedelta(days=365))

    st.sidebar.markdown("---")
    st.sidebar.header(f"👤 {t('sidebar.player_trader_access')}")

    # Player level input
    player_level = st.sidebar.number_input(
        t("sidebar.player_level"),
        min_value=1,
        max_value=79,
        value=79,
        key="player_level",
        help=t("sidebar.player_level_help"),
        on_change=lambda: save_cookie("player_level", st.session_state.player_level),
    )

    # Flea market access - automatically disabled if player level < 15
    flea_unlocked = player_level >= 15
    if flea_unlocked:
        if "flea_available" not in st.session_state:
            st.session_state.flea_available = True

        flea_available = st.sidebar.checkbox(
            t("sidebar.flea_market_access"),
            value=True,
            key="flea_available",
            help=t("sidebar.flea_help"),
            on_change=lambda: save_cookie(
                "flea_available", st.session_state.flea_available
            ),
        )
    else:
        flea_available = False
        st.sidebar.checkbox(
            t("sidebar.flea_market_access"),
            value=False,
            disabled=True,
            help=t("sidebar.flea_unlocks_at_15"),
            key="flea_disabled_display",
        )
        st.sidebar.caption(f"⚠️ {t('sidebar.flea_unlocks_at_15')}")

    # Initialize trader levels defaults if not present
    for trader_key, _ in TRADERS:
        session_key = f"trader_{trader_key}"
        if session_key not in st.session_state:
            st.session_state[session_key] = 4

    # Build trader_levels dict from session state
    trader_levels = {key: st.session_state[f"trader_{key}"] for key, _ in TRADERS}

    # Individual trader levels in an expander
    with st.sidebar.expander(t("sidebar.trader_levels"), expanded=False):
        # Quick preset buttons
        preset_col1, preset_col2 = st.columns(2)
        if preset_col1.button(
            t("sidebar.all_ll1"), key="traders_ll1", width="stretch"
        ):
            for trader_key, _ in TRADERS:
                key = f"trader_{trader_key}"
                st.session_state[key] = 1
                save_cookie(key, 1)
            st.rerun()
        if preset_col2.button(
            t("sidebar.all_ll4"), key="traders_ll4", width="stretch"
        ):
            for trader_key, _ in TRADERS:
                key = f"trader_{trader_key}"
                st.session_state[key] = 4
                save_cookie(key, 4)
            st.rerun()

        st.markdown("---")

        # Individual trader sliders
        for trader_key, trader_name in TRADERS:
            session_key = f"trader_{trader_key}"
            trader_levels[trader_key] = st.select_slider(
                trader_name,
                options=[1, 2, 3, 4],
                value=st.session_state[session_key],
                key=session_key,
                on_change=lambda k=session_key: save_cookie(k, st.session_state[k]),
            )

    # Show summary of constraints
    non_maxed = [name for key, name in TRADERS if trader_levels.get(key, 4) < 4]
    if non_maxed or not flea_available or player_level < 79:
        constraints_info = []
        if player_level < 79:
            constraints_info.append(t("sidebar.level", level=player_level))
        if non_maxed:
            constraints_info.append(
                t("sidebar.traders_below_ll4", count=len(non_maxed))
            )
        if not flea_available:
            constraints_info.append(t("sidebar.no_flea"))
        st.sidebar.caption(
            f"⚠️ {t('sidebar.limited')}: {', '.join(constraints_info)}"
        )

    return player_level, flea_available, trader_levels
