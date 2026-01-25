"""
Constraints sidebar components.
"""

from typing import Optional, Set, List

import streamlit as st

from i18n import t


def render_constraints(weapon_stats: dict) -> dict:
    """Render hard constraints section in sidebar.

    Args:
        weapon_stats: Weapon stats dictionary

    Returns:
        Dictionary of constraint values (None if not enabled)
    """
    st.sidebar.markdown("---")

    with st.sidebar.expander(f"🛡️ {t('optimize.constraints_header')}", expanded=False):
        # Budget constraint
        enable_budget = st.checkbox(t("constraints.budget_limit"), key="sb_budget_check")
        max_price = None
        if enable_budget:
            max_price = st.number_input(
                t("constraints.max_budget"),
                min_value=0,
                max_value=10000000,
                value=500000,
                step=50000,
                help=t("constraints.max_budget_help"),
                key="sb_max_price",
            )

        # Minimum ergonomics constraint
        enable_min_ergo = st.checkbox(
            t("constraints.min_ergonomics"), key="sb_ergo_check"
        )
        min_ergonomics = None
        if enable_min_ergo:
            min_ergonomics = st.slider(
                t("constraints.min_ergo"),
                min_value=0,
                max_value=100,
                value=50,
                help=t("constraints.min_ergo_help"),
                key="sb_min_ergo",
            )

        # Maximum recoil constraint
        enable_max_recoil = st.checkbox(
            t("constraints.max_recoil"), key="sb_recoil_check"
        )
        max_recoil_v = None
        if enable_max_recoil:
            naked_recoil = weapon_stats.get("naked_recoil_v", 100)
            max_recoil_v = st.slider(
                t("constraints.max_recoil_v"),
                min_value=20,
                max_value=int(naked_recoil),
                value=int(naked_recoil * 0.7),
                help=t("constraints.max_recoil_help", naked=f"{naked_recoil:.0f}"),
                key="sb_max_recoil",
            )

        # Minimum magazine capacity constraint
        enable_min_mag = st.checkbox(
            t("constraints.min_mag_capacity"), key="sb_mag_check"
        )
        min_mag_capacity = None
        if enable_min_mag:
            min_mag_capacity = st.number_input(
                t("constraints.min_mag"),
                min_value=5,
                max_value=100,
                value=30,
                step=5,
                help=t("constraints.min_mag_help"),
                key="sb_min_mag",
            )

        # Minimum sighting range constraint
        enable_min_sight = st.checkbox(
            t("constraints.min_sighting_range"), key="sb_sight_check"
        )
        min_sighting_range = None
        if enable_min_sight:
            min_sighting_range = st.number_input(
                t("constraints.min_sight"),
                min_value=50,
                max_value=1000,
                value=200,
                step=50,
                help=t("constraints.min_sight_help"),
                key="sb_min_sight",
            )

        # Maximum weight constraint
        enable_max_weight = st.checkbox(
            t("constraints.max_weight"), key="sb_weight_check"
        )
        max_weight = None
        if enable_max_weight:
            base_weight = weapon_stats.get("weight", 3.0)
            max_weight = st.number_input(
                t("constraints.max_weight_kg"),
                min_value=1.0,
                max_value=15.0,
                value=round(base_weight + 3.0, 1),
                step=0.5,
                format="%.1f",
                help=t("constraints.max_weight_help", base=f"{base_weight:.2f}"),
                key="sb_max_weight",
            )

    return {
        "max_price": max_price,
        "min_ergonomics": min_ergonomics,
        "max_recoil_v": max_recoil_v,
        "min_mag_capacity": min_mag_capacity,
        "min_sighting_range": min_sighting_range,
        "max_weight": max_weight,
    }


def render_include_exclude(mods: list, reachable_ids: set) -> dict:
    """Render include/exclude mods section in sidebar.

    Args:
        mods: List of all mods
        reachable_ids: Set of reachable item IDs for this weapon

    Returns:
        Dictionary with include/exclude categories and items
    """
    with st.sidebar.expander(f"➕/➖ {t('sidebar.include_exclude')}", expanded=False):
        # Filter by compatibility
        compatible_mods = [m for m in mods if m["id"] in reachable_ids]

        all_mod_names = sorted(
            list(set(m["name"] for m in compatible_mods if m.get("name")))
        )
        all_categories = sorted(
            list(
                set(
                    m.get("bsgCategory", {}).get("name")
                    for m in compatible_mods
                    if m.get("bsgCategory", {}).get("name")
                )
            )
        )
        mod_name_to_id = {m["name"]: m["id"] for m in compatible_mods if m.get("name")}

        sel_include_cats = st.multiselect(
            t("sidebar.require_categories"), all_categories, key="sb_inc_cat"
        )
        sel_exclude_cats = st.multiselect(
            t("sidebar.ban_categories"), all_categories, key="sb_exc_cat"
        )

        sel_include_items = st.multiselect(
            t("sidebar.require_items"), all_mod_names, key="sb_inc_item"
        )
        sel_exclude_items = st.multiselect(
            t("sidebar.ban_items"), all_mod_names, key="sb_exc_item"
        )

        # Convert to appropriate formats
        include_categories = (
            [[c] for c in sel_include_cats] if sel_include_cats else None
        )
        exclude_categories = set(sel_exclude_cats) if sel_exclude_cats else None
        include_items = (
            set(mod_name_to_id[n] for n in sel_include_items)
            if sel_include_items
            else None
        )
        exclude_items = (
            set(mod_name_to_id[n] for n in sel_exclude_items)
            if sel_exclude_items
            else None
        )

    return {
        "include_categories": include_categories,
        "exclude_categories": exclude_categories,
        "include_items": include_items,
        "exclude_items": exclude_items,
    }
