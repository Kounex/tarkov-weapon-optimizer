"""
Weapon selector sidebar component.
"""

from typing import List, Tuple

import streamlit as st

from i18n import t
from src.utils import get_image_url, get_caliber_display, get_category_name


def render_weapon_selector(
    guns: list,
    item_lookup: dict,
) -> Tuple[dict, dict, list, set]:
    """Render the weapon selection sidebar.

    Args:
        guns: List of all guns
        item_lookup: Item lookup dictionary

    Returns:
        Tuple of (selected_gun, weapon_stats, presets, reachable_ids)
    """
    st.sidebar.header(f"🔫 {t('sidebar.select_weapon')}")

    # Build category counts and caliber counts for all guns
    all_category_counts = {}
    all_caliber_counts = {}
    for gun in guns:
        cat = get_category_name(gun)
        cal = get_caliber_display(gun)
        if cat:
            all_category_counts[cat] = all_category_counts.get(cat, 0) + 1
        if cal:
            all_caliber_counts[cal] = all_caliber_counts.get(cal, 0) + 1

    # Initialize session state for cascading filters
    if "selected_types" not in st.session_state:
        st.session_state.selected_types = []
    if "selected_calibers" not in st.session_state:
        st.session_state.selected_calibers = []

    # Calculate available calibers based on selected types
    if st.session_state.selected_types:
        guns_of_selected_types = [
            g for g in guns if get_category_name(g) in st.session_state.selected_types
        ]
    else:
        guns_of_selected_types = guns

    available_caliber_counts = {}
    for gun in guns_of_selected_types:
        cal = get_caliber_display(gun)
        if cal:
            available_caliber_counts[cal] = available_caliber_counts.get(cal, 0) + 1

    # Calculate available categories based on selected calibers
    if st.session_state.selected_calibers:
        guns_of_selected_calibers = [
            g for g in guns if get_caliber_display(g) in st.session_state.selected_calibers
        ]
    else:
        guns_of_selected_calibers = guns

    available_category_counts = {}
    for gun in guns_of_selected_calibers:
        cat = get_category_name(gun)
        if cat:
            available_category_counts[cat] = available_category_counts.get(cat, 0) + 1

    # Build options with counts
    category_options = sorted(available_category_counts.keys())
    caliber_options = sorted(available_caliber_counts.keys())

    def format_with_count(option, counts):
        count = counts.get(option, 0)
        return f"{option} ({count})"

    # Gun type filter
    selected_types = st.sidebar.multiselect(
        t("sidebar.filter_gun_type"),
        category_options,
        default=st.session_state.selected_types,
        format_func=lambda x: format_with_count(x, available_category_counts),
        placeholder="All",
        key="type_filter",
    )

    # Update session state and rerun if changed
    if selected_types != st.session_state.selected_types:
        st.session_state.selected_types = selected_types
        if selected_types:
            new_available_calibers = set()
            for gun in guns:
                if get_category_name(gun) in selected_types:
                    cal = get_caliber_display(gun)
                    if cal:
                        new_available_calibers.add(cal)
            st.session_state.selected_calibers = [
                c for c in st.session_state.selected_calibers if c in new_available_calibers
            ]
        st.rerun()

    # Caliber filter
    selected_calibers = st.sidebar.multiselect(
        t("sidebar.filter_caliber"),
        caliber_options,
        default=st.session_state.selected_calibers,
        format_func=lambda x: format_with_count(x, available_caliber_counts),
        placeholder="All",
        key="caliber_filter",
    )

    # Update session state and rerun if changed
    if selected_calibers != st.session_state.selected_calibers:
        st.session_state.selected_calibers = selected_calibers
        if selected_calibers:
            new_available_types = set()
            for gun in guns:
                if get_caliber_display(gun) in selected_calibers:
                    cat = get_category_name(gun)
                    if cat:
                        new_available_types.add(cat)
            st.session_state.selected_types = [
                t for t in st.session_state.selected_types if t in new_available_types
            ]
        st.rerun()

    # Filter guns
    filtered_guns = guns
    if selected_types:
        filtered_guns = [g for g in filtered_guns if get_category_name(g) in selected_types]
    if selected_calibers:
        filtered_guns = [
            g for g in filtered_guns if get_caliber_display(g) in selected_calibers
        ]

    gun_options = {gun["name"]: gun for gun in filtered_guns}
    gun_names = sorted(gun_options.keys())

    # Show match count
    st.sidebar.caption(f"Showing {len(gun_names)} of {len(guns)} weapons")

    if not gun_names:
        st.sidebar.warning("No weapons match the selected filters.")
        st.stop()

    selected_gun_name = st.sidebar.selectbox(
        t("sidebar.choose_weapon"),
        gun_names,
        help=t("sidebar.choose_weapon"),
    )

    selected_gun = gun_options[selected_gun_name]
    weapon_id = selected_gun["id"]
    weapon_stats = item_lookup[weapon_id]["stats"]
    presets = item_lookup[weapon_id]["presets"]

    return selected_gun, weapon_stats, presets


def render_weapon_details(
    selected_gun: dict,
    weapon_stats: dict,
    presets: list,
) -> None:
    """Render weapon details in sidebar (image, base stats, presets).

    Args:
        selected_gun: Selected gun data
        weapon_stats: Weapon stats dictionary
        presets: List of available presets
    """
    # Display weapon image
    weapon_image_url = weapon_stats.get("default_preset_image") or get_image_url(
        selected_gun, prefer_high_res=True
    )
    if weapon_image_url:
        st.sidebar.image(weapon_image_url, width="stretch")

    # Show base weapon stats in expander
    with st.sidebar.expander(f"📊 {t('sidebar.base_stats')}", expanded=False):
        st.markdown(
            f"- {t('sidebar.ergonomics')}: {weapon_stats.get('naked_ergonomics', 0):.0f}"
        )
        st.markdown(
            f"- {t('sidebar.recoil_v')}: {weapon_stats.get('naked_recoil_v', 0):.0f}"
        )
        st.markdown(
            f"- {t('sidebar.recoil_h')}: {weapon_stats.get('naked_recoil_h', 0):.0f}"
        )

        # Caliber and fire rate
        caliber = weapon_stats.get("caliber", "")
        if caliber:
            caliber_display = caliber.replace("Caliber", "").strip()
            st.markdown(f"- {t('sidebar.caliber')}: {caliber_display}")

        fire_rate = weapon_stats.get("fire_rate", 0)
        if fire_rate:
            st.markdown(f"- {t('sidebar.fire_rate')}: {fire_rate} RPM")

        fire_modes = weapon_stats.get("fire_modes", [])
        if fire_modes:
            modes_display = ", ".join(fire_modes)
            st.markdown(f"- {t('sidebar.fire_modes')}: {modes_display}")

        # Weapon handling properties
        st.markdown("---")
        camera_snap = weapon_stats.get("camera_snap", 0)
        if camera_snap:
            st.markdown(f"- {t('sidebar.camera_snap')}: {camera_snap}")

        center_of_impact = weapon_stats.get("center_of_impact", 0)
        if center_of_impact:
            st.markdown(f"- {t('sidebar.center_of_impact')}: {center_of_impact}")

        deviation_max = weapon_stats.get("deviation_max", 0)
        if deviation_max:
            st.markdown(f"- {t('sidebar.deviation_max')}: {deviation_max}")

        deviation_curve = weapon_stats.get("deviation_curve", 0)
        if deviation_curve:
            st.markdown(f"- {t('sidebar.deviation_curve')}: {deviation_curve}")

        recoil_angle = weapon_stats.get("recoil_angle", 0)
        if recoil_angle:
            st.markdown(f"- {t('sidebar.recoil_angle')}: {recoil_angle}°")

        recoil_dispersion = weapon_stats.get("recoil_dispersion", 0)
        if recoil_dispersion:
            st.markdown(f"- {t('sidebar.recoil_dispersion')}: {recoil_dispersion}")

    # Show all presets info
    if presets:
        with st.sidebar.expander(
            f"📦 {t('sidebar.available_presets')} ({len(presets)})"
        ):
            for preset in presets:
                preset_name = preset.get("name", "Unknown")
                preset_price = preset.get("price", 0)
                preset_items = preset.get("items", [])
                st.markdown(f"**{preset_name}**")
                st.markdown(f"  - Price: ₽{preset_price:,}")
                st.markdown(f"  - Items: {len(preset_items)}")
                st.markdown("---")
