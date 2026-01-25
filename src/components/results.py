"""
Optimization results display component.
"""

from typing import Optional

import streamlit as st

from i18n import t
from src.core import calculate_total_stats
from src.utils import get_image_url
from src.costs import calculate_build_cost
from src.components.tables import display_mods_table


def display_optimization_results(
    result: dict,
    item_lookup: dict,
    weapon_stats: dict,
    presets: list,
    selected_gun: dict,
    constraints: Optional[dict] = None,
) -> bool:
    """Display optimization results.

    Args:
        result: Optimization result dictionary
        item_lookup: Item lookup dictionary
        weapon_stats: Weapon stats dictionary
        presets: List of available presets
        selected_gun: Selected gun data
        constraints: Optional constraints dictionary

    Returns:
        True if results were displayed, False if infeasible
    """
    if result["status"] == "infeasible":
        st.error(t("results.infeasible"))
        return False

    status_key = (
        f"results.{result['status']}"
        if result["status"] in ["optimal", "feasible"]
        else "results.feasible"
    )
    status_text = t(status_key)
    status_icon = "✅" if result["status"] == "optimal" else "⚠️"

    st.success(f"{status_icon} {t('results.optimization_status')} {status_text}")

    selected_items = result["selected_items"]
    selected_preset = result.get("selected_preset")

    # Calculate final stats
    final_stats = calculate_total_stats(weapon_stats, selected_items, item_lookup)

    # Display final stats
    st.subheader(t("results.final_stats"))
    col1, col2, col3, col4, col5 = st.columns(5)

    # Ergonomics
    raw_ergo = final_stats["ergonomics"]
    capped_ergo = max(0, min(100, raw_ergo))
    ergo_delta = raw_ergo - weapon_stats["naked_ergonomics"]
    with col1:
        st.metric(
            t("sidebar.ergonomics"),
            f"{capped_ergo:.1f}",
            f"{ergo_delta:+.1f}",
            help=f"Raw: {raw_ergo:.1f} (capped at 0-100). Delta from naked weapon.",
        )

    # Recoil Vertical
    recoil_v_delta = final_stats["recoil_vertical"] - weapon_stats["naked_recoil_v"]
    with col2:
        st.metric(
            t("sidebar.recoil_v"),
            f"{final_stats['recoil_vertical']:.1f}",
            f"{recoil_v_delta:+.1f}",
            delta_color="inverse",
            help="Delta from naked weapon.",
        )

    # Recoil Horizontal
    recoil_h_delta = final_stats["recoil_horizontal"] - weapon_stats["naked_recoil_h"]
    with col3:
        st.metric(
            t("sidebar.recoil_h"),
            f"{final_stats['recoil_horizontal']:.1f}",
            f"{recoil_h_delta:+.1f}",
            delta_color="inverse",
            help="Delta from naked weapon.",
        )

    # Total Weight
    with col4:
        st.metric(
            t("results.total_weight"),
            f"{final_stats['total_weight']:.2f} {t('units.kg')}",
            help=f"{t('results.base_weapon')}: {weapon_stats.get('weight', 0):.2f} {t('units.kg')}",
        )

    # Total Cost
    cost_data = calculate_build_cost(
        selected_items,
        selected_preset,
        item_lookup,
        weapon_stats,
        presets,
        selected_gun,
        fallback_base=result.get("fallback_base"),
    )

    total_cost = cost_data["total_cost"]
    delta_val = cost_data["mods_cost"]
    cost_composition = f"{cost_data['base_label']}: ₽{cost_data['base_cost']:,} + {t('results.additional_mods')}: ₽{cost_data['mods_cost']:,}"

    with col5:
        st.metric(
            t("results.total_build_cost"),
            f"₽{total_cost:,}",
            f"+₽{delta_val:,}",
            delta_color="off",
            help=f"{t('results.total_cost_help')}\n\n{cost_composition}",
        )

    # Display selected mods
    st.markdown("---")
    st.subheader(t("results.selected_build"))

    # Get preset info and items if preset was selected
    preset_info = None
    preset_item_ids = set()
    fallback_base = result.get("fallback_base")
    if selected_preset:
        preset_info = next((p for p in presets if p.get("id") == selected_preset), None)
        # If not found in purchasable presets, check all_presets (for fallback case)
        if not preset_info:
            all_presets = item_lookup[selected_gun["id"]].get("all_presets", [])
            preset_info = next(
                (p for p in all_presets if p.get("id") == selected_preset), None
            )
        if preset_info:
            preset_item_ids = set(preset_info.get("items", []))

    if selected_preset and preset_info:
        _display_preset_results(
            preset_info,
            preset_item_ids,
            selected_items,
            item_lookup,
            fallback_base,
            constraints,
        )
    elif selected_items:
        _display_naked_gun_results(
            selected_gun,
            weapon_stats,
            selected_items,
            item_lookup,
            fallback_base,
            constraints,
        )
    else:
        _display_no_mods_results(
            selected_gun,
            weapon_stats,
            fallback_base,
        )

    # Optimization details
    _display_optimization_details(result, final_stats, constraints)

    return True


def _display_preset_results(
    preset_info: dict,
    preset_item_ids: set,
    selected_items: list,
    item_lookup: dict,
    fallback_base: Optional[dict],
    constraints: Optional[dict],
) -> None:
    """Display results when a preset was selected."""
    is_fallback = fallback_base and fallback_base.get("type") == "preset"
    display_price = 0 if is_fallback else preset_info.get("price", 0)
    price_source = (
        "fallback (free)" if is_fallback else preset_info.get("price_source", "market")
    )

    st.markdown(f"**{t('results.preset')}:** {preset_info.get('name')}")
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(
            f"**{t('results.bundle_price')}:** ₽{display_price:,} ({price_source})"
        )
        st.markdown(f"**{t('results.includes')}:** {len(preset_item_ids)} items")
    with col2:
        if preset_info.get("image"):
            st.image(preset_info["image"], width=150)

    additional_items = [
        item_id for item_id in selected_items if item_id not in preset_item_ids
    ]
    if additional_items:
        st.markdown("---")
        st.markdown(f"**{t('results.additional_mods')}:**")
        display_mods_table(
            additional_items, item_lookup, show_price=True, constraints=constraints
        )

    with st.expander(
        t("results.items_in_preset", name=preset_info.get("name")), expanded=False
    ):
        display_mods_table(
            preset_item_ids, item_lookup, show_price=False, constraints=constraints
        )


def _display_naked_gun_results(
    selected_gun: dict,
    weapon_stats: dict,
    selected_items: list,
    item_lookup: dict,
    fallback_base: Optional[dict],
    constraints: Optional[dict],
) -> None:
    """Display results when building from naked gun."""
    st.markdown(f"**{t('results.naked_gun_mods')}**")

    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f"**{t('results.base_weapon')}:** {selected_gun['name']}")
        weapon_price = weapon_stats.get("price", 0)
        weapon_source = weapon_stats.get("price_source", "market")

        is_naked_fallback = fallback_base and fallback_base.get("type") == "naked"
        if is_naked_fallback:
            st.markdown(f"**{t('sidebar.price')}:** ₽0 (fallback - free)")
        elif weapon_source == "not_available":
            st.markdown(f"**{t('sidebar.price')}:** {t('results.not_available')}")
        else:
            st.markdown(f"**{t('sidebar.price')}:** ₽{weapon_price:,} ({weapon_source})")
    with col2:
        weapon_image_url = weapon_stats.get("default_preset_image") or get_image_url(
            selected_gun, prefer_high_res=True
        )
        if weapon_image_url:
            st.image(weapon_image_url, width=150)

    if selected_items:
        st.markdown("---")
        st.markdown(f"**{t('results.additional_mods')}:**")
        display_mods_table(
            selected_items, item_lookup, show_price=True, constraints=constraints
        )


def _display_no_mods_results(
    selected_gun: dict,
    weapon_stats: dict,
    fallback_base: Optional[dict],
) -> None:
    """Display results when no mods selected."""
    st.markdown(f"**{t('results.naked_gun')}**")
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f"**{t('results.base_weapon')}:** {selected_gun['name']}")
        weapon_price = weapon_stats.get("price", 0)
        weapon_source = weapon_stats.get("price_source", "market")

        is_naked_fallback = fallback_base and fallback_base.get("type") == "naked"
        if is_naked_fallback:
            st.markdown(f"**{t('sidebar.price')}:** ₽0 (fallback - free)")
        elif weapon_source == "not_available":
            st.markdown(f"**{t('sidebar.price')}:** {t('results.not_available')}")
        else:
            st.markdown(f"**{t('sidebar.price')}:** ₽{weapon_price:,} ({weapon_source})")

        st.info(t("results.no_mods_selected"))
    with col2:
        weapon_image_url = weapon_stats.get("default_preset_image") or get_image_url(
            selected_gun, prefer_high_res=True
        )
        if weapon_image_url:
            st.image(weapon_image_url, width=150)


def _display_optimization_details(
    result: dict,
    final_stats: dict,
    constraints: Optional[dict],
) -> None:
    """Display optimization details in expander."""
    with st.expander(t("results.optimization_details")):
        st.write(f"**{t('results.status')}:** {result['status']}")
        st.write(f"**{t('results.objective_value')}:** {result['objective_value']:.0f}")
        st.write(
            f"**{t('results.recoil_multiplier')}:** {final_stats['recoil_multiplier']:.4f}"
        )
        if constraints:
            if constraints.get("max_price"):
                st.write(
                    f"**{t('results.budget_constraint')}:** ₽{constraints['max_price']:,}"
                )
            if constraints.get("min_ergonomics"):
                st.write(
                    f"**{t('results.min_ergo_constraint')}:** {constraints['min_ergonomics']}"
                )
            if constraints.get("max_recoil_v"):
                st.write(
                    f"**{t('results.max_recoil_constraint')}:** {constraints['max_recoil_v']}"
                )
            if constraints.get("min_mag_capacity"):
                st.write(
                    f"**{t('results.min_mag_constraint')}:** {constraints['min_mag_capacity']} {t('units.rounds')}"
                )
            if constraints.get("min_sighting_range"):
                st.write(
                    f"**{t('results.min_sight_constraint')}:** {constraints['min_sighting_range']}{t('units.meters')}"
                )
            if constraints.get("max_weight"):
                st.write(
                    f"**{t('results.max_weight_constraint')}:** {constraints['max_weight']:.1f} {t('units.kg')}"
                )
            player_lvl = constraints.get("player_level")
            if player_lvl is not None:
                st.write(f"**{t('sidebar.player_level')}:** {player_lvl}")
            trader_lvls = constraints.get("trader_levels", {})
            flea = constraints.get("flea_available", True)
            if trader_lvls:
                levels_str = ", ".join(
                    [f"{k.capitalize()}: LL{v}" for k, v in trader_lvls.items()]
                )
                st.write(f"**{t('sidebar.trader_levels')}:** {levels_str}")

            flea_status = (
                t("results.available") if flea else t("results.not_available_short")
            )
            st.write(f"**{t('results.flea_market')}:** {flea_status}")
