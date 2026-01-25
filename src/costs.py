"""
Build cost calculation utilities.
"""

from typing import Optional, List, Set

from i18n import t


def calculate_build_cost(
    selected_items: List[str],
    selected_preset: Optional[str],
    item_lookup: dict,
    weapon_stats: dict,
    presets: list,
    selected_gun: dict,
    fallback_base: Optional[dict] = None,
) -> dict:
    """Calculate the actual total build cost and details.

    Args:
        selected_items: List of selected mod item IDs
        selected_preset: Selected preset ID (if any)
        item_lookup: Item lookup dictionary
        weapon_stats: Weapon stats dictionary
        presets: List of available presets
        selected_gun: Selected gun data
        fallback_base: Optional fallback base info

    Returns:
        Dict with: 'total_cost', 'base_cost', 'mods_cost', 'base_label'
    """
    weapon_base_price = weapon_stats.get("price", 0)

    # Check if dummy price (unavailable) - don't include in total
    if weapon_base_price > 100_000_000:
        weapon_base_price = 0

    base_cost = 0
    mods_cost = 0
    base_label = t("results.base_weapon")

    if selected_preset:
        # Find the preset info
        preset_info = next((p for p in presets if p.get("id") == selected_preset), None)

        # If not found in purchasable presets, check all_presets
        if not preset_info:
            all_presets = item_lookup.get(selected_gun["id"], {}).get("all_presets", [])
            preset_info = next(
                (p for p in all_presets if p.get("id") == selected_preset), None
            )

        if preset_info:
            preset_items = set(preset_info.get("items", []))

            # Handle fallback base logic if provided
            is_fallback = fallback_base and fallback_base.get("type") == "preset"
            preset_price = 0 if is_fallback else preset_info.get("price", 0)

            base_cost = preset_price
            base_label = t("results.preset")

            # Only count cost of items NOT in the preset
            mods_cost = sum(
                item_lookup[item_id]["stats"].get("price", 0)
                for item_id in selected_items
                if item_id not in preset_items and item_id in item_lookup
            )

            return {
                "total_cost": base_cost + mods_cost,
                "base_cost": base_cost,
                "mods_cost": mods_cost,
                "base_label": base_label,
            }

    # No preset - naked gun + all mod prices
    base_cost = weapon_base_price
    mods_cost = sum(
        item_lookup[item_id]["stats"].get("price", 0)
        for item_id in selected_items
        if item_id in item_lookup
    )

    return {
        "total_cost": base_cost + mods_cost,
        "base_cost": base_cost,
        "mods_cost": mods_cost,
        "base_label": base_label,
    }
