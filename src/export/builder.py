"""
Build export generation (JSON and Markdown).
"""

from datetime import datetime
from typing import Optional, Tuple

from src.core import calculate_total_stats


def generate_build_export(
    result: dict,
    item_lookup: dict,
    weapon_stats: dict,
    presets: list,
    selected_gun: dict,
    constraints: Optional[dict] = None,
) -> Tuple[dict, str]:
    """Generate exportable build data in JSON and Markdown formats.

    Args:
        result: Optimization result dictionary
        item_lookup: Item lookup dictionary
        weapon_stats: Weapon stats dictionary
        presets: List of available presets
        selected_gun: Selected gun data
        constraints: Optional constraints dictionary

    Returns:
        Tuple of (json_data, markdown_text)
    """
    selected_items = result["selected_items"]
    selected_preset = result.get("selected_preset")
    fallback_base = result.get("fallback_base")
    final_stats = calculate_total_stats(weapon_stats, selected_items, item_lookup)

    # Calculate total cost
    total_cost = final_stats["total_price"]
    weapon_base_price = weapon_stats.get("price", 0)

    # Check if dummy price (unavailable)
    if weapon_base_price > 100_000_000:
        weapon_base_price = 0

    preset_info = None

    if selected_preset:
        preset_info = next(
            (p for p in presets if p.get("id") == selected_preset), None
        )
        # If not found in purchasable presets, check all_presets
        if not preset_info:
            weapon_id = selected_gun["id"]
            all_presets = item_lookup[weapon_id].get("all_presets", [])
            preset_info = next(
                (p for p in all_presets if p.get("id") == selected_preset), None
            )
        if preset_info:
            preset_items = set(preset_info.get("items", []))
            individual_cost = sum(
                [
                    item_lookup[item_id]["stats"].get("price", 0)
                    for item_id in selected_items
                    if item_id not in preset_items and item_id in item_lookup
                ]
            )
            # Use price=0 if this is a fallback preset
            preset_price = (
                0
                if (fallback_base and fallback_base.get("type") == "preset")
                else preset_info.get("price", 0)
            )
            total_cost = preset_price + individual_cost
    else:
        total_cost = weapon_base_price + final_stats["total_price"]

    # Build JSON export
    export_preset_price = None
    if preset_info:
        export_preset_price = (
            0
            if (fallback_base and fallback_base.get("type") == "preset")
            else preset_info.get("price")
        )

    json_data = {
        "exported_at": datetime.now().isoformat(),
        "weapon": {
            "id": selected_gun["id"],
            "name": selected_gun["name"],
            "base_price": weapon_base_price,
        },
        "preset": (
            {
                "id": preset_info["id"] if preset_info else None,
                "name": preset_info["name"] if preset_info else None,
                "price": export_preset_price,
                "is_fallback": bool(
                    fallback_base and fallback_base.get("type") == "preset"
                ),
            }
            if selected_preset
            else None
        ),
        "mods": [
            {
                "id": item_id,
                "name": item_lookup[item_id]["data"]["name"],
                "ergonomics": item_lookup[item_id]["stats"].get("ergonomics", 0),
                "recoil_modifier": item_lookup[item_id]["stats"].get(
                    "recoil_modifier", 0
                ),
                "price": item_lookup[item_id]["stats"].get("price", 0),
            }
            for item_id in selected_items
            if item_id in item_lookup
        ],
        "final_stats": {
            "ergonomics": round(final_stats["ergonomics"], 1),
            "recoil_vertical": round(final_stats["recoil_vertical"], 1),
            "recoil_horizontal": round(final_stats["recoil_horizontal"], 1),
            "recoil_multiplier": round(final_stats["recoil_multiplier"], 4),
            "total_weight": round(final_stats["total_weight"], 2),
            "total_cost": total_cost,
        },
        "constraints": constraints,
        "optimization_status": result["status"],
    }

    # Build Markdown export
    md_lines = [
        f"# {selected_gun['name']} Build",
        f"*Exported: {datetime.now().strftime('%Y-%m-%d %H:%M')}*",
        "",
        "## Final Stats",
        "| Stat | Value |",
        "|------|-------|",
        f"| Ergonomics | {min(100, max(0, final_stats['ergonomics'])):.1f} |",
        f"| Recoil V | {final_stats['recoil_vertical']:.1f} |",
        f"| Recoil H | {final_stats['recoil_horizontal']:.1f} |",
        f"| Weight | {final_stats['total_weight']:.2f} kg |",
        f"| Total Cost | ₽{total_cost:,} |",
        "",
    ]

    if selected_preset and preset_info:
        md_preset_price = (
            0
            if (fallback_base and fallback_base.get("type") == "preset")
            else preset_info.get("price", 0)
        )
        fallback_note = (
            " (fallback - free)"
            if (fallback_base and fallback_base.get("type") == "preset")
            else ""
        )
        md_lines.extend(
            [
                "## Base Preset",
                f"**{preset_info['name']}** - ₽{md_preset_price:,}{fallback_note}",
                "",
            ]
        )

    # Additional mods
    additional_mods = selected_items
    if preset_info:
        preset_items = set(preset_info.get("items", []))
        additional_mods = [m for m in selected_items if m not in preset_items]

    if additional_mods:
        md_lines.extend(
            [
                "## Modifications",
                "| Name | Ergo | Recoil | Price |",
                "|------|------|--------|-------|",
            ]
        )
        for item_id in additional_mods:
            if item_id in item_lookup:
                item = item_lookup[item_id]
                name = item["data"]["name"]
                ergo = item["stats"].get("ergonomics", 0)
                recoil = item["stats"].get("recoil_modifier", 0) * 100
                price = item["stats"].get("price", 0)
                md_lines.append(
                    f"| {name} | {ergo:+.1f} | {recoil:+.1f}% | ₽{price:,} |"
                )
        md_lines.append("")

    if constraints:
        md_lines.extend(["## Constraints Used"])
        if constraints.get("max_price"):
            md_lines.append(f"- Budget: ₽{constraints['max_price']:,}")
        if constraints.get("min_ergonomics"):
            md_lines.append(f"- Min Ergonomics: {constraints['min_ergonomics']}")
        if constraints.get("max_recoil_v"):
            md_lines.append(f"- Max Recoil V: {constraints['max_recoil_v']}")
        if constraints.get("min_mag_capacity"):
            md_lines.append(
                f"- Min Mag Capacity: {constraints['min_mag_capacity']} rounds"
            )
        if constraints.get("min_sighting_range"):
            md_lines.append(
                f"- Min Sighting Range: {constraints['min_sighting_range']}m"
            )
        if constraints.get("max_weight"):
            md_lines.append(f"- Max Weight: {constraints['max_weight']:.1f} kg")
        player_lvl = constraints.get("player_level")
        if player_lvl is not None:
            md_lines.append(f"- Player Level: {player_lvl}")
        trader_lvls = constraints.get("trader_levels", {})
        flea = constraints.get("flea_available", True)
        if trader_lvls:
            for trader, level in trader_lvls.items():
                md_lines.append(f"- {trader.capitalize()}: LL{level}")
        md_lines.append(f"- Flea Market: {'Yes' if flea else 'No'}")

    markdown_text = "\n".join(md_lines)

    return json_data, markdown_text
