"""
Stats calculation logic.
"""


def calculate_total_stats(weapon_stats, selected_mods, item_lookup):
    """
    Calculate total weapon stats with selected mods.

    Stats are calculated from NAKED weapon + selected mods:
    - Ergonomics: naked + sum(mod ergonomics)  [flat addition]
    - Recoil: naked * (1 + sum(mod recoil_modifiers))  [percentage]
    - Weight: naked + sum(mod weights)  [flat addition]
    """
    total_ergo = weapon_stats["naked_ergonomics"]
    total_recoil_mod = 0.0
    total_price = 0
    total_weight = weapon_stats.get("weight", 0)  # Include base weapon weight

    for mod_id in selected_mods:
        if mod_id in item_lookup:
            stats = item_lookup[mod_id]["stats"]
            total_ergo += stats.get("ergonomics", 0)
            total_recoil_mod += stats.get("recoil_modifier", 0)
            total_price += stats.get("price", 0)
            total_weight += stats.get("weight", 0)

    # Apply recoil modifier to naked recoil
    recoil_multiplier = 1 + total_recoil_mod
    final_recoil_v = weapon_stats["naked_recoil_v"] * recoil_multiplier
    final_recoil_h = weapon_stats["naked_recoil_h"] * recoil_multiplier

    return {
        "ergonomics": total_ergo,
        "recoil_vertical": final_recoil_v,
        "recoil_horizontal": final_recoil_h,
        "recoil_multiplier": recoil_multiplier,
        "total_price": total_price,
        "total_weight": total_weight,
    }
