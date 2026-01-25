"""
Pareto frontier exploration logic.
"""

from loguru import logger
from src.core.stats import calculate_total_stats
from .solver import optimize_weapon


def _build_frontier_point(stats, result):
    """Helper to build a frontier point dict."""
    return {
        "ergo": int(stats["ergonomics"]),
        "recoil_pct": round((stats["recoil_multiplier"] - 1) * 100, 1),
        "recoil_v": round(stats["recoil_vertical"], 1),
        "recoil_h": round(stats["recoil_horizontal"], 1),
        "price": int(stats["total_price"]),
        "selected_items": result["selected_items"],
        "selected_preset": result.get("selected_preset"),
        "status": result["status"],
    }


def explore_pareto(
    weapon_id, item_lookup, compatibility_map,
    ignore="price",
    max_price=None,
    min_ergonomics=None,
    max_recoil_v=None,
    min_mag_capacity=None,
    min_sighting_range=None,
    max_weight=None,
    include_items=None,
    exclude_items=None,
    include_categories=None,
    exclude_categories=None,
    steps=10,
    trader_levels=None,
    flea_available=True,
    player_level=None
):
    """
    Explore the Pareto frontier between two dimensions, ignoring the third.
    """
    weapon_stats = item_lookup[weapon_id]["stats"]
    naked_recoil_v = weapon_stats.get("naked_recoil_v", 100)

    logger.info(f"Exploring Pareto frontier (ignore={ignore}, steps={steps})")
    frontier = []

    constraint_kwargs = {
        "trader_levels": trader_levels,
        "flea_available": flea_available,
        "player_level": player_level,
        "min_mag_capacity": min_mag_capacity,
        "min_sighting_range": min_sighting_range,
        "max_weight": max_weight,
        "include_items": include_items,
        "exclude_items": exclude_items,
        "include_categories": include_categories,
        "exclude_categories": exclude_categories,
    }

    RECOIL_WEIGHTS = {"ergo_weight": 0, "recoil_weight": 1, "price_weight": 0}
    ERGO_WEIGHTS = {"ergo_weight": 1, "recoil_weight": 0, "price_weight": 0}
    PRICE_WEIGHTS = {"ergo_weight": 0, "recoil_weight": 0, "price_weight": 1}

    if ignore == "price":
        result_low = optimize_weapon(
            weapon_id, item_lookup, compatibility_map,
            max_price=max_price, max_recoil_v=max_recoil_v,
            **RECOIL_WEIGHTS, **constraint_kwargs
        )
        result_high = optimize_weapon(
            weapon_id, item_lookup, compatibility_map,
            max_price=max_price, max_recoil_v=max_recoil_v,
            **ERGO_WEIGHTS, **constraint_kwargs
        )

        if result_low["status"] == "infeasible": return []

        stats_low = calculate_total_stats(weapon_stats, result_low["selected_items"], item_lookup)
        range_min = int(stats_low["ergonomics"])

        if result_high["status"] != "infeasible":
            stats_high = calculate_total_stats(weapon_stats, result_high["selected_items"], item_lookup)
            range_max = int(stats_high["ergonomics"])
        else:
            range_max = 100

        if min_ergonomics is not None:
            range_min = max(range_min, min_ergonomics)

        range_min = max(0, range_min)
        range_max = min(100, range_max)
        if range_max <= range_min: range_max = range_min + 1
        step_size = (range_max - range_min) / (steps - 1) if steps > 1 else 0

        for i in range(steps):
            target = int(range_min + i * step_size)
            result = optimize_weapon(
                weapon_id, item_lookup, compatibility_map,
                max_price=max_price, min_ergonomics=target, max_recoil_v=max_recoil_v,
                **RECOIL_WEIGHTS, **constraint_kwargs
            )
            if result["status"] != "infeasible":
                stats = calculate_total_stats(weapon_stats, result["selected_items"], item_lookup)
                frontier.append(_build_frontier_point(stats, result))

    elif ignore == "recoil":
        result_low = optimize_weapon(
            weapon_id, item_lookup, compatibility_map,
            max_price=max_price, max_recoil_v=max_recoil_v,
            **PRICE_WEIGHTS, **constraint_kwargs
        )
        result_high = optimize_weapon(
            weapon_id, item_lookup, compatibility_map,
            max_price=max_price, max_recoil_v=max_recoil_v,
            **ERGO_WEIGHTS, **constraint_kwargs
        )

        if result_low["status"] == "infeasible": return []

        stats_low = calculate_total_stats(weapon_stats, result_low["selected_items"], item_lookup)
        range_min = int(stats_low["ergonomics"])

        if result_high["status"] != "infeasible":
            stats_high = calculate_total_stats(weapon_stats, result_high["selected_items"], item_lookup)
            range_max = int(stats_high["ergonomics"])
        else:
            range_max = 100

        if min_ergonomics is not None:
            range_min = max(range_min, min_ergonomics)

        range_min = max(0, range_min)
        range_max = min(100, range_max)
        if range_max <= range_min: range_max = range_min + 1
        step_size = (range_max - range_min) / (steps - 1) if steps > 1 else 0

        for i in range(steps):
            target = int(range_min + i * step_size)
            result = optimize_weapon(
                weapon_id, item_lookup, compatibility_map,
                max_price=max_price, min_ergonomics=target, max_recoil_v=max_recoil_v,
                **PRICE_WEIGHTS, **constraint_kwargs
            )
            if result["status"] != "infeasible":
                stats = calculate_total_stats(weapon_stats, result["selected_items"], item_lookup)
                frontier.append(_build_frontier_point(stats, result))

    elif ignore == "ergo":
        result_low = optimize_weapon(
            weapon_id, item_lookup, compatibility_map,
            max_price=max_price, min_ergonomics=min_ergonomics,
            **RECOIL_WEIGHTS, **constraint_kwargs
        )
        result_high = optimize_weapon(
            weapon_id, item_lookup, compatibility_map,
            max_price=max_price, min_ergonomics=min_ergonomics,
            **PRICE_WEIGHTS, **constraint_kwargs
        )

        if result_low["status"] == "infeasible": return []

        stats_low = calculate_total_stats(weapon_stats, result_low["selected_items"], item_lookup)
        range_min = stats_low["recoil_vertical"]

        if result_high["status"] != "infeasible":
            stats_high = calculate_total_stats(weapon_stats, result_high["selected_items"], item_lookup)
            range_max = stats_high["recoil_vertical"]
        else:
            range_max = naked_recoil_v

        if max_recoil_v is not None:
            range_max = min(range_max, max_recoil_v)

        if range_max <= range_min: range_max = range_min + 1
        step_size = (range_max - range_min) / (steps - 1) if steps > 1 else 0

        for i in range(steps):
            target = range_min + i * step_size
            result = optimize_weapon(
                weapon_id, item_lookup, compatibility_map,
                max_price=max_price, min_ergonomics=min_ergonomics, max_recoil_v=target,
                **PRICE_WEIGHTS, **constraint_kwargs
            )
            if result["status"] != "infeasible":
                stats = calculate_total_stats(weapon_stats, result["selected_items"], item_lookup)
                frontier.append(_build_frontier_point(stats, result))

    seen = set()
    unique_frontier = []
    for point in frontier:
        key = (point["ergo"], point["recoil_v"], point["price"])
        if key not in seen:
            seen.add(key)
            unique_frontier.append(point)

    logger.info(f"Pareto frontier exploration complete: {len(unique_frontier)} unique points")
    return unique_frontier
