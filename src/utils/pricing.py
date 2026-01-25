"""
Pricing and offer display utilities.
"""

from typing import Optional, Tuple


def get_best_offer_display(
    stats: dict,
    trader_levels: Optional[dict] = None,
    flea_available: bool = True,
    player_level: Optional[int] = None,
) -> Tuple[str, int]:
    """Get a formatted string showing the best AVAILABLE offer source with trader level.

    Respects trader level and player level constraints to show what the player
    would actually pay and where they'd buy from.

    Args:
        stats: Item stats dictionary containing offers
        trader_levels: Dict mapping trader name to level (1-4)
        flea_available: Whether flea market is enabled
        player_level: Current player level (affects flea access)

    Returns:
        Tuple of (source_display, price)
    """
    offers = stats.get("offers", [])
    if not offers:
        return stats.get("price_source", "market"), stats.get("price", 0)

    min_level_flea = stats.get("min_level_flea", 0) or 15

    # Filter to only available offers based on constraints
    available_offers = []
    for offer in offers:
        source = offer.get("source", "")
        if source == "fleaMarket":
            if not flea_available:
                continue
            if player_level is not None and min_level_flea > player_level:
                continue
            available_offers.append(offer)
        else:
            # Trader offer - check level requirement
            vendor = offer.get("vendor_normalized", "").lower()
            required_level = offer.get("trader_level")
            if trader_levels:
                player_trader_level = trader_levels.get(vendor, 4)
                if required_level is not None and required_level > player_trader_level:
                    continue
            available_offers.append(offer)

    if not available_offers:
        return "Unavailable", 0

    # Find the best (lowest price) available offer
    best_offer = min(available_offers, key=lambda x: x.get("price", float("inf")))
    price = best_offer.get("price", 0)
    source = best_offer.get("source", "")

    if source == "fleaMarket":
        return f"Flea Lv{min_level_flea}", price
    else:
        vendor_name = best_offer.get("vendor_name", source)
        trader_level = best_offer.get("trader_level")
        if trader_level:
            return f"{vendor_name} LL{trader_level}", price
        return vendor_name, price
