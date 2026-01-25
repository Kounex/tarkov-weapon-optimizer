"""
Data extraction and normalization from API results.
"""

from loguru import logger


def has_valid_price(item):
    """Check if an item has at least one valid buyFor offer with price > 0."""
    buy_for = item.get("buyFor", []) or []
    for offer in buy_for:
        if isinstance(offer, dict) and (offer.get("priceRUB") or 0) > 0:
            return True
    return False


def extract_conflicting_items(item):
    """Extract list of conflicting item IDs from an item."""
    conflicts = item.get("conflictingItems", [])
    if conflicts:
        return [c["id"] for c in conflicts if isinstance(c, dict) and "id" in c]
    return []


def extract_slots(gun):
    """Extract slot information from a gun."""
    slots = []
    props = gun.get("properties")
    if props and props.get("slots"):
        for slot in props["slots"]:
            allowed_ids = []
            # filters is a dict with allowedItems key, not a list
            filters = slot.get("filters")
            if filters and isinstance(filters, dict):
                allowed_items = filters.get("allowedItems", [])
                for item in allowed_items:
                    if isinstance(item, dict):
                        allowed_ids.append(item["id"])
                    elif isinstance(item, str):
                        allowed_ids.append(item)
            slots.append(
                {
                    "id": slot["id"],
                    "name": slot["name"],
                    "nameId": slot["nameId"],
                    "required": slot.get("required", False),
                    "allowedItems": allowed_ids,
                }
            )
    return slots


def extract_slots_from_mod(mod):
    """Extract slot information from a mod."""
    slots = []
    props = mod.get("properties")
    if props and props.get("slots"):
        for slot in props["slots"]:
            allowed_ids = []
            # filters is a dict with allowedItems key, not a list
            filters = slot.get("filters")
            if filters and isinstance(filters, dict):
                allowed_items = filters.get("allowedItems", [])
                for item in allowed_items:
                    if isinstance(item, dict):
                        allowed_ids.append(item["id"])
                    elif isinstance(item, str):
                        allowed_ids.append(item)
            slots.append(
                {
                    "id": slot["id"],
                    "name": slot["name"],
                    "nameId": slot["nameId"],
                    "required": slot.get("required", False),
                    "allowedItems": allowed_ids,
                }
            )
    return slots


def extract_all_presets(gun, include_unpurchasable=False):
    """Extract all preset information from a gun.

    Returns a list of presets, each containing:
    - id: preset identifier
    - name: preset display name
    - items: list of item IDs in this preset
    - image: preset image URL
    - price: lowest price (0 if not purchasable)
    - price_source: lowest price source
    - offers: list of all offers with trader level info
    - purchasable: whether this preset can be bought (price > 0)

    Args:
        gun: Gun data from API
        include_unpurchasable: If True, include presets with no purchase offers
    """
    props = gun.get("properties", {}) or {}
    presets_data = props.get("presets", []) or []

    if not presets_data:
        return []

    presets = []
    for preset in presets_data:
        if not isinstance(preset, dict):
            continue

        # Extract items from containsItems
        preset_items = []
        contains_items = preset.get("containsItems", []) or []
        for contained in contains_items:
            if isinstance(contained, dict) and "item" in contained:
                item = contained["item"]
                if isinstance(item, dict) and "id" in item:
                    preset_items.append(item["id"])

        # Get image (prefer high-res images)
        preset_image = (
            preset.get("image512pxLink")
            or preset.get("imageLink")
            or preset.get("image8xLink")
            or preset.get("gridImageLink")
            or preset.get("baseImageLink")
        )

        # Extract all buyFor offers
        buy_for = preset.get("buyFor", []) or []
        offers = []
        for offer in buy_for:
            if not isinstance(offer, dict): continue
            price = offer.get("priceRUB") or 0
            if price <= 0: continue

            source = offer.get("source", "")
            vendor = offer.get("vendor", {}) or {}

            trader_level = None
            if source != "fleaMarket":
                trader_level = vendor.get("minTraderLevel") or 1

            offers.append({
                "price": price,
                "source": source,
                "vendor_name": vendor.get("name", ""),
                "vendor_normalized": vendor.get("normalizedName", ""),
                "trader_level": trader_level,
            })

        offers.sort(key=lambda x: x["price"])

        lowest_price = 0
        price_source = "not_available"

        if offers:
            lowest_price = offers[0]["price"]
            price_source = offers[0]["source"]

        purchasable = lowest_price > 0

        # Include preset if purchasable OR if we want all presets
        if purchasable or include_unpurchasable:
            presets.append(
                {
                    "id": preset.get("id", ""),
                    "name": preset.get("name", "") or preset.get("shortName", "Unknown"),
                    "items": preset_items,
                    "image": preset_image,
                    "price": lowest_price,
                    "price_source": price_source,
                    "offers": offers,
                    "purchasable": purchasable,
                }
            )

    return presets


def extract_gun_stats(gun):
    """Extract and normalize stats from a gun.

    Uses NAKED weapon stats (no attachments):
    - ergonomics: base weapon ergonomics without any mods
    - recoilVertical/recoilHorizontal: base recoil without any mods

    The defaultErgonomics/defaultRecoil* values include the default preset,
    so we explicitly use the non-default values for naked stats.

    Also extracts the defaultPreset grid image for display purposes.
    """
    props = gun.get("properties", {}) or {}

    # Get lowest buyFor price (TRADER ONLY, exclude flea market) - this is for the naked weapon
    buy_for = gun.get("buyFor", []) or []
    lowest_price = 0
    price_source = "basePrice"
    if buy_for:
        # Filter out flea market offers - only use trader prices
        trader_offers = [offer for offer in buy_for if isinstance(offer, dict) and offer.get("source") != "fleaMarket"]
        if trader_offers:
            min_offer = min(trader_offers, key=lambda x: x.get("priceRUB", float("inf")))
            lowest_price = min_offer.get("priceRUB", 0) or 0
            price_source = min_offer.get("source", "market") or "market"

    # If naked gun has no trader offers, set prohibitively high price
    # This forces optimizer to use presets instead of building from naked gun
    if lowest_price == 0:
        # Check if there are any purchasable presets
        presets_data = props.get("presets", []) or []
        has_preset = False
        for preset in presets_data:
            if not isinstance(preset, dict):
                continue
            preset_buy_for = preset.get("buyFor", []) or []
            preset_trader_offers = [
                offer for offer in preset_buy_for
                if isinstance(offer, dict) and offer.get("source") != "fleaMarket"
            ]
            if preset_trader_offers:
                has_preset = True
                break

        if has_preset:
            # Naked gun not purchasable, but presets exist
            # Set very high price to discourage naked gun builds
            lowest_price = 999999999  # Prohibitively expensive
            price_source = "not_available"
        else:
            # No presets and no trader offers - gun is not purchasable
            # Note: We intentionally do NOT fall back to basePrice - only real market prices
            lowest_price = 999999999  # Prohibitively expensive
            price_source = "not_available"

    # Get defaultPreset image
    default_preset = props.get("defaultPreset", {}) or {}
    default_preset_image = (
        default_preset.get("image512pxLink")
        or default_preset.get("imageLink")
        or default_preset.get("image8xLink")
        or default_preset.get("gridImageLink")
        or default_preset.get("gridImageLinkFallback")
        or default_preset.get("iconLink")
        or default_preset.get("iconLinkFallback")
    )

    return {
        # Naked weapon stats (without default preset)
        "naked_ergonomics": props.get("ergonomics", 0) or 0,
        "naked_recoil_v": props.get("recoilVertical", 0) or 0,
        "naked_recoil_h": props.get("recoilHorizontal", 0) or 0,
        # Default preset stats (for reference/display)
        "default_ergonomics": props.get("defaultErgonomics", 0) or 0,
        "default_recoil_v": props.get("defaultRecoilVertical", 0) or 0,
        "default_recoil_h": props.get("defaultRecoilHorizontal", 0) or 0,
        # Default preset image (grid image)
        "default_preset_image": default_preset_image,
        # Other stats
        "accuracy_modifier": gun.get("accuracyModifier", 0) or 0,
        "fire_rate": props.get("fireRate", 0) or 0,
        "fire_modes": props.get("fireModes", []) or [],
        "caliber": props.get("caliber", ""),
        "weight": gun.get("weight", 0) or 0,
        "width": gun.get("width", 0) or 0,
        "height": gun.get("height", 0) or 0,
        "sighting_range": props.get("sightingRange") or 0,
        # Category
        "category": gun.get("bsgCategory", {}).get("name", "") if gun.get("bsgCategory") else "",
        "category_id": gun.get("bsgCategory", {}).get("id", "") if gun.get("bsgCategory") else "",
        # Weapon handling properties
        "camera_snap": props.get("cameraSnap", 0) or 0,
        "center_of_impact": props.get("centerOfImpact", 0) or 0,
        "deviation_max": props.get("deviationMax", 0) or 0,
        "deviation_curve": props.get("deviationCurve", 0) or 0,
        "recoil_angle": props.get("recoilAngle", 0) or 0,
        "recoil_dispersion": props.get("recoilDispersion", 0) or 0,
        # Price info (naked weapon price, not including preset)
        "price": lowest_price,
        "price_source": price_source,
    }


def extract_mod_stats(mod):
    """Extract and normalize stats from a mod.

    Note: Top-level recoilModifier is an integer percentage (e.g., -5 = -5%)
          Properties recoilModifier is a decimal (e.g., -0.05 = -5%)
          We normalize to decimal format for calculations.
    """
    props = mod.get("properties", {}) or {}

    # Get ergonomics (flat value)
    ergo = mod.get("ergonomicsModifier", 0) or 0

    # Get recoil modifier - top level is integer %, props is decimal
    top_recoil = mod.get("recoilModifier", 0) or 0
    props_recoil = props.get("recoilModifier", 0) or 0

    # Normalize: if top_recoil is integer format (e.g., -5), convert to decimal (-0.05)
    if props_recoil != 0:
        recoil_mod = props_recoil  # Already decimal
    elif top_recoil != 0:
        recoil_mod = top_recoil / 100.0  # Convert integer % to decimal
    else:
        recoil_mod = 0

    # Extract all buyFor offers with trader level info
    buy_for = mod.get("buyFor", []) or []
    offers = []
    lowest_price = 0
    price_source = "market"

    for offer in buy_for:
        if not isinstance(offer, dict):
            continue
        price = offer.get("priceRUB") or 0
        if price <= 0:
            continue

        source = offer.get("source", "")
        vendor = offer.get("vendor", {}) or {}

        # Extract trader level (1-4) or None for flea
        trader_level = None
        if source == "fleaMarket":
            trader_level = None  # Flea market has no level requirement
        else:
            trader_level = vendor.get("minTraderLevel") or 1

        offers.append({
            "price": price,
            "source": source,
            "vendor_name": vendor.get("name", ""),
            "vendor_normalized": vendor.get("normalizedName", ""),
            "trader_level": trader_level,
        })

    # Sort offers by price for easy lookup
    offers.sort(key=lambda x: x["price"])

    if offers:
        lowest_price = offers[0]["price"]
        price_source = offers[0]["source"]

    return {
        # Flat ergonomics bonus/penalty
        "ergonomics": ergo,
        # Percentage recoil modifier as decimal (e.g., -0.05 for -5%)
        "recoil_modifier": recoil_mod,
        # Accuracy modifier
        "accuracy_modifier": mod.get("accuracyModifier", 0) or 0,
        # All available offers (sorted by price)
        "offers": offers,
        # Default price (lowest available)
        "price": lowest_price,
        # Price source (trader name or "market")
        "price_source": price_source,
        # Weight
        "weight": mod.get("weight", 0) or 0,
        # Grid size
        "width": mod.get("width", 0) or 0,
        "height": mod.get("height", 0) or 0,
        # Minimum player level required to buy from flea market
        "min_level_flea": mod.get("minLevelForFlea") or 0,
        # Magazine capacity (only for magazines)
        "capacity": props.get("capacity") or 0,
        # Sighting range in meters (only for scopes/sights)
        "sighting_range": props.get("sightingRange") or 0,
        # BSG Category name (e.g. "Silencer", "Scope")
        "category": mod.get("bsgCategory", {}).get("name"),
    }


def build_item_lookup(guns, mods):
    """Build a dictionary for O(1) item lookup by ID.

    Note: Mods without valid market prices are excluded.
    """
    logger.info("Building item lookup table...")
    lookup = {}
    skipped_mods = 0

    # Add guns
    for gun in guns:
        lookup[gun["id"]] = {
            "type": "gun",
            "data": gun,
            "slots": extract_slots(gun),
            "stats": extract_gun_stats(gun),
            "presets": extract_all_presets(gun),  # List of purchasable presets
            "all_presets": extract_all_presets(gun, include_unpurchasable=True),  # All presets for fallback
        }

    # Add mods (only those with valid prices)
    for mod in mods:
        if not has_valid_price(mod):
            skipped_mods += 1
            continue  # Skip mods without valid market price
        lookup[mod["id"]] = {
            "type": "mod",
            "data": mod,
            "slots": extract_slots_from_mod(mod),
            "stats": extract_mod_stats(mod),
            "conflicting_items": extract_conflicting_items(mod),
            "conflicting_slot_ids": mod.get("conflictingSlotIds", []) or [],
        }

    logger.info(f"Item lookup built: {len(guns)} guns, {len(mods) - skipped_mods} mods (skipped {skipped_mods} mods without valid prices)")
    logger.debug(f"Total items indexed: {len(lookup)}")
    return lookup
