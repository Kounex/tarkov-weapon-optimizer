"""
Item lookup and resolution utilities.
"""

from typing import Optional


def resolve_item_id(name_query: str, item_lookup: dict) -> Optional[str]:
    """Find item ID by name using fuzzy-ish matching.

    Performs exact match first (case-insensitive), then substring match.

    Args:
        name_query: Name to search for
        item_lookup: Item lookup dictionary

    Returns:
        Item ID if found, else None
    """
    query_lower = name_query.lower().strip()
    candidates_substring = []

    for item_id, item in item_lookup.items():
        name = item["data"]["name"]
        name_lower = name.lower()

        # Exact match (case-insensitive)
        if name_lower == query_lower:
            return item_id

        # Collect substring matches
        if query_lower in name_lower:
            candidates_substring.append(item_id)

    # Prefer shortest name (closest match)
    if candidates_substring:
        candidates_substring.sort(key=lambda i: len(item_lookup[i]["data"]["name"]))
        return candidates_substring[0]

    return None


def get_caliber_display(gun: dict) -> str:
    """Get formatted caliber display name from gun data.

    Args:
        gun: Gun data dictionary

    Returns:
        Caliber display string (e.g., "7.62x39mm")
    """
    props = gun.get("properties", {}) or {}
    caliber = props.get("caliber", "")
    return caliber.replace("Caliber", "").strip() if caliber else ""


def get_category_name(gun: dict) -> str:
    """Get category name from gun data.

    Args:
        gun: Gun data dictionary

    Returns:
        Category name string
    """
    category = gun.get("bsgCategory", {})
    return category.get("name", "") if category else ""
