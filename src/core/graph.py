"""
Graph building and traversal logic (BFS).
"""

from collections import deque
from loguru import logger


def build_compatibility_map(weapon_id, item_lookup):
    """
    Perform BFS from the weapon to find all reachable mods.
    Returns a dict of reachable items with their slot context.

    Key fix: Each slot has a unique ID per parent item, so we track slot ownership
    to properly handle dependency constraints.
    """
    logger.debug(f"Building compatibility map for weapon {weapon_id}")
    reachable = {}  # item_id -> {"item": item_data}
    slot_items = {}  # slot_id -> list of item_ids that can go in that slot
    item_to_slots = {}  # item_id -> list of slot_ids it owns
    slot_owner = {}  # slot_id -> item_id that owns this slot (or weapon_id)

    if weapon_id not in item_lookup:
        logger.error(f"Weapon {weapon_id} not found in item lookup")
        raise ValueError(f"Weapon {weapon_id} not found in item lookup")

    weapon = item_lookup[weapon_id]
    queue = deque()

    # Start BFS from weapon's slots
    for slot in weapon["slots"]:
        slot_id = slot["id"]
        slot_items[slot_id] = []
        slot_owner[slot_id] = weapon_id  # Weapon owns this slot

        for allowed_id in slot["allowedItems"]:
            # Skip the weapon itself - it can't be a mod of itself
            if allowed_id == weapon_id:
                continue
            if allowed_id in item_lookup:
                queue.append((allowed_id, slot_id))
                slot_items[slot_id].append(allowed_id)

    visited = set()

    while queue:
        item_id, parent_slot_id = queue.popleft()

        if item_id in visited:
            continue
        visited.add(item_id)

        if item_id not in item_lookup:
            continue

        item = item_lookup[item_id]
        reachable[item_id] = {
            "item": item,
        }

        # Track slots owned by this item
        item_to_slots[item_id] = []

        # Explore this item's slots
        for slot in item["slots"]:
            slot_id = slot["id"]
            slot_items[slot_id] = []
            slot_owner[slot_id] = item_id  # This item owns this slot
            item_to_slots[item_id].append(slot_id)

            for allowed_id in slot["allowedItems"]:
                if allowed_id in item_lookup:
                    slot_items[slot_id].append(allowed_id)
                    if allowed_id not in visited:
                        queue.append((allowed_id, slot_id))

    logger.debug(f"Compatibility map built: {len(reachable)} reachable mods, {len(slot_items)} slots")
    return {
        "reachable_items": reachable,
        "slot_items": slot_items,
        "item_to_slots": item_to_slots,
        "slot_owner": slot_owner,  # Maps slot_id -> owner item_id
    }
