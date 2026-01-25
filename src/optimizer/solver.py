"""
Solver logic using CP-SAT.
"""

from loguru import logger
from ortools.sat.python import cp_model

# Default trader levels (all maxed) - only traders who sell weapon mods
DEFAULT_TRADER_LEVELS = {
    "prapor": 4,
    "skier": 4,
    "peacekeeper": 4,
    "mechanic": 4,
    "jaeger": 4,
}


def get_available_price(stats, trader_levels=None, flea_available=True, player_level=None):
    """Get the best available price for an item given trader levels and flea access."""
    if trader_levels is None:
        trader_levels = DEFAULT_TRADER_LEVELS

    min_level_flea = stats.get("min_level_flea", 0)
    offers = stats.get("offers")
    
    if not offers:
        default_price = stats.get("price", 0)
        if default_price > 0 and flea_available:
            if player_level is not None and min_level_flea > player_level:
                return (0, None, False)
            return (default_price, stats.get("price_source", "market"), True)
        return (0, None, False)

    best_price = None
    best_source = None

    for offer in offers:
        price = offer["price"]
        source = offer["source"]
        required_level = offer["trader_level"]
        vendor = offer.get("vendor_normalized", "").lower()

        if source == "fleaMarket":
            if not flea_available:
                continue
            if player_level is not None and min_level_flea > player_level:
                continue
        else:
            trader_level = trader_levels.get(vendor, 4)
            if required_level is not None and required_level > trader_level:
                continue

        if best_price is None or price < best_price:
            best_price = price
            best_source = source

    if best_price is not None:
        return (best_price, best_source, True)
    return (0, None, False)


def _check_constraints_feasibility(
    weapon, item_lookup, compatibility_map,
    max_price=None, min_ergonomics=None, max_recoil_v=None, max_recoil_sum=None,
    min_mag_capacity=None, min_sighting_range=None, max_weight=None,
    include_items=None, exclude_items=None,
    include_categories=None, exclude_categories=None,
    trader_levels=None, flea_available=True
):
    """Check if constraints are feasible and return reasons if not."""
    reasons = []
    available_items = compatibility_map["reachable_items"]

    if include_items:
        for req_id in include_items:
            if req_id not in available_items:
                name = item_lookup.get(req_id, {}).get("data", {}).get("name", req_id)
                reasons.append(f"Required item '{name}' is not compatible with this weapon")

    if include_categories:
        for group in include_categories:
            group_names = [cat for cat in group if isinstance(cat, str)]
            if not group_names: continue
            if not any(item_lookup[i]["stats"].get("category", "") in group_names for i in available_items):
                reasons.append(f"No items found for required category group: {group_names}")

    if min_mag_capacity:
        if not any(item_lookup[i]["stats"].get("capacity", 0) >= min_mag_capacity for i in available_items):
            reasons.append(f"No magazine with capacity >= {min_mag_capacity} rounds available")

    if min_sighting_range:
        base_sighting = weapon["stats"].get("sighting_range", 0)
        if base_sighting < min_sighting_range:
            if not any(item_lookup[i]["stats"].get("sighting_range", 0) >= min_sighting_range for i in available_items):
                reasons.append(f"No sight with sighting range >= {min_sighting_range}m available")

    if max_weight is not None:
        base_weight = weapon["stats"].get("weight", 0)
        min_mod_weight = 0
        for item_id in available_items:
            w = item_lookup[item_id]["stats"].get("weight", 0)
            if w > 0 and (min_mod_weight == 0 or w < min_mod_weight):
                min_mod_weight = w
        if base_weight + min_mod_weight > max_weight:
            reasons.append(f"Weight exceeds limit even with lightest mods")

    return reasons if reasons else None


def optimize_weapon(
    weapon_id, item_lookup, compatibility_map,
    max_price=None, min_ergonomics=None, max_recoil_v=None, max_recoil_sum=None,
    min_mag_capacity=None, min_sighting_range=None, max_weight=None,
    include_items=None, exclude_items=None,
    include_categories=None, exclude_categories=None,
    ergo_weight=1.0, recoil_weight=1.0, price_weight=0.0,
    trader_levels=None, flea_available=True, player_level=None
):
    """Use CP-SAT solver to find optimal mod configuration."""
    if trader_levels is None: trader_levels = DEFAULT_TRADER_LEVELS

    weapon = item_lookup[weapon_id]
    logger.info(f"Starting optimization for {weapon['data'].get('name', weapon_id)}")

    feasibility_reasons = _check_constraints_feasibility(
        weapon, item_lookup, compatibility_map,
        max_price, min_ergonomics, max_recoil_v, max_recoil_sum,
        min_mag_capacity, min_sighting_range, max_weight,
        include_items, exclude_items, include_categories, exclude_categories,
        trader_levels, flea_available
    )
    if feasibility_reasons:
        return {"status": "infeasible", "reason": "; ".join(feasibility_reasons), "selected_items": [], "selected_preset": None, "objective_value": 0}

    reachable = compatibility_map["reachable_items"]
    slot_items = compatibility_map["slot_items"]
    slot_owner = compatibility_map["slot_owner"]

    # Build preset maps
    presets = weapon.get("presets", [])
    preset_items_map = {}
    item_to_presets = {}
    preset_prices_map = {}
    
    for i, preset in enumerate(presets):
        p_price, _, p_avail = get_available_price(
            {"offers": preset.get("offers", []), "price": preset.get("price", 0)},
            trader_levels, flea_available, player_level
        )
        if not p_avail: continue

        preset_id = preset.get("id", f"preset_{i}")
        preset_prices_map[preset_id] = p_price
        items_in_preset = set(preset.get("items", []))
        preset_items_map[preset_id] = items_in_preset
        for item_id in items_in_preset:
            if item_id not in item_to_presets: item_to_presets[item_id] = []
            item_to_presets[item_id].append(preset_id)

    # Filter available items
    available_items = {}
    item_prices = {}
    exclude_items_set = set(exclude_items) if exclude_items else set()
    exclude_categories_set = set(exclude_categories) if exclude_categories else set()

    for item_id in reachable:
        if item_id not in item_lookup or item_id in exclude_items_set: continue
        stats = item_lookup[item_id]["stats"]
        if stats.get("category") in exclude_categories_set: continue

        price, source, is_available = get_available_price(stats, trader_levels, flea_available, player_level)
        in_preset = item_id in item_to_presets
        
        if stats.get("price", 0) > 100_000_000:
            if not in_preset: continue
            price = 0
            is_available = False

        if not is_available and not in_preset: continue
        available_items[item_id] = reachable[item_id]
        item_prices[item_id] = (price, source, is_available)

    model = cp_model.CpModel()

    # Base vars
    base_vars = {}
    naked_price = weapon["stats"].get("price", 0)
    fallback_base = None

    if naked_price < 100_000_000:
        base_vars["naked"] = model.NewBoolVar("base_naked")

    for pid in preset_items_map:
        base_vars[pid] = model.NewBoolVar(f"base_{pid}")

    if base_vars:
        model.Add(sum(base_vars.values()) == 1)
    else:
        # Fallback
        all_presets = weapon.get("all_presets", [])
        if all_presets:
            fallback = all_presets[0]
            fid = fallback.get("id", "fallback")
            base_vars[fid] = model.NewBoolVar(f"base_{fid}")
            model.Add(base_vars[fid] == 1)
            
            f_items = set(fallback.get("items", []))
            preset_items_map[fid] = f_items
            preset_prices_map[fid] = 0
            for item_id in f_items:
                if item_id not in item_to_presets: item_to_presets[item_id] = []
                item_to_presets[item_id].append(fid)
                if item_id in reachable and item_id not in available_items:
                    available_items[item_id] = reachable[item_id]
                    item_prices[item_id] = (0, "fallback", False)
            
            fallback_base = {"type": "preset", "id": fid, "name": fallback.get("name"), "price": 0}
        else:
            base_vars["naked"] = model.NewBoolVar("base_naked")
            model.Add(base_vars["naked"] == 1)
            fallback_base = {"type": "naked", "price": 0}

    # Item vars and Buy vars
    x = {i: model.NewBoolVar(f"x_{i}") for i in available_items}
    buy = {i: model.NewBoolVar(f"buy_{i}") for i in available_items}

    for item_id in available_items:
        is_avail = item_prices[item_id][2]
        containing = [base_vars[pid] for pid in item_to_presets.get(item_id, []) if pid in base_vars]

        if not is_avail:
            if containing:
                model.Add(x[item_id] <= sum(containing))
            else:
                model.Add(x[item_id] == 0)

        if containing:
            any_preset = model.NewBoolVar(f"preset_has_{item_id}")
            model.AddMaxEquality(any_preset, containing)
            model.Add(buy[item_id] <= x[item_id])
            model.Add(buy[item_id] <= 1 - any_preset)
            model.Add(buy[item_id] >= x[item_id] - any_preset)
        else:
            model.Add(buy[item_id] == x[item_id])

    # Constraints
    if include_items:
        for i in include_items:
            if i in x: model.Add(x[i] == 1)
            else: model.Add(0 == 1)

    if include_categories:
        for group in include_categories:
            vars_in_group = [x[i] for i in x if item_lookup[i]["stats"].get("category") in group]
            if vars_in_group: model.Add(sum(vars_in_group) >= 1)
            else: model.Add(0 == 1)

    # Topology constraints (Mutex, Dependency)
    item_to_valid_slots = {i: [] for i in x}
    for slot_id, items in slot_items.items():
        owner_id = slot_owner.get(slot_id)
        is_base = (owner_id == weapon_id)
        if is_base or owner_id in x:
            for item_id in items:
                if item_id in x:
                    item_to_valid_slots[item_id].append((slot_id, owner_id, is_base))

    placed_in = {}
    items_needing_placement = set()
    for item_id, valid in item_to_valid_slots.items():
        if len(valid) > 1:
            items_needing_placement.add(item_id)
            placed_in[item_id] = {sf[0]: model.NewBoolVar(f"placed_{item_id[:8]}_{sf[0][:8]}") for sf in valid}

    for item_id in items_needing_placement:
        model.Add(sum(placed_in[item_id].values()) == x[item_id])

    # Slot Mutex
    for slot_id, items in slot_items.items():
        placements = []
        for item_id in items:
            if item_id not in x: continue
            if item_id in items_needing_placement:
                if slot_id in placed_in.get(item_id, {}):
                    placements.append(placed_in[item_id][slot_id])
            elif any(s[0] == slot_id for s in item_to_valid_slots[item_id]):
                placements.append(x[item_id])
        if placements:
            model.Add(sum(placements) <= 1)

    # Dependency
    for item_id, valid in item_to_valid_slots.items():
        if not valid:
            model.Add(x[item_id] == 0)
            continue
        
        has_base = any(is_b for _, _, is_b in valid)
        if item_id in items_needing_placement:
            for slot_id, owner_id, is_base in valid:
                if slot_id not in placed_in[item_id]: continue
                p_var = placed_in[item_id][slot_id]
                if not is_base:
                    if owner_id in x: model.Add(p_var <= x[owner_id])
                    else: model.Add(p_var == 0)
        elif not has_base:
            parents = [x[oid] for _, oid, _ in valid if oid in x]
            if not parents: model.Add(x[item_id] == 0)
            else:
                parent_or = model.NewBoolVar(f'parent_or_{item_id}')
                model.AddMaxEquality(parent_or, parents)
                model.Add(x[item_id] <= parent_or)

    # Conflict
    conflicts = set()
    for i in available_items:
        for c in item_lookup[i].get("conflicting_items", []):
            if c in x:
                pair = tuple(sorted([i, c]))
                if pair not in conflicts:
                    conflicts.add(pair)
                    model.Add(x[i] + x[c] <= 1)

    # Required slots
    for slot in weapon["slots"]:
        if slot.get("required"):
            slot_id = slot["id"]
            items = [i for i in slot_items.get(slot_id, []) if i in x]
            if items: model.Add(sum(x[i] for i in items) >= 1)

    for owner_id, slots in compatibility_map["item_to_slots"].items():
        if owner_id in x:
            for slot_id in slots:
                # Find if slot required
                s_data = next((s for s in item_lookup[owner_id]["slots"] if s["id"] == slot_id), None)
                if s_data and s_data.get("required"):
                    items = [i for i in slot_items.get(slot_id, []) if i in x]
                    if items:
                        model.Add(sum(x[i] for i in items) >= 1).OnlyEnforceIf(x[owner_id])

    # Constraint: Min Mag Capacity
    if min_mag_capacity:
        suitable_mags = [x[i] for i in x if item_lookup[i]["stats"].get("capacity", 0) >= min_mag_capacity]
        if suitable_mags:
            model.Add(sum(suitable_mags) >= 1)
        else:
            model.Add(0 == 1)

    # Constraint: Min Sighting Range
    if min_sighting_range:
        base_sighting = weapon["stats"].get("sighting_range", 0)
        # If base weapon already satisfies, no constraint needed on mods constraints
        if base_sighting < min_sighting_range:
            suitable_sights = [x[i] for i in x if item_lookup[i]["stats"].get("sighting_range", 0) >= min_sighting_range]
            if suitable_sights:
                model.Add(sum(suitable_sights) >= 1)
            else:
                model.Add(0 == 1)

    # Constraint: Max Weight
    if max_weight is not None:
        base_weight = weapon["stats"].get("weight", 0)
        weight_terms = []
        for i in x:
            w = item_lookup[i]["stats"].get("weight", 0)
            if w > 0:
                # Scale by 1000 to handle grams/floats
                weight_terms.append(int(w * 1000) * x[i])
        
        if weight_terms:
            # max_weight is float kg
            limit_g = int(max_weight * 1000)
            base_g = int(base_weight * 1000)
            model.Add(sum(weight_terms) <= limit_g - base_g)

    # Constraint: Min Ergonomics
    if min_ergonomics is not None:
        base_ergo = weapon["stats"].get("naked_ergonomics", 0)
        ergo_terms = []
        scale = 10000 # Use same scale as extraction
        for i in x:
            e = item_lookup[i]["stats"].get("ergonomics", 0)
            if e != 0:
                ergo_terms.append(int(e * scale) * x[i])
        
        if ergo_terms:
            req_add = int((min_ergonomics - base_ergo) * scale)
            model.Add(sum(ergo_terms) >= req_add)

    # Constraint: Max Recoil Vertical
    if max_recoil_v is not None:
        # Final = naked * (1 + sum(mod_recoils))
        # Final <= Limit  =>  1 + sum <= Limit/naked  => sum <= Limit/naked - 1
        naked_r = weapon["stats"].get("naked_recoil_v", 100)
        if naked_r > 0:
            recoil_terms = []
            scale = 10000
            for i in x:
                r = item_lookup[i]["stats"].get("recoil_modifier", 0)
                if r != 0:
                    recoil_terms.append(int(r * scale) * x[i])
            
            if recoil_terms:
                # Limit calculation: (max / naked) - 1
                # Scaled: ((max * scale) / naked) - scale
                limit_val = int(((max_recoil_v * scale) / naked_r) - scale)
                model.Add(sum(recoil_terms) <= limit_val)

    # Objectives
    # Objectives
    def get_price_terms():
        terms = []
        nk_price = 0 if fallback_base and fallback_base["type"] == "naked" else int(weapon["stats"].get("price", 0))
        if "naked" in base_vars: terms.append((nk_price, base_vars["naked"]))
        for pid, p_price in preset_prices_map.items():
            if pid in base_vars and p_price > 0: terms.append((int(p_price), base_vars[pid]))
        for i, p in item_prices.items():
            if i in buy and int(p[0]) > 0: terms.append((int(p[0]), buy[i]))
        return terms

    price_terms = get_price_terms()
    if max_price is not None and price_terms:
        model.Add(sum(c * v for c, v in price_terms) <= max_price)

    obj_terms = []
    if price_weight > 0:
        for c, v in price_terms: obj_terms.append(-1 * int(price_weight * c) * v)
    
    if recoil_weight > 0:
        nk_recoil = weapon["stats"]["naked_recoil_v"]
        scale = 10000
        for i in x:
            mod_recoil = item_lookup[i]["stats"].get("recoil_modifier", 0)
            if mod_recoil != 0:
                obj_terms.append(int(-1 * mod_recoil * nk_recoil * scale * recoil_weight) * x[i])

    if ergo_weight > 0:
        scale = 10000
        for i in x:
            mod_ergo = item_lookup[i]["stats"].get("ergonomics", 0)
            if mod_ergo != 0:
                obj_terms.append(int(mod_ergo * scale * ergo_weight) * x[i])

    model.Maximize(sum(obj_terms))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 10.0
    status = solver.Solve(model)

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        sel_items = [i for i in x if solver.Value(x[i]) == 1]
        sel_preset = next((pid for pid in base_vars if pid != "naked" and solver.Value(base_vars[pid]) == 1), None)
        return {
            "status": "optimal" if status == cp_model.OPTIMAL else "feasible",
            "selected_items": sel_items,
            "selected_preset": sel_preset,
            "objective_value": solver.ObjectiveValue(),
            "fallback_base": fallback_base,
        }
    
    return {"status": "infeasible", "selected_items": [], "selected_preset": None, "objective_value": 0}
