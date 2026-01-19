"""
Streamlit Web UI for Tarkov Weapon Mod Optimizer
"""

import json
import os
import sys
from datetime import datetime, timedelta

# Set Streamlit config directory to project directory (must be before streamlit import)
os.environ.setdefault("STREAMLIT_CONFIG_DIR", os.path.dirname(os.path.abspath(__file__)))

import altair as alt
import extra_streamlit_components as stx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from loguru import logger
from weapon_optimizer import (
    fetch_all_data,
    build_item_lookup,
    build_compatibility_map,
    optimize_weapon,
    calculate_total_stats,
    explore_pareto,
    set_log_level,
)
from i18n import t, language_selector, get_language

# Configure loguru for Streamlit (reduce noise for UI)
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    level="INFO",
    filter=lambda record: record["level"].name != "DEBUG",  # Filter out DEBUG in Streamlit
)

# Also log to file with rotation (if possible)
_log_dir = os.path.join(os.path.dirname(__file__), "logs")
try:
    os.makedirs(_log_dir, exist_ok=True)
    logger.add(
        os.path.join(_log_dir, "streamlit_app_{time}.log"),
        rotation="5 MB",
        retention="3 days",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    )
except (OSError, PermissionError):
    # File logging not available, continue with console only
    pass

# Page configuration
st.set_page_config(
    page_title="Tarkov Weapon Optimizer",
    page_icon="🔫",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Cached data loading functions
@st.cache_data(show_spinner=False)
def load_data():
    """Fetch all guns and mods from API (cached). Saves to debug file."""
    logger.info("Loading game data...")
    guns, mods = fetch_all_data()

    # Save to debug file
    debug_data = {
        "fetched_at": datetime.now().isoformat(),
        "guns_count": len(guns),
        "mods_count": len(mods),
        "guns": guns,
        "mods": mods,
    }
    with open("api_cache_debug.json", "w", encoding="utf-8") as f:
        json.dump(debug_data, f, indent=2, ensure_ascii=False)

    logger.info(f"Loaded {len(guns)} guns and {len(mods)} mods")
    return guns, mods


@st.cache_data(show_spinner=False)
def build_lookup(_guns, _mods):
    """Build item lookup dictionary (cached)."""
    return build_item_lookup(_guns, _mods)


@st.cache_data(show_spinner=False)
def get_compat_map(weapon_id, _item_lookup):
    """Build compatibility map for a weapon (cached per weapon_id)."""
    return build_compatibility_map(weapon_id, _item_lookup)


def get_resource_path(filename):
    """Get the correct path for bundled resources.

    When running as a PyInstaller bundle, resources are extracted to a
    temporary directory (sys._MEIPASS). This function returns the correct
    path whether running from source or as a bundled executable.

    Args:
        filename: Name of the resource file (e.g., "tasks.json")

    Returns:
        Full path to the resource file
    """
    if getattr(sys, 'frozen', False):
        return os.path.join(sys._MEIPASS, filename)
    return filename


@st.cache_data(show_spinner=False)
def load_tasks():
    """Load Gunsmith tasks from JSON file."""
    try:
        with open(get_resource_path("tasks.json"), "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return []


def resolve_item_id(name_query, item_lookup):
    """
    Find item ID by name using fuzzy-ish matching.
    Returns ID if found, else None.
    """
    # 1. Exact match (case-insensitive)
    query_lower = name_query.lower().strip()
    
    candidates_substring = []
    
    for item_id, item in item_lookup.items():
        name = item["data"]["name"]
        name_lower = name.lower()
        
        if name_lower == query_lower:
            return item_id
            
        if query_lower in name_lower:
            candidates_substring.append(item_id)
            
    # 2. Substring match
    # Prefer shortest name (closest match)
    if candidates_substring:
        candidates_substring.sort(key=lambda i: len(item_lookup[i]["data"]["name"]))
        return candidates_substring[0]
        
    return None


def get_image_url(item_data, prefer_high_res=False, prefer_icon=False):
    """Get image URL from item data with fallback chain."""
    if prefer_icon:
        # Prefer icon for compact display
        return (
            item_data.get("iconLink")
            or item_data.get("iconLinkFallback")
            or item_data.get("imageLink")
            or item_data.get("image512pxLink")
        )
    elif prefer_high_res:
        # Prefer high-res images
        return (
            item_data.get("image512pxLink")
            or item_data.get("image8xLink")
            or item_data.get("imageLink")
            or item_data.get("iconLink")
            or item_data.get("iconLinkFallback")
        )
    else:
        # Prefer standard images
        return (
            item_data.get("imageLink")
            or item_data.get("iconLink")
            or item_data.get("image512pxLink")
            or item_data.get("iconLinkFallback")
        )


def get_best_offer_display(stats, trader_levels=None, flea_available=True, player_level=None):
    """Get a formatted string showing the best AVAILABLE offer source with trader level.

    Respects trader level and player level constraints to show what the player
    would actually pay and where they'd buy from.
    """
    offers = stats.get('offers', [])
    if not offers:
        return stats.get('price_source', 'market'), stats.get('price', 0)

    min_level_flea = stats.get('min_level_flea', 0) or 15

    # Filter to only available offers based on constraints
    available_offers = []
    for offer in offers:
        source = offer.get('source', '')
        if source == 'fleaMarket':
            if not flea_available:
                continue
            if player_level is not None and min_level_flea > player_level:
                continue
            available_offers.append(offer)
        else:
            # Trader offer - check level requirement
            vendor = offer.get('vendor_normalized', '').lower()
            required_level = offer.get('trader_level')
            if trader_levels:
                player_trader_level = trader_levels.get(vendor, 4)
                if required_level is not None and required_level > player_trader_level:
                    continue
            available_offers.append(offer)

    if not available_offers:
        return "Unavailable", 0

    # Find the best (lowest price) available offer
    best_offer = min(available_offers, key=lambda x: x.get('price', float('inf')))
    price = best_offer.get('price', 0)
    source = best_offer.get('source', '')

    if source == 'fleaMarket':
        return f"Flea Lv{min_level_flea}", price
    else:
        vendor_name = best_offer.get('vendor_name', source)
        trader_level = best_offer.get('trader_level')
        if trader_level:
            return f"{vendor_name} LL{trader_level}", price
        return vendor_name, price


def display_mods_table(item_ids, item_lookup, show_price=True, constraints=None):
    """Display a markdown table of mods with their stats.

    Args:
        constraints: Optional dict with trader_levels, flea_available, player_level
                     to show accurate source/price based on what's actually available.
    """
    trader_levels = constraints.get("trader_levels") if constraints else None
    flea_available = constraints.get("flea_available", True) if constraints else True
    player_level = constraints.get("player_level") if constraints else None

    rows = []
    for item_id in item_ids:
        if item_id in item_lookup:
            item = item_lookup[item_id]
            stats = item["stats"]
            name = item['data']['name'].replace("|", "\\|")
            icon_url = get_image_url(item["data"], prefer_icon=True)
            ergo = stats.get('ergonomics', 0)
            recoil = stats.get('recoil_modifier', 0) * 100

            row = {
                "icon": f'<img src="{icon_url}" width="64" style="vertical-align:middle; min-width: 64px;">' if icon_url else "",
                "name": name,
                "ergo": f"{ergo:+.1f}" if ergo != 0 else "-",
                "recoil": f"{recoil:+.1f}%" if recoil != 0 else "-",
            }
            if show_price:
                source_display, price = get_best_offer_display(
                    stats, trader_levels, flea_available, player_level
                )
                row["price"] = f"₽{price:,}"
                row["source"] = source_display
            rows.append(row)

    if rows:
        if show_price:
            header = f"| | {t('table.name')} | {t('table.ergo')} | {t('table.recoil')} | {t('table.price')} | {t('table.source')} |"
            separator = "|:---:|:-----|:----:|:------:|------:|:------:|"
            lines = [header, separator]
            for row in rows:
                lines.append(f"| {row['icon']} | {row['name']} | {row['ergo']} | {row['recoil']} | {row['price']} | {row['source']} |")
        else:
            header = f"| | {t('table.name')} | {t('table.ergo')} | {t('table.recoil')} |"
            separator = "|:---:|:-----|:----:|:------:|"
            lines = [header, separator]
            for row in rows:
                lines.append(f"| {row['icon']} | {row['name']} | {row['ergo']} | {row['recoil']} |")

        st.markdown(
            """<style>
table { width: 75% !important; margin-left: auto !important; margin-right: auto !important; }
@media (max-width: 1200px) { table { width: 100% !important; } }
</style>""" + "\n" + "\n".join(lines),
            unsafe_allow_html=True
        )


def calculate_build_cost(selected_items, selected_preset, item_lookup, weapon_stats, presets, selected_gun, fallback_base=None):
    """
    Calculate the actual total build cost and details.
    
    Returns a dict with: 'total_cost', 'base_cost', 'mods_cost', 'base_label'.
    """
    weapon_base_price = weapon_stats.get("price", 0)
    
    # Check if dummy price (unavailable) - don't include in total
    if weapon_base_price > 100_000_000:
        weapon_base_price = 0
    
    base_cost = 0
    mods_cost = 0
    base_label = t('results.base_weapon')
    
    if selected_preset:
        # Find the preset info
        preset_info = next((p for p in presets if p.get("id") == selected_preset), None)
        
        # If not found in purchasable presets, check all_presets
        if not preset_info:
            all_presets = item_lookup.get(selected_gun["id"], {}).get("all_presets", [])
            preset_info = next((p for p in all_presets if p.get("id") == selected_preset), None)
        
        if preset_info:
            preset_items = set(preset_info.get("items", []))
            
            # Handle fallback base logic if provided
            is_fallback = fallback_base and fallback_base.get("type") == "preset"
            preset_price = 0 if is_fallback else preset_info.get("price", 0)
            
            base_cost = preset_price
            base_label = t('results.preset')
            
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
                "base_label": base_label
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
        "base_label": base_label
    }


def display_optimization_results(result, item_lookup, weapon_stats, presets, selected_gun, constraints=None):
    """Display optimization results. Returns True if results were displayed."""
    if result["status"] == "infeasible":
        st.error(t("results.infeasible"))
        return False

    status_key = f"results.{result['status']}" if result['status'] in ['optimal', 'feasible'] else "results.feasible"
    status_text = t(status_key)
    status_icon = "✅" if result["status"] == "optimal" else "⚠️"
    
    # "Optimization OPTIMAL" or similar construction
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

    # Total Cost (using unified builder)
    cost_data = calculate_build_cost(
        selected_items,
        selected_preset,
        item_lookup,
        weapon_stats,
        presets,
        selected_gun,
        fallback_base=result.get("fallback_base")
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
            preset_info = next((p for p in all_presets if p.get("id") == selected_preset), None)
        if preset_info:
            preset_item_ids = set(preset_info.get("items", []))

    if selected_preset and preset_info:
        # Check if this is a fallback preset (price=0)
        is_fallback = fallback_base and fallback_base.get("type") == "preset"
        display_price = 0 if is_fallback else preset_info.get('price', 0)
        price_source = "fallback (free)" if is_fallback else preset_info.get('price_source', 'market')

        st.markdown(f"**{t('results.preset')}:** {preset_info.get('name')}")
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"**{t('results.bundle_price')}:** ₽{display_price:,} ({price_source})")
            st.markdown(f"**{t('results.includes')}:** {len(preset_item_ids)} items")
        with col2:
            if preset_info.get("image"):
                st.image(preset_info["image"], width=150)

        additional_items = [item_id for item_id in selected_items if item_id not in preset_item_ids]
        if additional_items:
            st.markdown("---")
            st.markdown(f"**{t('results.additional_mods')}:**")
            display_mods_table(additional_items, item_lookup, show_price=True, constraints=constraints)

        with st.expander(t("results.items_in_preset", name=preset_info.get('name')), expanded=False):
            display_mods_table(preset_item_ids, item_lookup, show_price=False, constraints=constraints)

    elif selected_items:
        st.markdown(f"**{t('results.naked_gun_mods')}**")

        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"**{t('results.base_weapon')}:** {selected_gun['name']}")
            weapon_price = weapon_stats.get('price', 0)
            weapon_source = weapon_stats.get('price_source', 'market')

            # Check if this is a fallback naked gun (price=0)
            is_naked_fallback = fallback_base and fallback_base.get("type") == "naked"
            if is_naked_fallback:
                st.markdown(f"**{t('sidebar.price')}:** ₽0 (fallback - free)")
            elif weapon_source == "not_available":
                st.markdown(f"**{t('sidebar.price')}:** {t('results.not_available')}")
            else:
                st.markdown(f"**{t('sidebar.price')}:** ₽{weapon_price:,} ({weapon_source})")
        with col2:
            weapon_image_url = weapon_stats.get("default_preset_image") or get_image_url(selected_gun, prefer_high_res=True)
            if weapon_image_url:
                st.image(weapon_image_url, width=150)

        if selected_items:
            st.markdown("---")
            st.markdown(f"**{t('results.additional_mods')}:**")
            display_mods_table(selected_items, item_lookup, show_price=True, constraints=constraints)
    else:
        st.markdown(f"**{t('results.naked_gun')}**")
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"**{t('results.base_weapon')}:** {selected_gun['name']}")
            weapon_price = weapon_stats.get('price', 0)
            weapon_source = weapon_stats.get('price_source', 'market')

            # Check if this is a fallback naked gun (price=0)
            is_naked_fallback = fallback_base and fallback_base.get("type") == "naked"
            if is_naked_fallback:
                st.markdown(f"**{t('sidebar.price')}:** ₽0 (fallback - free)")
            elif weapon_source == "not_available":
                st.markdown(f"**{t('sidebar.price')}:** {t('results.not_available')}")
            else:
                st.markdown(f"**{t('sidebar.price')}:** ₽{weapon_price:,} ({weapon_source})")

            st.info(t("results.no_mods_selected"))
        with col2:
            weapon_image_url = weapon_stats.get("default_preset_image") or get_image_url(selected_gun, prefer_high_res=True)
            if weapon_image_url:
                st.image(weapon_image_url, width=150)

    # Optimization details
    with st.expander(t("results.optimization_details")):
        st.write(f"**{t('results.status')}:** {result['status']}")
        st.write(f"**{t('results.objective_value')}:** {result['objective_value']:.0f}")
        st.write(f"**{t('results.recoil_multiplier')}:** {final_stats['recoil_multiplier']:.4f}")
        if constraints:
            if constraints.get("max_price"):
                st.write(f"**{t('results.budget_constraint')}:** ₽{constraints['max_price']:,}")
            if constraints.get("min_ergonomics"):
                st.write(f"**{t('results.min_ergo_constraint')}:** {constraints['min_ergonomics']}")
            if constraints.get("max_recoil_v"):
                st.write(f"**{t('results.max_recoil_constraint')}:** {constraints['max_recoil_v']}")
            if constraints.get("min_mag_capacity"):
                st.write(f"**{t('results.min_mag_constraint')}:** {constraints['min_mag_capacity']} {t('units.rounds')}")
            if constraints.get("min_sighting_range"):
                st.write(f"**{t('results.min_sight_constraint')}:** {constraints['min_sighting_range']}{t('units.meters')}")
            if constraints.get("max_weight"):
                st.write(f"**{t('results.max_weight_constraint')}:** {constraints['max_weight']:.1f} {t('units.kg')}")
            player_lvl = constraints.get("player_level")
            if player_lvl is not None:
                st.write(f"**{t('sidebar.player_level')}:** {player_lvl}")
            trader_lvls = constraints.get("trader_levels", {})
            flea = constraints.get("flea_available", True)
            if trader_lvls:
                levels_str = ", ".join([f"{k.capitalize()}: LL{v}" for k, v in trader_lvls.items()])
                st.write(f"**{t('sidebar.trader_levels')}:** {levels_str}")
            
            flea_status = t('results.available') if flea else t('results.not_available_short')
            st.write(f"**{t('results.flea_market')}:** {flea_status}")

    return True


def generate_build_export(result, item_lookup, weapon_stats, presets, selected_gun, constraints=None):
    """Generate exportable build data in JSON and Markdown formats."""
    selected_items = result["selected_items"]
    selected_preset = result.get("selected_preset")
    fallback_base = result.get("fallback_base")
    final_stats = calculate_total_stats(weapon_stats, selected_items, item_lookup)

    # Calculate total cost
    total_cost = final_stats['total_price']
    weapon_base_price = weapon_stats.get("price", 0)

    # Check if dummy price (unavailable)
    if weapon_base_price > 100_000_000:
        weapon_base_price = 0

    preset_info = None

    if selected_preset:
        preset_info = next((p for p in presets if p.get("id") == selected_preset), None)
        # If not found in purchasable presets, check all_presets (for fallback case)
        if not preset_info:
            weapon_id = selected_gun["id"]
            all_presets = item_lookup[weapon_id].get("all_presets", [])
            preset_info = next((p for p in all_presets if p.get("id") == selected_preset), None)
        if preset_info:
            preset_items = set(preset_info.get("items", []))
            individual_cost = sum([
                item_lookup[item_id]["stats"].get("price", 0)
                for item_id in selected_items
                if item_id not in preset_items and item_id in item_lookup
            ])
            # Use price=0 if this is a fallback preset
            preset_price = 0 if (fallback_base and fallback_base.get("type") == "preset") else preset_info.get("price", 0)
            total_cost = preset_price + individual_cost
    else:
        total_cost = weapon_base_price + final_stats['total_price']

    # Build JSON export
    # Use fallback price=0 if applicable
    export_preset_price = None
    if preset_info:
        export_preset_price = 0 if (fallback_base and fallback_base.get("type") == "preset") else preset_info.get("price")

    json_data = {
        "exported_at": datetime.now().isoformat(),
        "weapon": {
            "id": selected_gun["id"],
            "name": selected_gun["name"],
            "base_price": weapon_base_price,
        },
        "preset": {
            "id": preset_info["id"] if preset_info else None,
            "name": preset_info["name"] if preset_info else None,
            "price": export_preset_price,
            "is_fallback": bool(fallback_base and fallback_base.get("type") == "preset"),
        } if selected_preset else None,
        "mods": [
            {
                "id": item_id,
                "name": item_lookup[item_id]["data"]["name"],
                "ergonomics": item_lookup[item_id]["stats"].get("ergonomics", 0),
                "recoil_modifier": item_lookup[item_id]["stats"].get("recoil_modifier", 0),
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
        f"| Stat | Value |",
        f"|------|-------|",
        f"| Ergonomics | {min(100, max(0, final_stats['ergonomics'])):.1f} |",
        f"| Recoil V | {final_stats['recoil_vertical']:.1f} |",
        f"| Recoil H | {final_stats['recoil_horizontal']:.1f} |",
        f"| Weight | {final_stats['total_weight']:.2f} kg |",
        f"| Total Cost | ₽{total_cost:,} |",
        "",
    ]

    if selected_preset and preset_info:
        md_preset_price = 0 if (fallback_base and fallback_base.get("type") == "preset") else preset_info.get('price', 0)
        fallback_note = " (fallback - free)" if (fallback_base and fallback_base.get("type") == "preset") else ""
        md_lines.extend([
            "## Base Preset",
            f"**{preset_info['name']}** - ₽{md_preset_price:,}{fallback_note}",
            "",
        ])

    # Additional mods
    additional_mods = selected_items
    if preset_info:
        preset_items = set(preset_info.get("items", []))
        additional_mods = [m for m in selected_items if m not in preset_items]

    if additional_mods:
        md_lines.extend([
            "## Modifications",
            "| Name | Ergo | Recoil | Price |",
            "|------|------|--------|-------|",
        ])
        for item_id in additional_mods:
            if item_id in item_lookup:
                item = item_lookup[item_id]
                name = item["data"]["name"]
                ergo = item["stats"].get("ergonomics", 0)
                recoil = item["stats"].get("recoil_modifier", 0) * 100
                price = item["stats"].get("price", 0)
                md_lines.append(f"| {name} | {ergo:+.1f} | {recoil:+.1f}% | ₽{price:,} |")
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
            md_lines.append(f"- Min Mag Capacity: {constraints['min_mag_capacity']} rounds")
        if constraints.get("min_sighting_range"):
            md_lines.append(f"- Min Sighting Range: {constraints['min_sighting_range']}m")
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


def main():
    # Initialize Cookie Manager for persistent settings
    # This must be done at the start of the app
    cookie_manager = stx.CookieManager(key="main_cookie_manager")

    # Helper to save cookies with 1 year expiration
    def save_cookie(key, value):
        cookie_manager.set(key, value, expires_at=datetime.now() + timedelta(days=365))

    logger.debug("Streamlit app main() started")

    # Language selector at top of sidebar
    with st.sidebar:
        language_selector(label="🌐 Language")
        st.markdown("---")

    # Title
    st.title(f"🔫 {t('app.title')}")
    st.markdown(t("app.subtitle"))

    # Load data with status indicator
    with st.status(t("status.loading"), expanded=False) as status:
        try:
            if status:
                status.update(label=t("status.fetching"))
            guns, mods = load_data()
            if status:
                status.update(label=t("status.building_lookup"))
            item_lookup = build_lookup(guns, mods)
            if status:
                status.update(label=t("status.loaded", guns=len(guns), mods=len(mods)), state="complete")
        except Exception as e:
            if status:
                status.update(label=t("status.failed_load"), state="error")
            st.error(f"{t('status.failed_load')}: {e}")
            st.stop()

    # Load tasks
    tasks = load_tasks()

    # Sidebar: Weapon Selection
    st.sidebar.header(f"🔫 {t('sidebar.select_weapon')}")

    # Helper to get caliber display name
    def get_caliber_display(gun):
        props = gun.get("properties", {}) or {}
        caliber = props.get("caliber", "")
        return caliber.replace("Caliber", "").strip() if caliber else ""

    # Helper to get category name
    def get_category_name(gun):
        category = gun.get("bsgCategory", {})
        return category.get("name", "") if category else ""

    # Build category counts and caliber counts for all guns
    all_category_counts = {}
    all_caliber_counts = {}
    for gun in guns:
        cat = get_category_name(gun)
        cal = get_caliber_display(gun)
        if cat:
            all_category_counts[cat] = all_category_counts.get(cat, 0) + 1
        if cal:
            all_caliber_counts[cal] = all_caliber_counts.get(cal, 0) + 1

    # Get selected types from session state (for cascading logic)
    if "selected_types" not in st.session_state:
        st.session_state.selected_types = []
    if "selected_calibers" not in st.session_state:
        st.session_state.selected_calibers = []

    # Calculate available calibers based on selected types (for cascading)
    if st.session_state.selected_types:
        guns_of_selected_types = [g for g in guns if get_category_name(g) in st.session_state.selected_types]
    else:
        guns_of_selected_types = guns

    available_caliber_counts = {}
    for gun in guns_of_selected_types:
        cal = get_caliber_display(gun)
        if cal:
            available_caliber_counts[cal] = available_caliber_counts.get(cal, 0) + 1

    # Calculate available categories based on selected calibers (for cascading)
    if st.session_state.selected_calibers:
        guns_of_selected_calibers = [g for g in guns if get_caliber_display(g) in st.session_state.selected_calibers]
    else:
        guns_of_selected_calibers = guns

    available_category_counts = {}
    for gun in guns_of_selected_calibers:
        cat = get_category_name(gun)
        if cat:
            available_category_counts[cat] = available_category_counts.get(cat, 0) + 1

    # Build options with counts
    category_options = sorted(available_category_counts.keys())
    caliber_options = sorted(available_caliber_counts.keys())

    # Format options with counts
    def format_with_count(option, counts):
        count = counts.get(option, 0)
        return f"{option} ({count})"

    # Gun type filter (multi-select) with counts
    selected_types = st.sidebar.multiselect(
        t("sidebar.filter_gun_type"),
        category_options,
        default=st.session_state.selected_types,
        format_func=lambda x: format_with_count(x, available_category_counts),
        placeholder="All",
        key="type_filter",
    )

    # Update session state and rerun if changed
    if selected_types != st.session_state.selected_types:
        st.session_state.selected_types = selected_types
        # Clear calibers that are no longer available
        if selected_types:
            new_available_calibers = set()
            for gun in guns:
                if get_category_name(gun) in selected_types:
                    cal = get_caliber_display(gun)
                    if cal:
                        new_available_calibers.add(cal)
            st.session_state.selected_calibers = [c for c in st.session_state.selected_calibers if c in new_available_calibers]
        st.rerun()

    # Caliber filter (multi-select) with counts
    selected_calibers = st.sidebar.multiselect(
        t("sidebar.filter_caliber"),
        caliber_options,
        default=st.session_state.selected_calibers,
        format_func=lambda x: format_with_count(x, available_caliber_counts),
        placeholder="All",
        key="caliber_filter",
    )

    # Update session state and rerun if changed
    if selected_calibers != st.session_state.selected_calibers:
        st.session_state.selected_calibers = selected_calibers
        # Clear types that are no longer available
        if selected_calibers:
            new_available_types = set()
            for gun in guns:
                if get_caliber_display(gun) in selected_calibers:
                    cat = get_category_name(gun)
                    if cat:
                        new_available_types.add(cat)
            st.session_state.selected_types = [t for t in st.session_state.selected_types if t in new_available_types]
        st.rerun()

    # Filter guns by selected types and calibers
    filtered_guns = guns
    if selected_types:
        filtered_guns = [g for g in filtered_guns if get_category_name(g) in selected_types]
    if selected_calibers:
        filtered_guns = [g for g in filtered_guns if get_caliber_display(g) in selected_calibers]

    gun_options = {gun["name"]: gun for gun in filtered_guns}
    gun_names = sorted(gun_options.keys())

    # Show match count
    st.sidebar.caption(f"Showing {len(gun_names)} of {len(guns)} weapons")

    if not gun_names:
        st.sidebar.warning("No weapons match the selected filters.")
        st.stop()

    selected_gun_name = st.sidebar.selectbox(
        t("sidebar.choose_weapon"),
        gun_names,
        help=t("sidebar.choose_weapon"),
    )

    selected_gun = gun_options[selected_gun_name]
    weapon_id = selected_gun["id"]
    weapon_stats = item_lookup[weapon_id]["stats"]
    presets = item_lookup[weapon_id]["presets"]

    # Build compatibility map immediately for sidebar filters
    # This ensures we only show compatible mods in the inclusion/exclusion lists
    compat_map = get_compat_map(weapon_id, item_lookup)
    reachable_ids = set(compat_map["reachable_items"].keys())

    # Display weapon image
    weapon_image_url = weapon_stats.get("default_preset_image") or get_image_url(selected_gun, prefer_high_res=True)
    if weapon_image_url:
        st.sidebar.image(weapon_image_url, width="stretch")

    # Show base weapon stats in a folded expander
    with st.sidebar.expander(f"📊 {t('sidebar.base_stats')}", expanded=False):
        # Basic stats
        st.markdown(f"- {t('sidebar.ergonomics')}: {weapon_stats.get('naked_ergonomics', 0):.0f}")
        st.markdown(f"- {t('sidebar.recoil_v')}: {weapon_stats.get('naked_recoil_v', 0):.0f}")
        st.markdown(f"- {t('sidebar.recoil_h')}: {weapon_stats.get('naked_recoil_h', 0):.0f}")

        # Caliber and fire rate
        caliber = weapon_stats.get('caliber', '')
        if caliber:
            # Format caliber name (remove "Caliber" prefix if present)
            caliber_display = caliber.replace("Caliber", "").strip()
            st.markdown(f"- {t('sidebar.caliber')}: {caliber_display}")

        fire_rate = weapon_stats.get('fire_rate', 0)
        if fire_rate:
            st.markdown(f"- {t('sidebar.fire_rate')}: {fire_rate} RPM")

        fire_modes = weapon_stats.get('fire_modes', [])
        if fire_modes:
            modes_display = ", ".join(fire_modes)
            st.markdown(f"- {t('sidebar.fire_modes')}: {modes_display}")

        # Weapon handling properties
        st.markdown("---")
        camera_snap = weapon_stats.get('camera_snap', 0)
        if camera_snap:
            st.markdown(f"- {t('sidebar.camera_snap')}: {camera_snap}")

        center_of_impact = weapon_stats.get('center_of_impact', 0)
        if center_of_impact:
            st.markdown(f"- {t('sidebar.center_of_impact')}: {center_of_impact}")

        deviation_max = weapon_stats.get('deviation_max', 0)
        if deviation_max:
            st.markdown(f"- {t('sidebar.deviation_max')}: {deviation_max}")

        deviation_curve = weapon_stats.get('deviation_curve', 0)
        if deviation_curve:
            st.markdown(f"- {t('sidebar.deviation_curve')}: {deviation_curve}")

        recoil_angle = weapon_stats.get('recoil_angle', 0)
        if recoil_angle:
            st.markdown(f"- {t('sidebar.recoil_angle')}: {recoil_angle}°")

        recoil_dispersion = weapon_stats.get('recoil_dispersion', 0)
        if recoil_dispersion:
            st.markdown(f"- {t('sidebar.recoil_dispersion')}: {recoil_dispersion}")

    # Show all presets info
    if presets:
        with st.sidebar.expander(f"📦 {t('sidebar.available_presets')} ({len(presets)})"):
            for preset in presets:
                preset_name = preset.get("name", "Unknown")
                preset_price = preset.get("price", 0)
                preset_items = preset.get("items", [])
                st.markdown(f"**{preset_name}**")
                st.markdown(f"  - Price: ₽{preset_price:,}")
                st.markdown(f"  - Items: {len(preset_items)}")
                st.markdown("---")

    # Player Level and Trader Settings
    st.sidebar.markdown("---")
    st.sidebar.header(f"👤 {t('sidebar.player_trader_access')}")

    # Define traders first (needed for cookie sync)
    traders = [
        ("prapor", "Prapor"),
        ("skier", "Skier"),
        ("peacekeeper", "Peacekeeper"),
        ("mechanic", "Mechanic"),
        ("jaeger", "Jaeger"),
    ]

    # Sync settings from cookies (run once when cookies are available)
    cookies = cookie_manager.get_all()
    
    # Only run sync if not yet confirmed synced
    if not st.session_state.get("cookies_synced", False):
        # Proceed only if cookies are loaded (not None) and we find our keys
        if cookies:
            found_data = False
            try:
                if "player_level" in cookies:
                    st.session_state.player_level = int(cookies["player_level"])
                    found_data = True
                
                if "flea_available" in cookies:
                    st.session_state.flea_available = str(cookies["flea_available"]).lower() == "true"
                    found_data = True
                    
                for t_key, _ in traders:
                    s_key = f"trader_{t_key}"
                    if s_key in cookies:
                        st.session_state[s_key] = int(cookies[s_key])
                        found_data = True
                        
                # Only mark as synced and rerun if we actually loaded data
                if found_data:
                    st.session_state.cookies_synced = True
                    st.rerun()
                    
            except Exception as e:
                logger.error(f"Error syncing cookies: {e}")

    # Player level input
    player_level = st.sidebar.number_input(
        t("sidebar.player_level"),
        min_value=1,
        max_value=79,
        value=79,
        key="player_level",
        help=t("sidebar.player_level_help"),
        on_change=lambda: save_cookie("player_level", st.session_state.player_level)
    )

    # Flea market access - automatically disabled if player level < 15
    flea_unlocked = player_level >= 15
    if flea_unlocked:
        # If accessing for the first time without defaults, ensure key exists with default
        if "flea_available" not in st.session_state:
             st.session_state.flea_available = True
             
        flea_available = st.sidebar.checkbox(
            t("sidebar.flea_market_access"),
            value=True, # fallback if key missing
            key="flea_available",
            help=t("sidebar.flea_help"),
            on_change=lambda: save_cookie("flea_available", st.session_state.flea_available)
        )
    else:
        flea_available = False
        st.sidebar.checkbox(
            t("sidebar.flea_market_access"),
            value=False,
            disabled=True,
            help=t("sidebar.flea_unlocks_at_15"),
            key="flea_disabled_display" # Unique key to prevent conflict
        )
        st.sidebar.caption(f"⚠️ {t('sidebar.flea_unlocks_at_15')}")

    # Initialize trader levels defaults if not present
    for trader_key, _ in traders:
        session_key = f"trader_{trader_key}"
        if session_key not in st.session_state:
            st.session_state[session_key] = 4

    # Build trader_levels dict from session state (always available)
    trader_levels = {key: st.session_state[f"trader_{key}"] for key, _ in traders}

    # Individual trader levels in an expander
    with st.sidebar.expander(t("sidebar.trader_levels"), expanded=False):
        # Quick preset buttons
        preset_col1, preset_col2 = st.columns(2)
        if preset_col1.button(t("sidebar.all_ll1"), key="traders_ll1", use_container_width=True):
            for trader_key, _ in traders:
                key = f"trader_{trader_key}"
                st.session_state[key] = 1
                save_cookie(key, 1)
            st.rerun()
        if preset_col2.button(t("sidebar.all_ll4"), key="traders_ll4", use_container_width=True):
            for trader_key, _ in traders:
                key = f"trader_{trader_key}"
                st.session_state[key] = 4
                save_cookie(key, 4)
            st.rerun()

        st.markdown("---")

        # Individual trader sliders
        for trader_key, trader_name in traders:
            session_key = f"trader_{trader_key}"
            trader_levels[trader_key] = st.select_slider(
                trader_name,
                options=[1, 2, 3, 4],
                value=st.session_state[session_key],
                key=session_key,
                on_change=lambda k=session_key: save_cookie(k, st.session_state[k])
            )

    # Show summary of constraints
    non_maxed = [name for key, name in traders if trader_levels.get(key, 4) < 4]
    if non_maxed or not flea_available or player_level < 79:
        constraints_info = []
        if player_level < 79:
            constraints_info.append(t("sidebar.level", level=player_level))
        if non_maxed:
            constraints_info.append(t("sidebar.traders_below_ll4", count=len(non_maxed)))
        if not flea_available:
            constraints_info.append(t("sidebar.no_flea"))
        st.sidebar.caption(f"⚠️ {t('sidebar.limited')}: {', '.join(constraints_info)}")

    st.sidebar.markdown("---")
    
    # Hard Constraints Sidebar Section
    with st.sidebar.expander(f"🛡️ {t('optimize.constraints_header')}", expanded=False):
        # Budget constraint
        enable_budget = st.checkbox(t("constraints.budget_limit"), key="sb_budget_check")
        max_price = None
        if enable_budget:
            max_price = st.number_input(
                t("constraints.max_budget"),
                min_value=0,
                max_value=10000000,
                value=500000,
                step=50000,
                help=t("constraints.max_budget_help"),
                key="sb_max_price",
            )

        # Minimum ergonomics constraint
        enable_min_ergo = st.checkbox(t("constraints.min_ergonomics"), key="sb_ergo_check")
        min_ergonomics = None
        if enable_min_ergo:
            min_ergonomics = st.slider(
                t("constraints.min_ergo"),
                min_value=0,
                max_value=100,
                value=50,
                help=t("constraints.min_ergo_help"),
                key="sb_min_ergo",
            )

        # Maximum recoil constraint
        enable_max_recoil = st.checkbox(t("constraints.max_recoil"), key="sb_recoil_check")
        max_recoil_v = None
        if enable_max_recoil:
            naked_recoil = weapon_stats.get("naked_recoil_v", 100)
            max_recoil_v = st.slider(
                t("constraints.max_recoil_v"),
                min_value=20,
                max_value=int(naked_recoil),
                value=int(naked_recoil * 0.7),
                help=t("constraints.max_recoil_help", naked=f"{naked_recoil:.0f}"),
                key="sb_max_recoil",
            )

        # Minimum magazine capacity constraint
        enable_min_mag = st.checkbox(t("constraints.min_mag_capacity"), key="sb_mag_check")
        min_mag_capacity = None
        if enable_min_mag:
            min_mag_capacity = st.number_input(
                t("constraints.min_mag"),
                min_value=5,
                max_value=100,
                value=30,
                step=5,
                help=t("constraints.min_mag_help"),
                key="sb_min_mag",
            )

        # Minimum sighting range constraint
        enable_min_sight = st.checkbox(t("constraints.min_sighting_range"), key="sb_sight_check")
        min_sighting_range = None
        if enable_min_sight:
            min_sighting_range = st.number_input(
                t("constraints.min_sight"),
                min_value=50,
                max_value=1000,
                value=200,
                step=50,
                help=t("constraints.min_sight_help"),
                key="sb_min_sight",
            )

        # Maximum weight constraint
        enable_max_weight = st.checkbox(t("constraints.max_weight"), key="sb_weight_check")
        max_weight = None
        if enable_max_weight:
            base_weight = weapon_stats.get("weight", 3.0)
            max_weight = st.number_input(
                t("constraints.max_weight_kg"),
                min_value=1.0,
                max_value=15.0,
                value=round(base_weight + 3.0, 1),
                step=0.5,
                format="%.1f",
                help=t("constraints.max_weight_help", base=f"{base_weight:.2f}"),
                key="sb_max_weight",
            )

    # Include/Exclude Mods Sidebar Section
    with st.sidebar.expander(f"➕/➖ {t('sidebar.include_exclude')}", expanded=False):
        # Prepare data for selection - FILTER by compatibility
        compatible_mods = [m for m in mods if m["id"] in reachable_ids]
        
        all_mod_names = sorted(list(set(m["name"] for m in compatible_mods if m.get("name"))))
        all_categories = sorted(list(set(m.get("bsgCategory", {}).get("name") for m in compatible_mods if m.get("bsgCategory", {}).get("name"))))
        mod_name_to_id = {m["name"]: m["id"] for m in compatible_mods if m.get("name")}

        sel_include_cats = st.multiselect(t("sidebar.require_categories"), all_categories, key="sb_inc_cat")
        sel_exclude_cats = st.multiselect(t("sidebar.ban_categories"), all_categories, key="sb_exc_cat")
        
        sel_include_items = st.multiselect(t("sidebar.require_items"), all_mod_names, key="sb_inc_item")
        sel_exclude_items = st.multiselect(t("sidebar.ban_items"), all_mod_names, key="sb_exc_item")

        # Convert to list of lists for AND logic (each sublist is OR, here size 1)
        include_categories = [[c] for c in sel_include_cats] if sel_include_cats else None
        exclude_categories = set(sel_exclude_cats) if sel_exclude_cats else None
        include_items = set(mod_name_to_id[n] for n in sel_include_items) if sel_include_items else None
        exclude_items = set(mod_name_to_id[n] for n in sel_exclude_items) if sel_exclude_items else None

    # Create tabs for Explore, Optimize, and Gunsmith
    tab_explore, tab_optimize, tab_gunsmith = st.tabs([
        f"📊 {t('tabs.explore')}", 
        f"🚀 {t('tabs.optimize')}",
        f"🛠️ {t('tabs.gunsmith')}"
    ])

    # ==================== EXPLORE TAB ====================
    with tab_explore:
        st.header(t("explore.header"))
        st.markdown(t("explore.description"))

        explore_tradeoff = st.selectbox(
            t("explore.tradeoff_label"),
            [t("explore.ergo_vs_recoil"), t("explore.ergo_vs_price"), t("explore.recoil_vs_price")],
            help=t("explore.tradeoff_label"),
        )
        
        st.markdown("---")

        # Explore button
        explore_button = st.button(f"📊 {t('explore.explore_btn')}", type="primary", key="explore_btn", width="stretch")

        if explore_button:
            logger.info(f"User started Pareto exploration for {selected_gun_name}")
            with st.status(t("status.exploring"), expanded=True) as status:
                # Build compatibility map (cached per weapon)
                try:
                    status.update(label=t("status.building_compat"))
                    compat_map = get_compat_map(weapon_id, item_lookup)
                    st.write(f"✓ {t('status.found_mods', count=len(compat_map['reachable_items']))}")
                except Exception as e:
                    status.update(label=t("status.failed"), state="error")
                    st.error(f"{t('status.failed_compat')}: {e}")
                    st.stop()

                try:
                    ignore_map = {
                        t("explore.ergo_vs_recoil"): "price",
                        t("explore.ergo_vs_price"): "recoil",
                        t("explore.recoil_vs_price"): "ergo",
                    }
                    status.update(label=t("status.running_passes"))
                    st.write(f"✓ {t('status.sampling', points=8)}")
                    frontier = explore_pareto(
                        weapon_id,
                        item_lookup,
                        compat_map,
                        ignore=ignore_map[explore_tradeoff],
                        max_price=max_price,
                        min_ergonomics=min_ergonomics,
                        max_recoil_v=max_recoil_v,
                        min_mag_capacity=min_mag_capacity,
                        min_sighting_range=min_sighting_range,
                        max_weight=max_weight,
                        include_items=include_items,
                        exclude_items=exclude_items,
                        include_categories=include_categories,
                        exclude_categories=exclude_categories,
                        steps=8,
                        trader_levels=trader_levels,
                        flea_available=flea_available,
                        player_level=player_level,
                    )
                    status.update(label=t("status.exploration_complete"), state="complete")
                except Exception as e:
                    status.update(label=t("status.exploration_failed"), state="error")
                    st.error(f"{t('status.exploration_failed')}: {e}")
                    st.stop()

            if not frontier:
                st.error(t("explore.no_feasible"))
            else:
                # Store frontier in session state for click interactions
                st.session_state.explore_frontier = frontier
                st.session_state.explore_ignore = ignore_map[explore_tradeoff]
                st.session_state.explore_weapon_id = weapon_id
                st.session_state.explore_selected_idx = None  # Clear selection on new exploration
                st.session_state.explore_constraints = {
                    "max_price": max_price,
                    "min_ergonomics": min_ergonomics,
                    "max_recoil_v": max_recoil_v,
                    "min_mag_capacity": min_mag_capacity,
                    "min_sighting_range": min_sighting_range,
                    "max_weight": max_weight,
                    "trader_levels": trader_levels,
                    "flea_available": flea_available,
                    "player_level": player_level,
                }

        # Display frontier if available in session state
        if "explore_frontier" in st.session_state and st.session_state.get("explore_weapon_id") == weapon_id:
            frontier = st.session_state.explore_frontier
            
            # Recalculate prices ensuring consistency between graph and detail view
            for point in frontier:
                cost_data = calculate_build_cost(
                    point["selected_items"],
                    point.get("selected_preset"),
                    item_lookup,
                    weapon_stats,
                    presets,
                    selected_gun
                )
                point["price"] = cost_data["total_cost"]
                
            ignore = st.session_state.explore_ignore
            stored_constraints = st.session_state.explore_constraints

            if ignore == "price":
                chart_x, chart_y = t("chart.ergonomics"), t("chart.recoil_v")
                x_data = [p["ergo"] for p in frontier]
                y_data = [p["recoil_v"] for p in frontier]
                tip = t("explore.tip_recoil_at_ergo")
            elif ignore == "recoil":
                chart_x, chart_y = t("chart.ergonomics"), t("chart.price")
                x_data = [p["ergo"] for p in frontier]
                y_data = [p["price"] for p in frontier]
                tip = t("explore.tip_price_at_ergo")
            else:
                chart_x, chart_y = t("chart.recoil_v"), t("chart.price")
                x_data = [p["recoil_v"] for p in frontier]
                y_data = [p["price"] for p in frontier]
                tip = t("explore.tip_price_at_recoil")

            # Show active constraints
            constraints_display = []
            if stored_constraints.get("max_price"):
                constraints_display.append(t("constraints.budget_le", value=f"{stored_constraints['max_price']:,}"))
            if stored_constraints.get("min_ergonomics"):
                constraints_display.append(t("constraints.ergo_ge", value=stored_constraints['min_ergonomics']))
            if stored_constraints.get("max_recoil_v"):
                constraints_display.append(t("constraints.recoil_le", value=stored_constraints['max_recoil_v']))
            if stored_constraints.get("min_mag_capacity"):
                constraints_display.append(f"{t('constraints.min_mag')}: {stored_constraints['min_mag_capacity']}")
            if stored_constraints.get("min_sighting_range"):
                constraints_display.append(f"{t('constraints.min_sight')}: {stored_constraints['min_sighting_range']}")
            if stored_constraints.get("max_weight"):
                constraints_display.append(f"{t('constraints.max_weight')}: {stored_constraints['max_weight']}")

            if constraints_display:
                st.info(f"{t('explore.active_constraints')}: {', '.join(constraints_display)}")

            # Create Plotly chart with click selection
            fig = go.Figure()

            # Add line connecting points
            fig.add_trace(go.Scatter(
                x=x_data,
                y=y_data,
                mode='lines',
                line=dict(color='rgba(99, 110, 250, 0.5)', width=2),
                hoverinfo='skip',
                showlegend=False,
            ))

            # Add clickable points with index as custom data
            hover_texts = [
                f"{chart_x}: {x}<br>{chart_y}: {y:,.0f}" if isinstance(y, (int, float)) and y > 1000 
                else f"{chart_x}: {x}<br>{chart_y}: {y}"
                for x, y in zip(x_data, y_data)
            ]
            
            fig.add_trace(go.Scatter(
                x=x_data,
                y=y_data,
                mode='markers',
                marker=dict(size=12, color='rgb(99, 110, 250)', line=dict(width=2, color='white')),
                hovertemplate='%{hovertext}<extra></extra>',
                hovertext=hover_texts,
                customdata=list(range(len(frontier))),
                showlegend=False,
            ))

            # Highlight selected point if any
            selected_idx = st.session_state.get("explore_selected_idx")
            if selected_idx is not None and 0 <= selected_idx < len(frontier):
                fig.add_trace(go.Scatter(
                    x=[x_data[selected_idx]],
                    y=[y_data[selected_idx]],
                    mode='markers',
                    marker=dict(size=18, color='red', symbol='circle', line=dict(width=2, color='white')),
                    hoverinfo='skip',
                    showlegend=False,
                ))

            fig.update_layout(
                xaxis_title=chart_x,
                yaxis_title=chart_y,
                height=350,
                margin=dict(l=50, r=20, t=20, b=50),
                xaxis=dict(zeroline=False),
                yaxis=dict(zeroline=False),
            )

            # Display chart with click selection
            event = st.plotly_chart(
                fig,
                use_container_width=True,
                on_select="rerun",
                selection_mode="points",
                key="pareto_frontier_chart",
            )

            # Handle click events
            if event and event.selection and event.selection.get("points"):
                point = event.selection["points"][0]
                if "customdata" in point:
                    clicked_idx = point["customdata"]
                    if clicked_idx != st.session_state.get("explore_selected_idx"):
                        st.session_state.explore_selected_idx = clicked_idx
                        st.rerun()

            # Display as sortable table
            col_ergo = t("table.ergo")
            col_recoil_pct = t("table.recoil_pct")
            col_recoil_v = t("table.recoil_v")
            col_recoil_h = t("table.recoil_h")
            col_price = t("table.price")

            frontier_df = pd.DataFrame([
                {
                    col_ergo: point["ergo"],
                    col_recoil_pct: f"{point['recoil_pct']:+.1f}%",
                    col_recoil_v: round(point["recoil_v"], 1),
                    col_recoil_h: round(point["recoil_h"], 1),
                    col_price: point["price"],
                }
                for point in frontier
            ])

            st.dataframe(
                frontier_df,
                column_config={
                    col_ergo: st.column_config.NumberColumn(col_ergo, format="%.1f"),
                    col_recoil_pct: st.column_config.TextColumn(col_recoil_pct),
                    col_recoil_v: st.column_config.NumberColumn(col_recoil_v, format="%.1f"),
                    col_recoil_h: st.column_config.NumberColumn(col_recoil_h, format="%.1f"),
                    col_price: st.column_config.NumberColumn(col_price, format="₽ %d"),
                },
                hide_index=True,
                use_container_width=True,
            )

            st.caption(tip)

            # Display build details for selected point
            if selected_idx is not None and 0 <= selected_idx < len(frontier):
                st.markdown("---")
                st.subheader(f"🔧 {t('explore.selected_build')}")
                
                selected_point = frontier[selected_idx]
                
                # Build a result dict compatible with display_optimization_results
                result = {
                    "status": selected_point.get("status", "optimal"),
                    "selected_items": selected_point.get("selected_items", []),
                    "selected_preset": selected_point.get("selected_preset"),
                    "objective_value": 0,  # Not applicable for Pareto exploration points
                }
                
                # Build constraints dict for display
                display_constraints = {
                    "max_price": stored_constraints.get("max_price"),
                    "min_ergonomics": stored_constraints.get("min_ergonomics"),
                    "max_recoil_v": stored_constraints.get("max_recoil_v"),
                    "min_mag_capacity": stored_constraints.get("min_mag_capacity"),
                    "min_sighting_range": stored_constraints.get("min_sighting_range"),
                    "max_weight": stored_constraints.get("max_weight"),
                    "trader_levels": stored_constraints.get("trader_levels"),
                    "flea_available": stored_constraints.get("flea_available"),
                    "player_level": stored_constraints.get("player_level"),
                }
                
                display_optimization_results(
                    result, item_lookup, weapon_stats, presets, selected_gun, display_constraints
                )
            else:
                st.info(f"💡 {t('explore.click_point_hint')}")

    # ==================== OPTIMIZE TAB ====================
    with tab_optimize:
        st.header(t("optimize.header"))

        st.subheader(t("optimize.weights_header"))

        # Initialize weights in session state if not present
        if "weight_ergo" not in st.session_state:
            st.session_state.weight_ergo = 33
        if "weight_recoil" not in st.session_state:
            st.session_state.weight_recoil = 67
        if "weight_price" not in st.session_state:
            st.session_state.weight_price = 0

        # Preset buttons
        preset_cols = st.columns(4)
        if preset_cols[0].button(t("optimize.preset_ergo"), help=t("optimize.preset_ergo_help"), key="preset_ergo"):
            st.session_state.weight_ergo = 98
            st.session_state.weight_recoil = 1
            st.session_state.weight_price = 1
            st.rerun()
        if preset_cols[1].button(t("optimize.preset_recoil"), help=t("optimize.preset_recoil_help"), key="preset_recoil"):
            st.session_state.weight_ergo = 1
            st.session_state.weight_recoil = 98
            st.session_state.weight_price = 1
            st.rerun()
        if preset_cols[2].button(t("optimize.preset_price"), help=t("optimize.preset_price_help"), key="preset_price"):
            st.session_state.weight_ergo = 1
            st.session_state.weight_recoil = 1
            st.session_state.weight_price = 98
            st.rerun()
        if preset_cols[3].button(t("optimize.preset_balanced"), help=t("optimize.preset_balanced_help"), key="preset_balanced"):
            st.session_state.weight_ergo = 34
            st.session_state.weight_recoil = 33
            st.session_state.weight_price = 33
            st.rerun()

        # Get current weights from session state
        w_ergo = st.session_state.weight_ergo
        w_recoil = st.session_state.weight_recoil
        w_price = st.session_state.weight_price

        # Create ternary plot
        fig = go.Figure()

        # Generate clickable grid points
        grid_a, grid_b, grid_c = [], [], []
        for a in range(0, 101, 10):
            for b in range(0, 101 - a, 10):
                c = 100 - a - b
                grid_a.append(a)
                grid_b.append(b)
                grid_c.append(c)

        # Add clickable grid
        fig.add_trace(go.Scatterternary(
            a=grid_a,
            b=grid_b,
            c=grid_c,
            mode='markers',
            marker=dict(size=8, color='lightgray', opacity=0.3),
            hovertemplate=f"{t('optimize.preset_ergo')}: %{{a}}%<br>{t('optimize.preset_recoil')}: %{{b}}%<br>{t('optimize.preset_price')}: %{{c}}%<extra></extra>",
            showlegend=False,
        ))

        # Add current position marker
        fig.add_trace(go.Scatterternary(
            a=[w_ergo],
            b=[w_recoil],
            c=[w_price],
            mode='markers',
            marker=dict(size=18, color='red', symbol='circle', line=dict(width=2, color='white')),
            name='Current',
            hovertemplate=f"<b>Current</b><br>{t('optimize.preset_ergo')}: %{{a}}%<br>{t('optimize.preset_recoil')}: %{{b}}%<br>{t('optimize.preset_price')}: %{{c}}%<extra></extra>",
        ))

        # Add corner labels
        fig.add_trace(go.Scatterternary(
            a=[100, 0, 0],
            b=[0, 100, 0],
            c=[0, 0, 100],
            mode='text',
            text=[t("optimize.preset_ergo"), t("optimize.preset_recoil"), t("optimize.preset_price")],
            textposition=['top center', 'bottom left', 'bottom right'],
            textfont=dict(size=11, color='gray'),
            showlegend=False,
            hoverinfo='skip',
        ))

        fig.update_layout(
            ternary=dict(
                sum=100,
                aaxis=dict(title="", showticklabels=False, linewidth=1),
                baxis=dict(title="", showticklabels=False, linewidth=1),
                caxis=dict(title="", showticklabels=False, linewidth=1),
            ),
            showlegend=False,
            margin=dict(l=30, r=30, t=30, b=10),
            height=280,
        )

        # Display chart with click selection
        event = st.plotly_chart(
            fig,
            width="stretch",
            on_select="rerun",
            selection_mode="points",
            key="ternary_weights",
        )

        # Handle click events
        if event and event.selection and event.selection.get("points"):
            point = event.selection["points"][0]
            if "a" in point and "b" in point and "c" in point:
                new_a = round(point["a"])
                new_b = round(point["b"])
                new_c = round(point["c"])
                if (new_a, new_b, new_c) != (w_ergo, w_recoil, w_price):
                    st.session_state.weight_ergo = new_a
                    st.session_state.weight_recoil = new_b
                    st.session_state.weight_price = new_c
                    st.rerun()

        st.caption(f"{t('optimize.preset_ergo')}: {w_ergo}% | {t('optimize.preset_recoil')}: {w_recoil}% | {t('optimize.preset_price')}: {w_price}%")

        # Manual weight sliders
        with st.expander(f"⚙️ {t('optimize.manual_weights')}", expanded=False):
            st.slider(t("optimize.preset_ergo"), 0, 100, key="weight_ergo", help=t("optimize.preset_ergo_help"))
            st.slider(t("optimize.preset_recoil"), 0, 100, key="weight_recoil", help=t("optimize.preset_recoil_help"))
            st.slider(t("optimize.preset_price"), 0, 100, key="weight_price", help=t("optimize.preset_price_help"))
            st.info(t("optimize.weight_info"))

        # Convert percentages to weights
        total = w_ergo + w_recoil + w_price
        if total > 0:
            ergo_weight = (w_ergo / 100) * 2
            recoil_weight = (w_recoil / 100) * 2
            price_weight = (w_price / 100) * 2
        else:
            ergo_weight, recoil_weight, price_weight = 1.0, 1.0, 0.0

        # Optimize button
        st.markdown("---")
        optimize_button = st.button(f"🚀 {t('optimize.optimize_btn')}", type="primary", key="optimize_btn", width="stretch")

        if optimize_button:
            logger.info(f"User started optimization for {selected_gun_name} (weights: ergo={w_ergo}%, recoil={w_recoil}%, price={w_price}%)")
            with st.status(t("status.optimizing"), expanded=True) as status:
                # Build compatibility map (cached per weapon)
                try:
                    status.update(label=t("status.building_compat"))
                    compat_map = get_compat_map(weapon_id, item_lookup)
                    st.write(f"✓ {t('status.found_mods', count=len(compat_map['reachable_items']))}")
                except Exception as e:
                    status.update(label=t("status.failed"), state="error")
                    st.error(f"{t('status.failed_compat')}: {e}")
                    st.stop()

                # Run optimization
                try:
                    status.update(label=t("status.running_solver"))
                    st.write(f"✓ {t('status.building_model')}")
                    result = optimize_weapon(
                        weapon_id,
                        item_lookup,
                        compat_map,
                        max_price=max_price,
                        min_ergonomics=min_ergonomics,
                        max_recoil_v=max_recoil_v,
                        min_mag_capacity=min_mag_capacity,
                        min_sighting_range=min_sighting_range,
                        max_weight=max_weight,
                        include_items=include_items,
                        exclude_items=exclude_items,
                        include_categories=include_categories,
                        exclude_categories=exclude_categories,
                        ergo_weight=ergo_weight,
                        recoil_weight=recoil_weight,
                        price_weight=price_weight,
                        trader_levels=trader_levels,
                        flea_available=flea_available,
                        player_level=player_level,
                    )
                    if result["status"] == "infeasible":
                        status.update(label=t("status.no_solution"), state="error")
                    else:
                        status_key = f"results.{result['status']}" if result['status'] in ['optimal', 'feasible'] else "results.feasible"
                        status.update(label=f"{t('results.optimization_status')} {t(status_key)}", state="complete")
                except Exception as e:
                    status.update(label=t("status.optimization_failed"), state="error")
                    st.error(f"{t('status.optimization_failed')}: {e}")
                    st.stop()

            # Display results
            # Helper to map IDs to names for display/export
            id_to_name = lambda ids: sorted([item_lookup[i]["data"]["name"] for i in ids if i in item_lookup]) if ids else None
            
            constraints = {
                "max_price": max_price,
                "min_ergonomics": min_ergonomics,
                "max_recoil_v": max_recoil_v,
                "min_mag_capacity": min_mag_capacity,
                "min_sighting_range": min_sighting_range,
                "max_weight": max_weight,
                "include_items": id_to_name(include_items),
                "exclude_items": id_to_name(exclude_items),
                "include_categories": sorted(list(include_categories)) if include_categories else None,
                "exclude_categories": sorted(list(exclude_categories)) if exclude_categories else None,
                "trader_levels": trader_levels,
                "flea_available": flea_available,
                "player_level": player_level,
            }
            display_optimization_results(
                result, item_lookup, weapon_stats, presets, selected_gun, constraints
            )

            # Export buttons
            if result["status"] != "infeasible":
                st.markdown("---")
                st.subheader(t("export.header"))

                json_data, markdown_text = generate_build_export(
                    result, item_lookup, weapon_stats, presets, selected_gun, constraints
                )

                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        label=f"📥 {t('export.download_json')}",
                        data=json.dumps(json_data, indent=2),
                        file_name=f"{selected_gun['name'].replace(' ', '_')}_build.json",
                        mime="application/json",
                    )
                with col2:
                    st.download_button(
                        label=f"📥 {t('export.download_markdown')}",
                        data=markdown_text,
                        file_name=f"{selected_gun['name'].replace(' ', '_')}_build.md",
                        mime="text/markdown",
                    )

        else:
            # Initial state for Optimize tab
            st.info(f"{t('optimize.how_it_works')} {t('optimize.how_step_3').replace('**', '')}")
            st.markdown(f"""
            **{t('optimize.how_it_works')}**
            1. {t('optimize.how_step_1')}
            2. {t('optimize.how_step_2')}
            3. {t('optimize.how_step_3')}

            **{t('optimize.tip')}** {t('optimize.tip_text')}
            """)


    # ==================== GUNSMITH TAB ====================
    with tab_gunsmith:
        st.header(t("gunsmith.header"))
        
        task_options = {task["task_name"]: task for task in tasks}
        selected_task_name = st.selectbox(t("gunsmith.select_task"), list(task_options.keys()))
        
        if selected_task_name:
            task = task_options[selected_task_name]
            st.markdown(f"**{t('gunsmith.task_requirements')}**")
            
            # Show requirements
            req_cols = st.columns(3)
            with req_cols[0]:
                st.write(f"**{t('gunsmith.weapon')}:** {task['weapon_name']}")
            with req_cols[1]:
                constraints = task.get("constraints", {})
                if constraints:
                    # Map task keys to locale keys
                    key_map = {
                        "min_ergonomics": "constraints.min_ergonomics",
                        "max_recoil_v": "constraints.max_recoil",
                        "max_recoil_sum": "constraints.max_recoil_sum",
                        "max_weight": "constraints.max_weight",
                        "min_mag_capacity": "constraints.min_mag_capacity",
                        "min_sighting_range": "constraints.min_sighting_range"
                    }
                    for k, v in constraints.items():
                        label = t(key_map.get(k, k))
                        st.write(f"- {label}: {v}")
            with req_cols[2]:
                required_items = task.get("required_items", [])
                required_categories = task.get("required_categories", [])
                required_groups = task.get("required_category_groups", [])
                
                if required_items or required_categories or required_groups:
                    st.write(f"**{t('gunsmith.required_items')}:**")
                    for item in required_items:
                        st.write(f"- {item}")
                    for cat in required_categories:
                        st.write(f"- Category: {cat}")
                    for group in required_groups:
                        st.write(f"- One of: {', '.join(group)}")
            
            st.markdown("---")
            
            if st.button(t("gunsmith.optimize_btn"), type="primary"):
                logger.info(f"User started Gunsmith task optimization for {selected_task_name}")
                target_gun_id = None
                for gun in guns:
                    if gun["name"] == task["weapon_name"]:
                        target_gun_id = gun["id"]
                        break
                
                if not target_gun_id:
                    st.error(t("gunsmith.weapon_not_found", name=task["weapon_name"]))
                else:
                    task_include_items = set()
                    missing_items = []
                    for item_name in task.get("required_items", []):
                        resolved_id = resolve_item_id(item_name, item_lookup)
                        if resolved_id:
                            task_include_items.add(resolved_id)
                        else:
                            missing_items.append(item_name)
                    
                    # Resolved categories
                    task_include_categories = []
                    if "required_category_groups" in task:
                        task_include_categories.extend(task["required_category_groups"])
                    if "required_categories" in task:
                        # Convert simple list to list of lists (AND logic)
                        for cat in task["required_categories"]:
                            task_include_categories.append([cat])
                    
                    if missing_items:
                        for m in missing_items:
                            st.warning(t("gunsmith.missing_item", name=m))
                    
                    with st.status(t("status.optimizing"), expanded=True) as status:
                        try:
                            status.update(label=t("status.building_compat"))
                            target_compat_map = get_compat_map(target_gun_id, item_lookup)
                            
                            status.update(label=t("status.running_solver"))
                            c = task.get("constraints", {})
                            
                            result = optimize_weapon(
                                target_gun_id,
                                item_lookup,
                                target_compat_map,
                                max_price=c.get("max_price"),
                                min_ergonomics=c.get("min_ergonomics"),
                                max_recoil_v=c.get("max_recoil_v"),
                                max_recoil_sum=c.get("max_recoil_sum"),
                                min_mag_capacity=c.get("min_mag_capacity"),
                                min_sighting_range=c.get("min_sighting_range"),
                                max_weight=c.get("max_weight"),
                                include_items=task_include_items,
                                include_categories=task_include_categories,
                                price_weight=100,
                                ergo_weight=0,
                                recoil_weight=0,
                                trader_levels=trader_levels,
                                flea_available=flea_available,
                                player_level=player_level,
                            )
                            
                            if result["status"] == "infeasible":
                                status.update(label=t("status.no_solution"), state="error")
                            else:
                                status.update(label=t("results.optimal"), state="complete")
                                
                                target_stats = item_lookup[target_gun_id]["stats"]
                                target_presets = item_lookup[target_gun_id]["presets"]
                                target_gun_data = item_lookup[target_gun_id]["data"]
                                
                                task_constraints = c.copy()
                                task_constraints["include_items"] = sorted([item_lookup[i]["data"]["name"] for i in task_include_items])
                                task_constraints["include_categories"] = sorted(list(task_include_categories))
                                
                                display_optimization_results(
                                    result, item_lookup, target_stats, target_presets, target_gun_data, task_constraints
                                )
                                
                        except Exception as e:
                            status.update(label="Error", state="error")
                            st.error(f"Error: {e}")


if __name__ == "__main__":
    main()
