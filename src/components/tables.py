"""
Table display components for mods and items.
"""

from typing import List, Optional

import streamlit as st

from i18n import t
from src.utils import get_image_url, get_best_offer_display


def display_mods_table(
    item_ids: List[str],
    item_lookup: dict,
    show_price: bool = True,
    constraints: Optional[dict] = None,
) -> None:
    """Display a markdown table of mods with their stats.

    Args:
        item_ids: List of item IDs to display
        item_lookup: Item lookup dictionary
        show_price: Whether to show price column
        constraints: Optional dict with trader_levels, flea_available, player_level
                     to show accurate source/price based on what's actually available
    """
    trader_levels = constraints.get("trader_levels") if constraints else None
    flea_available = constraints.get("flea_available", True) if constraints else True
    player_level = constraints.get("player_level") if constraints else None

    rows = []
    for item_id in item_ids:
        if item_id in item_lookup:
            item = item_lookup[item_id]
            stats = item["stats"]
            name = item["data"]["name"].replace("|", "\\|")
            icon_url = get_image_url(item["data"], prefer_icon=True)
            ergo = stats.get("ergonomics", 0)
            recoil = stats.get("recoil_modifier", 0) * 100

            row = {
                "icon": f'<img src="{icon_url}" width="64" style="vertical-align:middle; min-width: 64px;">'
                if icon_url
                else "",
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
                lines.append(
                    f"| {row['icon']} | {row['name']} | {row['ergo']} | {row['recoil']} | {row['price']} | {row['source']} |"
                )
        else:
            header = f"| | {t('table.name')} | {t('table.ergo')} | {t('table.recoil')} |"
            separator = "|:---:|:-----|:----:|:------:|"
            lines = [header, separator]
            for row in rows:
                lines.append(
                    f"| {row['icon']} | {row['name']} | {row['ergo']} | {row['recoil']} |"
                )

        st.markdown(
            """<style>
table { width: 75% !important; margin-left: auto !important; margin-right: auto !important; }
@media (max-width: 1200px) { table { width: 100% !important; } }
</style>"""
            + "\n"
            + "\n".join(lines),
            unsafe_allow_html=True,
        )
