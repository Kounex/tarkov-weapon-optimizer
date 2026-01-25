"""
Explore (Pareto frontier) tab component.
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from loguru import logger

from i18n import t
from src.optimizer import explore_pareto
from src.data import get_compat_map
from src.costs import calculate_build_cost
from src.components import display_optimization_results


def render_explore_tab(
    selected_gun: dict,
    weapon_id: str,
    weapon_stats: dict,
    presets: list,
    item_lookup: dict,
    hard_constraints: dict,
    include_exclude: dict,
    player_level: int,
    flea_available: bool,
    trader_levels: dict,
) -> None:
    """Render the Explore (Pareto frontier) tab.

    Args:
        selected_gun: Selected gun data
        weapon_id: Weapon ID
        weapon_stats: Weapon stats dictionary
        presets: List of available presets
        item_lookup: Item lookup dictionary
        hard_constraints: Hard constraints dictionary
        include_exclude: Include/exclude settings dictionary
        player_level: Player level
        flea_available: Whether flea market is available
        trader_levels: Trader levels dictionary
    """
    st.header(t("explore.header"))
    st.markdown(t("explore.description"))

    explore_tradeoff = st.selectbox(
        t("explore.tradeoff_label"),
        [
            t("explore.ergo_vs_recoil"),
            t("explore.ergo_vs_price"),
            t("explore.recoil_vs_price"),
        ],
        help=t("explore.tradeoff_label"),
    )

    st.markdown("---")

    # Explore button
    explore_button = st.button(
        f"📊 {t('explore.explore_btn')}",
        type="primary",
        key="explore_btn",
        width="stretch",
    )

    if explore_button:
        logger.info(f"User started Pareto exploration for {selected_gun['name']}")
        with st.status(t("status.exploring"), expanded=True) as status:
            try:
                status.update(label=t("status.building_compat"))
                compat_map = get_compat_map(weapon_id, item_lookup)
                st.write(
                    f"✓ {t('status.found_mods', count=len(compat_map['reachable_items']))}"
                )
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
                    max_price=hard_constraints.get("max_price"),
                    min_ergonomics=hard_constraints.get("min_ergonomics"),
                    max_recoil_v=hard_constraints.get("max_recoil_v"),
                    min_mag_capacity=hard_constraints.get("min_mag_capacity"),
                    min_sighting_range=hard_constraints.get("min_sighting_range"),
                    max_weight=hard_constraints.get("max_weight"),
                    include_items=include_exclude.get("include_items"),
                    exclude_items=include_exclude.get("exclude_items"),
                    include_categories=include_exclude.get("include_categories"),
                    exclude_categories=include_exclude.get("exclude_categories"),
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
            # Store frontier in session state
            st.session_state.explore_frontier = frontier
            st.session_state.explore_ignore = ignore_map[explore_tradeoff]
            st.session_state.explore_weapon_id = weapon_id
            st.session_state.explore_selected_idx = None
            st.session_state.explore_constraints = {
                "max_price": hard_constraints.get("max_price"),
                "min_ergonomics": hard_constraints.get("min_ergonomics"),
                "max_recoil_v": hard_constraints.get("max_recoil_v"),
                "min_mag_capacity": hard_constraints.get("min_mag_capacity"),
                "min_sighting_range": hard_constraints.get("min_sighting_range"),
                "max_weight": hard_constraints.get("max_weight"),
                "trader_levels": trader_levels,
                "flea_available": flea_available,
                "player_level": player_level,
            }

    # Display frontier if available
    if (
        "explore_frontier" in st.session_state
        and st.session_state.get("explore_weapon_id") == weapon_id
    ):
        _display_frontier(
            selected_gun,
            weapon_stats,
            presets,
            item_lookup,
        )


def _display_frontier(
    selected_gun: dict,
    weapon_stats: dict,
    presets: list,
    item_lookup: dict,
) -> None:
    """Display the Pareto frontier chart and details."""
    frontier = st.session_state.explore_frontier

    # Recalculate prices for consistency
    for point in frontier:
        cost_data = calculate_build_cost(
            point["selected_items"],
            point.get("selected_preset"),
            item_lookup,
            weapon_stats,
            presets,
            selected_gun,
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
    _display_constraints_info(stored_constraints)

    # Create Plotly chart
    fig = _create_frontier_chart(frontier, x_data, y_data, chart_x, chart_y)

    # Display chart
    event = st.plotly_chart(
        fig,
        width="stretch",
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

    # Display data table
    _display_frontier_table(frontier)
    st.caption(tip)

    # Display selected build details
    selected_idx = st.session_state.get("explore_selected_idx")
    if selected_idx is not None and 0 <= selected_idx < len(frontier):
        _display_selected_build(
            frontier[selected_idx],
            selected_gun,
            weapon_stats,
            presets,
            item_lookup,
            stored_constraints,
        )
    else:
        st.info(f"💡 {t('explore.click_point_hint')}")


def _display_constraints_info(stored_constraints: dict) -> None:
    """Display active constraints info."""
    constraints_display = []
    if stored_constraints.get("max_price"):
        constraints_display.append(
            t("constraints.budget_le", value=f"{stored_constraints['max_price']:,}")
        )
    if stored_constraints.get("min_ergonomics"):
        constraints_display.append(
            t("constraints.ergo_ge", value=stored_constraints["min_ergonomics"])
        )
    if stored_constraints.get("max_recoil_v"):
        constraints_display.append(
            t("constraints.recoil_le", value=stored_constraints["max_recoil_v"])
        )
    if stored_constraints.get("min_mag_capacity"):
        constraints_display.append(
            f"{t('constraints.min_mag')}: {stored_constraints['min_mag_capacity']}"
        )
    if stored_constraints.get("min_sighting_range"):
        constraints_display.append(
            f"{t('constraints.min_sight')}: {stored_constraints['min_sighting_range']}"
        )
    if stored_constraints.get("max_weight"):
        constraints_display.append(
            f"{t('constraints.max_weight')}: {stored_constraints['max_weight']}"
        )

    if constraints_display:
        st.info(f"{t('explore.active_constraints')}: {', '.join(constraints_display)}")


def _create_frontier_chart(
    frontier: list,
    x_data: list,
    y_data: list,
    chart_x: str,
    chart_y: str,
) -> go.Figure:
    """Create the Pareto frontier Plotly chart."""
    fig = go.Figure()

    # Add line connecting points
    fig.add_trace(
        go.Scatter(
            x=x_data,
            y=y_data,
            mode="lines",
            line=dict(color="rgba(99, 110, 250, 0.5)", width=2),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    # Add clickable points
    hover_texts = [
        f"{t('table.ergo')}: {p['ergo']}<br>{t('table.recoil_v')}: {p['recoil_v']:.1f}<br>{t('table.price')}: ₽{p['price']:,}"
        for p in frontier
    ]

    fig.add_trace(
        go.Scatter(
            x=x_data,
            y=y_data,
            mode="markers",
            marker=dict(
                size=12, color="rgb(99, 110, 250)", line=dict(width=2, color="white")
            ),
            hovertemplate="%{hovertext}<extra></extra>",
            hovertext=hover_texts,
            customdata=list(range(len(frontier))),
            showlegend=False,
        )
    )

    # Highlight selected point
    selected_idx = st.session_state.get("explore_selected_idx")
    if selected_idx is not None and 0 <= selected_idx < len(frontier):
        fig.add_trace(
            go.Scatter(
                x=[x_data[selected_idx]],
                y=[y_data[selected_idx]],
                mode="markers",
                marker=dict(
                    size=18,
                    color="red",
                    symbol="circle",
                    line=dict(width=2, color="white"),
                ),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    fig.update_layout(
        xaxis_title=chart_x,
        yaxis_title=chart_y,
        height=350,
        margin=dict(l=50, r=20, t=20, b=50),
        xaxis=dict(zeroline=False),
        yaxis=dict(zeroline=False),
    )

    return fig


def _display_frontier_table(frontier: list) -> None:
    """Display the frontier data as a sortable table."""
    col_ergo = t("table.ergo")
    col_recoil_pct = t("table.recoil_pct")
    col_recoil_v = t("table.recoil_v")
    col_recoil_h = t("table.recoil_h")
    col_price = t("table.price")

    frontier_df = pd.DataFrame(
        [
            {
                col_ergo: point["ergo"],
                col_recoil_pct: f"{point['recoil_pct']:+.1f}%",
                col_recoil_v: round(point["recoil_v"], 1),
                col_recoil_h: round(point["recoil_h"], 1),
                col_price: f"₽ {point['price']:,}",
            }
            for point in frontier
        ]
    )

    st.dataframe(
        frontier_df,
        column_config={
            col_ergo: st.column_config.NumberColumn(col_ergo, format="%.1f"),
            col_recoil_pct: st.column_config.TextColumn(col_recoil_pct),
            col_recoil_v: st.column_config.NumberColumn(col_recoil_v, format="%.1f"),
            col_recoil_h: st.column_config.NumberColumn(col_recoil_h, format="%.1f"),
            col_price: st.column_config.TextColumn(col_price),
        },
        hide_index=True,
        width="stretch",
    )


def _display_selected_build(
    selected_point: dict,
    selected_gun: dict,
    weapon_stats: dict,
    presets: list,
    item_lookup: dict,
    stored_constraints: dict,
) -> None:
    """Display build details for selected point."""
    st.markdown("---")
    st.subheader(f"🔧 {t('explore.selected_build')}")

    result = {
        "status": selected_point.get("status", "optimal"),
        "selected_items": selected_point.get("selected_items", []),
        "selected_preset": selected_point.get("selected_preset"),
        "objective_value": 0,
    }

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
