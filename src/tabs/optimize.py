"""
Optimize tab component.
"""

import json

import plotly.graph_objects as go
import streamlit as st
from loguru import logger

from i18n import t
from src.optimizer import optimize_weapon
from src.data import get_compat_map
from src.components import display_optimization_results
from src.export import generate_build_export


def render_optimize_tab(
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
    """Render the Optimize tab.

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
    st.header(t("optimize.header"))
    st.subheader(t("optimize.weights_header"))

    # Initialize weights in session state
    if "weight_ergo" not in st.session_state:
        st.session_state.weight_ergo = 33
    if "weight_recoil" not in st.session_state:
        st.session_state.weight_recoil = 67
    if "weight_price" not in st.session_state:
        st.session_state.weight_price = 0

    # Preset buttons
    _render_preset_buttons()

    # Get current weights
    w_ergo = st.session_state.weight_ergo
    w_recoil = st.session_state.weight_recoil
    w_price = st.session_state.weight_price

    # Ternary plot
    _render_ternary_plot(w_ergo, w_recoil, w_price)

    st.caption(
        f"{t('optimize.preset_ergo')}: {w_ergo}% | {t('optimize.preset_recoil')}: {w_recoil}% | {t('optimize.preset_price')}: {w_price}%"
    )

    # Manual weight sliders
    with st.expander(f"⚙️ {t('optimize.manual_weights')}", expanded=False):
        st.slider(
            t("optimize.preset_ergo"),
            0,
            100,
            key="weight_ergo",
            help=t("optimize.preset_ergo_help"),
        )
        st.slider(
            t("optimize.preset_recoil"),
            0,
            100,
            key="weight_recoil",
            help=t("optimize.preset_recoil_help"),
        )
        st.slider(
            t("optimize.preset_price"),
            0,
            100,
            key="weight_price",
            help=t("optimize.preset_price_help"),
        )
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
    optimize_button = st.button(
        f"🚀 {t('optimize.optimize_btn')}",
        type="primary",
        key="optimize_btn",
        width="stretch",
    )

    if optimize_button:
        _run_optimization(
            selected_gun,
            weapon_id,
            weapon_stats,
            presets,
            item_lookup,
            hard_constraints,
            include_exclude,
            player_level,
            flea_available,
            trader_levels,
            ergo_weight,
            recoil_weight,
            price_weight,
            w_ergo,
            w_recoil,
            w_price,
        )
    else:
        # Initial state help text
        st.info(
            f"{t('optimize.how_it_works')} {t('optimize.how_step_3').replace('**', '')}"
        )
        st.markdown(
            f"""
        **{t('optimize.how_it_works')}**
        1. {t('optimize.how_step_1')}
        2. {t('optimize.how_step_2')}
        3. {t('optimize.how_step_3')}

        **{t('optimize.tip')}** {t('optimize.tip_text')}
        """
        )


def _render_preset_buttons() -> None:
    """Render weight preset buttons."""
    preset_cols = st.columns(4)
    if preset_cols[0].button(
        t("optimize.preset_ergo"),
        help=t("optimize.preset_ergo_help"),
        key="preset_ergo",
    ):
        st.session_state.weight_ergo = 98
        st.session_state.weight_recoil = 1
        st.session_state.weight_price = 1
        st.rerun()
    if preset_cols[1].button(
        t("optimize.preset_recoil"),
        help=t("optimize.preset_recoil_help"),
        key="preset_recoil",
    ):
        st.session_state.weight_ergo = 1
        st.session_state.weight_recoil = 98
        st.session_state.weight_price = 1
        st.rerun()
    if preset_cols[2].button(
        t("optimize.preset_price"),
        help=t("optimize.preset_price_help"),
        key="preset_price",
    ):
        st.session_state.weight_ergo = 1
        st.session_state.weight_recoil = 1
        st.session_state.weight_price = 98
        st.rerun()
    if preset_cols[3].button(
        t("optimize.preset_balanced"),
        help=t("optimize.preset_balanced_help"),
        key="preset_balanced",
    ):
        st.session_state.weight_ergo = 34
        st.session_state.weight_recoil = 33
        st.session_state.weight_price = 33
        st.rerun()


def _render_ternary_plot(w_ergo: int, w_recoil: int, w_price: int) -> None:
    """Render the ternary plot for weight selection."""
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
    fig.add_trace(
        go.Scatterternary(
            a=grid_a,
            b=grid_b,
            c=grid_c,
            mode="markers",
            marker=dict(size=8, color="lightgray", opacity=0.3),
            hovertemplate=f"{t('optimize.preset_ergo')}: %{{a}}%<br>{t('optimize.preset_recoil')}: %{{b}}%<br>{t('optimize.preset_price')}: %{{c}}%<extra></extra>",
            showlegend=False,
        )
    )

    # Add current position marker
    fig.add_trace(
        go.Scatterternary(
            a=[w_ergo],
            b=[w_recoil],
            c=[w_price],
            mode="markers",
            marker=dict(
                size=18, color="red", symbol="circle", line=dict(width=2, color="white")
            ),
            name="Current",
            hovertemplate=f"<b>Current</b><br>{t('optimize.preset_ergo')}: %{{a}}%<br>{t('optimize.preset_recoil')}: %{{b}}%<br>{t('optimize.preset_price')}: %{{c}}%<extra></extra>",
        )
    )

    # Add corner labels
    fig.add_trace(
        go.Scatterternary(
            a=[100, 0, 0],
            b=[0, 100, 0],
            c=[0, 0, 100],
            mode="text",
            text=[
                t("optimize.preset_ergo"),
                t("optimize.preset_recoil"),
                t("optimize.preset_price"),
            ],
            textposition=["top center", "bottom left", "bottom right"],
            textfont=dict(size=11, color="gray"),
            showlegend=False,
            hoverinfo="skip",
        )
    )

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


def _run_optimization(
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
    ergo_weight: float,
    recoil_weight: float,
    price_weight: float,
    w_ergo: int,
    w_recoil: int,
    w_price: int,
) -> None:
    """Run the optimization and display results."""
    logger.info(
        f"User started optimization for {selected_gun['name']} (weights: ergo={w_ergo}%, recoil={w_recoil}%, price={w_price}%)"
    )

    with st.status(t("status.optimizing"), expanded=True) as status:
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
            status.update(label=t("status.running_solver"))
            st.write(f"✓ {t('status.building_model')}")
            result = optimize_weapon(
                weapon_id,
                item_lookup,
                compat_map,
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
                status_key = (
                    f"results.{result['status']}"
                    if result["status"] in ["optimal", "feasible"]
                    else "results.feasible"
                )
                status.update(
                    label=f"{t('results.optimization_status')} {t(status_key)}",
                    state="complete",
                )
        except Exception as e:
            status.update(label=t("status.optimization_failed"), state="error")
            st.error(f"{t('status.optimization_failed')}: {e}")
            st.stop()

    # Helper to map IDs to names
    def id_to_name(ids):
        return (
            sorted([item_lookup[i]["data"]["name"] for i in ids if i in item_lookup])
            if ids
            else None
        )

    constraints = {
        "max_price": hard_constraints.get("max_price"),
        "min_ergonomics": hard_constraints.get("min_ergonomics"),
        "max_recoil_v": hard_constraints.get("max_recoil_v"),
        "min_mag_capacity": hard_constraints.get("min_mag_capacity"),
        "min_sighting_range": hard_constraints.get("min_sighting_range"),
        "max_weight": hard_constraints.get("max_weight"),
        "include_items": id_to_name(include_exclude.get("include_items")),
        "exclude_items": id_to_name(include_exclude.get("exclude_items")),
        "include_categories": (
            sorted(list(include_exclude.get("include_categories")))
            if include_exclude.get("include_categories")
            else None
        ),
        "exclude_categories": (
            sorted(list(include_exclude.get("exclude_categories")))
            if include_exclude.get("exclude_categories")
            else None
        ),
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
