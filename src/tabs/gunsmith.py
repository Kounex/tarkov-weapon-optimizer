"""
Gunsmith tasks tab component.
"""

import streamlit as st
from loguru import logger

from i18n import t
from src.optimizer import optimize_weapon
from src.data import get_compat_map, load_tasks
from src.utils import resolve_item_id
from src.components import display_optimization_results


def render_gunsmith_tab(
    guns: list,
    item_lookup: dict,
    player_level: int,
    flea_available: bool,
    trader_levels: dict,
) -> None:
    """Render the Gunsmith tasks tab.

    Args:
        guns: List of all guns
        item_lookup: Item lookup dictionary
        player_level: Player level
        flea_available: Whether flea market is available
        trader_levels: Trader levels dictionary
    """
    st.header(t("gunsmith.header"))

    tasks = load_tasks()
    task_options = {task["task_name"]: task for task in tasks}
    selected_task_name = st.selectbox(
        t("gunsmith.select_task"), list(task_options.keys())
    )

    if not selected_task_name:
        return

    task = task_options[selected_task_name]
    st.markdown(f"**{t('gunsmith.task_requirements')}**")

    # Show requirements
    _display_task_requirements(task)

    st.markdown("---")

    if st.button(t("gunsmith.optimize_btn"), type="primary"):
        _run_gunsmith_optimization(
            task,
            guns,
            item_lookup,
            player_level,
            flea_available,
            trader_levels,
            selected_task_name,
        )


def _display_task_requirements(task: dict) -> None:
    """Display task requirements in columns."""
    req_cols = st.columns(3)
    with req_cols[0]:
        st.write(f"**{t('gunsmith.weapon')}:** {task['weapon_name']}")
    with req_cols[1]:
        constraints = task.get("constraints", {})
        if constraints:
            key_map = {
                "min_ergonomics": "constraints.min_ergonomics",
                "max_recoil_v": "constraints.max_recoil",
                "max_recoil_sum": "constraints.max_recoil_sum",
                "max_weight": "constraints.max_weight",
                "min_mag_capacity": "constraints.min_mag_capacity",
                "min_sighting_range": "constraints.min_sighting_range",
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


def _run_gunsmith_optimization(
    task: dict,
    guns: list,
    item_lookup: dict,
    player_level: int,
    flea_available: bool,
    trader_levels: dict,
    selected_task_name: str,
) -> None:
    """Run Gunsmith task optimization."""
    logger.info(f"User started Gunsmith task optimization for {selected_task_name}")

    # Find target gun
    target_gun_id = None
    for gun in guns:
        if gun["name"] == task["weapon_name"]:
            target_gun_id = gun["id"]
            break

    if not target_gun_id:
        st.error(t("gunsmith.weapon_not_found", name=task["weapon_name"]))
        return

    # Resolve required items
    task_include_items = set()
    missing_items = []
    for item_name in task.get("required_items", []):
        resolved_id = resolve_item_id(item_name, item_lookup)
        if resolved_id:
            task_include_items.add(resolved_id)
        else:
            missing_items.append(item_name)

    # Resolve categories
    task_include_categories = []
    if "required_category_groups" in task:
        task_include_categories.extend(task["required_category_groups"])
    if "required_categories" in task:
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
                task_constraints["include_items"] = sorted(
                    [item_lookup[i]["data"]["name"] for i in task_include_items]
                )
                task_constraints["include_categories"] = sorted(
                    list(task_include_categories)
                )

                display_optimization_results(
                    result,
                    item_lookup,
                    target_stats,
                    target_presets,
                    target_gun_data,
                    task_constraints,
                )

        except Exception as e:
            status.update(label="Error", state="error")
            st.error(f"Error: {e}")
