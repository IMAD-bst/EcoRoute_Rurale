import streamlit as st
import pandas as pd
from ui.translations import translate, PROGRESS_KEYS
import json
from streamlit_folium import st_folium
from ui.map_view import create_optimization_map
from ui.charts import render_charts
from pathlib import Path
from engine.network import load_douars
from engine.pipeline import run_pipeline


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="EcoRoute Rurale",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)
language = st.sidebar.selectbox(
    "🌐 Language / Langue / اللغة",
    options=["fr", "ar", "en"],
    format_func=lambda x: {
        "fr": "🇫🇷 Français",
        "ar": "🇲🇦 العربية",
        "en": "🇬🇧 English"
    }[x],
    key="language"
)


def t(key):
    return translate(key, language)

def persistent_widget(widget, label, *args, key, **kwargs):
    """Keep values independently of translated widget identity."""
    saved = f"saved_{key}"
    widget_key = f"widget_{key}_{language}"
    if saved in st.session_state:
        st.session_state[widget_key] = st.session_state[saved]
        kwargs.pop("value", None)
        kwargs.pop("default", None)
    def save():
        st.session_state[saved] = st.session_state[widget_key]
    value = widget(label, *args, key=widget_key, on_change=save, **kwargs)
    st.session_state[saved] = value
    return value



if language == "ar":

    st.markdown(
        """
        <style>
        .stMain, [data-testid="stSidebar"] {
            direction: rtl;
            text-align: right;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

BASE_DIR = Path(__file__).resolve().parent


# ==================================================
# LOAD DATA
# ==================================================

@st.cache_data
def get_settlements():
    return load_douars()


@st.cache_data
def get_demo():
    return pd.read_csv(
        BASE_DIR / "data" / "demo" / "farmers.csv"
    )


douars = get_settlements()
demo = get_demo()

names = douars.set_index("osm_id")["name"].to_dict()

demo_quantities = demo.set_index(
    "osm_id"
)["quantity_kg"].to_dict()


# ==================================================
# HEADER
# ==================================================

st.title("🌿 EcoRoute Rurale")

st.caption(t("subtitle"))
st.divider()


# ==================================================
# SIDEBAR
# ==================================================

st.sidebar.header(t("configuration"))
scenario_mode = persistent_widget(st.sidebar.radio, 
    t("mode"),
    ["demo", "custom"], format_func=t
, key="mode")

max_distance = persistent_widget(st.sidebar.number_input, 
    t("radius"),
    min_value=0.0,
    value=3.0,
    step=0.5
, key="radius")

fuel_price = persistent_widget(st.sidebar.number_input, 
    t("fuel"),
    min_value=0.0,
    value=12.0,
    step=0.5
, key="fuel")


# ==================================================
# PRICING SETTINGS
# ==================================================

with st.sidebar.expander(t("economic")):

    baseline_fixed_cost = persistent_widget(st.number_input, 
        t("baseline_fixed"),
        min_value=0.0,
        value=50.0
    , key="baseline_fixed")

    baseline_cost_per_km = persistent_widget(st.number_input, 
        t("baseline_km"),
        min_value=0.0,
        value=3.0
    , key="baseline_km")

    local_cost = persistent_widget(st.number_input, 
        t("local_rate"),
        min_value=0.0,
        value=0.01,
        step=0.005,
        format="%.3f"
    , key="local_rate")


with st.sidebar.expander(t("solidarity")):

    minimum_savings = persistent_widget(st.slider, 
        t("min_pct"),
        min_value=0,
        max_value=100,
        value=20
    , key="min_pct")

    max_alpha = persistent_widget(st.slider, 
        t("alpha"),
        min_value=0.0,
        max_value=0.99,
        value=0.95,
        step=0.01
    , key="alpha")


# ==================================================
# MAIN CONFIGURATION
# ==================================================

tab1, tab2, tab3 = st.tabs([
    t("tab1"),
    t("tab2"),
    t("tab3")
])


# ==================================================
# TAB 1: SETTLEMENTS AND PRODUCTION
# ==================================================

with tab1:

    st.subheader(t("select_title"))

    st.caption(
        t("available").format(n=len(douars))
    )

    default_ids = (
        demo["osm_id"].tolist()
        if scenario_mode == "demo"
        else []
    )

    selected_ids = persistent_widget(st.multiselect, 
        t("select"),
        options=douars["osm_id"].tolist(),
        default=default_ids,
        format_func=lambda x: f"{names[x]} — {x}",
        key=f"settlements_{scenario_mode}"
    )

    production = pd.DataFrame({
        "osm_id": selected_ids,
        "name": [names[x] for x in selected_ids],
        "quantity_kg": [
            int(demo_quantities.get(x, 500))
            if scenario_mode == "demo"
            else 500
            for x in selected_ids
        ]
    })

    quantity_store = st.session_state.setdefault(f"quantities_{scenario_mode}", {})
    production["quantity_kg"] = [quantity_store.get(i, q) for i, q in zip(selected_ids, production["quantity_kg"])]
    editor_identity = (language, scenario_mode, tuple(selected_ids))
    if st.session_state.get("production_identity") != editor_identity:
        st.session_state["production_identity"] = editor_identity
        st.session_state["production_base"] = production.copy()
        st.session_state["production_revision"] = st.session_state.get("production_revision", 0) + 1
    production = st.data_editor(
        st.session_state["production_base"],
        hide_index=True,
        disabled=["osm_id", "name"],
        column_config={
            "osm_id": None,
            "name": st.column_config.TextColumn(
                t("settlement")
            ),
            "quantity_kg": st.column_config.NumberColumn(
                t("quantity"),
                min_value=1,
                step=1,
                required=True
            )
        },
        key=f"production_editor_{st.session_state['production_revision']}",
        use_container_width=True
    )

    quantity_store.update(production.set_index("osm_id")["quantity_kg"].to_dict())

    col1, col2 = st.columns(2)

    col1.metric(
        t("selected"),
        len(selected_ids)
    )

    col2.metric(
        t("total_production"),
        f"{production['quantity_kg'].sum():,.0f} kg"
    )


# ==================================================
# TAB 2: FLEET CONFIGURATION
# ==================================================

with tab2:

    st.subheader(t("fleet_title"))

    fleet_data = pd.DataFrame([
        {
            "id": 1,
            "capacity_kg": 4000,
            "consumption_l_100km": 18.0,
            "fixed_cost": 150.0,
            "maintenance_per_km": 0.0
        },
        {
            "id": 2,
            "capacity_kg": 3000,
            "consumption_l_100km": 14.0,
            "fixed_cost": 120.0,
            "maintenance_per_km": 0.0
        }
    ])

    if "fleet_saved" not in st.session_state:
        st.session_state["fleet_saved"] = fleet_data.copy()
    if st.session_state.get("fleet_language") != language:
        st.session_state["fleet_language"] = language
        st.session_state["fleet_base"] = st.session_state["fleet_saved"].copy()
        st.session_state["fleet_revision"] = st.session_state.get("fleet_revision", 0) + 1
    fleet_data = st.data_editor(
        st.session_state["fleet_base"],
        key=f"fleet_editor_{st.session_state['fleet_revision']}",
        num_rows="dynamic",
        hide_index=True,
        use_container_width=True,
        column_config={
            "id": st.column_config.NumberColumn(
                t("truck_id"),
                min_value=1,
                step=1,
                required=True
            ),
            "capacity_kg": st.column_config.NumberColumn(
                t("capacity"),
                min_value=1,
                step=1,
                required=True
            ),
            "consumption_l_100km": st.column_config.NumberColumn(
                t("consumption"),
                min_value=0.1,
                required=True
            ),
            "fixed_cost": st.column_config.NumberColumn(
                t("fixed"),
                min_value=0.0,
                required=True
            ),
            "maintenance_per_km": st.column_config.NumberColumn(
                t("maintenance"),
                min_value=0.0,
                required=True
            )
        }
    )

    st.session_state["fleet_saved"] = fleet_data.copy()
    max_trips = st.number_input(t("max_trips"), min_value=1, value=1, step=1, key="max_trips")
    st.caption(t("trip_note"))
    total_capacity = fleet_data["capacity_kg"].sum() * max_trips
    total_production = production["quantity_kg"].sum()

    col1, col2, col3 = st.columns(3)

    col1.metric(
        t("available_trucks"),
        len(fleet_data)
    )

    col2.metric(
        t("trip_capacity"),
        f"{total_capacity:,.0f} kg"
    )

    col3.metric(
        t("remaining"),
        f"{total_capacity - total_production:,.0f} kg"
    )

    if total_production > total_capacity:
        st.error(t("insufficient"))

    elif total_capacity > 0:

        utilization = total_production / total_capacity

        st.progress(
            float(utilization),
            text=t("utilization").format(v=utilization)
        )


# ==================================================
# TAB 3: DEPARTURE AND DESTINATION
# ==================================================

with tab3:

    col1, col2 = st.columns(2)

    with col1:

        st.subheader(t("departure"))

        departure = {
            "name": persistent_widget(st.text_input, 
                t("dep_name"),
                value="Chefchaouen"
            , key="dep_name"),
            "latitude": persistent_widget(st.number_input, 
                t("dep_lat"),
                value=35.1715,
                format="%.6f"
            , key="dep_lat"),
            "longitude": persistent_widget(st.number_input, 
                t("dep_lon"),
                value=-5.2697,
                format="%.6f"
            , key="dep_lon")
        }

    with col2:

        st.subheader(t("destination"))

        destination = {
            "name": persistent_widget(st.text_input, 
                t("dest_name"),
                value="Chefchaouen"
            , key="dest_name"),
            "latitude": persistent_widget(st.number_input, 
                t("dest_lat"),
                value=35.1715,
                format="%.6f"
            , key="dest_lat"),
            "longitude": persistent_widget(st.number_input, 
                t("dest_lon"),
                value=-5.2697,
                format="%.6f"
            , key="dest_lon")
        }


# ==================================================
# OPTIMIZATION BUTTON
# ==================================================

st.sidebar.divider()

run_button = st.sidebar.button(
    t("run"),
    type="primary",
    use_container_width=True
)

scenario_signature = json.dumps({
    "production": production.to_dict("records"),
    "fleet": fleet_data.to_dict("records"),
    "departure": departure, "destination": destination,
    "settings": [fuel_price, max_distance, baseline_fixed_cost,
                 baseline_cost_per_km, local_cost, minimum_savings, max_alpha, max_trips]
}, sort_keys=True, default=str)

if run_button:

    # Remove previous results before a new run
    # Keep previous successful results until a new run succeeds.

    if not selected_ids:

        st.error(t("no_selection"))

    elif production["quantity_kg"].isna().any():

        st.error(t("missing_production"))

    elif fleet_data.empty or fleet_data.isna().any().any():

        st.error(t("missing_fleet"))

    elif total_production > total_capacity:

        st.error(t("insufficient"))

    else:

        progress_bar = st.progress(0)

        status_text = st.empty()

        def update_progress(message, progress):

            progress_bar.progress(float(progress))

            status_text.text(t(PROGRESS_KEYS.get(message, "running")))

        try:

            with st.spinner(t("running")):

                result = run_pipeline(

                    osm_ids=selected_ids,

                    quantities=production.set_index(
                        "osm_id"
                    )["quantity_kg"].to_dict(),

                    trucks=fleet_data.to_dict(
                        orient="records"
                    ),

                    departure=departure,

                    destination=destination,

                    fuel_price=fuel_price,

                    max_distance_km=max_distance,

                    baseline_fixed_cost=baseline_fixed_cost,

                    baseline_cost_per_km=baseline_cost_per_km,

                    local_cost_per_kg_km=local_cost,

                    minimum_savings_pct=minimum_savings,

                    max_alpha=max_alpha,

                    max_trips=max_trips,
                    progress_callback=update_progress

                )

            st.session_state["optimization_result"] = result
            st.session_state["result_minimum_savings"] = minimum_savings
            st.session_state["result_signature"] = scenario_signature

            progress_bar.empty()

            status_text.empty()

        except Exception as error:

            progress_bar.empty()

            status_text.empty()

            st.error(t("error"))
            with st.expander(t("technical")):
                st.code(str(error))
# ==================================================
# OPTIMIZATION RESULTS
# ==================================================

if "optimization_result" in st.session_state:

    result = st.session_state["optimization_result"]

    kpis = result["kpis"]
    if st.session_state.get("result_signature") != scenario_signature:
        st.warning(t("stale"))

    st.divider()

    st.subheader(t("overview"))

    if result["pricing"]["feasible"]:

        st.success(t("success"))

    else:

        st.warning(t("pricing_infeasible"))

    # ----------------------------------------------
    # Main indicators
    # ----------------------------------------------

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        t("production"),
        f"{kpis['total_production_kg']:,.0f} kg"
    )

    c2.metric(
        t("collection_points"),
        kpis["number_of_collection_points"],
        delta=f"-{kpis['collection_points_reduction_pct']:.0f}%",
        delta_color="inverse"
    )

    c3.metric(
        t("transport_cost"),
        f"{kpis['total_transportation_cost']:,.2f} MAD"
    )

    c4.metric(
        t("distance"),
        f"{kpis['total_distance_km']:,.2f} km"
    )

    # ----------------------------------------------
    # Economic and social indicators
    # ----------------------------------------------

    if result["pricing"]["feasible"]:

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            t("savings"),
            f"{kpis['global_savings_pct']:.2f}%",
            delta=f"{kpis['total_net_savings']:,.2f} MAD"
        )

        c2.metric(
            t("effective_cost"),
            f"{kpis['total_effective_cost']:,.2f} MAD"
        )

        c3.metric(
            t("minimum_savings"),
            f"{kpis['minimum_individual_savings_pct']:.2f}%"
        )

        c4.metric(
            t("redistribution"),
            f"{kpis['redistributed_amount']:,.2f} MAD"
        )
        # ----------------------------------------------
    # Interactive optimization map
    # ----------------------------------------------

    st.divider()

    map_titles = {
        "fr": "🗺️ Carte interactive des itinéraires",
        "en": "🗺️ Interactive route map",
        "ar": "🗺️ خريطة المسارات التفاعلية"
    }

    st.subheader(t("map"))

    optimization_map = create_optimization_map(
        result,
        language=language
    )

    st_folium(
    optimization_map,
    height=750,
    use_container_width=True,
    returned_objects=[],
    key=f"optimization_map_{language}"
    )
    # ----------------------------------------------
    # Optimized routes
    # ----------------------------------------------

    st.divider()

    st.subheader(t("routes"))
    st.caption(f"{t('trip_count')}: {result['routing'].get('number_of_trips', len(result['routing']['routes']))}")

    for route in result["routing"]["routes"]:

        with st.container(border=True):

            st.markdown(
                f"### 🚚 {t('truck')} {route.get('truck_label', route['truck_id'])}"
            )

            st.write(
                " → ".join(route["route_names"])
            )

            c1, c2, c3 = st.columns(3)

            c1.metric(
                t("load"),
                f"{route['load_kg']:,} / "
                f"{route['capacity_kg']:,} kg"
            )

            c2.metric(
                t("distance"),
                f"{route['distance_km']:.2f} km"
            )

            c3.metric(
                t("cost"),
                f"{route['total_cost']:.2f} MAD"
            )

            st.progress(
                float(route["utilization_pct"] / 100),
                text=f"{route['utilization_pct']:.2f}%"
            )

    # ----------------------------------------------
    # Individual pricing
    # ----------------------------------------------

    if result["pricing"]["feasible"]:

        st.divider()

        st.subheader(t("pricing"))

        pricing_table = result["pricing"]["results"][[

            "name",

            "quantity_kg",

            "isolation_score",

            "baseline_cost",

            "solidarity_price",

            "local_collection_cost",

            "effective_cost",

            "savings_pct"

        ]].copy()

        pricing_table["isolation_score"] *= 100
        pricing_table = pricing_table.rename(columns={
            "name": t("settlement"), "quantity_kg": t("quantity"),
            "isolation_score": t("isolation"), "baseline_cost": t("baseline"),
            "solidarity_price": t("solidarity_price"),
            "local_collection_cost": t("local_collection"),
            "effective_cost": t("effective_column"), "savings_pct": t("savings_pct")
        })
        st.caption(t("estimated"))
        st.dataframe(
            pricing_table.round(2),
            hide_index=True,
            use_container_width=True
        )
    # ==============================================
    # Economic and social charts
    # ==============================================

    render_charts(
        result,
        language=language,
        minimum_target=st.session_state.get(
            "result_minimum_savings"
        )
    )