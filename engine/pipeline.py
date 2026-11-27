
# ==================================================
# EcoRoute Rurale - Integration Pipeline
# ==================================================

from time import perf_counter

from engine.geometry import reconstruct_routes

from engine.network import (
    select_douars,
    configure_location,
    build_distance_matrices,
    build_routing_locations
)

from engine.configuration import (
    configure_production,
    configure_fleet,
    check_fleet_capacity
)

from engine.clustering import (
    optimize_capacitated_clustering
)

from engine.isolation import (
    compute_isolation_scores
)

from engine.routing import (
    prepare_collection_points,
    optimize_routes,
    validate_routes
)

from engine.pricing import (
    prepare_pricing_data,
    compute_baseline_costs,
    compute_local_collection_costs,
    optimize_solidarity,
    compute_pricing_kpis
)


# --------------------------------------------------
# Complete optimization pipeline
# --------------------------------------------------

def run_pipeline(
    osm_ids,
    quantities,
    trucks,
    departure,
    destination,
    fuel_price=12.0,
    max_distance_km=3.0,
    baseline_fixed_cost=50.0,
    baseline_cost_per_km=3.0,
    actual_baseline_costs=None,
    local_cost_per_kg_km=0.01,
    minimum_savings_pct=20.0,
    max_alpha=0.95,
    isolation_weights=(0.3, 0.3, 0.4),
    clustering_time_limit=15,
    routing_time_limit=10,
    progress_callback=None,
    max_trips=1
):

    pipeline_start = perf_counter()

    timings = {}

    def notify(message, progress):

        print(message)

        if progress_callback is not None:

            progress_callback(message, progress)

    # ----------------------------------------------
    # Stage 1: Scenario configuration
    # ----------------------------------------------

    notify("Configuring scenario...", 0.05)

    start = perf_counter()

    selected = select_douars(osm_ids)

    farmers = configure_production(
        selected,
        quantities
    )

    fleet = configure_fleet(
        trucks,
        fuel_price
    )

    physical_fleet = fleet
    if isinstance(max_trips, bool) or int(max_trips) != max_trips or max_trips < 1:
        raise ValueError("max_trips must be a positive integer.")
    if max_trips > 1:
        fleet = [dict(truck, id=i * int(max_trips) + trip,
                      physical_truck_id=truck["id"], trip_number=trip)
                 for i, truck in enumerate(physical_fleet)
                 for trip in range(1, int(max_trips) + 1)]
    feasibility = check_fleet_capacity(farmers, fleet)

    if not feasibility["preliminary_feasible"]:

        raise ValueError(
            "; ".join(feasibility["errors"])
        )

    departure = configure_location(
        **departure
    )

    destination = configure_location(
        **destination
    )

    timings["configuration"] = (
        perf_counter() - start
    )

    # ----------------------------------------------
    # Stage 2: Settlement distance matrices
    # ----------------------------------------------

    notify("Computing road distances...", 0.15)

    start = perf_counter()

    douar_locations = selected[
        ["osm_id", "road_node"]
    ].rename(columns={"osm_id": "id"})

    D, T = build_distance_matrices(
        douar_locations
    )

    timings["distance_matrices"] = (
        perf_counter() - start
    )

    # ----------------------------------------------
    # Stage 3: Capacitated clustering
    # ----------------------------------------------

    notify("Optimizing collection points...", 0.30)

    start = perf_counter()

    clustering = optimize_capacitated_clustering(
        D,
        farmers,
        fleet,
        max_distance_km=max_distance_km,
        time_limit=clustering_time_limit
    )

    assignments = clustering["assignments"]

    collection_points = prepare_collection_points(
        assignments,
        selected
    )

    timings["clustering"] = (
        perf_counter() - start
    )

    # ----------------------------------------------
    # Stage 4: Territorial isolation
    # ----------------------------------------------

    notify("Computing isolation scores...", 0.45)

    start = perf_counter()

    isolation = compute_isolation_scores(
        selected,
        D,
        destination,
        weights=isolation_weights
    )

    timings["isolation"] = (
        perf_counter() - start
    )

    # ----------------------------------------------
    # Stage 5: Routing matrices
    # ----------------------------------------------

    notify("Preparing routing matrices...", 0.60)

    start = perf_counter()

    locations = build_routing_locations(
        collection_points,
        departure,
        destination
    )

    routing_D, routing_T = build_distance_matrices(
        locations
    )

    timings["routing_matrices"] = (
        perf_counter() - start
    )

    # ----------------------------------------------
    # Stage 6: Vehicle routing optimization
    # ----------------------------------------------

    notify("Optimizing truck routes...", 0.75)

    start = perf_counter()

    routing = optimize_routes(
        locations,
        routing_D,
        routing_T,
        fleet,
        time_limit=routing_time_limit
    )

    validation = validate_routes(
        routing,
        locations
    )

    if not validation["valid"]:

        raise RuntimeError(
            "Routing validation failed."
        )

    for route in routing["routes"]:
        slot = next(t for t in fleet if t["id"] == route["truck_id"])
        route["physical_truck_id"] = slot.get("physical_truck_id", slot["id"])
        route["trip_number"] = slot.get("trip_number", 1)
    routing["number_of_trips"] = len(routing["routes"])
    routing["used_trucks"] = len({r["physical_truck_id"] for r in routing["routes"]})
    routing["available_trucks"] = len(physical_fleet)
    timings["routing"] = (
        perf_counter() - start
    )

    # ----------------------------------------------
    # Stage 6.5: Synchronize final truck assignments
    # ----------------------------------------------

    start = perf_counter()

    truck_mapping = {
        collection_id: route["truck_id"]
        for route in routing["routes"]
        for collection_id in route["collection_ids"]
    }

    assignments = assignments.copy()

    # Preserve the initial clustering assignment
    assignments["clustering_truck_id"] = (
        assignments["truck_id"]
    )

    # Assign the actual truck selected by routing
    assignments["truck_id"] = (
        assignments["collection_id"].map(truck_mapping)
    )

    collection_points["truck_id"] = (
        collection_points["osm_id"].map(truck_mapping)
    )

    if assignments["truck_id"].isna().any():

        raise RuntimeError(
            "Missing final truck assignments."
        )

    if collection_points["truck_id"].isna().any():

        raise RuntimeError(
            "Missing collection point truck assignments."
        )

    # Verify consistency between assignments and routes
    assigned_loads = assignments.groupby(
        "truck_id"
    )["quantity_kg"].sum().to_dict()

    for route in routing["routes"]:

        truck_id = route["truck_id"]

        if assigned_loads.get(truck_id, 0) != route["load_kg"]:

            raise RuntimeError(
                f"Load mismatch for truck {truck_id}."
            )

    slot_labels = {r["truck_id"]: (str(r["physical_truck_id"]) if max_trips == 1 else f'{r["physical_truck_id"]} / #{r["trip_number"]}') for r in routing["routes"]}
    assignments["truck_label"] = assignments["truck_id"].map(slot_labels)
    collection_points["truck_label"] = collection_points["truck_id"].map(slot_labels)
    for route in routing["routes"]:
        route["truck_label"] = slot_labels[route["truck_id"]]
    clustering["assignments"] = assignments

    timings["assignment_sync"] = (
        perf_counter() - start
    )

    # ----------------------------------------------
    # Stage 7: Fair pricing
    # ----------------------------------------------

    notify("Optimizing solidarity pricing...", 0.90)

    start = perf_counter()

    pricing_data = prepare_pricing_data(
        farmers,
        assignments,
        isolation
    )

    pricing_data = compute_baseline_costs(
        pricing_data,
        fixed_cost=baseline_fixed_cost,
        cost_per_km=baseline_cost_per_km,
        actual_costs=actual_baseline_costs
    )

    pricing_data = compute_local_collection_costs(
        pricing_data,
        local_cost_per_kg_km=local_cost_per_kg_km
    )

    pricing = optimize_solidarity(
        pricing_data,
        routing["total_cost"],
        minimum_savings_pct=minimum_savings_pct,
        max_alpha=max_alpha
    )

    pricing_kpis = None

    if pricing["feasible"]:

        pricing_kpis = compute_pricing_kpis(
            pricing["results"]
        )

    timings["pricing"] = (
        perf_counter() - start
    )

    # ----------------------------------------------
    # Stage 7.5: Route geometry reconstruction
    # ----------------------------------------------

    notify("Reconstructing road geometries...", 0.97)

    start = perf_counter()

    geometry = reconstruct_routes(
        routing,
        locations
    )

    timings["geometry"] = (
        perf_counter() - start
    )

    # ----------------------------------------------
    # Stage 8: Global results
    # ----------------------------------------------

    kpis = {

        "number_of_douars": len(selected),

        "number_of_collection_points": len(
            collection_points
        ),

        "collection_points_reduction_pct": (
            1 - len(collection_points) / len(selected)
        ) * 100,

        "total_production_kg": int(
            farmers["quantity_kg"].sum()
        ),

        "available_trucks": len(physical_fleet),
        "number_of_trips": routing["number_of_trips"],

        "used_trucks": routing["used_trucks"],

        "total_distance_km": routing[
            "total_distance_km"
        ],

        "total_transportation_cost": routing[
            "total_cost"
        ],

        "average_collection_distance_km": clustering[
            "average_distance_km"
        ],

        "maximum_collection_distance_km": clustering[
            "maximum_distance_km"
        ],

        "pricing_feasible": pricing["feasible"]

    }

    if pricing_kpis is not None:

        kpis.update(pricing_kpis)

    timings["total"] = (
        perf_counter() - pipeline_start
    )

    notify("Optimization completed.", 1.0)

    # ----------------------------------------------
    # Return complete optimization results
    # ----------------------------------------------

    return {

        "status": (
            "success"
            if pricing["feasible"]
            else "pricing_infeasible"
        ),

        "selected": selected,

        "farmers": farmers,

        "fleet": physical_fleet,
        "trip_slots": fleet,

        "departure": departure,

        "destination": destination,

        "fleet_feasibility": feasibility,

        "distance_matrix": D,

        "duration_matrix": T,

        "clustering": clustering,

        "assignments": assignments,

        "collection_points": collection_points,

        "isolation": isolation,

        "routing_locations": locations,

        "routing_distance_matrix": routing_D,

        "routing_duration_matrix": routing_T,

        "routing": routing,

        "geometry": geometry,

        "routing_validation": validation,

        "pricing": pricing,

        "kpis": kpis,

        "timings": timings

    }
