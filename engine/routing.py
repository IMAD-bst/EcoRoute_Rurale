# ==================================================
# EcoRoute Rurale - Routing Engine
# ==================================================

import numpy as np
import pandas as pd

from time import perf_counter

from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2

from engine.network import (
    build_distance_matrices,
    build_routing_locations,
    configure_location
)


# --------------------------------------------------
# Prepare collection points
# --------------------------------------------------

def prepare_collection_points(assignments, selected):

    collection_points = assignments.groupby(
        "collection_id"
    ).agg(

        quantity_kg=("quantity_kg", "sum"),

        number_of_douars=("osm_id", "count"),

        average_distance_km=("distance_km", "mean"),

        maximum_distance_km=("distance_km", "max")

    ).reset_index()

    collection_points = collection_points.rename(
        columns={"collection_id": "osm_id"}
    )

    metadata = selected[[
        "osm_id",
        "name",
        "latitude",
        "longitude",
        "road_node"
    ]]

    collection_points = collection_points.merge(

        metadata,

        on="osm_id",

        how="left",

        validate="one_to_one"

    )

    if collection_points["road_node"].isna().any():

        raise ValueError(
            "Missing collection point road nodes."
        )

    if (
        collection_points["quantity_kg"].sum()
        != assignments["quantity_kg"].sum()
    ):

        raise ValueError(
            "Production conservation error."
        )

    return collection_points
# --------------------------------------------------
# Heterogeneous capacitated vehicle routing
# --------------------------------------------------

def optimize_routes(
    locations,
    distance_matrix,
    duration_matrix,
    trucks,
    time_limit=10
):

    locations = locations.reset_index(drop=True)

    ids = locations["id"].tolist()

    n = len(locations)

    num_vehicles = len(trucks)

    start_node = ids.index("departure")

    end_node = ids.index("destination")

    # Align matrices with location order

    D = distance_matrix.loc[ids, ids].to_numpy(
        dtype=float
    )

    T = duration_matrix.loc[ids, ids].to_numpy(
        dtype=float
    )

    if not np.isfinite(D).all():
        raise ValueError("Unreachable routing locations.")

    if not np.isfinite(T).all():
        raise ValueError("Invalid travel durations.")

    demands = locations["quantity_kg"].astype(int).tolist()

    capacities = [
        truck["capacity_kg"]
        for truck in trucks
    ]

    if sum(demands) > sum(capacities):
        raise ValueError("Insufficient fleet capacity.")

    # Monetary precision: 0.0001 MAD

    COST_SCALE = 10000

    # Routing model

    manager = pywrapcp.RoutingIndexManager(
        n,
        num_vehicles,
        [start_node] * num_vehicles,
        [end_node] * num_vehicles
    )

    routing = pywrapcp.RoutingModel(manager)

    # --------------------------------------------------
    # Individual transportation costs
    # --------------------------------------------------

    def make_cost_callback(cost_per_km):

        cost_matrix = np.rint(
            D * cost_per_km * COST_SCALE
        ).astype(np.int64)

        def cost_callback(from_index, to_index):

            i = manager.IndexToNode(from_index)

            j = manager.IndexToNode(to_index)

            return int(cost_matrix[i, j])

        return cost_callback

    for k, truck in enumerate(trucks):

        callback_index = routing.RegisterTransitCallback(
            make_cost_callback(truck["cost_per_km"])
        )

        routing.SetArcCostEvaluatorOfVehicle(
            callback_index,
            k
        )

        routing.SetFixedCostOfVehicle(

            int(round(
                truck["fixed_cost"] * COST_SCALE
            )),

            k
        )

    # --------------------------------------------------
    # Capacity constraints
    # --------------------------------------------------

    def demand_callback(from_index):

        node = manager.IndexToNode(from_index)

        return int(demands[node])

    demand_callback_index = (
        routing.RegisterUnaryTransitCallback(
            demand_callback
        )
    )

    routing.AddDimensionWithVehicleCapacity(

        demand_callback_index,

        0,

        capacities,

        True,

        "Capacity"

    )

    # --------------------------------------------------
    # Search configuration
    # --------------------------------------------------

    search_parameters = (
        pywrapcp.DefaultRoutingSearchParameters()
    )

    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy
        .PARALLEL_CHEAPEST_INSERTION
    )

    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic
        .GUIDED_LOCAL_SEARCH
    )

    search_parameters.time_limit.FromMilliseconds(
        max(1, int(time_limit * 1000))
    )

    search_parameters.log_search = False

    # --------------------------------------------------
    # Solve
    # --------------------------------------------------

    start = perf_counter()

    solution = routing.SolveWithParameters(
        search_parameters
    )

    elapsed = perf_counter() - start

    if solution is None:

        raise RuntimeError(
            "No routing solution found. "
            f"Solver status: {routing.status()}"
        )

    # --------------------------------------------------
    # Extract optimized routes
    # --------------------------------------------------

    routes = []

    for k, truck in enumerate(trucks):

        if not routing.IsVehicleUsed(solution, k):
            continue

        index = routing.Start(k)

        route_nodes = []

        total_distance = 0.0

        total_duration = 0.0

        total_load = 0

        while not routing.IsEnd(index):

            node = manager.IndexToNode(index)

            route_nodes.append(node)

            total_load += demands[node]

            next_index = solution.Value(
                routing.NextVar(index)
            )

            next_node = manager.IndexToNode(
                next_index
            )

            total_distance += D[node, next_node]

            total_duration += T[node, next_node]

            index = next_index

        route_nodes.append(
            manager.IndexToNode(index)
        )

        # Transportation cost breakdown

        fuel_liters = (

            total_distance

            * truck["consumption_l_100km"]

            / 100

        )

        fuel_cost = (
            fuel_liters * truck["fuel_price"]
        )

        maintenance_cost = (

            total_distance

            * truck["maintenance_per_km"]

        )

        total_cost = (

            fuel_cost

            + maintenance_cost

            + truck["fixed_cost"]

        )

        routes.append({

            "truck_id": truck["id"],

            "route_ids": [
                ids[node]
                for node in route_nodes
            ],

            "route_names": [
                locations.iloc[node]["name"]
                for node in route_nodes
            ],

            "collection_ids": [
                ids[node]
                for node in route_nodes[1:-1]
            ],

            "load_kg": total_load,

            "capacity_kg": truck["capacity_kg"],

            "utilization_pct": (
                total_load
                / truck["capacity_kg"]
                * 100
            ),

            "distance_km": total_distance,

            "duration_min": total_duration,

            "fuel_liters": fuel_liters,

            "fuel_cost": fuel_cost,

            "maintenance_cost": maintenance_cost,

            "fixed_cost": truck["fixed_cost"],

            "total_cost": total_cost

        })

    # --------------------------------------------------
    # Global optimization results
    # --------------------------------------------------

    return {

        "routes": routes,

        "used_trucks": len(routes),

        "available_trucks": num_vehicles,

        "total_load_kg": sum(
            r["load_kg"] for r in routes
        ),

        "total_distance_km": sum(
            r["distance_km"] for r in routes
        ),

        "total_duration_min": sum(
            r["duration_min"] for r in routes
        ),

        "maximum_route_duration_min": max(
            (r["duration_min"] for r in routes),
            default=0
        ),

        "total_fuel_liters": sum(
            r["fuel_liters"] for r in routes
        ),

        "total_cost": sum(
            r["total_cost"] for r in routes
        ),

        "solver_objective": (
            solution.ObjectiveValue() / COST_SCALE
        ),

        "solver_status": routing.status(),

        "computation_time_s": elapsed

    }
# --------------------------------------------------
# Validate optimized transportation plan
# --------------------------------------------------

def validate_routes(optimization, locations):

    routes = optimization["routes"]

    expected_points = locations.loc[
        ~locations["id"].isin(
            ["departure", "destination"]
        ),
        "id"
    ].tolist()

    visited_points = [

        point

        for route in routes

        for point in route["collection_ids"]

    ]

    expected_load = int(
        locations["quantity_kg"].sum()
    )

    checks = {

        "all_collection_points_visited": (
            set(visited_points) == set(expected_points)
        ),

        "no_duplicate_visits": (
            len(visited_points) == len(set(visited_points))
        ),

        "capacity_constraints": all(

            route["load_kg"] <= route["capacity_kg"]

            for route in routes
        ),

        "production_conservation": (
            optimization["total_load_kg"]
            == expected_load
        ),

        "correct_departure": all(

            route["route_ids"][0] == "departure"

            for route in routes
        ),

        "correct_destination": all(

            route["route_ids"][-1] == "destination"

            for route in routes
        ),

        "unique_trucks": (
            len(routes)
            == len({r["truck_id"] for r in routes})
        ),

        "cost_consistency": np.isclose(

            optimization["total_cost"],

            optimization["solver_objective"],

            atol=0.001 * max(1, len(locations)),

            rtol=0

        )

    }

    return {

        "checks": checks,

        "valid": all(checks.values())

    }