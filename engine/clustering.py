# ==================================================
# EcoRoute Rurale - Clustering Engine
# ==================================================

import numpy as np
import pandas as pd

from time import perf_counter
from ortools.sat.python import cp_model

from engine.network import (
    load_douars,
    select_douars,
    build_distance_matrices
)


# --------------------------------------------------
# Build coverage matrix
# --------------------------------------------------

def build_coverage_matrix(D, max_distance_km=3.0):

    if max_distance_km < 0:
        raise ValueError("Invalid maximum distance.")

    if D.empty or not D.index.equals(D.columns):
        raise ValueError("Invalid distance matrix.")

    coverage = D.to_numpy() <= max_distance_km

    if not coverage.any(axis=1).all():
        raise ValueError("Some settlements cannot be covered.")

    return coverage


# --------------------------------------------------
# Greedy Set Cover
# --------------------------------------------------

def greedy_set_cover(D, max_distance_km=3.0):

    coverage = build_coverage_matrix(
        D,
        max_distance_km
    )

    n = len(D)

    uncovered = np.ones(n, dtype=bool)

    centers = []

    while uncovered.any():

        # Number of uncovered settlements
        # covered by each candidate center

        gains = coverage[uncovered, :].sum(axis=0)

        best_center = int(np.argmax(gains))

        if gains[best_center] == 0:
            raise ValueError("No feasible coverage.")

        centers.append(best_center)

        uncovered &= ~coverage[:, best_center]

    return D.columns[centers].tolist()


# --------------------------------------------------
# Assign settlements to collection points
# --------------------------------------------------

def assign_douars(D, centers):

    distances = D.loc[:, centers]

    assignments = distances.idxmin(axis=1)

    results = pd.DataFrame({
        "osm_id": D.index,
        "collection_id": assignments.values,
        "distance_km": distances.min(axis=1).values
    })

    cluster_ids = {
        center: i + 1
        for i, center in enumerate(centers)
    }

    results["cluster"] = (
        results["collection_id"].map(cluster_ids)
    )

    return results
# --------------------------------------------------
# Hybrid clustering optimization with CP-SAT
# --------------------------------------------------

def optimize_collection_points(
    D,
    max_distance_km=3.0,
    time_limit=10.0
):

    coverage = build_coverage_matrix(
        D, max_distance_km
    )

    n = len(D)

    # Initial feasible solution
    greedy_centers = greedy_set_cover(
        D, max_distance_km
    )

    best_centers = greedy_centers.copy()

    # ==============================================
    # Stage 1: Minimize the number of centers
    # ==============================================

    model = cp_model.CpModel()

    y = [
        model.NewBoolVar(f"center_{j}")
        for j in range(n)
    ]

    for i in range(n):

        candidates = np.flatnonzero(coverage[i])

        model.Add(
            sum(y[j] for j in candidates) >= 1
        )

    # Greedy solution as upper bound
    model.Add(
        sum(y) <= len(greedy_centers)
    )

    # Provide the initial solution
    for j in range(n):

        model.AddHint(
            y[j],
            int(D.columns[j] in greedy_centers)
        )

    model.Minimize(sum(y))

    solver = cp_model.CpSolver()

    solver.parameters.max_time_in_seconds = time_limit

    solver.parameters.num_search_workers = 4

    status1 = solver.Solve(model)

    if status1 in (
        cp_model.OPTIMAL,
        cp_model.FEASIBLE
    ):

        best_centers = [
            D.columns[j]
            for j in range(n)
            if solver.Value(y[j]) == 1
        ]

    minimum_centers = len(best_centers)

    # ==============================================
    # Stage 2: Minimize farmer travel distances
    # ==============================================

    model2 = cp_model.CpModel()

    y2 = [
        model2.NewBoolVar(f"center_{j}")
        for j in range(n)
    ]

    x = {}

    objective_terms = []

    for i in range(n):

        candidates = np.flatnonzero(coverage[i])

        for j in candidates:

            x[i, j] = model2.NewBoolVar(
                f"assign_{i}_{j}"
            )

            model2.Add(x[i, j] <= y2[j])

            distance_m = int(
                round(D.iloc[i, j] * 1000)
            )

            objective_terms.append(
                distance_m * x[i, j]
            )

        # Each settlement has exactly one center
        model2.Add(
            sum(x[i, j] for j in candidates) == 1
        )

    # Keep the number of centers fixed
    model2.Add(sum(y2) == minimum_centers)

    # Provide the previous solution as a hint
    for j in range(n):

        model2.AddHint(
            y2[j],
            int(D.columns[j] in best_centers)
        )

    model2.Minimize(sum(objective_terms))

    solver2 = cp_model.CpSolver()

    solver2.parameters.max_time_in_seconds = time_limit

    solver2.parameters.num_search_workers = 4

    status2 = solver2.Solve(model2)

    if status2 in (
        cp_model.OPTIMAL,
        cp_model.FEASIBLE
    ):

        candidate_centers = [
            D.columns[j]
            for j in range(n)
            if solver2.Value(y2[j]) == 1
        ]

        old_cost = D[best_centers].min(axis=1).sum()

        new_cost = D[candidate_centers].min(axis=1).sum()

        if new_cost <= old_cost:

            best_centers = candidate_centers

    results = assign_douars(D, best_centers)

    return {
        "centers": best_centers,
        "assignments": results,
        "number_of_centers": len(best_centers),
        "average_distance_km": results["distance_km"].mean(),
        "maximum_distance_km": results["distance_km"].max(),
        "total_distance_km": results["distance_km"].sum(),
        "stage1_status": solver.StatusName(status1),
        "stage2_status": solver2.StatusName(status2),
        "minimum_centers_proven": status1 == cp_model.OPTIMAL
    }
# --------------------------------------------------
# Capacitated clustering with heterogeneous fleet
# --------------------------------------------------

def optimize_capacitated_clustering(
    D,
    farmers,
    trucks,
    max_distance_km=3.0,
    time_limit=15.0
):

    from engine.configuration import check_fleet_capacity

    feasibility = check_fleet_capacity(
        farmers,
        trucks
    )

    if not feasibility["preliminary_feasible"]:
        raise ValueError(
            "; ".join(feasibility["errors"])
        )

    coverage = build_coverage_matrix(
        D,
        max_distance_km
    )

    ids = D.index.tolist()

    n = len(ids)

    m = len(trucks)

    quantities = farmers.set_index(
        "osm_id"
    ).loc[ids, "quantity_kg"].astype(int).tolist()

    capacities = [
        truck["capacity_kg"]
        for truck in trucks
    ]

    model = cp_model.CpModel()

    # Collection point activation

    y = {
        j: model.NewBoolVar(f"center_{j}")
        for j in range(n)
    }

    # Collection point - truck assignment

    z = {
        (j, k): model.NewBoolVar(f"center_{j}_truck_{k}")
        for j in range(n)
        for k in range(m)
    }

    # Settlement - collection point - truck assignment

    x = {}

    for i in range(n):

        candidates = np.flatnonzero(coverage[i])

        for j in candidates:

            for k in range(m):

                if quantities[i] <= capacities[k]:

                    x[i, j, k] = model.NewBoolVar(
                        f"assign_{i}_{j}_{k}"
                    )

    # Each settlement is assigned exactly once

    for i in range(n):

        variables = [
            var
            for (a, j, k), var in x.items()
            if a == i
        ]

        model.Add(sum(variables) == 1)

    # Each active center is assigned to one truck

    for j in range(n):

        model.Add(
            sum(z[j, k] for k in range(m)) == y[j]
        )

    # Assignment consistency

    for (i, j, k), var in x.items():

        model.Add(var <= z[j, k])

    # No empty active collection points

    for j in range(n):

        variables = [
            var
            for (i, b, k), var in x.items()
            if b == j
        ]

        model.Add(sum(variables) >= y[j])

    # Individual truck capacity

    for k in range(m):

        model.Add(

            sum(
                quantities[i] * var
                for (i, j, t), var in x.items()
                if t == k
            )

            <= capacities[k]
        )

    # Stage 1: minimize collection points

    model.Minimize(sum(y.values()))

    solver = cp_model.CpSolver()

    solver.parameters.max_time_in_seconds = time_limit

    solver.parameters.num_search_workers = 4

    status1 = solver.Solve(model)

    if status1 not in (
        cp_model.OPTIMAL,
        cp_model.FEASIBLE
    ):

        raise RuntimeError(
            f"Clustering status: {solver.StatusName(status1)}"
        )

    best_assignments = [
        (i, j, k)
        for (i, j, k), var in x.items()
        if solver.Value(var)
    ]

    minimum_centers = sum(
        solver.Value(var)
        for var in y.values()
    )

    # Stage 2: minimize farmer travel distance

    model.Add(
        sum(y.values()) == minimum_centers
    )

    # Reuse the first solution

    for var in y.values():
        model.AddHint(var, solver.Value(var))

    for var in z.values():
        model.AddHint(var, solver.Value(var))

    for var in x.values():
        model.AddHint(var, solver.Value(var))

    model.Minimize(

        sum(
            int(round(D.iloc[i, j] * 1000)) * var
            for (i, j, k), var in x.items()
        )

    )

    solver2 = cp_model.CpSolver()

    solver2.parameters.max_time_in_seconds = time_limit

    solver2.parameters.num_search_workers = 4

    status2 = solver2.Solve(model)

    if status2 in (
        cp_model.OPTIMAL,
        cp_model.FEASIBLE
    ):

        candidate_assignments = [
            (i, j, k)
            for (i, j, k), var in x.items()
            if solver2.Value(var)
        ]

        old_distance = sum(
            D.iloc[i, j]
            for i, j, k in best_assignments
        )

        new_distance = sum(
            D.iloc[i, j]
            for i, j, k in candidate_assignments
        )

        if new_distance <= old_distance:
            best_assignments = candidate_assignments

    # Build final results

    records = []

    for i, j, k in best_assignments:

        records.append({

            "osm_id": ids[i],

            "collection_id": ids[j],

            "truck_id": trucks[k]["id"],

            "quantity_kg": quantities[i],

            "distance_km": D.iloc[i, j]

        })

    results = pd.DataFrame(records)

    centers = results["collection_id"].unique().tolist()

    cluster_ids = {
        center: index + 1
        for index, center in enumerate(centers)
    }

    results["cluster"] = (
        results["collection_id"].map(cluster_ids)
    )

    return {

        "assignments": results,

        "centers": centers,

        "number_of_centers": len(centers),

        "average_distance_km": results["distance_km"].mean(),

        "maximum_distance_km": results["distance_km"].max(),

        "total_distance_km": results["distance_km"].sum(),

        "stage1_status": solver.StatusName(status1),

        "stage2_status": solver2.StatusName(status2),

        "minimum_centers_proven": status1 == cp_model.OPTIMAL

    }


# --------------------------------------------------
# Module test - Hybrid clustering
# --------------------------------------------------

if __name__ == "__main__":

    from pathlib import Path

    BASE_DIR = Path(__file__).resolve().parent.parent

    demo = pd.read_csv(
        BASE_DIR / "data" / "demo" / "selected_douars.csv"
    )

    selected = select_douars(
        demo["osm_id"].tolist()
    )

    locations = selected[
        ["osm_id", "road_node"]
    ].rename(columns={"osm_id": "id"})

    D, T = build_distance_matrices(locations)

    MAX_DISTANCE_KM = 3.0

    # Greedy reference
    greedy_centers = greedy_set_cover(
        D, MAX_DISTANCE_KM
    )

    greedy_results = assign_douars(
        D, greedy_centers
    )

    # Hybrid optimization
    start = perf_counter()

    optimization = optimize_collection_points(
        D,
        max_distance_km=MAX_DISTANCE_KM,
        time_limit=10.0
    )

    elapsed = perf_counter() - start

    results = optimization["assignments"]

    names = selected.set_index("osm_id")["name"]

    results["douar"] = results["osm_id"].map(names)

    results["collection_point"] = (
        results["collection_id"].map(names)
    )

    print("\n===== OPTIMIZED CLUSTERS =====")

    print(results[[
        "douar",
        "collection_point",
        "distance_km",
        "cluster"
    ]].to_string(index=False))

    print("\n===== COMPARISON =====")

    comparison = pd.DataFrame({

        "Greedy": [
            len(greedy_centers),
            greedy_results["distance_km"].mean(),
            greedy_results["distance_km"].max(),
            greedy_results["distance_km"].sum()
        ],

        "CP-SAT": [
            optimization["number_of_centers"],
            optimization["average_distance_km"],
            optimization["maximum_distance_km"],
            optimization["total_distance_km"]
        ]

    }, index=[
        "Collection points",
        "Average distance (km)",
        "Maximum distance (km)",
        "Total distance (km)"
    ])

    print(comparison.round(3))

    print("\n===== SOLVER STATUS =====")

    print("Stage 1:", optimization["stage1_status"])

    print("Stage 2:", optimization["stage2_status"])

    print(
        "Minimum centers proven:",
        optimization["minimum_centers_proven"]
    )

    print(f"Optimization time: {elapsed:.4f} s")

    # Validation
    assert len(results) == len(selected)

    assert results["osm_id"].is_unique

    assert (
        results["distance_km"] <= MAX_DISTANCE_KM
    ).all()

    assert (
        optimization["number_of_centers"]
        <= len(greedy_centers)
    )

    print("\nHYBRID CLUSTERING: OK")