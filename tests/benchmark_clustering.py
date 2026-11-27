# ==================================================
# EcoRoute Rurale - 100 Settlements Benchmark
# ==================================================

import os
import psutil
import numpy as np
import pandas as pd

from time import perf_counter

from engine.network import (
    load_douars,
    select_douars,
    build_distance_matrices,
    build_spatial_index
)

from engine.clustering import (
    greedy_set_cover,
    assign_douars,
    optimize_collection_points
)

N_DOUARS = 100
MAX_DISTANCE_KM = 3.0
TIME_LIMIT = 10.0

process = psutil.Process(os.getpid())


def memory_mb():
    return process.memory_info().rss / 1024**2


# --------------------------------------------------
# 1. Select 100 nearby settlements
# --------------------------------------------------

global_start = perf_counter()

douars = load_douars()

tree, node_ids, transformer = build_spatial_index()

x, y = transformer.transform(
    douars["longitude"].to_numpy(),
    douars["latitude"].to_numpy()
)

reference_x = x[0]
reference_y = y[0]

distances = np.hypot(
    x - reference_x,
    y - reference_y
)

ordered_indices = np.argsort(distances)

selected_parts = []

for index in ordered_indices:

    osm_id = douars.iloc[index]["osm_id"]

    try:

        connected = select_douars([osm_id])

        selected_parts.append(connected)

    except ValueError as error:

        print(f"Skipped {osm_id}: {error}")

    if len(selected_parts) == N_DOUARS:
        break

if len(selected_parts) != N_DOUARS:
    raise RuntimeError("Not enough valid settlements.")

selected = pd.concat(
    selected_parts,
    ignore_index=True
)

print("\n===== SCENARIO =====")
print(f"Settlements: {len(selected)}")
print(f"Maximum farmer distance: {MAX_DISTANCE_KM} km")
print(f"Memory usage: {memory_mb():.2f} MB")


# --------------------------------------------------
# 2. Build road matrices
# --------------------------------------------------

locations = selected[
    ["osm_id", "road_node"]
].rename(columns={"osm_id": "id"})

start = perf_counter()

D, T = build_distance_matrices(locations)

matrix_time = perf_counter() - start

print("\n===== ROAD MATRICES =====")

print(f"Matrix shape: {D.shape}")
print(f"Matrix time: {matrix_time:.4f} s")
print(f"Memory usage: {memory_mb():.2f} MB")

print(
    "Unreachable pairs:",
    np.isinf(D.to_numpy()).sum()
)


# --------------------------------------------------
# 3. Greedy clustering
# --------------------------------------------------

start = perf_counter()

greedy_centers = greedy_set_cover(
    D,
    MAX_DISTANCE_KM
)

greedy_results = assign_douars(
    D,
    greedy_centers
)

greedy_time = perf_counter() - start

print("\n===== GREEDY =====")

print(f"Centers: {len(greedy_centers)}")
print(f"Time: {greedy_time:.6f} s")


# --------------------------------------------------
# 4. Hybrid CP-SAT optimization
# --------------------------------------------------

start = perf_counter()

optimization = optimize_collection_points(
    D,
    max_distance_km=MAX_DISTANCE_KM,
    time_limit=TIME_LIMIT
)

optimization_time = perf_counter() - start

results = optimization["assignments"]


# --------------------------------------------------
# 5. Compare both algorithms
# --------------------------------------------------

comparison = pd.DataFrame({

    "Greedy": [
        len(greedy_centers),
        greedy_results["distance_km"].mean(),
        greedy_results["distance_km"].max(),
        greedy_results["distance_km"].sum(),
        greedy_time
    ],

    "Hybrid": [
        optimization["number_of_centers"],
        optimization["average_distance_km"],
        optimization["maximum_distance_km"],
        optimization["total_distance_km"],
        optimization_time
    ]

}, index=[
    "Collection points",
    "Average distance (km)",
    "Maximum distance (km)",
    "Total distance (km)",
    "Computation time (s)"
])

print("\n===== COMPARISON =====")
print(comparison.round(4))


# --------------------------------------------------
# 6. Validation
# --------------------------------------------------

assert len(results) == N_DOUARS

assert results["osm_id"].is_unique

assert (
    results["distance_km"] <= MAX_DISTANCE_KM
).all()

assert (
    optimization["number_of_centers"]
    <= len(greedy_centers)
)

assert results["collection_id"].isin(
    optimization["centers"]
).all()


# --------------------------------------------------
# 7. Final benchmark
# --------------------------------------------------

reduction = (
    1 - optimization["number_of_centers"] / N_DOUARS
) * 100

print("\n===== FINAL BENCHMARK =====")

print(f"Settlements: {N_DOUARS}")

print(
    f"Collection points: "
    f"{optimization['number_of_centers']}"
)

print(f"Stop reduction: {reduction:.2f}%")

print(f"Matrix time: {matrix_time:.4f} s")

print(f"Greedy time: {greedy_time:.6f} s")

print(f"Hybrid time: {optimization_time:.4f} s")

print(
    f"Stage 1: {optimization['stage1_status']}"
)

print(
    f"Stage 2: {optimization['stage2_status']}"
)

print(
    f"Minimum centers proven: "
    f"{optimization['minimum_centers_proven']}"
)

print(f"Memory usage: {memory_mb():.2f} MB")

print(
    f"Total execution time: "
    f"{perf_counter() - global_start:.2f} s"
)

print("\n100 SETTLEMENTS BENCHMARK: OK")