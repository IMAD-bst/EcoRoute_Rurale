import numpy as np
import pandas as pd

from pathlib import Path
from time import perf_counter

from engine.network import (
    select_douars,
    build_distance_matrices
)

from engine.configuration import (
    configure_production,
    configure_fleet
)

from engine.clustering import (
    optimize_capacitated_clustering
)


BASE_DIR = Path(__file__).resolve().parent.parent

demo = pd.read_csv(
    BASE_DIR / "data" / "demo" / "farmers.csv"
)

selected = select_douars(
    demo["osm_id"].tolist()
)

quantities = demo.set_index(
    "osm_id"
)["quantity_kg"].to_dict()

farmers = configure_production(
    selected,
    quantities
)

trucks = configure_fleet(

    [
        {
            "id": 1,
            "capacity_kg": 4000,
            "consumption_l_100km": 18,
            "fixed_cost": 150
        },
        {
            "id": 2,
            "capacity_kg": 3000,
            "consumption_l_100km": 14,
            "fixed_cost": 120
        }
    ],

    fuel_price=12

)

locations = selected[
    ["osm_id", "road_node"]
].rename(columns={"osm_id": "id"})

D, T = build_distance_matrices(locations)

# Run capacitated clustering

start = perf_counter()

optimization = optimize_capacitated_clustering(

    D,
    farmers,
    trucks,
    max_distance_km=3.0,
    time_limit=15.0

)

elapsed = perf_counter() - start

results = optimization["assignments"]

names = selected.set_index("osm_id")["name"]

results["douar"] = results["osm_id"].map(names)

results["collection_point"] = (
    results["collection_id"].map(names)
)

print("\n===== CAPACITATED CLUSTERING =====")

print(results[[
    "douar",
    "collection_point",
    "truck_id",
    "quantity_kg",
    "distance_km"
]].round(3).to_string(index=False))


# Collection point summary

collection_points = results.groupby(
    ["collection_id", "collection_point", "truck_id"]
).agg(

    quantity_kg=("quantity_kg", "sum"),

    number_of_douars=("osm_id", "count"),

    maximum_distance_km=("distance_km", "max")

).reset_index()

print("\n===== COLLECTION POINTS =====")

print(
    collection_points.round(3)
    .to_string(index=False)
)


# Truck capacity summary

loads = results.groupby(
    "truck_id"
)["quantity_kg"].sum()

fleet_summary = pd.DataFrame(trucks)

fleet_summary["load_kg"] = (
    fleet_summary["id"].map(loads).fillna(0)
)

fleet_summary["utilization_pct"] = (

    fleet_summary["load_kg"]

    / fleet_summary["capacity_kg"] * 100

)

print("\n===== TRUCK CAPACITY =====")

print(fleet_summary[[
    "id",
    "capacity_kg",
    "load_kg",
    "utilization_pct"
]].round(2).to_string(index=False))


# Validation

assert len(results) == len(farmers)

assert results["osm_id"].is_unique

assert set(results["osm_id"]) == set(farmers["osm_id"])

assert results["quantity_kg"].sum() == 6328

assert (results["distance_km"] <= 3.0).all()

assert (
    fleet_summary["load_kg"]
    <= fleet_summary["capacity_kg"]
).all()

assert (
    results.groupby("collection_id")["truck_id"]
    .nunique() == 1
).all()

expected_quantities = farmers.set_index(
    "osm_id"
)["quantity_kg"]

actual_quantities = results.set_index(
    "osm_id"
)["quantity_kg"]

assert (
    actual_quantities.sort_index()
    == expected_quantities.sort_index()
).all()


print("\n===== PERFORMANCE =====")

print(
    f"Collection points: "
    f"{optimization['number_of_centers']}"
)

print(
    f"Average distance: "
    f"{optimization['average_distance_km']:.4f} km"
)

print(
    f"Maximum distance: "
    f"{optimization['maximum_distance_km']:.4f} km"
)

print(
    f"Total distance: "
    f"{optimization['total_distance_km']:.4f} km"
)

print(f"Stage 1: {optimization['stage1_status']}")

print(f"Stage 2: {optimization['stage2_status']}")

print(
    f"Minimum centers proven: "
    f"{optimization['minimum_centers_proven']}"
)

print(f"Computation time: {elapsed:.4f} s")

print("\nCAPACITATED CLUSTERING: OK")