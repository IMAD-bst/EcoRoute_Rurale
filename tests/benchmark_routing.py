import pandas as pd

from pathlib import Path

from engine.network import (
    select_douars,
    build_distance_matrices,
    configure_location,
    build_routing_locations
)

from engine.configuration import (
    configure_production,
    configure_fleet
)

from engine.clustering import (
    optimize_capacitated_clustering
)

from engine.routing import (
    prepare_collection_points,
    optimize_routes,
    validate_routes
)


BASE_DIR = Path(__file__).resolve().parent.parent

# --------------------------------------------------
# Production configuration
# --------------------------------------------------

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

# --------------------------------------------------
# Fleet configuration
# --------------------------------------------------

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

# --------------------------------------------------
# Clustering
# --------------------------------------------------

douar_locations = selected[
    ["osm_id", "road_node"]
].rename(columns={"osm_id": "id"})

D, T = build_distance_matrices(
    douar_locations
)

clustering = optimize_capacitated_clustering(

    D,

    farmers,

    trucks,

    max_distance_km=3.0,

    time_limit=15

)

collection_points = prepare_collection_points(

    clustering["assignments"],

    selected

)

# --------------------------------------------------
# Independent departure and destination
# --------------------------------------------------

departure = configure_location(

    name="Chefchaouen",

    latitude=35.1715,

    longitude=-5.2697

)

destination = configure_location(

    name="Chefchaouen",

    latitude=35.1715,

    longitude=-5.2697

)

# --------------------------------------------------
# Routing matrices
# --------------------------------------------------

locations = build_routing_locations(

    collection_points,

    departure,

    destination

)

routing_D, routing_T = build_distance_matrices(
    locations
)

# --------------------------------------------------
# Optimize routes
# --------------------------------------------------

optimization = optimize_routes(

    locations,

    routing_D,

    routing_T,

    trucks,

    time_limit=10

)

# --------------------------------------------------
# Display routes
# --------------------------------------------------

for route in optimization["routes"]:

    print(
        f"\n===== TRUCK {route['truck_id']} ====="
    )

    print(
        " -> ".join(route["route_names"])
    )

    for key in [

        "load_kg",
        "capacity_kg",
        "utilization_pct",
        "distance_km",
        "duration_min",
        "fuel_liters",
        "fuel_cost",
        "maintenance_cost",
        "fixed_cost",
        "total_cost"

    ]:

        print(f"{key}: {route[key]:.4f}")

# --------------------------------------------------
# Global results
# --------------------------------------------------

print("\n===== GLOBAL RESULTS =====")

for key, value in optimization.items():

    if key != "routes":

        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

# --------------------------------------------------
# Validation
# --------------------------------------------------

validation = validate_routes(
    optimization,
    locations
)

print("\n===== VALIDATION =====")

for name, passed in validation["checks"].items():

    print(
        f"{name}: {'OK' if passed else 'FAILED'}"
    )

assert validation["valid"]

print("\nROUTING ENGINE: OK")