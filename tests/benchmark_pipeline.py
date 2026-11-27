import numpy as np
import pandas as pd

from pathlib import Path

from engine.pipeline import run_pipeline


BASE_DIR = Path(__file__).resolve().parent.parent

# --------------------------------------------------
# Scenario inputs
# --------------------------------------------------

demo = pd.read_csv(
    BASE_DIR / "data" / "demo" / "farmers.csv"
)

osm_ids = demo["osm_id"].tolist()

quantities = demo.set_index(
    "osm_id"
)["quantity_kg"].to_dict()

trucks = [
    {
        "id": 1,
        "capacity_kg": 4000,
        "consumption_l_100km": 18,
        "fixed_cost": 150,
        "maintenance_per_km": 0
    },
    {
        "id": 2,
        "capacity_kg": 3000,
        "consumption_l_100km": 14,
        "fixed_cost": 120,
        "maintenance_per_km": 0
    }
]

departure = {
    "name": "Chefchaouen",
    "latitude": 35.1715,
    "longitude": -5.2697
}

destination = {
    "name": "Chefchaouen",
    "latitude": 35.1715,
    "longitude": -5.2697
}


# --------------------------------------------------
# Execute complete pipeline
# --------------------------------------------------

result = run_pipeline(

    osm_ids=osm_ids,

    quantities=quantities,

    trucks=trucks,

    departure=departure,

    destination=destination,

    fuel_price=12,

    max_distance_km=3,

    baseline_fixed_cost=50,

    baseline_cost_per_km=3,

    local_cost_per_kg_km=0.01,

    minimum_savings_pct=20,

    max_alpha=0.95,

    isolation_weights=(0.3, 0.3, 0.4),

    clustering_time_limit=15,

    routing_time_limit=10

)


# --------------------------------------------------
# Global status
# --------------------------------------------------

print("\n===== PIPELINE STATUS =====")

print(result["status"])


# --------------------------------------------------
# Global KPIs
# --------------------------------------------------

print("\n===== GLOBAL KPIs =====")

for key, value in result["kpis"].items():

    if isinstance(value, (float, np.floating)):
        print(f"{key}: {value:.4f}")
    else:
        print(f"{key}: {value}")


# --------------------------------------------------
# Collection points
# --------------------------------------------------

print("\n===== COLLECTION POINTS =====")

print(
    result["collection_points"][[
        "name",
        "quantity_kg",
        "number_of_douars",
        "maximum_distance_km"
    ]].round(4).to_string(index=False)
)


# --------------------------------------------------
# Truck routes
# --------------------------------------------------

print("\n===== OPTIMIZED ROUTES =====")

for route in result["routing"]["routes"]:

    print(f"\nTruck {route['truck_id']}")

    print(" -> ".join(route["route_names"]))

    print(f"Load: {route['load_kg']} kg")

    print(f"Distance: {route['distance_km']:.4f} km")

    print(f"Cost: {route['total_cost']:.4f} MAD")


# --------------------------------------------------
# Individual pricing
# --------------------------------------------------

if result["pricing"]["feasible"]:

    print("\n===== INDIVIDUAL PRICING =====")

    print(
        result["pricing"]["results"][[
            "name",
            "quantity_kg",
            "isolation_score",
            "baseline_cost",
            "solidarity_price",
            "local_collection_cost",
            "effective_cost",
            "savings_pct"
        ]].round(4).to_string(index=False)
    )

    print(
        f"\nOptimal alpha: "
        f"{result['pricing']['alpha']:.6f}"
    )

else:

    print("\nPRICING INFEASIBLE")

    print(result["pricing"]["reason"])


# --------------------------------------------------
# Execution times
# --------------------------------------------------

print("\n===== PERFORMANCE =====")

for stage, duration in result["timings"].items():

    print(f"{stage}: {duration:.4f} s")


# --------------------------------------------------
# End-to-end validation
# --------------------------------------------------

kpis = result["kpis"]

assignments = result["assignments"]

checks = {

    "pipeline_success": (
        result["status"] == "success"
    ),

    "all_douars_selected": (
        len(result["selected"]) == 10
    ),

    "all_douars_assigned": (
        len(assignments) == 10
        and assignments["osm_id"].is_unique
        and set(assignments["osm_id"]) == set(osm_ids)
    ),

    "production_conservation": (
        kpis["total_production_kg"] == 6328
        and result["routing"]["total_load_kg"] == 6328
    ),

    "collection_distance_respected": (
        assignments["distance_km"].max() <= 3 + 1e-9
    ),

    "minimum_collection_points": (
        kpis["number_of_collection_points"] == 6
    ),

    "routing_validation": (
        result["routing_validation"]["valid"]
    ),

    "reference_cost_recovered": np.isclose(
        kpis["total_transportation_cost"],
        711.9922,
        atol=0.01,
        rtol=0
    ),

    "pricing_feasible": (
        result["pricing"]["feasible"]
    )

}

if result["pricing"]["feasible"]:

    prices = result["pricing"]["results"]

    checks.update({

        "full_cost_recovery": np.isclose(
            prices["solidarity_price"].sum(),
            result["routing"]["total_cost"],
            atol=1e-7,
            rtol=0
        ),

        "minimum_savings_respected": (
            prices["savings_pct"].min() >= 20 - 1e-7
        ),

        "all_douars_save_money": (
            prices["net_savings"] > 0
        ).all(),

        "balanced_redistribution": np.isclose(
            prices["solidarity_discount"].sum(),
            0,
            atol=1e-7
        )

    })


print("\n===== END-TO-END VALIDATION =====")

for name, passed in checks.items():

    print(f"{name}: {'OK' if passed else 'FAILED'}")

assert all(checks.values()), "Pipeline validation failed."

print("\nINTEGRATION PIPELINE: OK")
print("\n===== ROUTE GEOMETRIES =====")

for route in result["geometry"]["routes"]:

    print(f"\nTruck {route['truck_id']}")

    print(
        f"Geometry points: "
        f"{route['number_of_geometry_points']}"
    )

    print(
        f"Optimized distance: "
        f"{route['distance_km']:.6f} km"
    )

    print(
        f"Reconstructed distance: "
        f"{route['geometry_distance_km']:.6f} km"
    )

    assert np.isclose(
        route["distance_km"],
        route["geometry_distance_km"],
        atol=0.001,
        rtol=0
    )

print(
    f"\nGeometry computation time: "
    f"{result['geometry']['computation_time_s']:.4f} s"
)

print("\nGEOMETRY ENGINE: OK")