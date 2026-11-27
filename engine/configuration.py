# ==================================================
# EcoRoute Rurale - Scenario Configuration
# ==================================================

import numpy as np
import pandas as pd

from numbers import Real


# --------------------------------------------------
# Numeric input validation
# --------------------------------------------------

def validate_number(value, name, minimum=0):

    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be numeric.")

    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite.")

    if value < minimum:
        raise ValueError(
            f"{name} must be >= {minimum}."
        )

    return float(value)


# --------------------------------------------------
# Configure agricultural production
# --------------------------------------------------

def configure_production(selected, quantities):

    selected = selected.copy()

    if selected.empty:
        raise ValueError("No settlements selected.")

    if not selected["osm_id"].is_unique:
        raise ValueError("Duplicate settlement IDs.")

    selected_ids = set(selected["osm_id"])

    provided_ids = set(quantities)

    missing = selected_ids - provided_ids

    extra = provided_ids - selected_ids

    if missing:
        raise ValueError(
            f"Missing production quantities: {missing}"
        )

    if extra:
        raise ValueError(
            f"Unknown settlement IDs: {extra}"
        )

    validated = {}

    for osm_id, quantity in quantities.items():

        quantity = validate_number(
            quantity,
            f"Production of {osm_id}",
            minimum=1
        )

        if not quantity.is_integer():
            raise ValueError(
                "Production must be expressed in whole kg."
            )

        validated[osm_id] = int(quantity)

    selected["quantity_kg"] = (
        selected["osm_id"].map(validated)
    )

    return selected
# --------------------------------------------------
# Configure heterogeneous truck fleet
# --------------------------------------------------

def configure_fleet(trucks, fuel_price):

    fuel_price = validate_number(
        fuel_price,
        "Fuel price"
    )

    if not trucks:
        raise ValueError("The fleet cannot be empty.")

    configured = []

    used_ids = set()

    for truck in trucks:

        truck_id = truck["id"]

        if truck_id in used_ids:
            raise ValueError(
                f"Duplicate truck ID: {truck_id}"
            )

        used_ids.add(truck_id)

        capacity = validate_number(
            truck["capacity_kg"],
            "Truck capacity",
            minimum=1
        )

        if not capacity.is_integer():
            raise ValueError(
                "Capacity must be expressed in whole kg."
            )

        consumption = validate_number(
            truck["consumption_l_100km"],
            "Fuel consumption"
        )

        if consumption == 0:
            raise ValueError(
                "Fuel consumption must be positive."
            )

        fixed_cost = validate_number(
            truck.get("fixed_cost", 0),
            "Fixed cost"
        )

        maintenance = validate_number(
            truck.get("maintenance_per_km", 0),
            "Maintenance cost"
        )

        fuel_cost_per_km = (
            consumption * fuel_price / 100
        )

        cost_per_km = (
            fuel_cost_per_km + maintenance
        )

        configured.append({

            "id": truck_id,

            "capacity_kg": int(capacity),

            "consumption_l_100km": consumption,

            "fuel_price": fuel_price,

            "fixed_cost": fixed_cost,

            "maintenance_per_km": maintenance,

            "fuel_cost_per_km": fuel_cost_per_km,

            "cost_per_km": cost_per_km

        })

    return configured
# --------------------------------------------------
# Preliminary fleet feasibility
# --------------------------------------------------

def check_fleet_capacity(farmers, trucks):

    total_production = int(
        farmers["quantity_kg"].sum()
    )

    total_capacity = sum(
        truck["capacity_kg"]
        for truck in trucks
    )

    max_truck_capacity = max(
        truck["capacity_kg"]
        for truck in trucks
    )

    largest_production = int(
        farmers["quantity_kg"].max()
    )

    errors = []

    if total_production > total_capacity:

        errors.append(
            "Total production exceeds fleet capacity."
        )

    if largest_production > max_truck_capacity:

        errors.append(
            "A settlement exceeds every truck capacity."
        )

    return {

        "total_production_kg": total_production,

        "total_capacity_kg": total_capacity,

        "remaining_capacity_kg": (
            total_capacity - total_production
        ),

        "max_truck_capacity_kg": max_truck_capacity,

        "largest_production_kg": largest_production,

        "utilization_pct": (
            total_production / total_capacity * 100
        ),

        "preliminary_feasible": len(errors) == 0,

        "errors": errors

    }
# --------------------------------------------------
# Module test
# --------------------------------------------------

if __name__ == "__main__":

    from pathlib import Path

    BASE_DIR = Path(__file__).resolve().parent.parent

    demo = pd.read_csv(
        BASE_DIR / "data" / "demo" / "farmers.csv"
    )

    # Individual production inputs

    selected = demo[
        ["osm_id", "name"]
    ].copy()

    quantities = demo.set_index(
        "osm_id"
    )["quantity_kg"].to_dict()

    farmers = configure_production(
        selected,
        quantities
    )

    # User-defined fleet

    trucks = [

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

    ]

    FUEL_PRICE = 12

    fleet = configure_fleet(
        trucks,
        FUEL_PRICE
    )

    feasibility = check_fleet_capacity(
        farmers,
        fleet
    )

    print("\n===== PRODUCTION =====")

    print(
        farmers[["name", "quantity_kg"]]
        .to_string(index=False)
    )

    print("\n===== FLEET =====")

    print(
        pd.DataFrame(fleet)[[
            "id",
            "capacity_kg",
            "consumption_l_100km",
            "fixed_cost",
            "cost_per_km"
        ]].to_string(index=False)
    )

    print("\n===== CAPACITY CHECK =====")

    for key, value in feasibility.items():

        print(f"{key}: {value}")

    assert feasibility["total_production_kg"] == 6328

    assert feasibility["total_capacity_kg"] == 7000

    assert feasibility["preliminary_feasible"]

    assert np.isclose(fleet[0]["cost_per_km"], 2.16)

    assert np.isclose(fleet[1]["cost_per_km"], 1.68)

    # Test an insufficient fleet

    insufficient_fleet = configure_fleet(
        [trucks[0]],
        FUEL_PRICE
    )

    invalid_scenario = check_fleet_capacity(
        farmers,
        insufficient_fleet
    )

    assert not invalid_scenario["preliminary_feasible"]

    print("\nINSUFFICIENT CAPACITY DETECTION: OK")

    print("\nCONFIGURATION ENGINE: OK")