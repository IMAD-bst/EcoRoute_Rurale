from engine.network import (
    select_douars,
    configure_location
)

from engine.isolation import (
    compute_market_isolation
)
import numpy as np
import pandas as pd

from pathlib import Path

from engine.pricing import (
    prepare_pricing_data,
    compute_baseline_costs,
    compute_local_collection_costs,
    optimize_solidarity,
    compute_pricing_kpis
)

BASE_DIR = Path(__file__).resolve().parent.parent

DEMO_DIR = BASE_DIR / "data" / "demo"

# --------------------------------------------------
# Reference scenario
# --------------------------------------------------

farmers = pd.read_csv(
    DEMO_DIR / "farmers.csv"
)

assignments = pd.read_csv(
    DEMO_DIR / "clustering_results.csv"
)

isolation = pd.read_csv(
    DEMO_DIR / "final_isolation_scores.csv"
)

# Recover missing market distances

selected = select_douars(
    farmers["osm_id"].tolist()
)

destination = configure_location(
    name="Chefchaouen",
    latitude=35.1715,
    longitude=-5.2697
)

market_data = compute_market_isolation(
    selected,
    destination
)

isolation = isolation.merge(
    market_data[[
        "osm_id",
        "distance_to_market_km"
    ]],
    on="osm_id",
    how="left",
    validate="one_to_one"
)

print("\n===== MARKET DISTANCES =====")

print(
    isolation[[
        "osm_id",
        "distance_to_market_km"
    ]].to_string(index=False)
)

TOTAL_COST = 711.9922

MINIMUM_SAVINGS = 20.0

MAX_ALPHA = 0.95

# --------------------------------------------------
# Prepare pricing inputs
# --------------------------------------------------

pricing_data = prepare_pricing_data(
    farmers,
    assignments,
    isolation
)

pricing_data = compute_baseline_costs(

    pricing_data,

    fixed_cost=50,

    cost_per_km=3

)

pricing_data = compute_local_collection_costs(

    pricing_data,

    local_cost_per_kg_km=0.01

)

# --------------------------------------------------
# Optimize solidarity
# --------------------------------------------------

optimization = optimize_solidarity(

    pricing_data,

    TOTAL_COST,

    minimum_savings_pct=MINIMUM_SAVINGS,

    max_alpha=MAX_ALPHA

)

if not optimization["feasible"]:

    raise RuntimeError(
        optimization["reason"]
    )

results = optimization["results"]

kpis = compute_pricing_kpis(results)

# --------------------------------------------------
# Individual results
# --------------------------------------------------

print("\n===== INDIVIDUAL PRICING =====")

columns = [

    "name",

    "quantity_kg",

    "isolation_score",

    "baseline_cost",

    "proportional_price",

    "solidarity_price",

    "local_collection_cost",

    "effective_cost",

    "savings_pct"

]

print(
    results[columns]
    .round(3)
    .to_string(index=False)
)

# --------------------------------------------------
# Global indicators
# --------------------------------------------------

print("\n===== PRICING KPIs =====")

for key, value in kpis.items():

    if isinstance(value, (float, np.floating)):
        print(f"{key}: {value:.4f}")
    else:
        print(f"{key}: {value}")

# --------------------------------------------------
# Optimization performance
# --------------------------------------------------

print("\n===== SOLIDARITY OPTIMIZATION =====")

print(f"Optimal alpha: {optimization['alpha']:.6f}")

print(f"Lower bound: {optimization['lower_bound']:.6f}")

print(f"Upper bound: {optimization['upper_bound']:.6f}")

print(f"Limiting douar: {optimization['limiting_douar']}")

print(
    f"Computation time: "
    f"{optimization['computation_time_s']:.6f} s"
)

# --------------------------------------------------
# Validation
# --------------------------------------------------

checks = {

    "full_cost_recovery": np.isclose(
        results["solidarity_price"].sum(),
        TOTAL_COST
    ),

    "minimum_savings_respected": (
        results["savings_pct"].min()
        >= MINIMUM_SAVINGS - 1e-7
    ),

    "all_settlements_save_money": (
        results["net_savings"] > 0
    ).all(),

    "non_negative_prices": (
        results["solidarity_price"] >= 0
    ).all(),

    "balanced_redistribution": np.isclose(
        results["solidarity_discount"].sum(),
        0,
        atol=1e-7
    ),

    "production_conservation": (
        results["quantity_kg"].sum() == 6328
    ),

    "valid_alpha": (
        0 <= optimization["alpha"] <= MAX_ALPHA
    )

}

print("\n===== VALIDATION =====")

for name, passed in checks.items():

    print(
        f"{name}: {'OK' if passed else 'FAILED'}"
    )

assert all(checks.values())

# --------------------------------------------------
# Impossible savings scenario
# --------------------------------------------------

impossible = optimize_solidarity(

    pricing_data,

    TOTAL_COST,

    minimum_savings_pct=99,

    max_alpha=MAX_ALPHA

)

assert not impossible["feasible"]

print("\nINFEASIBLE SCENARIO DETECTION: OK")

print("\nFAIR PRICING ENGINE: OK")