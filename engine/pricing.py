# ==================================================
# EcoRoute Rurale - Fair Pricing Engine
# ==================================================

import numpy as np
import pandas as pd

from time import perf_counter


# --------------------------------------------------
# Prepare individual pricing inputs
# --------------------------------------------------

def prepare_pricing_data(
    farmers,
    assignments,
    isolation
):

    results = farmers[
        ["osm_id", "name", "quantity_kg"]
    ].copy()

    results = results.merge(

        assignments[[
            "osm_id",
            "collection_id",
            "distance_km"
        ]],

        on="osm_id",
        how="left",
        validate="one_to_one"

    )

    results = results.merge(

        isolation[[
            "osm_id",
            "distance_to_market_km",
            "time_to_market_min",
            "isolation_score",
            "isolation_level"
        ]],

        on="osm_id",
        how="left",
        validate="one_to_one"

    )

    if results.isna().any().any():

        raise ValueError(
            "Missing pricing input data."
        )

    if not results["isolation_score"].between(
        0, 1
    ).all():

        raise ValueError(
            "Invalid isolation scores."
        )

    if (results["quantity_kg"] <= 0).any():

        raise ValueError(
            "Invalid production quantities."
        )

    return results
# --------------------------------------------------
# Conventional transportation baseline
# --------------------------------------------------

def compute_baseline_costs(
    pricing_data,
    fixed_cost=50.0,
    cost_per_km=3.0,
    actual_costs=None
):

    results = pricing_data.copy()

    if actual_costs is not None:

        expected_ids = set(results["osm_id"])

        if set(actual_costs) != expected_ids:

            raise ValueError(
                "Baseline costs must match selected settlements."
            )

        results["baseline_cost"] = (
            results["osm_id"].map(actual_costs)
        )

        results["baseline_source"] = "Actual"

    else:

        if not np.isfinite(
            [fixed_cost, cost_per_km]
        ).all():

            raise ValueError("Invalid baseline rates.")

        if fixed_cost < 0 or cost_per_km < 0:

            raise ValueError("Negative baseline rates.")

        results["baseline_cost"] = (

            fixed_cost

            + cost_per_km
            * results["distance_to_market_km"]

        )

        results["baseline_source"] = "Estimated"

    costs = results["baseline_cost"].to_numpy(
        dtype=float
    )

    if not np.isfinite(costs).all():

        raise ValueError("Invalid baseline costs.")

    if (costs <= 0).any():

        raise ValueError(
            "Baseline costs must be positive."
        )

    return results
# --------------------------------------------------
# Local collection transportation cost
# --------------------------------------------------

def compute_local_collection_costs(
    pricing_data,
    local_cost_per_kg_km=0.01
):

    results = pricing_data.copy()

    if (
        not np.isfinite(local_cost_per_kg_km)
        or local_cost_per_kg_km < 0
    ):

        raise ValueError(
            "Invalid local transportation rate."
        )

    results["local_collection_cost"] = (

        results["quantity_kg"]

        * results["distance_km"]

        * local_cost_per_kg_km

    )

    return results
# --------------------------------------------------
# Compute solidarity-based transportation prices
# --------------------------------------------------

def compute_solidarity_prices(
    pricing_data,
    total_cost,
    alpha
):

    results = pricing_data.copy()

    if not np.isfinite(total_cost) or total_cost < 0:
        raise ValueError("Invalid transportation cost.")

    if not np.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("Invalid solidarity coefficient.")

    quantities = results["quantity_kg"].to_numpy(
        dtype=float
    )

    isolation = results["isolation_score"].to_numpy(
        dtype=float
    )

    weights = quantities * (1 - alpha * isolation)

    if weights.sum() <= 0:
        raise ValueError("Invalid solidarity weights.")

    # Standard proportional pricing

    results["proportional_price"] = (
        total_cost * quantities / quantities.sum()
    )

    # Solidarity pricing

    results["solidarity_weight"] = weights

    results["solidarity_price"] = (
        total_cost * weights / weights.sum()
    )

    # Redistribution

    results["solidarity_discount"] = (

        results["proportional_price"]

        - results["solidarity_price"]

    )

    # Effective transportation cost

    results["effective_cost"] = (

        results["solidarity_price"]

        + results["local_collection_cost"]

    )

    # Individual savings

    results["net_savings"] = (

        results["baseline_cost"]

        - results["effective_cost"]

    )

    results["savings_pct"] = (

        results["net_savings"]

        / results["baseline_cost"]

        * 100

    )

    results["price_per_kg"] = (

        results["solidarity_price"]

        / results["quantity_kg"]

    )

    return results
# --------------------------------------------------
# Analytical solidarity coefficient optimization
# --------------------------------------------------

def optimize_solidarity(
    pricing_data,
    total_cost,
    minimum_savings_pct=20.0,
    max_alpha=0.95
):

    start = perf_counter()

    if not np.isfinite(total_cost) or total_cost < 0:
        raise ValueError("Invalid transportation cost.")

    if not 0 <= minimum_savings_pct <= 100:
        raise ValueError("Invalid minimum savings.")

    if not 0 <= max_alpha < 1:
        raise ValueError("Maximum alpha must be in [0, 1).")

    if pricing_data.empty:
        raise ValueError("No pricing data.")

    q = pricing_data["quantity_kg"].to_numpy(
        dtype=float
    )

    I = pricing_data["isolation_score"].to_numpy(
        dtype=float
    )

    B = pricing_data["baseline_cost"].to_numpy(
        dtype=float
    )

    L = pricing_data["local_collection_cost"].to_numpy(
        dtype=float
    )

    if not np.isfinite(
        np.concatenate([q, I, B, L])
    ).all():
        raise ValueError("Non-finite pricing inputs.")

    if (q <= 0).any() or (B <= 0).any():
        raise ValueError("Invalid quantities or baseline costs.")

    if (I < 0).any() or (I > 1).any():
        raise ValueError("Invalid isolation scores.")

    if (L < 0).any():
        raise ValueError("Negative local collection costs.")

    Q = q.sum()

    H = np.dot(q, I)

    s = minimum_savings_pct / 100

    U = (1 - s) * B - L

    lower_bound = 0.0

    upper_bound = float(max_alpha)

    limiting_douar = None

    tolerance = 1e-9

    # Necessary global feasibility conditions

    if (U < 0).any():

        return {
            "feasible": False,
            "reason": "Local collection cost exceeds an individual budget."
        }

    if U.sum() < total_cost - tolerance:

        return {
            "feasible": False,
            "reason": "Individual budgets cannot cover transportation costs."
        }

    # Linear constraints on alpha

    for i in range(len(q)):

        a = U[i] * H - total_cost * q[i] * I[i]

        b = U[i] * Q - total_cost * q[i]

        if abs(a) <= tolerance:

            if b < -tolerance:

                return {
                    "feasible": False,
                    "reason": "An individual pricing constraint is impossible."
                }

        elif a > 0:

            bound = b / a

            if bound < upper_bound:

                upper_bound = bound

                limiting_douar = pricing_data.iloc[i]["osm_id"]

        else:

            lower_bound = max(
                lower_bound,
                b / a
            )

    # Check feasible interval

    if lower_bound > upper_bound:

        return {
            "feasible": False,
            "reason": "No feasible solidarity coefficient.",
            "lower_bound": lower_bound,
            "upper_bound": upper_bound
        }

    # Maximum feasible solidarity

    alpha = upper_bound

    results = compute_solidarity_prices(
        pricing_data,
        total_cost,
        alpha
    )

    # Numerical verification

    if (
        results["savings_pct"].min()
        < minimum_savings_pct - 1e-7
    ):

        raise RuntimeError(
            "Minimum savings constraint violated."
        )

    if not np.isclose(
        results["solidarity_price"].sum(),
        total_cost,
        atol=1e-7,
        rtol=0
    ):

        raise RuntimeError(
            "Transportation cost recovery failed."
        )

    return {

        "feasible": True,

        "alpha": alpha,

        "lower_bound": lower_bound,

        "upper_bound": upper_bound,

        "limiting_douar": limiting_douar,

        "minimum_savings_pct": minimum_savings_pct,

        "results": results,

        "computation_time_s": perf_counter() - start

    }
# --------------------------------------------------
# Economic and social impact indicators
# --------------------------------------------------

def compute_pricing_kpis(results):

    baseline = results["baseline_cost"].sum()

    transportation = results["solidarity_price"].sum()

    local_cost = results["local_collection_cost"].sum()

    effective_cost = results["effective_cost"].sum()

    savings = results["net_savings"].sum()

    discounts = results["solidarity_discount"]

    beneficiaries = discounts > 1e-8

    contributors = discounts < -1e-8

    high_isolation = (
        results["isolation_level"] == "High"
    )

    redistributed = discounts.clip(lower=0).sum()

    high_isolation_support = discounts[
        high_isolation
    ].clip(lower=0).sum()

    return {

        "total_baseline_cost": baseline,

        "total_transportation_cost": transportation,

        "total_local_collection_cost": local_cost,

        "total_effective_cost": effective_cost,

        "total_net_savings": savings,

        "global_savings_pct": (
            savings / baseline * 100
        ),

        "minimum_individual_savings_pct": (
            results["savings_pct"].min()
        ),

        "maximum_individual_savings_pct": (
            results["savings_pct"].max()
        ),

        "settlements_saving_money": int(
            (results["net_savings"] > 0).sum()
        ),

        "total_settlements": len(results),

        "redistributed_amount": redistributed,

        "solidarity_beneficiaries": int(
            beneficiaries.sum()
        ),

        "solidarity_contributors": int(
            contributors.sum()
        ),

        "high_isolation_support": high_isolation_support,

        "high_isolation_support_pct": (

            high_isolation_support / redistributed * 100

            if redistributed > 1e-8 else 0.0

        )

    }
