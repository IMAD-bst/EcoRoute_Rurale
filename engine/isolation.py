# ==================================================
# EcoRoute Rurale - Isolation Engine
# ==================================================

import heapq
import numpy as np
import pandas as pd

from functools import lru_cache
from time import perf_counter

from engine.network import (
    load_road_network,
    load_douars,
    select_douars,
    build_distance_matrices,
    configure_location,
    targeted_dijkstra
)


# --------------------------------------------------
# Min-Max normalization
# --------------------------------------------------

def normalize(values):

    values = pd.Series(values, dtype=float)

    if not np.isfinite(values.to_numpy()).all():
        raise ValueError("Non-finite isolation values.")

    minimum = values.min()
    maximum = values.max()

    if maximum == minimum:

        return pd.Series(
            0.0,
            index=values.index
        )

    return (
        (values - minimum)
        / (maximum - minimum)
    )


# --------------------------------------------------
# Geographic isolation
# --------------------------------------------------

def compute_neighbor_isolation(D, k=3):

    if D.empty or not D.index.equals(D.columns):
        raise ValueError("Invalid distance matrix.")

    if k < 1:
        raise ValueError("k must be positive.")

    n = len(D)

    distances = D.to_numpy(
        dtype=float,
        copy=True
    )

    # Exclude the settlement itself
    np.fill_diagonal(distances, np.inf)

    mean_distances = []

    for i in range(n):

        reachable = distances[i][
            np.isfinite(distances[i])
        ]

        if len(reachable) == 0:

            if n == 1:
                mean_distances.append(0.0)
                continue

            raise ValueError(
                f"No reachable neighbors: {D.index[i]}"
            )

        nearest = np.sort(reachable)[:k]

        mean_distances.append(
            nearest.mean()
        )

    results = pd.DataFrame({

        "osm_id": D.index,

        "mean_neighbor_distance_km": mean_distances

    })

    results["geographic_isolation"] = normalize(
        results["mean_neighbor_distance_km"]
    )

    return results
# --------------------------------------------------
# Identify major road nodes
# --------------------------------------------------

MAJOR_ROADS = {
    "trunk",
    "primary",
    "secondary",
    "tertiary",
    "trunk_link",
    "primary_link",
    "secondary_link",
    "tertiary_link"
}


@lru_cache(maxsize=1)
def get_major_road_nodes():

    G = load_road_network()

    major_nodes = set()

    for u, v, data in G.edges(data=True):

        if data.get("highway") in MAJOR_ROADS:

            major_nodes.add(u)
            major_nodes.add(v)

    if not major_nodes:
        raise ValueError("No major roads found.")

    return frozenset(major_nodes)
# --------------------------------------------------
# Distance to nearest major road
# --------------------------------------------------

def compute_road_isolation(selected):

    G = load_road_network()

    reverse_G = G.reverse(copy=False)

    major_nodes = get_major_road_nodes()

    remaining = set(
        selected["road_node"].astype(int)
    )

    best = {
        node: 0.0
        for node in major_nodes
    }

    queue = [
        (0.0, node)
        for node in major_nodes
    ]

    heapq.heapify(queue)

    found = {}

    while queue and remaining:

        distance, u = heapq.heappop(queue)

        if distance != best[u]:
            continue

        if u in remaining:

            found[u] = distance

            remaining.remove(u)

        if not remaining:
            break

        for v, edges in reverse_G[u].items():

            edge_length = min(
                float(edge["length"])
                for edge in edges.values()
            )

            candidate = distance + edge_length

            if candidate < best.get(v, np.inf):

                best[v] = candidate

                heapq.heappush(
                    queue,
                    (candidate, v)
                )

    results = selected[
        ["osm_id", "road_node"]
    ].copy()

    results["distance_to_major_road_km"] = (
        results["road_node"].map(found) / 1000
    )

    if results["distance_to_major_road_km"].isna().any():

        raise ValueError(
            "Some settlements cannot reach major roads."
        )

    results["road_isolation"] = normalize(
        results["distance_to_major_road_km"]
    )

    return results
# --------------------------------------------------
# Economic isolation - Market accessibility
# --------------------------------------------------

def compute_market_isolation(selected, destination):

    G = load_road_network()

    reverse_G = G.reverse(copy=False)

    market_node = destination["road_node"]

    douar_nodes = selected["road_node"].tolist()

    routes = targeted_dijkstra(
        reverse_G,
        market_node,
        douar_nodes
    )

    results = selected[["osm_id"]].copy()

    results["distance_to_market_km"] = [

        routes.get(node, (np.inf, np.inf))[0] / 1000

        for node in douar_nodes
    ]

    results["time_to_market_min"] = [

        routes.get(node, (np.inf, np.inf))[1] / 60

        for node in douar_nodes
    ]

    if not np.isfinite(
        results["time_to_market_min"]
    ).all():

        raise ValueError(
            "Some settlements cannot reach the market."
        )

    results["market_isolation"] = normalize(
        results["time_to_market_min"]
    )

    return results
# --------------------------------------------------
# Composite territorial isolation index
# --------------------------------------------------

def compute_isolation_scores(
    selected,
    D,
    destination,
    weights=(0.3, 0.3, 0.4),
    k=3
):

    weights = np.asarray(weights, dtype=float)

    if weights.shape != (3,):
        raise ValueError("Exactly three weights required.")

    if not np.isfinite(weights).all():
        raise ValueError("Invalid weights.")

    if (weights < 0).any():
        raise ValueError("Weights must be non-negative.")

    if not np.isclose(weights.sum(), 1.0):
        raise ValueError("Weights must sum to 1.")

    # Compute the three components

    geographic = compute_neighbor_isolation(
        D, k=k
    )

    road = compute_road_isolation(
        selected
    )

    market = compute_market_isolation(
        selected,
        destination
    )

    # Merge all components

    results = selected[
        ["osm_id", "name"]
    ].copy()

    results = results.merge(
        geographic,
        on="osm_id",
        validate="one_to_one"
    )

    results = results.merge(
        road,
        on="osm_id",
        validate="one_to_one"
    )

    results = results.merge(
        market,
        on="osm_id",
        validate="one_to_one"
    )

    # Composite isolation score

    results["isolation_score"] = (

        weights[0] * results["geographic_isolation"]

        + weights[1] * results["road_isolation"]

        + weights[2] * results["market_isolation"]

    )

    # Isolation classification

    results["isolation_level"] = np.select(

        [
            results["isolation_score"] >= 0.7,
            results["isolation_score"] >= 0.3
        ],

        [
            "High",
            "Medium"
        ],

        default="Low"
    )

    results["isolation_pct"] = (
        results["isolation_score"] * 100
    )

    return results.sort_values(
        "isolation_score",
        ascending=False
    ).reset_index(drop=True)
# --------------------------------------------------
# Module test - Complete Isolation Engine
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

    # Configure market destination

    destination = configure_location(
        name="Chefchaouen",
        latitude=35.1715,
        longitude=-5.2697
    )

    # Compute complete isolation index

    start = perf_counter()

    isolation = compute_isolation_scores(
        selected,
        D,
        destination,
        weights=(0.3, 0.3, 0.4)
    )

    elapsed = perf_counter() - start

    print("\n===== FINAL ISOLATION SCORES =====")

    columns = [
        "name",
        "geographic_isolation",
        "road_isolation",
        "market_isolation",
        "isolation_pct",
        "isolation_level"
    ]

    print(
        isolation[columns]
        .round(3)
        .to_string(index=False)
    )

    print("\n===== ISOLATION DISTRIBUTION =====")

    print(
        isolation["isolation_level"].value_counts()
    )

    print("\n===== PERFORMANCE =====")

    print(f"Settlements: {len(isolation)}")

    print(f"Computation time: {elapsed:.4f} s")

    print(
        f"Average isolation: "
        f"{isolation['isolation_pct'].mean():.2f}%"
    )

    # Validation

    assert len(isolation) == len(selected)

    assert isolation["osm_id"].is_unique

    for column in [
        "geographic_isolation",
        "road_isolation",
        "market_isolation",
        "isolation_score"
    ]:

        assert isolation[column].between(
            0, 1
        ).all()

    assert isolation["isolation_level"].isin(
        ["Low", "Medium", "High"]
    ).all()

    print("\nISOLATION ENGINE: OK")