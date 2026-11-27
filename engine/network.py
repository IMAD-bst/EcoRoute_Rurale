# ==================================================
# EcoRoute Rurale - Network Engine
# ==================================================
import heapq
from time import perf_counter
from pyproj import Transformer
from pathlib import Path
from functools import lru_cache

import pandas as pd
import numpy as np
import networkx as nx
import geopandas as gpd
import sqlite3

from scipy.spatial import cKDTree


# --------------------------------------------------
# Project paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data" / "processed"

# Prefer the compressed network on both local and cloud environments.
GRAPH_PATH = DATA_DIR / "roads.graphml.gz"

if not GRAPH_PATH.is_file():
    GRAPH_PATH = DATA_DIR / "roads.graphml"

if not GRAPH_PATH.is_file():
    raise FileNotFoundError(
        f"Road network missing in {DATA_DIR}. "
        "Expected roads.graphml.gz or roads.graphml."
    )
DOUARS_PATH = DATA_DIR / "douars.csv"
BOUNDARY_PATH = DATA_DIR / "boundary.geojson"
CACHE_PATH = DATA_DIR / "routing_cache.db"

graph_stat = GRAPH_PATH.stat()

CACHE_VERSION = (
    f"shortest_distance_v1:"
    f"{graph_stat.st_size}:"
    f"{graph_stat.st_mtime_ns}"
)


# --------------------------------------------------
# Load road network
# --------------------------------------------------

@lru_cache(maxsize=1)
def load_road_network():

    print("Loading road network...")

    G = nx.read_graphml(
        GRAPH_PATH,
        node_type=int,
        force_multigraph=True
    )

    print(f"Nodes: {G.number_of_nodes():,}")
    print(f"Edges: {G.number_of_edges():,}")

    return G


# --------------------------------------------------
# Load all available settlements
# --------------------------------------------------

@lru_cache(maxsize=1)
def load_douars():

    douars = pd.read_csv(DOUARS_PATH)

    print(f"Available settlements: {len(douars)}")

    return douars


# --------------------------------------------------
# Load geographical boundary
# --------------------------------------------------

@lru_cache(maxsize=1)
def load_boundary():

    return gpd.read_file(BOUNDARY_PATH)
# --------------------------------------------------
# Build spatial index
# --------------------------------------------------

@lru_cache(maxsize=1)
def build_spatial_index():

    G = load_road_network()

    boundary = load_boundary()

    utm_crs = boundary.estimate_utm_crs()

    transformer = Transformer.from_crs(
        "EPSG:4326",
        utm_crs,
        always_xy=True
    )

    node_ids = list(G.nodes)

    longitudes = np.array([
        float(G.nodes[n]["x"]) for n in node_ids
    ])

    latitudes = np.array([
        float(G.nodes[n]["y"]) for n in node_ids
    ])

    x, y = transformer.transform(
        longitudes,
        latitudes
    )

    coordinates = np.column_stack((x, y))

    tree = cKDTree(coordinates)

    print("Spatial index created successfully.")

    return tree, node_ids, transformer


# --------------------------------------------------
# Connect any location to the road network
# --------------------------------------------------

def snap_location(latitude, longitude, max_snap_m=500):

    tree, node_ids, transformer = build_spatial_index()

    G = load_road_network()

    x, y = transformer.transform(
        longitude,
        latitude
    )

    distance, index = tree.query([x, y])

    if distance > max_snap_m:

        raise ValueError(
            f"No road node within {max_snap_m} meters. "
            f"Nearest node: {distance:.2f} m"
        )

    node = node_ids[index]

    return {
        "latitude": latitude,
        "longitude": longitude,
        "road_node": node,
        "snap_distance_m": float(distance),
        "road_latitude": float(G.nodes[node]["y"]),
        "road_longitude": float(G.nodes[node]["x"])
    }
# --------------------------------------------------
# Select and connect settlements
# --------------------------------------------------

def select_douars(osm_ids):

    douars = load_douars()

    osm_ids = list(dict.fromkeys(osm_ids))

    missing_ids = set(osm_ids) - set(douars["osm_id"])

    if missing_ids:

        raise ValueError(
            f"Unknown settlements: {missing_ids}"
        )

    selected = (
        douars.set_index("osm_id")
        .loc[osm_ids]
        .reset_index()
        .copy()
    )

    connections = []

    for row in selected.itertuples():

        connection = snap_location(
            row.latitude,
            row.longitude
        )

        connections.append(connection)

    selected["road_node"] = [
        c["road_node"] for c in connections
    ]

    selected["snap_distance_m"] = [
        c["snap_distance_m"] for c in connections
    ]

    return selected
# --------------------------------------------------
# Targeted Dijkstra algorithm
# --------------------------------------------------

def targeted_dijkstra(G, source, targets):

    remaining = set(targets)

    best = {source: (0.0, 0.0)}

    found = {}

    queue = [(0.0, 0.0, source)]

    while queue and remaining:

        distance, duration, u = heapq.heappop(queue)

        if (distance, duration) != best[u]:
            continue

        # Check whether the current node is a target
        if u in remaining:

            found[u] = (distance, duration)

            remaining.remove(u)

        # Stop when all destinations are reached
        if not remaining:
            break

        # Explore neighboring nodes
        for v, edges in G[u].items():

            for edge in edges.values():

                new_distance = distance + float(
                    edge["length"]
                )

                new_duration = duration + float(
                    edge["travel_time"]
                )

                candidate = (
                    new_distance,
                    new_duration
                )

                if candidate < best.get(
                    v, (np.inf, np.inf)
                ):

                    best[v] = candidate

                    heapq.heappush(
                        queue,
                        (new_distance, new_duration, v)
                    )

    return found
# --------------------------------------------------
# Dynamic road distance and duration matrices
# --------------------------------------------------

def build_distance_matrices(locations):

    initialize_cache()

    labels = locations["id"].tolist()

    nodes = locations["road_node"].tolist()

    n = len(nodes)

    distances = np.full((n, n), np.inf)

    durations = np.full((n, n), np.inf)

    cache = {}

    start_time = perf_counter()

    for i, source in enumerate(nodes):

        print(f"Processing location {i + 1}/{n}")

        # Avoid duplicate calculations
        if source not in cache:

            cache[source] = cached_dijkstra(
                source,
                nodes
            )

        for j, target in enumerate(nodes):

            if target in cache[source]:

                distance, duration = cache[source][target]

                distances[i, j] = distance / 1000

                durations[i, j] = duration / 60

    distance_matrix = pd.DataFrame(
        distances,
        index=labels,
        columns=labels
    )

    duration_matrix = pd.DataFrame(
        durations,
        index=labels,
        columns=labels
    )

    elapsed = perf_counter() - start_time

    print(f"Computation time: {elapsed:.2f} seconds")

    return distance_matrix, duration_matrix
# --------------------------------------------------
# Configure departure or destination
# --------------------------------------------------

def configure_location(name, latitude, longitude):

    location = snap_location(
        latitude=latitude,
        longitude=longitude
    )

    location["name"] = name

    return location
# --------------------------------------------------
# Build routing locations
# --------------------------------------------------

def build_routing_locations(
    collection_points,
    departure,
    destination
):

    locations = [
        {
            "id": "departure",
            "name": departure["name"],
            "road_node": departure["road_node"],
            "quantity_kg": 0
        }
    ]

    for point in collection_points.itertuples():

        locations.append({
            "id": point.osm_id,
            "name": point.name,
            "road_node": point.road_node,
            "quantity_kg": point.quantity_kg
        })

    locations.append({
        "id": "destination",
        "name": destination["name"],
        "road_node": destination["road_node"],
        "quantity_kg": 0
    })

    return pd.DataFrame(locations)
# --------------------------------------------------
# Persistent routing cache
# --------------------------------------------------

def initialize_cache():

    with sqlite3.connect(CACHE_PATH) as conn:

        conn.execute("""
            CREATE TABLE IF NOT EXISTS road_distances (
                version TEXT NOT NULL,
                source INTEGER NOT NULL,
                target INTEGER NOT NULL,
                distance_m REAL NOT NULL,
                duration_s REAL NOT NULL,
                PRIMARY KEY (version, source, target)
            )
        """)


def get_cached_distances(source, targets):

    targets = set(map(int, targets))

    cached = {}

    with sqlite3.connect(CACHE_PATH) as conn:

        rows = conn.execute("""
            SELECT target, distance_m, duration_s
            FROM road_distances
            WHERE version = ? AND source = ?
        """, (CACHE_VERSION, int(source)))

        for target, distance, duration in rows:

            if target in targets:

                cached[target] = (
                    distance,
                    duration
                )

    return cached


def save_cached_distances(source, results):

    records = [
        (
            CACHE_VERSION,
            int(source),
            int(target),
            float(distance),
            float(duration)
        )
        for target, (distance, duration)
        in results.items()
    ]

    with sqlite3.connect(CACHE_PATH) as conn:

        conn.executemany("""
            INSERT OR REPLACE INTO road_distances
            VALUES (?, ?, ?, ?, ?)
        """, records)
# --------------------------------------------------
# Cached targeted Dijkstra
# --------------------------------------------------

def cached_dijkstra(source, targets):

    targets = set(map(int, targets))

    source = int(source)

    cached = get_cached_distances(
        source,
        targets
    )

    missing = targets - set(cached)

    if missing:

        G = load_road_network()

        computed = targeted_dijkstra(
            G,
            source,
            missing
        )

        save_cached_distances(
            source,
            computed
        )

        cached.update(computed)

        print(
            f"Cache hits: {len(targets) - len(missing)} | "
            f"Computed: {len(computed)}"
        )

    else:

        print(f"Cache hits: {len(targets)} | Computed: 0")

    return cached
# --------------------------------------------------
# Module test - Cache performance
# --------------------------------------------------

if __name__ == "__main__":

    douars = load_douars()

    test_ids = douars["osm_id"].head(5).tolist()

    selected = select_douars(test_ids)

    locations = selected[
        ["osm_id", "road_node"]
    ].rename(columns={"osm_id": "id"})

    print("\n===== FIRST EXECUTION =====")

    start = perf_counter()

    D1, T1 = build_distance_matrices(locations)

    time1 = perf_counter() - start

    print("\n===== SECOND EXECUTION =====")

    start = perf_counter()

    D2, T2 = build_distance_matrices(locations)

    time2 = perf_counter() - start

    assert np.allclose(D1, D2)
    assert np.allclose(T1, T2)

    print("\n===== PERFORMANCE RESULTS =====")

    print(f"First execution: {time1:.4f} s")
    print(f"Second execution: {time2:.4f} s")

    if time2 > 0:

        print(f"Speedup: {time1 / time2:.2f}x")

    print(f"Cache file: {CACHE_PATH.exists()}")

    print("\nPERSISTENT CACHE: OK")