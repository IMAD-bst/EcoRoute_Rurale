# ==================================================
# EcoRoute Rurale - Route Geometry Engine
# ==================================================

import networkx as nx
import numpy as np

from time import perf_counter
from functools import lru_cache

from engine.network import load_road_network


# --------------------------------------------------
# Reconstruct shortest road segment
# --------------------------------------------------

@lru_cache(maxsize=2048)
def reconstruct_segment(source, target):

    G = load_road_network()

    source = int(source)
    target = int(target)

    distance, nodes = nx.bidirectional_dijkstra(
        G,
        source,
        target,
        weight="length"
    )

    coordinates = tuple(
        (
            float(G.nodes[node]["y"]),
            float(G.nodes[node]["x"])
        )
        for node in nodes
    )

    return {
        "distance_km": distance / 1000,
        "coordinates": coordinates,
        "number_of_nodes": len(nodes)
    }


# --------------------------------------------------
# Reconstruct complete truck routes
# --------------------------------------------------

def reconstruct_routes(routing, locations):

    start = perf_counter()

    road_nodes = locations.set_index(
        "id"
    )["road_node"].to_dict()

    reconstructed = []

    for route in routing["routes"]:

        route_ids = route["route_ids"]

        coordinates = []

        total_distance = 0.0

        for source_id, target_id in zip(
            route_ids[:-1],
            route_ids[1:]
        ):

            segment = reconstruct_segment(
                road_nodes[source_id],
                road_nodes[target_id]
            )

            segment_coordinates = list(
                segment["coordinates"]
            )

            if coordinates:
                coordinates.extend(
                    segment_coordinates[1:]
                )
            else:
                coordinates.extend(
                    segment_coordinates
                )

            total_distance += segment["distance_km"]

        distance_consistent = np.isclose(
            total_distance,
            route["distance_km"],
            atol=0.001,
            rtol=0
        )

        if not distance_consistent:

            raise RuntimeError(
                f"Route distance mismatch for "
                f"truck {route['truck_id']}."
            )

        reconstructed.append({

            **route,

            "coordinates": coordinates,

            "geometry_distance_km": total_distance,

            "number_of_geometry_points": len(
                coordinates
            )

        })

    return {

        "routes": reconstructed,

        "computation_time_s": (
            perf_counter() - start
        )

    }