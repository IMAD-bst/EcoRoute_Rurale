
# ==================================================
# EcoRoute Rurale - Interactive Map
# ==================================================

import folium

from html import escape
from engine.network import load_boundary


MAP_TRANSLATIONS = {

    "en": {
        "settlements": "Settlements",
        "collection_points": "Collection points",
        "departure": "Departure",
        "destination": "Destination",
        "truck": "Truck",
        "production": "Production",
        "isolation": "Isolation",
        "distance": "Distance",
        "cost": "Cost",
        "boundary": "Province boundary",
        "legend": "Map legend",
        "high": "High isolation",
        "medium": "Medium isolation",
        "low": "Low isolation",
        "endpoints": "Departure and destination",
        "assigned_truck": "Assigned truck"
    },

    "fr": {
        "settlements": "Douars",
        "collection_points": "Points de collecte",
        "departure": "Départ",
        "destination": "Destination",
        "truck": "Camion",
        "production": "Production",
        "isolation": "Isolement",
        "distance": "Distance",
        "cost": "Coût",
        "boundary": "Limites de la province",
        "legend": "Légende",
        "high": "Isolement élevé",
        "medium": "Isolement moyen",
        "low": "Isolement faible",
        "endpoints": "Départ et destination",
        "assigned_truck": "Camion affecté"
    },

    "ar": {
        "settlements": "الدواوير",
        "collection_points": "نقاط التجميع",
        "departure": "نقطة الانطلاق",
        "destination": "الوجهة",
        "truck": "الشاحنة",
        "production": "الإنتاج",
        "isolation": "العزلة",
        "distance": "المسافة",
        "cost": "التكلفة",
        "boundary": "حدود الإقليم",
        "legend": "مفتاح الخريطة",
        "high": "عزلة مرتفعة",
        "medium": "عزلة متوسطة",
        "low": "عزلة منخفضة",
        "endpoints": "الانطلاق والوجهة",
        "assigned_truck": "الشاحنة المخصصة"
    }
}


ROUTE_COLORS = [
    "#2563EB",
    "#16A34A",
    "#9333EA",
    "#0891B2",
    "#DB2777",
    "#92400E",
    "#4F46E5",
    "#64748B"
]

ISOLATION_COLORS = {
    "High": "#DC2626",
    "Medium": "#F59E0B",
    "Low": "#22C55E"
}


def create_optimization_map(result, language="fr"):

    labels = MAP_TRANSLATIONS[language]

    direction = "rtl" if language == "ar" else "ltr"

    selected = result["selected"]
    collection_points = result["collection_points"]
    departure = result["departure"]
    destination = result["destination"]

    routes = result["geometry"]["routes"]

    # ----------------------------------------------
    # Prepare settlement information
    # ----------------------------------------------

    farmers = result["farmers"].merge(
        result["isolation"][
            ["osm_id", "isolation_score", "isolation_level"]
        ],
        on="osm_id",
        validate="one_to_one"
    )

    farmers = farmers.merge(
        result["assignments"][
            ["osm_id", "collection_id", "truck_id", "truck_label"]
        ],
        on="osm_id",
        validate="one_to_one"
    )

    # ----------------------------------------------
    # Initialize map
    # ----------------------------------------------

    m = folium.Map(
        location=[selected["latitude"].mean(), selected["longitude"].mean()],
        zoom_start=10,
        tiles="OpenStreetMap",
        control_scale=True,
        prefer_canvas=True
    )

    # ----------------------------------------------
    # Province boundary
    # ----------------------------------------------

    boundary = load_boundary().to_crs("EPSG:4326")

    folium.GeoJson(
        boundary.geometry.__geo_interface__,
        name=labels["boundary"],
        style_function=lambda feature: {
            "color": "#475569",
            "weight": 2,
            "fillOpacity": 0.03,
            "dashArray": "6 4"
        }
    ).add_to(m)

    bounds = []

    # ----------------------------------------------
    # Truck routes
    # ----------------------------------------------

    for index, route in enumerate(routes):

        color = ROUTE_COLORS[
            index % len(ROUTE_COLORS)
        ]

        truck_id = route.get("truck_label", route["truck_id"])

        route_layer = folium.FeatureGroup(
            name=f"{labels['truck']} {truck_id}"
        )

        popup = f"""
        <div dir="{direction}">
            <b>{labels['truck']} {truck_id}</b><br>
            {labels['production']}: {route['load_kg']} kg<br>
            {labels['distance']}: {route['distance_km']:.2f} km<br>
            {labels['cost']}: {route['total_cost']:.2f} MAD
        </div>
        """

        folium.PolyLine(
            locations=route["coordinates"],
            color=color,
            weight=6 if index % 2 == 0 else 4,
            opacity=0.9,
            dash_array=None if index % 2 == 0 else "10 7",
            tooltip=f"{labels['truck']} {truck_id}",
            popup=folium.Popup(popup, max_width=300)
        ).add_to(route_layer)

        route_layer.add_to(m)

        bounds.extend(route["coordinates"])

    # ----------------------------------------------
    # Settlements
    # ----------------------------------------------

    settlement_layer = folium.FeatureGroup(
        name=labels["settlements"]
    )

    for row in farmers.itertuples():

        color = ISOLATION_COLORS[
            row.isolation_level
        ]

        location = [
            row.latitude,
            row.longitude
        ]

        popup = f"""
        <div dir="{direction}">
            <b>{escape(str(row.name))}</b><br>
            {labels['production']}: {row.quantity_kg} kg<br>
            {labels['isolation']}: {row.isolation_score * 100:.1f}%<br>
            {labels['assigned_truck']}: {row.truck_label}
        </div>
        """

        folium.CircleMarker(
            location=location,
            radius=6,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.85,
            weight=2,
            tooltip=escape(str(row.name)),
            popup=folium.Popup(popup, max_width=300)
        ).add_to(settlement_layer)

        bounds.append(location)

    settlement_layer.add_to(m)

    # ----------------------------------------------
    # Collection points
    # ----------------------------------------------

    collection_layer = folium.FeatureGroup(
        name=labels["collection_points"]
    )

    stop_mapping = {
        collection_id: (index + 1, ROUTE_COLORS[route_index % len(ROUTE_COLORS)])
        for route_index, route in enumerate(routes)
        for index, collection_id in enumerate(route["collection_ids"])
    }

    for row in collection_points.itertuples():

        stop_number, stop_color = stop_mapping[row.osm_id]

        location = [
            row.latitude,
            row.longitude
        ]

        popup = f"""
        <div dir="{direction}">
            <b>{escape(str(row.name))}</b><br>
            {labels['collection_points']}<br>
            {labels['production']}: {row.quantity_kg} kg<br>
            {labels['settlements']}: {row.number_of_douars}<br>
            {labels['assigned_truck']}: {row.truck_label}
        </div>
        """

        folium.Marker(
            location=location,
            icon=folium.DivIcon(
                icon_size=(26, 26),
                icon_anchor=(13, 13),
                html=f"""<div style="width:26px;height:26px;
                    border:2px solid white;border-radius:50%;
                    background:{stop_color};color:white;
                    text-align:center;line-height:22px;
                    font-size:13px;font-weight:bold;
                    box-shadow:0 1px 4px #0006;">{stop_number}</div>"""
            ),
            tooltip=f"{stop_number}. {escape(str(row.name))}",
            popup=folium.Popup(popup, max_width=300)
        ).add_to(collection_layer)

        bounds.append(location)

    collection_layer.add_to(m)

    # ----------------------------------------------
    # Departure and destination
    # ----------------------------------------------

    endpoint_layer = folium.FeatureGroup(
        name=labels["endpoints"]
    )

    same_location = (
        departure["road_node"] ==
        destination["road_node"]
    )

    endpoints = [
        (departure, labels["departure"], "green", "play"),
        (destination, labels["destination"], "red", "flag")
    ]

    if same_location:

        endpoints = [
            (
                departure,
                f"{labels['departure']} / {labels['destination']}",
                "darkred",
                "home"
            )
        ]

    for point, label, color, icon in endpoints:

        location = [
            point["road_latitude"],
            point["road_longitude"]
        ]

        folium.Marker(
            location=location,
            tooltip=label,
            popup=f"{label}: {escape(str(point['name']))}",
            icon=folium.Icon(
                color=color,
                icon=icon,
                prefix="fa"
            )
        ).add_to(endpoint_layer)

        bounds.append(location)

    endpoint_layer.add_to(m)

    # ----------------------------------------------
    # Map legend
    # ----------------------------------------------

    legend = f"""
    <div dir="{direction}" style="
        position: fixed;
        bottom: 40px;
        left: 15px;
        max-height: 180px;
        overflow-y: auto;
        z-index: 9999;
        background: white;
        padding: 10px 12px;
        border-radius: 8px;
        border: 1px solid #CBD5E1;
        color: #1E293B;
        font-size: 12px;
        line-height: 1.5;
        box-shadow: 0 2px 8px #00000022;
    ">
        <b>{labels['legend']}</b><br>

        <span style="color:#DC2626">●</span>
        {labels['high']}<br>

        <span style="color:#F59E0B">●</span>
        {labels['medium']}<br>

        <span style="color:#22C55E">●</span>
        {labels['low']}<br>

        <span style="background:#475569;color:white;border-radius:50%;padding:1px 5px">1</span>
        {labels['collection_points']}<br>
    """

    for index, route in enumerate(routes):

        color = ROUTE_COLORS[
            index % len(ROUTE_COLORS)
        ]

        legend += f"""
        <span style="color:{color}">━━</span>
        {labels['truck']} {route.get('truck_label', route['truck_id'])}<br>
        """

    legend += "</div>"

    m.get_root().html.add_child(
        folium.Element(legend)
    )

    # ----------------------------------------------
    # Layer controls and automatic zoom
    # ----------------------------------------------

    folium.LayerControl(
        collapsed=True
    ).add_to(m)

    if bounds:

        m.fit_bounds(
            bounds,
            padding=(30, 30)
        )

    return m
