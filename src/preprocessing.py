from math import asin, cos, radians, sin, sqrt

import numpy as np
import pandas as pd
import torch

INPUT_FILENAME = "data/Bird migration dataset.xls"


def haversine(lon1, lat1, lon2, lat2):
    """
    Calculates the distance between two points in km using latitude/longitude coordinates.

    Args:
        lon1, lat1: Longitude and latitude of node 1 in the trajectory
        lon2, lat2: Longitude and latitude of node 2 in the trajectory

    Returns:
        c * r: Distance in kilometers between the two nodes
    """
    # Convert latitude and longitude from degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371  # Radius of Earth in kilometers
    return c * r


def compute_spatial_temporal_features(route_data):
    """
    Estimate when birds arrive at each stopover/node along migration route by allocating time proportionally to distance travelled.

    Args:
        route_data (pd.DataFrame): DataFrame containing the migration route data

    Returns:
        route_data (pd.DataFrame): DataFrame with added spatial and temporal features...
            distance_next_node_km: distance between current and next node in trajectory,
            cumulative_distance: ^^ accumulated,
            route_progress: percentage of trajectory completed,
            duration_h: estimated time spent travelling to the next node (hours),
            next_longitude: longitude of the next node,
            next_latitude: latitude of the next node
    """
    route_data = route_data.copy()
    route_data = route_data.sort_values("ID").reset_index(drop=True)

    next_latitude = route_data["GPS_yy"].shift(-1)
    next_longitude = route_data["GPS_xx"].shift(-1)

    # We can compute the rough spatial features of the route:
    route_data["distance_next_node_km"] = haversine(
        route_data["GPS_xx"], route_data["GPS_yy"], next_longitude, next_latitude
    )
    route_data["cumulative_distance"] = (
        route_data["distance_next_node_km"].shift(1, fill_value=0).cumsum()
    )
    route_data["route_progress"] = (
        route_data["cumulative_distance"] / route_data["cumulative_distance"].iloc[-1]
    )

    # Temporal feature estimation (allocating time proportionally to distance travelled)
    start_month = route_data.iloc[0]["Migration start month"]
    end_month = route_data.iloc[0]["Migration end month"]
    duration_months = end_month - start_month
    if duration_months < 0:
        duration_months += 12
    if duration_months == 0:
        duration_months = 1  # Assume at least 1 month if start and end are the same
    route_data["duration_h"] = (
        (
            route_data["distance_next_node_km"]
            / route_data["distance_next_node_km"].sum()
        )
        * duration_months
        * 30
        * 24
    )  # Convert months to hours, assumes 30 days/month

    route_data["next_longitude"] = route_data["GPS_xx"].shift(-1)
    route_data["next_latitude"] = route_data["GPS_yy"].shift(-1)

    return route_data


def preprocess(file_path, minimum_nodes=5, migration_route=1):
    """
    Preprocess the bird migration dataset.

    Args:
        file_path (str): Path to the Excel file containing the dataset.
        minimum_nodes (int): Minimum number of nodes required for a migratory route to be retained.
        migration_route (int): The migration route to filter by (default is 1).

    Returns:
        processed_trajectories: A list of DataFrames where each DataFrame corresponds to a processed migratory route code
    """
    print("Preprocessing data...")
    df = pd.read_excel(file_path)
    print(f"Original dataset: {len(df)} entries.")

    # 1. Filter by migration route
    df.rename(columns={df.columns[22]: "Migration routes"}, inplace=True)
    df = df[df["Migration routes"].isin([migration_route])]
    print(f"After filtering by migration route {migration_route}: {len(df)} entries.")

    # 2. Removing sparse routes (ie. migratory routes with fewer than a minimum number of nodes / stopovers)
    route_counts = df.groupby("Migratory route codes").size()
    valid_routes = route_counts[route_counts >= minimum_nodes].index
    df = df[df["Migratory route codes"].isin(valid_routes)]
    print(
        f"After removing sparse routes (minimum {minimum_nodes} nodes): {len(df)} entries."
    )

    # 3. Processing migration route features
    processed_trajectories = []
    for route_code in df[
        "Migratory route codes"
    ].unique():  # Iterate over each unique migratory route code
        route_data = df[df["Migratory route codes"] == route_code].copy()
        route_data = compute_spatial_temporal_features(route_data)
        processed_trajectories.append(route_data)

    print(f"Processed {len(processed_trajectories)} trajectories.")

    return processed_trajectories


if __name__ == "__main__":
    for i in range(1, 22):
        processed_trajectories = preprocess(INPUT_FILENAME, 5, i)
        all_data = pd.concat(processed_trajectories, ignore_index=True)

        # Write to Excel
        all_data.to_excel(f"data/all/processed_bird_migration_{i}.xlsx", index=False)
        print(f"Saved processed data to processed_bird_migration_{i}.xlsx")
