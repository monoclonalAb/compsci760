import pandas as pd
import numpy as np
import torch
from math import radians, cos, sin, asin, sqrt

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculates the distance between two points in km using latitude/longitude coordinates.
    """
    # Convert latitude and longitude from degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371  # Radius of Earth in kilometers
    return c * r


def compute_temporal_distances(route_data):
    """
    Estimate when birds arrive at each stopover/node along migration route by allocating time proportionally to distance travelled.
    """
    route_data = route_data.copy()
    distances = []
    cumulative_distances = [0]

    next_latitude = route_data['GPS_yy'].shift(-1)
    next_longitude = route_data['GPS_xx'].shift(-1)

    # We can compute the rough spatial features of the route:
    route_data['distance_to_next'] = haversine(
        route_data['GPS_xx'], route_data['GPS_yy'],
        next_longitude, next_latitude)
    route_data['cumulative_distance'] = route_data['distance_to_next'].cumsum().fillna(0)
    route_data['route_progress'] = route_data['cumulative_distance'] / route_data['cumulative_distance'].iloc[-1]

    # Temporal feature estimation: ???
    
    return route_data
    
def preprocess(file_path, minimum_nodes = 5, migration_route = 1):
    """
    Preprocess the bird migration dataset.
    
    Args:
        file_path (str): Path to the Excel file containing the dataset.
        minimum_nodes (int): Minimum number of nodes required for a migratory route to be retained.
        migration_route (int): The migration route to filter by (default is 1).

    Returns:
        ??
    """
    print("Preprocessing data...")
    df = pd.read_excel(file_path)
    print(f"Original dataset: {len(df)} entries.")

    # 1. Filter by migration route
    df.rename(columns={df.columns[21]: "Migration routes"}, inplace=True)
    df = df[df['Migration routes'].isin([migration_route])]
    print(f"After filtering by migration route {migration_route}: {len(df)} entries.")

    # 2. Removing sparse routes (ie. migratory routes with fewer than a minimum number of nodes / stopovers)
    route_counts = df.groupby('Migratory route codes').size()
    valid_routes = route_counts[route_counts >= minimum_nodes].index
    df = df[df['Migratory route codes'].isin(valid_routes)]
    print(f"After removing sparse routes (minimum {minimum_nodes} nodes): {len(df)} entries.")

    # 3. Processing migration route features
    processed_routes = []
    for route in df['Migratory route codes'].unique(): # Iterate over each unique migratory route code
        route_data = df[df['Migratory route codes'] == route].copy()
        route_data = compute_temporal_distances(route_data)
        processed_routes.append(route_data)

    
    return df

if __name__ == "__main__":
    data = preprocess('data/Bird migration dataset.xls')
    print(data)