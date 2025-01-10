import os
import time
import json
from typing import Tuple
import pandas as pd #type: ignore
import numpy as np #type: ignore
from gurobipy import Model, GRB, quicksum #type: ignore

def model_run(
        requests: pd.DataFrame,
        vehicles: list,
        initial_positions: dict,
        distance_duration_lookup: dict,
        stations: list,
        P_CS: float = 0.3, 
        C_V: float = 0.2, 
        C_S: float = 0.5,
        L: list = [-2, -1, 0, 1, 2, 5],
        verbose: bool = False) -> Tuple[dict, pd.DataFrame, pd.DataFrame, dict]:
    """Run the optimization model
    
    Parameters:
    parameters (dict): A dictionary containing the following
        requests (pd.DataFrame): A DataFrame containing the requests
        vehicles (list): A list of vehicle IDs
        initial_positions (dict): A dictionary containing the initial positions of the vehicles
        P_CS (float): The price per minute
        C_V (float): The cost per km
        C_S (float): The relocation cost per minute
        L (list): A list of pricing levels

    Returns:
    Tuple: A tuple containing the following
        financials (dict): A dictionary containing the financials
        vehicle_df (pd.DataFrame): A DataFrame containing vehicle data
        requests_served_df (pd.DataFrame): A DataFrame containing requests served
        performance_metrics (dict): A dictionary containing the performance metrics
    """
    ############################
    # Model initialization
    ############################
    start_time = time.time()
    model = Model("Carsharing_Profit_Maximization")
    model.setParam('OutputFlag', 0)
    model.reset()
    # Set Gurobi parameters
    # https://docs.gurobi.com/projects/optimizer/en/current/concepts/parameters/guidelines.html#secparameterguidelines
    model.setParam('MIPFocus', 1)
    model.setParam('Threads', 1)
    ############################
    # Decision variables
    ############################ 
    # Request fulfillment decision
    y = model.addVars(requests['index'], stations, L, vehicles, vtype=GRB.BINARY, name="y")
    # Pricing decision variables
    alpha = model.addVars(stations, stations, L, vtype=GRB.BINARY, name="alpha")
    # Vehicle assignment variables
    s = model.addVars(vehicles, stations, vtype=GRB.BINARY, name="s")
    ############################
    # Preprocessing
    ############################   
    # Initialize dictionaries
    B_r = {}     # Feasible (station, level) pairs for each request
    R_ri = {}    # Earlier requests for each request-station pair
    I_r = {}      # Set of requests stations that are within walking distance of request r
    R_ri = {}    # Earlier requests for each request-station pair
    # Define B_r: Feasible (station, level) pairs for each request
    for r in requests['index']:
        applicable_fees = requests.at[r, 'fee']
        origin_stations = requests.at[r, 'start_stations']
        B_r[r] = list(zip(origin_stations, applicable_fees))
    # Define R_ri: Earlier requests for each request-station pair
    for r in requests['index']:
        origin_stations = set(requests.at[r, 'start_stations'])
        for origin_station in origin_stations:  # Find earlier requests at the same origin station
            R_ri[(r, origin_station)] = [
                r_0 for r_0 in requests['index'] if r_0 < r 
                for station in set(requests.at[r_0, 'start_stations']) 
                if station == origin_station and r_0 < r
            ]
    # Define I_r: Set of requests stations that are within walking distance of request r
    for r in requests['index']:
        origin_stations = set(requests.at[r, 'start_stations'])
        I_r[r] = [
            i for i in stations for origin_station in origin_stations
            if distance_duration_lookup[(origin_station, i)][0] <= 1
            ]
    ############################
    # Model objective function
    ############################  
    # Revenue per Request (RN_{ril})
    RN_ril = quicksum(                                                                                 
        y[r, i, l, v] * (                                                                               
            P_CS * distance_duration_lookup[(i, j)][1]      # price per min times duration
            + l -                                           # plus the level                             
            C_V * distance_duration_lookup[(i, j)][0]       # minus the cost per km times distance
        ) 
        for v in vehicles
        for r in requests['index']
        for (i, j, l) in zip(
            requests.at[r, 'start_stations'], # Origin stations
            requests.at[r, 'end_stations'],   # Destination stations
            requests.at[r, 'fee']  # Pricing levels
        )
        if (i, l) in B_r[r]
    )
    # Relocation Cost (C_{vi})
    C_vi = quicksum(
        s[v, i] * (                                                                              
            C_V * distance_duration_lookup[(initial_positions[v], i)][0]     # cost per km times distance
            + C_S * distance_duration_lookup[(initial_positions[v], i)][1])  # plus cost per minute times duration 
        for v in vehicles 
        for i in stations
    )
    fR = RN_ril - C_vi
    model.setObjective(fR, GRB.MAXIMIZE)
    ############################
    # Constraints
    ############################ 
    # Constraint 1: Each vehicle must be located (i.e., either relocated or unmoved) at a certain station i
    for v in vehicles:
        model.addConstr(
            quicksum(s[v, i] for i in stations) == 1,
            name=f"Constraint1b_{v}"
        )

    # Constraint 2: Exactly one pricing level to each pair (i, j)
    for i in stations:
        for j in stations:
            if i != j:
                model.addConstr(
                    quicksum(alpha[i, j, l] for l in L) == 1,
                    name=f"Constraint1c_{i}_{j}")

    # Constraint 3: each shared vehicle may serve at most one request
    for v in vehicles:
        model.addConstr(
            quicksum(y[r, i, l, v] 
                for r, row in requests.iterrows()
                for i, l in zip(row['start_stations'], row['fee'])
                if (i, l) in B_r[r]
            ) <= 1,
            name=f"Constraint_4b_{v}"
        )
        
    # Constraint 4: Each request is satisfied at most by one origin station i and a corresponding pricing level l
    for r, row in requests.iterrows():
        model.addConstr(
            quicksum(y[r, i, l, v] 
                for v in vehicles
                for i, l in zip(row['start_stations'], row['fee'])
                if (i, l) in B_r[r]
            ) <= 1,
            name=f"Constraint_4b_{r}")
        
    # Constraint 5: request r can only be served at origin station at l, when the price is offered for all requests
    for r, row in requests.iterrows():
        for i, j, l in zip(row['start_stations'], row['end_stations'], row['fee']):
            if (i, l) in B_r[r]:
                model.addConstr(
                    quicksum(y[r, i, l, v] for v in vehicles) <= alpha[i, j, l],
                    name=f"Constraint_4c_{r}_{i}_{l}"
                )

    # Constraint 6: shared vehicle v can only be used to serve request r at station i ∈ Ir only if this vehicle is made available at station i and has not served any other requests r0 that come earlier than r at station i
    for r, row in requests.iterrows():
        for origin_station in row['start_stations']:
            for v in vehicles:
                for i in I_r[r]:
                    model.addConstr(
                        quicksum(
                            y[r, _i, _l, v] 
                            for _i, _l in zip(requests.at[r, 'start_stations'], requests.at[r, 'fee'])
                            if (_i, _l) in B_r[r]
                        )
                        + quicksum(
                            quicksum(
                                y[r0, _i, _l, v] 
                                for _i, _l in zip(requests.at[r0, 'start_stations'], requests.at[r0, 'fee'])
                                if (_i, _l) in B_r[r0]
                            )
                            for r0 in R_ri[(r, origin_station)]
                        )
                        <= s[v, i],  # Availability constraint
                        name=f"Constraint_4d_{r}_{v}_{i}"
                    )

    # Constraint 7: each request r at must be satisfied by the available vehicle v if level l has been offered to pair (i, j(r)), unless the request r has been satisfied by another vehicle v1 or vehicle v has been assigned to another precedent request r0 ∈Rri.
    for r, row in requests.iterrows():
        for v in vehicles: 
            if v in I_r[r]:
                for i, l in zip(requests.at[r, 'start_stations'], row['fee']):  # Use start stations and fees
                    for j in requests.at[r, 'end_stations']:
                        if (i, l) in B_r[r]:
                            model.addConstr(
                                y[r, i, l, v] 
                                + quicksum(
                                    y[r, i, l, v1] 
                                    for v1 in vehicles if v1 != v
                                )
                                + quicksum(
                                    y[r0, i, l, v] 
                                    for r0 in R_ri[(r, j)] if (i, l) in B_r[r0]
                                )
                                >= s[v, i] + alpha[i, j, l] - 1,
                                name=f"Constraint_4e_{r}_{v}_{i}_{l}"
                            )
    
    model.update()  
    model_initiation_time = time.time() - start_time
    ############################
    # Model optimization
    ############################
    start_time = time.time()
    model.optimize()

    if model.status != GRB.OPTIMAL:
        print("Model is infeasible")
        model.computeIIS()
    if model.status == GRB.OPTIMAL:
        print('Optimal objective value:', model.ObjVal) if verbose else None
    
    model_processing_time = time.time() - start_time
    ############################
    # Results verification
    ############################ 
    start_time = time.time()
    # Check for requests that are served by multiple combinations
    for r in requests['index']:
        served_combinations = [
            (i, l, v) for v in vehicles for i in stations for l in L
            if y[r, i, l, v].x > 0.5
        ]
        if len(served_combinations) > 1:
            raise Exception(f"Request {r} is served by multiple combinations: {served_combinations}")

    # Make sure each request is served by at most one combination
    for r in requests['index']:
        total_y = sum(y[r, i, l, v].x for v in vehicles for (i, l) in B_r[r])
        # print(f"Total y for request {r}: {total_y}")
        if total_y > 1.01:  # Allowing a small tolerance
            raise Exception(f"Request {r} is served by multiple combinations: {total_y}")

    # Check for fractional variable values
    fractional_vars = []
    for r in requests['index']:
        for v in vehicles:
            for (i, l) in B_r[r]:
                val = y[r, i, l, v].x
                if 0 < val < 1:
                    fractional_vars.append((r, i, l, v, val))
    if fractional_vars:
        print("FAULT: Fractional variable values detected:")
        for var in fractional_vars:
            print(f"y[{var[0]}, {var[1]}, {var[2]}, {var[3]}] = {var[4]}")
        raise Exception("Fractional variable values detected")
    ############################
    # Results extraction
    ############################ 
    # Create list with requests served
    processed_requests = set()
    requests_served = []
    for r, row in requests.iterrows():
        for v in vehicles:
            for i, l in zip(row['start_stations'], row['fee']):
                if (i, l) in B_r[r]:
                    for j in row['end_stations']:
                        if (i, j) in distance_duration_lookup:
                            if (r, v, i, j, l) not in processed_requests:
                                if y[r, i, l, v].x > 0.5:
                                    processed_requests.add((r, v, i, j, l))  # Mark as processed
                                    requests_served.append((r, v, i, j, l))  # Add request details
    # Get relocations from served requests
    costs_from_relocations = 0
    for r_served in requests_served:
        r, v, r_css_o, r_css_d, l = r_served
        if initial_positions[v] != r_css_o:
            relocation_time_cost = distance_duration_lookup[(initial_positions[v], r_css_o)][1] * C_S
            relocation_distance_cost = distance_duration_lookup[(initial_positions[v], r_css_o)][0] * C_V
            earnings_from_relocation = P_CS * distance_duration_lookup[(r_css_o, r_css_d)][1] + l - C_V * distance_duration_lookup[(r_css_o, r_css_d)][0]
            costs_from_relocations += relocation_time_cost + relocation_distance_cost
            
            print(f"{v} is relocated from {initial_positions[v][:12]} to {r_css_o[:12]}, "
                f"costing {round(relocation_time_cost + relocation_distance_cost, 2)}, "
                f"earning {round(earnings_from_relocation, 2)}, "
                f"i.e. netting {round(earnings_from_relocation - (relocation_time_cost + relocation_distance_cost), 2)}") if verbose else None

    print(f"Total relocation costs: {costs_from_relocations}") if verbose else None
    ############################
    # Result Analysis
    ############################      
    # calculate the revenue from the requests served
    revenue_dd = 0 # income from time minus costs from distance
    income_from_fees = 0 # income from fees
    cnt = 0
    for r_served_items in requests_served:
        r_served, v, r_css_o, r_css_d, l = r_served_items  # Unpack served request details
        cnt += 1
        distance, duration = distance_duration_lookup[(r_css_o, r_css_d)]
        revenue_dd += P_CS * duration - C_V * distance
        income_from_fees += l

    objVal = model.ObjVal
    expected_val = revenue_dd + income_from_fees - costs_from_relocations
    tolerance = 1e-1

    print(f"Income from r_dd: {revenue_dd}") if verbose else None
    print(f"Income from fees: {income_from_fees}") if verbose else None
    print(f"objective value: {objVal}, total revenue: {round(revenue_dd + income_from_fees)}, aggregate costs: {costs_from_relocations}") if verbose else None
    if abs(objVal - expected_val) < tolerance:
        print("Success, numbers match (within rounding tolerance)!") if verbose else None
    else:
        # Print the difference for debugging
        diff = objVal - expected_val
        print(f"Difference is {diff:.6f} (tolerance: {tolerance})")
        raise Exception(
            f"Objective value ({objVal}) does not match the expected value ({expected_val}) "
            f"within tolerance {tolerance}"
        )
    
    vehicle_data = []
    for v in vehicles:
        # Find assigned position
        assigned_positions = [i for i in stations if s[v, i].x > 0.5]
        if len(assigned_positions) != 1:
            print(f"Vehicle {v} assigned to multiple or no positions") if verbose else None
            continue
        assigned_position = assigned_positions[0]
        initial_position = initial_positions[v]
        moved = assigned_position != initial_position

        # Check if vehicle was used
        used = any(y[r, i, l, v].x > 0.5 for r in requests['index'] for i in stations for l in L)

        vehicle_data.append({
            'vehicle': v,
            'initial_position': initial_position,
            'assigned_position': assigned_position,
            'moved': moved,
            'used': used
        })
    ############################
    # Outputs
    ############################ 
    financials = {
        'objective_value': objVal,
        'income_from_time_fee': revenue_dd,
        'income_from_fee_level': income_from_fees,
        'aggregate_costs': costs_from_relocations,
    }

    vehicle_df = pd.DataFrame(vehicle_data)

    requests_served_df = pd.DataFrame(requests_served, columns=['request', 'vehicle', 'origin', 'destination', 'fee'])
    
    model_postprocessing_time = time.time() - start_time

    performance_metrics = {
        'model_status': model.status,
        'model_NumConstrs': model.NumConstrs,
        'model_NumVars': model.NumVars,
        'model_initiation_time': model_initiation_time,
        'model_processing_time': model_processing_time,
        'model_postprocessing_time': model_postprocessing_time,
        'model_runtime': model.Runtime,
        'model_mip_gap': model.MIPGap,
        'model_obj_val': round(model.ObjVal, 3)
    }

    return (financials, vehicle_df, requests_served_df, json.dumps(performance_metrics))

def format_requests_df(requests, addresses):
    df = requests.copy()
    df = df.groupby('id').agg(list).reset_index()
    df['index'] = df.index

    def row_station_names(df_row_index: float, type: str) -> list:
        """Get the names of the start- or end stations for a request"""
        relevant_stations = []
        if type == 'start':
            col_lat = 'station_start_lat'
            col_lon = 'station_start_lon'
        elif type == 'end':
            col_lat = 'station_end_lat'
            col_lon = 'station_end_lon'
        else:
            raise ValueError("type must be either 'start' or 'end'")
        lats, lons = df.at[df_row_index, col_lat], df.at[df_row_index, col_lon]
        for lat, lon in zip(lats, lons):
            station_addresses = addresses[(addresses['lat'] == lat) & (addresses['lng'] == lon)]['css_title'].values
            relevant_stations.extend(station_addresses)
        return relevant_stations

    # create new columns for start and end stations
    df['start_stations'] = df['index'].apply(lambda x: row_station_names(x, 'start'))
    df['end_stations'] = df['index'].apply(lambda x: row_station_names(x, 'end'))
    return df


if __name__ in "__main__":
    import os, sys
    sys.path.append(
        os.path.abspath(
            os.path.join(
                os.path.dirname(__file__), '..') # add parent directory to path
                ))
    from helpers.utility_calculation import calculate_feasible_combinations
    # Read datasets
    preprocessed_data = pd.read_csv('./requests/od_travel_data.csv', sep=';', encoding='utf-8')
    addresses = pd.read_csv('data/20_css_cop_latlng.csv', sep=';', index_col=0)
    distance_matrix = pd.read_csv('data/20_css_distance_matrix_regenerate.csv', sep=';', index_col=0)
    # Extract data 
    stations = addresses['css_title'].unique()
    # Create a dictionary for distance and duration lookup
    distance_duration_lookup = {
        (row['origin_css'], row['destination_css']): (row['distance'], row['duration']) #? (distance, duration)
        for _, row in distance_matrix.iterrows()}

    for location in set(distance_matrix['origin_css']).union(distance_matrix['destination_css']):
        distance_duration_lookup[(location, location)] = (0, 0)

    num_requests = 2500
    num_vehicles = 20
    od_travel_data = preprocessed_data.sample(num_requests)
    vehicles = ['v{}'.format(i) for i in range(num_vehicles)]
    initial_positions = {vehicle: stations[i % len(stations)] for i, vehicle in enumerate(vehicles)}

    requests, m_star = calculate_feasible_combinations(od_travel_data, beta_V_k=-0.01, beta_P_k=-0.5)

    requests = format_requests_df(requests, addresses)

    results = model_run(requests, vehicles, initial_positions, distance_duration_lookup, stations, verbose=True)

    financials, vehicle_df, requests_served, performance_metrics = results