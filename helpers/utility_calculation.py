import ast
import numpy as np
import pandas as pd
from typing import Tuple
from itertools import product

def calculate_feasible_combinations(
        od_travel_data: pd.DataFrame,
        beta_C_k:float=-1, 
        beta_V_k:float=-0.29,  # Vehicle time sensitivity
        beta_P_k:float=-0.29,  # Public transit time sensitivity
        beta_A_k:float=-0.72,  # Access time sensitivity
        beta_W_k:float=-0.72,  # Walk time sensitivity
        beta_B_k:float=-0.39,  # Bicycle time sensitivity
        P_CS:float=0.5, 
        L:list=[0, 1, 2, 5],  
        M:list=['Walk', 'Bike', 'PublicTransit'], 
        access_time_range:list=[1, 3], 
        walking_speed:float=5, 
        biking_speed:float=15, 
        driving_speed:float=45, 
        biking_cost_per_km:float=0.05,
        mode_choice_model:str='MNL',
        use_random_access_time:bool=False,
        verbose:bool=False) -> Tuple[pd.DataFrame, dict]:
    """
    Calculates feasible combinations given a set of parameters
    
    Parameters:
    beta_C_k (float): Coefficient for cost sensitivity.
    beta_V_k (float): Coefficient for vehicles time sensitivity.
    beta_A_k (float): Coefficient for access time sensitivity.
    beta_W_k (float): Coefficient for walking preference.
    beta_B_k (float): Coefficient for biking preference.
    beta_P_k (float): Coefficient for public transit preference.
    P_CS (float): Per-minute cost of car-sharing service.
    L (list): Fee-levels for car-sharing service.
    M (list): Modes of transportation considered ('Walk', 'Bike', 'PublicTransit').
    access_time_range (list): Minimum and maximum access time.
    walking_speed (float): Average walking speed in km/h.
    biking_speed (float): Average biking speed in km/h.
    driving_speed (float): Average driving speed in km/h.
    biking_cost_per_km (float): Cost of biking per kilometer in currency units.
    mode_choice_model (str): Mode choice model to use ('max' or 'MNL').
    use_random_access_time (bool): Whether to use random access time.
    verbose (bool): Whether to print debug information.
    Returns:
    Feasible combinations based on the provided parameters.
    """
    import ast
    import numpy as np
    import pandas as pd
    from typing import Tuple

    def T_A_bike_k():
        return np.random.uniform(access_time_range[0], access_time_range[1])

    def travel_time(distance_in_meters, travel_speed_kmh) -> float:
        """Distance in meters, travel speed in km/h -> time in minutes"""
        time = (( distance_in_meters / 1000 ) / travel_speed_kmh ) * 60
        return time

    # Utility of alternative transportation methods
    def utility_given_mode(mode: str, mode_data: Tuple, verbose=False) -> float:
        """Returns utility for given mode given a customer request
        input:
        - mode: str, mode of transport
        - mode_data: relevant for mode
        """
        if mode == "Walk":
            walking_distance_value = mode_data
            walking_time = travel_time(walking_distance_value, walking_speed)
            # Calculate utilities
            utility_walking_time_sensitivity = beta_W_k * walking_time
            return utility_walking_time_sensitivity
        elif mode == "Bike":
            walking_distance_value = mode_data
            biking_time = travel_time(walking_distance_value, biking_speed) 
            biking_cost =  biking_cost_per_km * walking_distance_value * 10**(-3) # convert to km
            # Calculate utilities
            access_time_1 = T_A_bike_k() if use_random_access_time else 1
            access_time_2 = T_A_bike_k() if use_random_access_time else 1
            print(f"bike access times {(access_time_1, access_time_2)}") if verbose else None
            utility_bicycle_access_time_sensitivity = beta_A_k * (access_time_1 + access_time_2)
            utility_bicycle_cost_sensitivity = beta_C_k * biking_cost
            utility_bicycle_time_sensitivity = beta_B_k * biking_time
            print(f"bike cost time access_time: {(utility_bicycle_cost_sensitivity, utility_bicycle_time_sensitivity, utility_bicycle_access_time_sensitivity)}") if verbose else None
            return utility_bicycle_time_sensitivity + utility_bicycle_cost_sensitivity + utility_bicycle_access_time_sensitivity
        elif mode == "PublicTransit":
            public_travel_price, public_travel_time, public_travel_access_time, public_travel_transfer_time = mode_data
            utility_cost_sensitivity = beta_C_k * public_travel_price
            utility_time_sensitivity = beta_P_k * public_travel_time
            utility_access_time_sensitivity = beta_A_k * public_travel_access_time + public_travel_transfer_time
            print(f"pt cost time access_time: {(utility_cost_sensitivity, utility_time_sensitivity, utility_access_time_sensitivity)}") if verbose else None
            return utility_cost_sensitivity + utility_time_sensitivity + utility_access_time_sensitivity
        else:
            raise ValueError("Invalid mode")

    def calculate_utility_for_modes(alternative_trans_data, verbose = False):
        """Returns the utility for each mode given a customer request"""
        walking_distance_value = alternative_trans_data['walking_distance_value']
        public_travel_price = alternative_trans_data['public_travel_price']
        public_travel_time = alternative_trans_data['public_travel_time']
        public_travel_access_time = alternative_trans_data['public_travel_access_time']
        public_travel_transfer_time = alternative_trans_data['public_travel_transfer_time']
        print(f"alternative transport distance {walking_distance_value}") if verbose else None
        U_bar_mk = {} # Dict to store utility for each mode
        for mode in M:
            if mode == 'PublicTransit':
                U_bar_mk[mode] = utility_given_mode(mode, (public_travel_price, public_travel_time, public_travel_access_time, public_travel_transfer_time), verbose)
            elif mode == 'Bike':
                U_bar_mk[mode] = utility_given_mode(mode, (walking_distance_value), verbose)
            elif mode == 'Walk':
                U_bar_mk[mode] = utility_given_mode(mode, (walking_distance_value), verbose)
            print(f"mode {mode}, utility {U_bar_mk.get(mode, None)}") if verbose else None
        return U_bar_mk

    def utility_kij(travel_data: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """ Calculate utility for car-sharing service given travel data"""
        # Extract travel data
        walking_dist_from_origin_to_station, request_driving_dist, walking_dist_from_station_to_destination = travel_data
        # Calculate and return utility
        utility_cost_sensitivity = beta_C_k * P_CS * travel_time(request_driving_dist, driving_speed)
        utility_time_sensitivity = beta_V_k * travel_time(request_driving_dist, driving_speed)
        utility_access_time_sensitivity = beta_A_k * (travel_time(walking_dist_from_origin_to_station, walking_speed) + travel_time(walking_dist_from_station_to_destination, walking_speed))
        return (utility_access_time_sensitivity, utility_time_sensitivity, utility_cost_sensitivity)

    def utility_kijl(utility_kij: Tuple[float, float, float], _pricing_level: float) -> float:
        """ Calculate utility for car-sharing service given travel data 
        input: Tuple with utility_access_time_sensitivity, utility_time_sensitivity, utility_cost_sensitivity"""
        # Extract travel data
        utility_access_time_sensitivity, utility_time_sensitivity, utility_cost_sensitivity = utility_kij
        # Calculate and return utility
        utility_cost_sensitivity += beta_C_k * (_pricing_level)
        return sum((utility_access_time_sensitivity, utility_time_sensitivity, utility_cost_sensitivity))

    def maximum_utility_for_modes(U_bar_mk, verbose = False):
        """Returns the best mode and utility for a given customer request"""
        print(U_bar_mk) if verbose else None
        best_mode = max(U_bar_mk, key=U_bar_mk.get)
        return best_mode, U_bar_mk[best_mode]

    def calculate_maximum_utilities_with_carsharing(
        cs_travel_data: dict, i_stations, j_stations, verbose = False):
        u_kijl = {}
        print(f"cs_travel_data: {cs_travel_data}") if verbose else None
        for (i_idx, j_idx), travel_info in cs_travel_data.items():
            walk1_val = travel_info[0]
            drive_val = travel_info[1]
            walk2_val = travel_info[2]
            base_utility = utility_kij((walk1_val, drive_val, walk2_val))
            # find best utility
            for l in L:
                this_utility = utility_kijl(base_utility, l)
                if l not in u_kijl or this_utility > u_kijl[l][0]:
                    u_kijl[l] = [
                        this_utility,
                        i_stations[i_idx][1],
                        j_stations[j_idx][1]
                    ]
        print(f"u_kijl: {u_kijl}") if verbose else None
        return u_kijl

    def calculate_MNL_probabilities(utility_dict):
        exp_utility_dict = np.exp(list(utility_dict.values()))
        sum_exp_utility_dict = np.sum(exp_utility_dict)
        probabilities = {
            mode: exp_u / sum_exp_utility_dict
            for mode, exp_u in zip(utility_dict.keys(), exp_utility_dict)}
        return probabilities

    def single_user_choice_MNL(u_kijl, U_bar_mk, m_star, verbose=False):
        """ Calculate feasible pricing levels using MNL model. We test at drop-off fee equal 0. If carsharing is chosen, the user will continue to choose carsharing unless the best alternative is better"""
        # Calculate probabilities
        comparison_dict = {mode: U_bar_mk[mode] for mode in M}
        #comparison_dict.pop(min(comparison_dict, key=comparison_dict.get)) # drop lowest key (walk)
        comparison_dict.update({"Carsharing": u_kijl[L[L.index(0)]][0]}) # add carsharing
        print(f"comparison dict: {comparison_dict}") if verbose else None
        probabilities = calculate_MNL_probabilities(comparison_dict)
        print(f"probabilities: {probabilities}") if verbose else None
        initial_choice = np.random.choice(list(probabilities.keys()), p=list(probabilities.values()))        
        print(f"initial choice: {initial_choice}") if verbose else None
        L_kil = (
            {0: u_kijl[L[L.index(0)]]} | {l: u_kijl[l] for l in L if u_kijl[l][0] > m_star[1]}
            if initial_choice == "Carsharing"
            else {})
        return L_kil

    # logical loop for each possible user
    def single_user_choice_simulation(request_row, verbose = False):
        alternative_trans = request_row.alternative_transportation_data
        i_stations = request_row.i_stations
        j_stations = request_row.j_stations
        cs_travel_data = request_row.cs_travel_data
        # Calculate utilities for cs and modes
        u_kijl = calculate_maximum_utilities_with_carsharing(cs_travel_data, i_stations, j_stations, verbose) # utility and station at each level
        U_bar_mk = calculate_utility_for_modes(alternative_trans, verbose) # utility for modes
        m_star = maximum_utility_for_modes(U_bar_mk, verbose) # best alternative
        if mode_choice_model == 'max':
            # For simple maximization, calculate feasibility at each pricing level
            L_kil = {l: u_kijl[l] for l in L if u_kijl[l][0] > m_star[1]}
        elif mode_choice_model == 'MNL':
            # for MNL, calculate choice probabilities
            L_kil = single_user_choice_MNL(u_kijl, U_bar_mk, m_star, verbose)
        else:
            raise ValueError("Invalid mode choice model")
        print(f"L_kil {L_kil}") if verbose else None
        return L_kil, m_star

    def preprocess_requests(requests, verbose = False):
        B_r = {} # Set of all feasible combinations for all requests
        m_star = {} # Set of all maximum alternative utilities for all requests
        # Preprocess columns for requests
        for r_idx, row in enumerate(requests.itertuples()):
            B_r_k, m_star_k = single_user_choice_simulation(row, verbose)
            B_r[r_idx] = B_r_k
            m_star_k_dict = {
                'origin': row.origin_node,
                'destination': row.destination_node,
                'best_alternative_mode': m_star_k[0],
                'best_alternative_utility': m_star_k[1]
            }
            m_star[r_idx] = m_star_k_dict
        return B_r, m_star

    def format_Br_df(res_B_r):
        # Create dataframe with feasible combinations
        records = []
        for outer_key, inner_dict in res_B_r.items():
            for fee, details in inner_dict.items():
                # details[0] is the numeric value, details[1] and details[2] are coordinate tuples
                utility = details[0]
                coord1_lat, coord1_lon = details[1]
                coord2_lat, coord2_lon = details[2]
                records.append({
                    'id': outer_key,
                    'fee': fee,
                    'utility': utility,
                    'station_start_lat': coord1_lat,
                    'station_start_lon': coord1_lon,
                    'station_end_lat': coord2_lat,
                    'station_end_lon': coord2_lon
                })

        feasible_combinations = pd.DataFrame(records)
        return feasible_combinations

    # Main logic
    B_r, m_star = preprocess_requests(od_travel_data, verbose)
    feasible_combinations = format_Br_df(B_r)
    return feasible_combinations, m_star

def preprocess_requests(requests: pd.DataFrame) -> pd.DataFrame:
    def safe_literal_eval(x):
        if isinstance(x, str):
            return ast.literal_eval(x)
        return x
    eval_columns = [
        'alternative_transportation_data',
        'i_stations',
        'j_stations',
        'cs_travel_data']
    requests[eval_columns] = requests[eval_columns].map(safe_literal_eval)
    return requests

if __name__ in "__main__":
    # Test the function
    raw_od_data = pd.read_csv("requests/od_travel_data_revised.csv", sep=';', encoding='utf-8')
    od_travel_data = preprocess_requests(raw_od_data)
    od_travel_data = od_travel_data.sample(500)
    feasible_combinations, m_star = calculate_feasible_combinations(od_travel_data, 
        beta_C_k=-1,
        beta_V_k=-0.25,
        beta_P_k=-0.30,
        beta_A_k=-0.35, 
        beta_W_k=-0.6, 
        beta_B_k=-0.4,
        verbose=False)
    
    print(len(feasible_combinations['id'].unique()))