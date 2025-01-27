import os, sys
sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__), '..') # add parent directory to path
            ))

# packages
import ast
import time
import traceback
import pandas as pd
from tqdm import tqdm
from itertools import product
from concurrent.futures import ProcessPoolExecutor

# local imports
from helpers.optimization_model import model_run
from helpers.utility_calculation import calculate_feasible_combinations

# Pandas settings
pd.options.mode.chained_assignment = None  # Suppress SettingWithCopyWarning
pd.set_option('display.max_columns', None) # Show all columns

#### Define experiment parameters

### fleet specs
## will be varied.
num_vehicles_options  = [25]
P_CS_options = [0.2, 0.35, 0.5, 0.65, 0.8]          # Per-minute price of carsharing service

# held constant
L_options = [0, 1, 2]                               # Fee-levels, not varied between experiments.

## customer preferences
beta_C_k_options = [-1.20, -1.10, -1, -0.90, -0.80] # Cost sensitivity
beta_V_k_options = [-0.20]                          # Vehicle time sensitivity
beta_P_k_options = [-0.30]                          # Public transit time sensitivity
beta_B_k_options = [-0.5, -0.4, -0.3]             # Bike time sensitivity                          

# held constant
beta_A_k_options = [-0.3]                           # Access time sensitivity

### Repeat each experiment a number of times
repetitions_list = list(range(1, 6))

# Make sure that the output directory exists
os.makedirs('data', exist_ok=True)
use_max_workers = os.cpu_count() - 4

# Read datasets
od_travel_data_raw = pd.read_csv('./data/od_travel_data_revised.csv', sep=';', encoding='utf-8')
addresses = pd.read_csv('data/20_css_cop_latlng.csv', sep=';', index_col=0)
distance_matrix = pd.read_csv('data/20_css_distance_matrix_regenerate.csv', sep=';', index_col=0)

# Extract data 
stations = addresses['css_title'].unique() # 20 stations

# Create a dictionary for distance and duration lookup
distance_duration_lookup = {
    (row['origin_css'], row['destination_css']): (row['distance'], row['duration']) #? (distance, duration)
    for _, row in distance_matrix.iterrows()}

for location in set(distance_matrix['origin_css']).union(distance_matrix['destination_css']):
    distance_duration_lookup[(location, location)] = (0, 0)

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

od_travel_data = preprocess_requests(od_travel_data_raw)

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

def calculate_usage_metrics(financials, vehicle_df, requests_served, requests):
    # get the objective value
    objective_value = financials.get('objective_value')
    # get the aggregated vehicle relocation costs
    vehicle_relocation_costs = financials.get('aggregate_costs')
    # calculate utilization ratio
    utilization_ratio = vehicle_df['used'].sum() / len(vehicle_df)
    # calculate acceptance ratio
    acceptance_ratio = len(requests_served) / len(requests)
    # calculate vehicle utilization ratio
    def categorize_vehicle(row):
        if not row['moved'] and not row['used']:
            return 'Parked'
        elif row['moved'] and row['used']:
            return 'Moved and Used'
        elif not row['moved'] and row['used']:
            return 'Used'
        else:
            raise ValueError("Vehicle cannot be moved and not used")
    vehicle_df['category'] = vehicle_df.apply(categorize_vehicle, axis=1)
    category_counts = vehicle_df['category'].value_counts()
    category_percentages = (category_counts / len(vehicle_df)) * 100
    # Reorder the categories
    categories = ['Parked', 'Moved and Used', 'Used']
    vehicle_usage_category_distribution =  [category_percentages.get(cat, 0) for cat in categories]
    return objective_value, vehicle_relocation_costs, utilization_ratio, acceptance_ratio, vehicle_usage_category_distribution

def run_experiment(
        num_vehicles: int = 20,
        P_CS: float = 0.3, 
        C_V: float = 0.2, 
        C_S: float = 0.5, 
        L: list = [0, 1, 2],
        beta_C_k:float=-1,     # Cost sensitivity
        beta_V_k:float=-0.29,  # Vehicle time sensitivity
        beta_P_k:float=-0.29,  # Public transit time sensitivity
        beta_A_k:float=-0.72,  # Access time sensitivity
        beta_B_k:float=-0.39,  # Bicycle time sensitivity
        M:list=['Walk', 'Bike', 'PublicTransit'], 
        access_time_range:list=[1, 2], 
        walking_speed:float=5, 
        biking_speed:float=17.5,
        driving_speed:float=45, 
        biking_cost_per_km:float=0.05,
        mode_choice_model:str='MNL',
        use_random_access_time:bool=False,
        verbose:bool=False):
    start_time = time.time()
    vehicles = ['v{}'.format(i) for i in range(num_vehicles)]
    initial_positions = {vehicle: stations[i % len(stations)] for i, vehicle in enumerate(vehicles)}
    
    od_travel_data_sample = od_travel_data.sample(1000)

    # get feasible combinations
    requests, m_star = calculate_feasible_combinations(
        od_travel_data=od_travel_data_sample, beta_C_k=beta_C_k, beta_V_k=beta_V_k, beta_P_k=beta_P_k, beta_A_k=beta_A_k, beta_W_k=beta_A_k, beta_B_k=beta_B_k, 
        L=L, P_CS=P_CS, M=M, 
        access_time_range=access_time_range, walking_speed=walking_speed, biking_speed=biking_speed, driving_speed=driving_speed, biking_cost_per_km=biking_cost_per_km,
        mode_choice_model=mode_choice_model, use_random_access_time=use_random_access_time)
    if len(requests) == 0:
        return 0, 0, 0, 0, 0, 0
    try:
        requests = format_requests_df(requests, addresses)
        model_prep_time = time.time() - start_time
        # run model
        results = model_run(requests, vehicles, initial_positions, distance_duration_lookup, stations, P_CS, C_V, C_S, L, verbose=verbose) 
        start_time = time.time()
        # unpack results
        financials, vehicle_df, requests_served, performance_metrics = results
        # Calculate metrics
        objective_value, vehicle_relocation_costs, utilization_ratio, acceptance_ratio, vehicle_usage_distribution = calculate_usage_metrics(financials, vehicle_df, requests_served, requests)
        metrics_dict = eval(performance_metrics) # format string to dict
        process_end_time = time.time() - start_time
        # Update performance metrics
        metrics_dict['relocation_costs'] = vehicle_relocation_costs
        metrics_dict['model_prep_time'] = model_prep_time
        metrics_dict['model_postprocessing_time_2'] = process_end_time
    except Exception as e:
        print(f"Error in run_experiment: {e}")
        traceback.print_exc()
        raise e
    return [len(requests), objective_value, utilization_ratio, acceptance_ratio, vehicle_usage_distribution, str(metrics_dict)]

def run_experiment_wrapper(args):
    repetition, num_vehicles, p_cs, beta_C_k, beta_V_k, beta_P_k, beta_B_k, beta_A_k, L_option = args
    try:
        results = run_experiment(num_vehicles=num_vehicles, P_CS=p_cs, L=L_option,
                                 beta_C_k=beta_C_k, beta_V_k=beta_V_k, beta_P_k=beta_P_k, 
                                 beta_B_k=beta_B_k, beta_A_k=beta_A_k)        
    except Exception as e:
        print(f"Error in run_experiment_wrapper: {e}")
        traceback.print_exc()
        print(f"Arguments: {args}")
        results = [0, 0, 0, 0, 0, "Error"]
    try:
        return_object = { 
                'repetition': repetition,
                'num_vehicles': num_vehicles,
                'P_cs': p_cs,
                'beta_C_k': beta_C_k,
                'beta_V_k': beta_V_k,
                'beta_P_k': beta_P_k,
                'beta_B_k': beta_B_k,
                'beta_A_k': beta_A_k,
                'num_cs_requests': results[0],
                'objective_value': results[1],
                'utilization_ratio': results[2],
                'acceptance_ratio': results[3],
                'vehicle_usage_distribution': results[4],
                'performance_metrics': results[5]}
    except Exception as e:
        print(f"Error in run_experiment_wrapper: {e}")
        print(f"Arguments: {args}")
        print(f"Results: {results}")
        raise e
    return return_object
        
def run_experiments():
    options = list(product(repetitions_list, num_vehicles_options, P_CS_options, 
                            beta_C_k_options, beta_V_k_options, beta_P_k_options, beta_B_k_options, beta_A_k_options))
    options = [(*opt, L_options) for opt in options] # add static parameters
    results = []
    with ProcessPoolExecutor(max_workers=use_max_workers) as executor:
            results = list(tqdm(executor.map(run_experiment_wrapper, options),
                                total=len(options), desc="Running experiments"))
    return pd.DataFrame(results)

if __name__ in "__main__":
    print("Running experiments")
    experiment_df = run_experiments()
    experiment_df.to_csv('data/factorial_output_complex_01.csv', index=False, sep=';', encoding='utf-8')