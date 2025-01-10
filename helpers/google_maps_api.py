import requests
import json

def get_transit_time(API_KEY, origin_lat, origin_lng, destination_lat, destination_lng, departure_time='now'):
    url = 'https://maps.googleapis.com/maps/api/directions/json'
    params = {
        'origin': f'{origin_lat},{origin_lng}',
        'destination': f'{destination_lat},{destination_lng}',
        'mode': 'transit',
        'departure_time': departure_time,
        'key': API_KEY}
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        if data['status'] == 'OK':
            route = data['routes'][0]
            leg = route['legs'][0]
            total_duration = leg['duration']['value']
            steps = leg['steps']
            access_time = 0
            transit_segments = []
            transfer_times = []
            egress_time = 0
            for idx, step in enumerate(steps):
                travel_mode = step['travel_mode']
                duration = step['duration']['value']
                if travel_mode == 'WALKING':
                    if idx == 0:
                        access_time += duration
                    elif idx == len(steps) - 1:
                        egress_time += duration
                    else:
                        if steps[idx + 1]['travel_mode'] == 'TRANSIT':
                            transfer_times.append(duration)
                        else:
                            egress_time += duration
                elif travel_mode == 'TRANSIT':
                    transit_details = step['transit_details']
                    transit_segment = {
                        'line_name': transit_details['line'].get('short_name', transit_details['line'].get('name')),
                        'vehicle_type': transit_details['line']['vehicle']['type'],
                        'departure_stop': transit_details['departure_stop']['name'],
                        'arrival_stop': transit_details['arrival_stop']['name'],
                        'num_stops': transit_details['num_stops'],
                        'departure_time': transit_details['departure_time']['text'],
                        'arrival_time': transit_details['arrival_time']['text'],
                        'duration': duration
                    }
                    transit_segments.append(transit_segment)
            details = {
                'total_duration_minutes': total_duration / 60,
                'access_time_minutes': access_time / 60,
                'egress_time_minutes': egress_time / 60,
                'transfer_times_minutes': [t / 60 for t in transfer_times],
                'transit_segments': transit_segments}
            return details, data
        else:
            error_message = data.get('error_message', data['status'])
            print(f"Error fetching data: {error_message}\n{data}")
            return 1, error_message
    except requests.exceptions.RequestException as e:
        raise ValueError(f"Error fetching data: {e}")

def details_from_response(json_data):
    try:
        route = json_data['routes'][0]
        leg = route['legs'][0]
        total_duration = leg['duration']['value']
        steps = leg['steps']
        access_time = 0
        transit_segments = []
        transfer_times = []
        egress_time = 0
        for idx, step in enumerate(steps):
            travel_mode = step['travel_mode']
            duration = step['duration']['value']
            if travel_mode == 'WALKING':
                if idx == 0:
                    access_time += duration
                elif idx == len(steps) - 1:
                    egress_time += duration
                else:
                    if steps[idx + 1]['travel_mode'] == 'TRANSIT':
                        transfer_times.append(duration)
                    else:
                        egress_time += duration
            elif travel_mode == 'TRANSIT':
                transit_details = step['transit_details']
                transit_segment = {
                    'line_name': transit_details['line'].get('short_name', transit_details['line'].get('name')),
                    'vehicle_type': transit_details['line']['vehicle']['type'],
                    'departure_stop': transit_details['departure_stop']['name'],
                    'arrival_stop': transit_details['arrival_stop']['name'],
                    'num_stops': transit_details['num_stops'],
                    'departure_time': transit_details['departure_time']['text'],
                    'arrival_time': transit_details['arrival_time']['text'],
                    'duration': duration
                }
                transit_segments.append(transit_segment)
        details = {
            'total_duration_minutes': total_duration / 60,
            'access_time_minutes': access_time / 60,
            'egress_time_minutes': egress_time / 60,
            'transfer_times_minutes': [t / 60 for t in transfer_times],
            'transit_segments': transit_segments}
        return details
    except Exception as e:
        raise ValueError(f"Error extracting details: {e}")

if __name__ == '__main__':
    #? Example usage
    origin_lat = 55.679737
    origin_lng = 12.533444
    destination_lat = 55.692133
    destination_lng = 12.509493
    departure_time = 'now'
    departure_time = 1736247600
    details, data = get_transit_time(
        origin_lat, origin_lng,
        destination_lat, destination_lng,
        departure_time=departure_time)
    with open('tmp/data.json', 'w') as f:
        json.dump(data, f, indent=4)
    with open('tmp/details.json', 'w') as f:
        json.dump(details, f, indent=4)
