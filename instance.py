class Instance():
    def __init__(self, 
                 dataset, 
                 name, 
                 days, 
                 capacity, 
                 max_trip_distance, 
                 depot_coordinate, 
                 vehicle_cost, 
                 vehicle_day_cost, 
                 distance_cost,
                 tools,
                 coordinates,
                 requests,
                 distance,
    ):
        self.dataset = dataset
        self.name = name
        self.days = days
        self.capacity = capacity
        self.max_trip_distance = max_trip_distance
        self.depot_coordinate = depot_coordinate
        self.vehicle_cost = vehicle_cost
        self.vehicle_day_cost = vehicle_day_cost
        self.distance_cost = distance_cost
        self.tools = tools
        self.coordinates = coordinates
        self.requests = requests
        self.distance = distance

    def _print(self):
        for k, v in vars(self).items():
            print(f"{k}: {v}")
