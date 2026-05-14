with open('B3_cvae_solution.txt') as f:
    lines = f.readlines()

multitrip_routes = []
single_trip_routes = 0
current_day = None
in_vehicles = False
route_count = 0

for line in lines:
    original_line = line.strip()
    if original_line.startswith('DAY ='):
        current_day = int(original_line.split('=')[1].strip())
        in_vehicles = True
    elif original_line.startswith('NUMBER_OF_VEHICLES'):
        in_vehicles = False
    elif in_vehicles and '\tR\t' in original_line:
        route_count += 1
        parts = original_line.split('\t')
        if len(parts) >= 4:
            vnum = int(parts[0])
            # The route stops are from index 3 to the second-to-last element
            # Last element is "D" (cost marker)
            tokens = parts[3:-1]  # This should give us the route stops
            
            # Count depot visits (0s)
            depot_count = tokens.count('0')
            
            if route_count <= 5:  # Debug first 5 routes
                print(f"Route {route_count}: vnum={vnum}, tokens={tokens}, depot_count={depot_count}")
            
            # Check if this is a multi-trip route (has intermediate 0s, so depot_count > 2)
            if depot_count > 2:
                multitrip_routes.append({
                    'day': current_day,
                    'vehicle': vnum,
                    'depot_count': depot_count,
                    'trips': depot_count - 1
                })
            elif tokens:
                single_trip_routes += 1

print(f"\nTotal routes processed: {route_count}")
print(f"Single-trip routes: {single_trip_routes}")
print(f"Multi-trip vehicles: {len(multitrip_routes)}")
if multitrip_routes:
    print(f"\nMulti-trip vehicles (first 10):")
    for r in multitrip_routes[:10]:
        print(f"  Day {r['day']}, Vehicle {r['vehicle']}: {r['trips']} trips")
