with open('B3_cvae_solution.txt') as f:
    lines = f.readlines()

multitrip_routes = []
single_trip_routes = 0
current_day = None
route_count = 0

for line in lines:
    line = line.rstrip('\n')
    if line.startswith('DAY ='):
        current_day = int(line.split('=')[1].strip())
    elif '\tR\t' in line:
        route_count += 1
        parts = line.split('\t')
        vnum = int(parts[0])
        # The route includes everything from the 'R' marker onwards
        # Format: vehicle_id \t R \t stop1 \t stop2 ... \t stopN
        # The last part is the cost line (1 \t D \t cost), which is on next line
        # So we take all elements from index 2 onwards (skip vehicle_id and R)
        tokens = parts[2:]  # This includes all the stops from the first depot
        
        # Count depot visits (0s)
        depot_count = tokens.count('0')
        
        if route_count <= 5:  # Debug first 5 routes
            print(f"Route {route_count}: vnum={vnum}, tokens={tokens}, depot_count={depot_count}")
        
        # Multi-trip: if it has more than 2 depots (start and end), there are intermediate trips
        if depot_count > 2:
            multitrip_routes.append({
                'day': current_day,
                'vehicle': vnum,
                'depot_count': depot_count,
                'trips': depot_count - 1
            })
        else:
            single_trip_routes += 1

print(f"\nTotal routes processed: {route_count}")
print(f"Single-trip routes: {single_trip_routes}")
print(f"Multi-trip vehicles: {len(multitrip_routes)}")
if multitrip_routes:
    print(f"\nMulti-trip vehicles (first 15):")
    for r in multitrip_routes[:15]:
        print(f"  Day {r['day']}, Vehicle {r['vehicle']}: {r['trips']} trips ({r['depot_count']} depots)")
