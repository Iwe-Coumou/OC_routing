with open('B3_cvae_solution.txt') as f:
    lines = f.readlines()

multitrip_routes = []
single_trip_routes = 0
current_day = None
in_vehicles = False

for line in lines:
    line = line.strip()
    if line.startswith('DAY ='):
        current_day = int(line.split('=')[1].strip())
        in_vehicles = True
    elif line.startswith('NUMBER_OF_VEHICLES'):
        in_vehicles = False
    elif in_vehicles and '\tR\t' in line:
        parts = line.split('\t')
        vnum = int(parts[0])
        tokens = parts[3:-1]
        
        # Count depot visits (0s)
        depot_count = tokens.count('0')
        
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

print(f"Single-trip routes: {single_trip_routes}")
print(f"Multi-trip vehicles: {len(multitrip_routes)}")
if multitrip_routes:
    print(f"\nMulti-trip vehicles (showing first 15):")
    for r in multitrip_routes[:15]:
        print(f"  Day {r['day']}, Vehicle {r['vehicle']}: {r['trips']} trips")
else:
    print("\nNo multi-trip vehicles found.")
