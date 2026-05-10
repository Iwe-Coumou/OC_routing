import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def build_sequential_gif(instance_path, solution_path, output_path):
    # 1. Parse Coordinates
    # 1. Parse Coordinates (ROBUST)
    coords = {}
    with open(instance_path, "r") as f:
        in_coords = False
        for line in f:
            line = line.strip()
            
            if not line:
                continue # Ignore stray blank lines instead of stopping
                
            if line.startswith("COORDINATES"): 
                in_coords = True
                continue
            elif in_coords and line[0].isalpha(): 
                in_coords = False # Stop only when hitting the next word (e.g., "REQUESTS")
            
            if in_coords:
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        coords[int(parts[0])] = (float(parts[1]), float(parts[2]))
                    except ValueError:
                        pass


    #IWE
    # 2. Parse Routes
    routes_by_day = {}
    current_day = None
    with open(solution_path, "r") as f:
        for line in f:
            line = line.strip()
            
            if line.startswith("DAY"):
                current_day = int(line.split("=")[1].strip())
                routes_by_day[current_day] = []
            elif line:
                parts = line.split()
                # Check if it is a routing line (e.g., "1 R 0 10 158 0")
                if len(parts) >= 3 and parts[1] == "R":
                    # parts[2:] contains just the nodes: ['0', '10', '158', ...]
                    route_nodes = [abs(int(n)) for n in parts[2:]]
                    routes_by_day[current_day].append(route_nodes)

    days = sorted(list(routes_by_day.keys()))
    if not days:
        raise ValueError("No valid routing days found.")

    # 3. Setup Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    xs = [c[0] for c in coords.values()]
    ys = [c[1] for c in coords.values()]
    ax.scatter(xs, ys, c='lightgray', s=30, label='Locations')
    ax.scatter(coords[0][0], coords[0][1], c='red', s=100, marker='s', label='Depot')
    ax.grid(True, linestyle='--', alpha=0.5)

    # Allocate enough lines for the busiest day
    max_vehicles = max(len(routes) for routes in routes_by_day.values())
    colors = plt.cm.tab20(np.linspace(0, 1, max_vehicles))
    lines, points = [], []
    
    for i in range(max_vehicles):
        line, = ax.plot([], [], lw=2, color=colors[i % 20], alpha=0.7)
        point, = ax.plot([], [], marker='o', color=colors[i % 20], markersize=8)
        lines.append(line)
        points.append(point)

    # 4. Calculate Timeline
    frames_per_segment = 1
    pause_frames = 30  # 1-second pause at the end of each day
    day_frame_limits = []
    
    for day in days:
        routes = routes_by_day[day]
        max_route_len = max([len(r) for r in routes] + [0])
        active_frames = max(1, (max_route_len - 1) * frames_per_segment)
        total_day_frames = active_frames + pause_frames
        day_frame_limits.append({
            "day": day,
            "routes": routes,
            "active_frames": active_frames,
            "total_frames": total_day_frames
        })

    total_animation_frames = sum(d["total_frames"] for d in day_frame_limits)

    def init():
        for line, point in zip(lines, points):
            line.set_data([], [])
            point.set_data([], [])
        return lines + points

    # 5. Animation Logic
    def update(frame):
        # Determine which day this frame belongs to
        current_frame = frame
        current_day_data = None
        
        for d_data in day_frame_limits:
            if current_frame < d_data["total_frames"]:
                current_day_data = d_data
                break
            current_frame -= d_data["total_frames"]
            
        day = current_day_data["day"]
        routes = current_day_data["routes"]
        active_frames = current_day_data["active_frames"]
        
        ax.set_title(f"Vehicle Routes - Day {day}", fontsize=16, fontweight='bold')
        
        # Calculate movement within the current day
        # Calculate movement within the current day
        for i in range(max_vehicles):
            if i < len(routes):
                route = routes[i]
                if len(route) == 0:
                    continue
                    
                if current_frame < active_frames:
                    # Truck is moving
                    segment_idx = current_frame // frames_per_segment
                    progress = (current_frame % frames_per_segment) / frames_per_segment
                    
                    if segment_idx < len(route) - 1:
                        # THE FAILSAFE: Use .get() to fallback to Node 0 if a coordinate is missing
                        start_node = route[segment_idx]
                        end_node = route[segment_idx + 1]
                        start_coord = coords.get(start_node, coords[0])
                        end_coord = coords.get(end_node, coords[0])
                        
                        curr_x = start_coord[0] + (end_coord[0] - start_coord[0]) * progress
                        curr_y = start_coord[1] + (end_coord[1] - start_coord[1]) * progress
                        
                        history_x = [coords.get(n, coords[0])[0] for n in route[:segment_idx + 1]] + [curr_x]
                        history_y = [coords.get(n, coords[0])[1] for n in route[:segment_idx + 1]] + [curr_y]
                    else:
                        # Reached the end before other longer routes
                        history_x = [coords.get(n, coords[0])[0] for n in route]
                        history_y = [coords.get(n, coords[0])[1] for n in route]
                else:
                    # Pause period: lock at final nodes
                    history_x = [coords.get(n, coords[0])[0] for n in route]
                    history_y = [coords.get(n, coords[0])[1] for n in route]
                    
                lines[i].set_data(history_x, history_y)
                points[i].set_data([history_x[-1]], [history_y[-1]])
            else:
                # Hide lines for vehicles not used on this day
                lines[i].set_data([], [])
                points[i].set_data([], [])
                
        return lines + points

    # 6. Render and Save
    print(f"Rendering {len(days)} days into one animation... (Total frames: {total_animation_frames})")
    ani = animation.FuncAnimation(fig, update, frames=total_animation_frames, init_func=init, blit=True, interval=50)
    
    writer = animation.PillowWriter(fps=30)
    ani.save(output_path, writer=writer)
    print(f"SUCCESS: Full timeline animation saved to {output_path}")

if __name__ == "__main__":
    # Define paths
    base_dir = r"C:\Users\iweyn\Documents\Uni\Year_3\Combinatorial Optimization\Routing Project\OC_routing"
    instance_file = os.path.join(base_dir, "instances/B2.txt")
    solution_file = os.path.join(base_dir, "instances/B2_solution.txt")
    output_file = os.path.join(base_dir, "full_timeline_animation_B2.gif")
    
    build_sequential_gif(instance_file, solution_file, output_file)