import matplotlib.pyplot as plt
import numpy as np
import math
from matplotlib.animation import FuncAnimation

from environment import DebrisMap
from rrt_planner import TailBaseRRT
from robot_model import SnakeRobotModel
import config
import json

def calculate_straight_state_from_head(head_x, head_y, yaw_deg):
    yaw_rad = np.deg2rad(yaw_deg)
    dist_to_j3= 9.0
    j3_x = head_x - dist_to_j3 * math.cos(yaw_rad)
    j3_y = head_y - dist_to_j3 * math.sin(yaw_rad)
    return np.array([j3_x, j3_y, yaw_rad, 0, 0, 0])

def interpolate_arc_path(path_data, env, steps_per_node=10):
    """
    Smart Interpolator with collision validation:
    - Detects 'Turn in Place' (Parking maneuvers)
    - Generates fluid animation for turns
    - Disables interpolation for driving to avoid unphysical sliding
    - Rejects interpolated frames that collide with the inflated zone
    """
    anim_frames = []
    last_valid = path_data[0][0]
    
    for i in range(len(path_data) - 1):
        s1, dir1 = path_data[i] 
        s2, _ = path_data[i+1]
        
        # 1. Check for Turn-In-Place (Position same, Angle diff)
        dist_move = np.linalg.norm(s2[:2] - s1[:2])
        dth = s2[2] - s1[2]
        while dth > math.pi: dth -= 2*math.pi
        while dth < -math.pi: dth += 2*math.pi
        
        # 2. Interpolate
        for t in np.linspace(0, 1, steps_per_node):
            if dist_move < 0.1 and abs(dth) > 0.01:
                # Pure Rotation (Turn in Place)
                interp_state = s1.copy()
                interp_state[2] = s1[2] + t * dth
                # Joints might change too
                interp_state[3:] = s1[3:] + t * (s2[3:] - s1[3:])
            else:
                # Curved Driving (Non-holonomic arc approximation)
                interp_state = s1.copy()
                heading = s1[2] + t * dth
                interp_state[2] = heading
                d_t = t * dist_move
                interp_state[0] = s1[0] + d_t * math.cos(heading)
                interp_state[1] = s1[1] + d_t * math.sin(heading)
                interp_state[3:] = s1[3:] + t * (s2[3:] - s1[3:])
                
            # 3. Collision check: only show frames that are physically valid
            if SnakeRobotModel.is_valid_state(interp_state, env):
                anim_frames.append(interp_state)
                last_valid = interp_state
            else:
                # Snap to last valid state to prevent visual wall penetration
                anim_frames.append(last_valid)
            
    anim_frames.append(path_data[-1][0])
    return anim_frames

def main():
    SELECTED_MAP = "complex_map"
    env = DebrisMap(45, 70, map_type=SELECTED_MAP)
    # Start the head at (16.0, 22.0) facing North. 
    # Tail sits perfectly safe at (16.0, 10.0), clearing the bottom wall.
    START_STATE = calculate_straight_state_from_head(17.0, 22.0, 90)

    # Goal perfectly centered in the final vertical corridor, facing North.
    GOAL_STATE = calculate_straight_state_from_head(26.0, 60.0, 90)

    planner = TailBaseRRT(env, START_STATE, GOAL_STATE)
    
    print("\n[START] Starting Hybrid RRT (Arc Cruise + Parking Mode)...")
    frame_count = 0
    
    while not planner.finished:
        if frame_count > config.MAX_ITER:
            print("[WARN] Max iterations reached.")
            break
        if planner.step():
            break
        frame_count += 1
        if frame_count % 500 == 0:
            print(f"   Iteration: {frame_count} | Nodes: {len(planner.nodes)}")

    if not planner.path:
        print("\n[FAIL] Failed to find a path.")
        return

    print("\n[OK] Path Found! Generating Animation...")
    anim_frames = interpolate_arc_path(planner.path, env, steps_per_node=15)
    
    # --- JSON Export Logic ---
    SPROCKET_PITCH_RADIUS = 3.0
    circumference = 2 * math.pi * SPROCKET_PITCH_RADIUS
    MAX_RPM = 30.0
    
    commands = []
    prev_state = None
    
    for i, state in enumerate(anim_frames):
        x, y, yaw_rad, q1, q2, q3 = state
        
        body = SnakeRobotModel.get_body_from_tail_base(state, env)
        if not body:
            continue
        pitch_angles_deg = []
        for j in range(len(body)-1):
            x1, y1 = body[j]
            x2, y2 = body[j+1]
            z1 = env.get_elevation(x1, y1)
            z2 = env.get_elevation(x2, y2)
            dist_xy = math.hypot(x2 - x1, y2 - y1)
            if dist_xy > 1e-5:
                pitch = math.degrees(math.atan2(z2 - z1, dist_xy))
            else:
                pitch = 0.0
            pitch_angles_deg.append(pitch)
        
        if prev_state is None:
            dist_head = 0.0
            dist_link2 = 0.0
            revolutions = 0.0
        else:
            body_old = SnakeRobotModel.get_body_from_tail_base(prev_state)
            body_new = SnakeRobotModel.get_body_from_tail_base(state)
            
            # Segment 1 (Head) distance
            head_old_x, head_old_y = body_old[1]
            head_new_x, head_new_y = body_new[1]
            dist_head = math.hypot(head_new_x - head_old_x, head_new_y - head_old_y)
            
            # Segment 3 (Link 2) distance
            j3_old_x, j3_old_y = body_old[3]
            j3_new_x, j3_new_y = body_new[3]
            dist_link2 = math.hypot(j3_new_x - j3_old_x, j3_new_y - j3_old_y)
            
            # Determine the direction by checking the dot product of the movement vector against the base's heading (yaw_rad)
            prev_x, prev_y = prev_state[:2]
            dx = x - prev_x
            dy = y - prev_y
            dot = dx * math.cos(yaw_rad) + dy * math.sin(yaw_rad)
            if dot < 0:
                dist_head = -dist_head
                dist_link2 = -dist_link2
                
            max_dist = max(abs(dist_head), abs(dist_link2))
            revolutions = max_dist / circumference
            
        # Calculate step duration based on MAX_RPM (RPM = rev/min)
        if abs(revolutions) > 1e-6:
            step_duration_ms = int(abs(revolutions) * 60000.0 / MAX_RPM)
        else:
            step_duration_ms = 0
            
        # Ensure a minimum duration to avoid divide-by-zero on pure stationary turns
        step_duration_ms = max(50, step_duration_ms)
        
        command = {
            "step_index": i,
            "step_duration_ms": step_duration_ms,
            "base_coordinates": {
                "x": round(float(x), 4),
                "y": round(float(y), 4),
                "yaw_rad": round(float(yaw_rad), 4),
                "yaw_deg": round(float(np.degrees(yaw_rad)), 4)
            },
            "dc_motor_commands": {
                "segment1_head_distance_units": round(float(dist_head), 4),
                "segment3_link2_distance_units": round(float(dist_link2), 4)
            },
            "servo_yaw_commands": {
                "q1_deg": round(max(-config.JOINT_LIMIT, min(config.JOINT_LIMIT, float(state[3]))), 2),
                "q2_deg": round(max(-config.JOINT_LIMIT, min(config.JOINT_LIMIT, float(state[4]))), 2),
                "q3_deg": round(max(-config.JOINT_LIMIT, min(config.JOINT_LIMIT, float(state[5]))), 2)
            },
            "servo_pitch_commands": {
                "q1_pitch_deg": round(pitch_angles_deg[0] - pitch_angles_deg[1], 2),
                "q2_pitch_deg": round(pitch_angles_deg[1] - pitch_angles_deg[2], 2),
                "q3_pitch_deg": round(pitch_angles_deg[2] - pitch_angles_deg[3], 2)
            }
        }
        commands.append(command)
        prev_state = state

    with open("robot_path_commands.json", "w") as f:
        json.dump(commands, f, indent=4)
        
    print("[EXPORT] Saved robot_path_commands.json for STM32 control.")
    # --------------------------
    
    # --- VISUALIZATION SETUP ---
    fig, (ax_top, ax_side) = plt.subplots(1, 2, figsize=(14, 8), gridspec_kw={'width_ratios': [1.5, 1]})
    
    # 1. Top-Down View
    ax_top.set_xlim(0, env.width)
    ax_top.set_ylim(0, env.height)
    # Use 'terrain' colormap so Z-heights are visually distinct
    ax_top.imshow(env.raw_grid, cmap='terrain', origin='lower', vmin=0, vmax=10.0)
    ax_top.imshow(env.planning_grid, cmap='Reds', alpha=0.2, origin='lower')

    line_width_envelope, = ax_top.plot([], [], color='cyan', lw=15, alpha=0.3, solid_capstyle='butt')
    line_body, = ax_top.plot([], [], color='blue', lw=3)
    scat_joints = ax_top.scatter([], [], color='white', edgecolors='black', s=30, zorder=16)
    scat_head = ax_top.scatter([], [], color='gold', edgecolors='black', marker='D', s=50, zorder=17)
    
    # 2. Side Profile (Z-Elevation) View
    ax_side.set_xlim(-2, config.NUM_SEGMENTS * config.SEGMENT_LENGTH + 2)
    ax_side.set_ylim(-1, 5)
    ax_side.set_title("Side Profile (Z-Elevation Climbing)")
    ax_side.set_xlabel("Distance Along Robot Spine")
    ax_side.set_ylabel("Z Height (Terrain)")
    ax_side.grid(True)
    
    line_profile, = ax_side.plot([], [], color='blue', lw=4, marker='o', markersize=8, markerfacecolor='white', markeredgecolor='black')
    head_profile, = ax_side.plot([], [], color='gold', marker='D', markersize=10, markeredgecolor='black', ls='')

    # Plot Start and Goal visual markers
    # Using the coordinates defined in START_STATE and GOAL_STATE
    start_x, start_y = START_STATE[0], START_STATE[1]
    goal_x, goal_y = GOAL_STATE[0], GOAL_STATE[1]
    
    ax_top.scatter([start_x], [start_y], color='blue', marker='s', s=100, edgecolors='black', zorder=20)
    ax_top.scatter([goal_x], [goal_y], color='green', marker='X', s=150, edgecolors='black', zorder=20)
    
    # Add text labels
    ax_top.text(start_x + 2, start_y, "Initial State", color='white', weight='bold', fontsize=10, ha='left', va='center', zorder=25, bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', pad=2))
    ax_top.text(goal_x + 2, goal_y, "Goal State", color='white', weight='bold', fontsize=10, ha='left', va='center', zorder=25, bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', pad=2))

    def init():
        return line_width_envelope, line_body, scat_joints, scat_head, line_profile, head_profile

    def update(frame_idx):
        state = anim_frames[frame_idx]
        body = SnakeRobotModel.get_body_from_tail_base(state, env)
        if not body:
            return line_width_envelope, line_body, scat_joints, scat_head, line_profile, head_profile
        bx, by = zip(*body)
        
        # Update Top-Down View
        line_width_envelope.set_data(bx, by)
        line_body.set_data(bx, by)
        scat_joints.set_offsets(body[1:-1])
        scat_head.set_offsets([body[0]])
        
        # Update Side Profile View
        z_coords = [env.get_elevation(x, y) for x, y in body]
        
        # Calculate distance along the spine for the X-axis of the side plot
        spine_dist = [0.0]
        for i in range(1, len(body)):
            dist_xy = math.hypot(body[i][0] - body[i-1][0], body[i][1] - body[i-1][1])
            spine_dist.append(spine_dist[-1] + dist_xy)
            
        # Reverse distances so Head is on the right side of the plot
        spine_dist = [max(spine_dist) - d for d in spine_dist]
        
        line_profile.set_data(spine_dist, z_coords)
        head_profile.set_data([spine_dist[0]], [z_coords[0]]) 
        
        ax_top.set_title(f"2.5D Top-Down View | Progress: {int(frame_idx/len(anim_frames)*100)}%")
        return line_width_envelope, line_body, scat_joints, scat_head, line_profile, head_profile

    anim = FuncAnimation(fig, update, frames=len(anim_frames), init_func=init, 
                         interval=10, blit=True, repeat=False)
    plt.show()

if __name__ == "__main__":
    main()