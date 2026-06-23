import matplotlib
import numpy as np
import math
import json
import sys
import argparse

from environment import DebrisMap
from rrt_planner import TailBaseRRT
from robot_model import SnakeRobotModel
import config

def calculate_straight_state_from_head(head_x, head_y, yaw_deg):
    yaw_rad = np.deg2rad(yaw_deg)
    dist_to_j3 = config.L_HEAD + config.L_SEG2 + config.L_SEG3
    j3_x = head_x - dist_to_j3 * math.cos(yaw_rad)
    j3_y = head_y - dist_to_j3 * math.sin(yaw_rad)
    return np.array([j3_x, j3_y, yaw_rad, 0, 0, 0])

def interpolate_arc_path(path_data, env, steps_per_node=10):
    """
    Smart Interpolator with collision validation:
    - Detects Arcs
    - Detects 'Turn in Place' (Parking maneuvers)
    - Generates fluid animation for both
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
                # Drive (Linear/Arc approx)
                # Since we are essentially connecting valid states, linear interp of state
                # combined with angle interp creates the visual arc.
                interp_state = s1 + t * (s2 - s1)
                interp_state[2] = s1[2] + t * dth
            
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
    # --- CLI argument parsing (supports dynamic start state for replanning) ---
    parser = argparse.ArgumentParser(description="Snake Robot RRT* Path Planner")
    parser.add_argument('--start_x',       type=float, default=None,
                        help='Base (J3) X coordinate on the map')
    parser.add_argument('--start_y',       type=float, default=None,
                        help='Base (J3) Y coordinate on the map')
    parser.add_argument('--start_yaw_rad', type=float, default=None,
                        help='Base heading in radians')
    parser.add_argument('--start_q1',      type=float, default=None,
                        help='Joint 1 angle in degrees')
    parser.add_argument('--start_q2',      type=float, default=None,
                        help='Joint 2 angle in degrees')
    parser.add_argument('--start_q3',      type=float, default=None,
                        help='Joint 3 angle in degrees')
    parser.add_argument('--out',           type=str, default='robot_path_commands.json',
                        help='Output JSON file path')
    args = parser.parse_args()

    # Detect replan mode: all six start-state args must be provided together
    start_args = [args.start_x, args.start_y, args.start_yaw_rad,
                  args.start_q1, args.start_q2, args.start_q3]
    replan_mode = all(v is not None for v in start_args)

    # Disable interactive matplotlib backend when replanning headlessly
    if replan_mode:
        matplotlib.use('Agg')

    SELECTED_MAP = "complex_map"
    env = DebrisMap(55, 85, map_type=SELECTED_MAP)

    if replan_mode:
        # --- REPLAN MODE: start from the robot's current physical state ---
        START_STATE = np.array([
            args.start_x, args.start_y, args.start_yaw_rad,
            args.start_q1, args.start_q2, args.start_q3,
        ])
        print(f"[REPLAN] Start state from feeder: {START_STATE}")
    else:
        # --- NORMAL MODE: fixed starting pose ---
        # Start the head at (17.0, 22.0) facing North.
        # Tail sits perfectly safe at (17.0, 10.0), clearing the bottom wall.
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

    print(f"\n[OK] Docking Complete! Total Cost: {planner.final_node.cost:.2f}")
    print("[OK] Path Found! Generating Animation...")
    anim_frames = interpolate_arc_path(planner.path, env, steps_per_node=15)
    
    # --- JSON Export Logic ---
    SPROCKET_PITCH_RADIUS = 0.7
    circumference = 2 * math.pi * SPROCKET_PITCH_RADIUS
    MAX_RPM = 30.0
    
    commands = []
    prev_state = None
    
    for i, state in enumerate(anim_frames):
        x, y, yaw_rad, q1, q2, q3 = state
        
        if prev_state is None:
            dist_head = 0.0
            dist_link2 = 0.0
            revolutions = 0.0
        else:
            body_old = SnakeRobotModel.get_body_from_seg3_base(prev_state)
            body_new = SnakeRobotModel.get_body_from_seg3_base(state)
            
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
                "q1_pitch_deg": 0.0,
                "q2_pitch_deg": 0.0,
                "q3_pitch_deg": 0.0
            }
        }
        commands.append(command)
        prev_state = state

    with open(args.out, "w") as f:
        json.dump(commands, f, indent=4)
        
    print(f"[EXPORT] Saved {args.out} for STM32 control.")

    # --- In replan mode: no animation, just exit after writing JSON ---
    if replan_mode:
        print("[REPLAN] Path written. Exiting.")
        sys.exit(0)

    # --------------------------
    # NORMAL MODE: interactive animation
    # --------------------------
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    fig, ax = plt.subplots(figsize=(7, 10))
    ax.set_xlim(0, env.width)
    ax.set_ylim(0, env.height)
    
    ax.imshow(env.raw_grid, cmap='Greys', origin='lower', vmin=0, vmax=1)
    ax.imshow(env.planning_grid, cmap='Reds', alpha=0.2, origin='lower')
    
    def draw_ghost(s, c):
        b = SnakeRobotModel.get_body_from_seg3_base(s)
        bx, by = zip(*b)
        ax.plot(bx, by, color=c, lw=2, alpha=0.4)
    draw_ghost(START_STATE, 'green')
    draw_ghost(GOAL_STATE, 'red')
    
    # Width envelope: shows the robot's actual physical footprint
    # We must calculate the exact points-per-data-unit using the true axes bounding box,
    # otherwise default figure margins make the line ~30% too thick.
    fig.canvas.draw()
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width_in_points = config.SNAKE_WIDTH * (bbox.width / env.width) * 72
    
    line_width_envelope, = ax.plot([], [], color='cyan', lw=width_in_points,
                                   alpha=0.75, solid_capstyle='butt', zorder=14)
    line_body, = ax.plot([], [], color='blue', lw=3, zorder=15)
    scat_joints = ax.scatter([], [], color='white', edgecolors='black', s=30, zorder=16)
    scat_head = ax.scatter([], [], color='gold', edgecolors='black', marker='D', s=50, zorder=17)
    
    trail_x, trail_y = [], []
    line_trail, = ax.plot([], [], color='lime', lw=2, alpha=0.5)

    def init():
        return line_width_envelope, line_body, line_trail, scat_joints, scat_head

    def update(frame_idx):
        state = anim_frames[frame_idx]
        body = SnakeRobotModel.get_body_from_seg3_base(state)
        bx, by = zip(*body)
        
        line_width_envelope.set_data(bx, by)
        line_body.set_data(bx, by)
        scat_joints.set_offsets(body[1:-1])
        scat_head.set_offsets([body[0]])
        
        trail_x.append(body[0][0])
        trail_y.append(body[0][1])
        line_trail.set_data(trail_x, trail_y)
        
        ax.set_title(f"Simulation: {int(frame_idx/len(anim_frames)*100)}%")
        return line_width_envelope, line_body, line_trail, scat_joints, scat_head

    anim = FuncAnimation(fig, update, frames=len(anim_frames), init_func=init, 
                         interval=20, blit=True, repeat=False)
    plt.show()

if __name__ == "__main__":
    main()
