import matplotlib.pyplot as plt
import numpy as np
import math
from matplotlib.animation import FuncAnimation

from environment import DebrisMap
from rrt_planner import TailBaseRRT
from robot_model import SnakeRobotModel
import config

def get_straight_state_tail_centered(tail_x, tail_y, yaw_deg):
    """
    Returns state for a straight snake.
    State: [x_tail, y_tail, theta_tail, q1, q2, q3, q4]
    """
    yaw_rad = np.deg2rad(yaw_deg)
    return np.array([tail_x, tail_y, yaw_rad, 0.0, 0.0, 0.0, 0.0])

def interpolate_arc_path(path_data, steps_per_node=10):
    """
    Smart Interpolator (Restored):
    - Detects 'Turn in Place' vs 'Drive'
    - Generates fluid animation frames
    """
    anim_frames = []
    
    for i in range(len(path_data) - 1):
        s1 = path_data[i] 
        s2 = path_data[i+1]
        
        dist_move = np.linalg.norm(s2[:2] - s1[:2])
        
        # Handle Angle Wrap for Theta (Index 2)
        dth = s2[2] - s1[2]
        while dth > math.pi: dth -= 2*math.pi
        while dth < -math.pi: dth += 2*math.pi
        
        for t in np.linspace(0, 1, steps_per_node, endpoint=False):
            # Linear interpolation of state
            interp_state = s1 + t * (s2 - s1)
            # Correct interpolation for Yaw
            interp_state[2] = s1[2] + t * dth
            anim_frames.append(interp_state)
            
    anim_frames.append(path_data[-1])
    return anim_frames

def main():
    env = DebrisMap(70, 70)
    
    # --- SETUP (Safe Coordinates) ---
    # Start: Bottom-Left
    START_STATE = get_straight_state_tail_centered(10.0, 10.0, 0)
    # Goal: Top-Right
    GOAL_STATE = get_straight_state_tail_centered(50.0, 60.0, 180)

    planner = TailBaseRRT(env, START_STATE, GOAL_STATE)
    
    print("\n🔍 Starting Tail-Base RRT (Non-Holonomic)...")
    
    frame_count = 0
    while not planner.finished:
        if frame_count > config.MAX_ITER:
            print("⚠️ Max iterations reached.")
            break
        if planner.step():
            break
        frame_count += 1
        if frame_count % 1000 == 0:
            print(f"   Iteration: {frame_count} | Nodes: {len(planner.nodes)}")

    if not planner.path:
        print("\n❌ Failed to find a path.")
        return

    print(f"\n✅ Path Found with {len(planner.path)} nodes! Generating Animation...")
    anim_frames = interpolate_arc_path(planner.path, steps_per_node=8)
    
    # --- VISUALIZATION (Restored from Older Version) ---
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(0, env.width)
    ax.set_ylim(0, env.height)
    
    ax.imshow(env.raw_grid, cmap='Greys', origin='lower', vmin=0, vmax=1)
    ax.imshow(env.planning_grid, cmap='Reds', alpha=0.2, origin='lower')
    
    def draw_ghost(s, c):
        # Using the new Tail-Based method, but keeping the visual style
        b = SnakeRobotModel.get_body_from_tail_state(s)
        bx, by = zip(*b)
        ax.plot(bx, by, color=c, lw=2, alpha=0.4)
    draw_ghost(START_STATE, 'green')
    draw_ghost(GOAL_STATE, 'red')
    
    # Restored Plot Objects
    line_body, = ax.plot([], [], color='blue', lw=3, zorder=15)
    scat_joints = ax.scatter([], [], color='white', edgecolors='black', s=30, zorder=16)
    scat_head = ax.scatter([], [], color='gold', edgecolors='black', marker='D', s=50, zorder=17)
    
    # Restored Trail
    trail_x, trail_y = [], []
    line_trail, = ax.plot([], [], color='lime', lw=2, alpha=0.5)

    def init():
        return line_body, line_trail, scat_joints, scat_head

    def update(frame_idx):
        state = anim_frames[frame_idx]
        body = SnakeRobotModel.get_body_from_tail_state(state)
        bx, by = zip(*body)
        
        line_body.set_data(bx, by)
        scat_joints.set_offsets(body[1:-1])
        scat_head.set_offsets([body[0]])
        
        # Trace the Head (Index 0 in the reconstructed body)
        trail_x.append(body[0][0])
        trail_y.append(body[0][1])
        line_trail.set_data(trail_x, trail_y)
        
        ax.set_title(f"Simulation: {int(frame_idx/len(anim_frames)*100)}%")
        return line_body, line_trail, scat_joints, scat_head

    anim = FuncAnimation(fig, update, frames=len(anim_frames), init_func=init, 
                         interval=20, blit=True, repeat=False)
    plt.show()

if __name__ == "__main__":
    main()