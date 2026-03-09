import matplotlib.pyplot as plt
import numpy as np
import math
from matplotlib.animation import FuncAnimation

from environment import DebrisMap
from rrt_planner import TailBaseRRT
from robot_model import SnakeRobotModel
import config

def calculate_straight_state_from_head(head_x, head_y, yaw_deg):
    yaw_rad = np.deg2rad(yaw_deg)
    dist_to_j4 = 12.0
    j4_x = head_x - dist_to_j4 * math.cos(yaw_rad)
    j4_y = head_y - dist_to_j4 * math.sin(yaw_rad)
    return np.array([j4_x, j4_y, yaw_rad, 0, 0, 0, 0])

def interpolate_arc_path(path_data, steps_per_node=10):
    """
    Smart Interpolator:
    - Detects Arcs
    - Detects 'Turn in Place' (Parking maneuvers)
    - Generates fluid animation for both
    """
    anim_frames = []
    
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
                
            anim_frames.append(interp_state)
            
    anim_frames.append(path_data[-1][0])
    return anim_frames

def main():
    env = DebrisMap(70, 70)
    START_STATE = calculate_straight_state_from_head(35.0, 20.0, 90)
    GOAL_STATE = calculate_straight_state_from_head(60.0, 60.0, 0)

    planner = TailBaseRRT(env, START_STATE, GOAL_STATE)
    
    print("\n🔍 Starting Hybrid RRT (Arc Cruise + Parking Mode)...")
    frame_count = 0
    
    while not planner.finished:
        if frame_count > config.MAX_ITER:
            print("⚠️ Max iterations reached.")
            break
        if planner.step():
            break
        frame_count += 1
        if frame_count % 500 == 0:
            print(f"   Iteration: {frame_count} | Nodes: {len(planner.nodes)}")

    if not planner.path:
        print("\n❌ Failed to find a path.")
        return

    print("\n✅ Path Found! Generating Animation...")
    anim_frames = interpolate_arc_path(planner.path, steps_per_node=15)
    
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(0, env.width)
    ax.set_ylim(0, env.height)
    
    ax.imshow(env.raw_grid, cmap='Greys', origin='lower', vmin=0, vmax=1)
    ax.imshow(env.planning_grid, cmap='Reds', alpha=0.2, origin='lower')
    
    def draw_ghost(s, c):
        b = SnakeRobotModel.get_body_from_tail_base(s)
        bx, by = zip(*b)
        ax.plot(bx, by, color=c, lw=2, alpha=0.4)
    draw_ghost(START_STATE, 'green')
    draw_ghost(GOAL_STATE, 'red')
    
    line_body, = ax.plot([], [], color='blue', lw=3, zorder=15)
    scat_joints = ax.scatter([], [], color='white', edgecolors='black', s=30, zorder=16)
    scat_head = ax.scatter([], [], color='gold', edgecolors='black', marker='D', s=50, zorder=17)
    
    trail_x, trail_y = [], []
    line_trail, = ax.plot([], [], color='lime', lw=2, alpha=0.5)

    def init():
        return line_body, line_trail, scat_joints, scat_head

    def update(frame_idx):
        state = anim_frames[frame_idx]
        body = SnakeRobotModel.get_body_from_tail_base(state)
        bx, by = zip(*body)
        
        line_body.set_data(bx, by)
        scat_joints.set_offsets(body[1:-1])
        scat_head.set_offsets([body[0]])
        
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