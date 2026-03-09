import numpy as np

# --- ROBOT PHYSICAL DIMENSIONS ---
NUM_SEGMENTS = 5        # Head + 4 Links
NUM_JOINTS = 4          # J1, J2, J3, J4
SEGMENT_LENGTH = 2.5    # Reduced from 3.0 to fit better
SNAKE_WIDTH = 3.0       # Reduced width slightly
INFLATION_RADIUS = int(SNAKE_WIDTH / 2.0 + 1.0)

# --- CONSTRAINTS ---
JOINT_LIMIT = 50.0          # Degrees (+/-)
MAX_JOINT_CHANGE = 20.0     # Degrees per step
RRT_STEP_SIZE = 2.0         # Increased step size for faster exploration
MAX_TURN_ANGLE = np.deg2rad(45) # Increased steering capability

# --- RRT SETTINGS ---
MAX_ITER = 300000
GOAL_POS_TOLERANCE = 5.0    # Relaxed tolerance (The snake is big)
GOAL_ANGLE_TOLERANCE = 30.0 # Degrees (checked at JOINTS)

# --- KD-TREE WEIGHTS ---
W_POS = 1.0
W_YAW = 0.5
W_JOINT = 0.1