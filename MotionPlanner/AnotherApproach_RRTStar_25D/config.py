import numpy as np

# --- ROBOT PHYSICAL DIMENSIONS ---
NUM_SEGMENTS = 4       # Head + 4 Links
NUM_JOINTS = 3      # J1, J2, J3
SEGMENT_LENGTH = 3.0    
SNAKE_WIDTH = 4.0       
INFLATION_RADIUS = int(SNAKE_WIDTH / 2.0 + 1.0)

# --- CONSTRAINTS ---
JOINT_LIMIT = 37.0          # Degrees (+/-)
MAX_JOINT_CHANGE = 15.0     # Degrees per step
RRT_STEP_SIZE = 2.0         # Max euclidean step for base (was 1.5)
MAX_TURN_ANGLE = np.deg2rad(25) # Max rotation for base per step

# --- RRT SETTINGS ---
MAX_ITER = 1000000
GOAL_POS_TOLERANCE = 0.8    # Units (checked at HEAD)
GOAL_ANGLE_TOLERANCE = 20.0 # Degrees (checked at JOINTS)

# --- KD-TREE WEIGHTS ---
# State: [x_base, y_base, yaw_base, q1, q2, q3]
# We weight position higher to guide expansion, yaw/joints lower.
W_POS = 1.0
W_YAW = 0.5
W_JOINT = 0.1

# --- RRT* COST OPTIMIZATIONS ---
RRT_STAR_K = 15            # Number of neighbors to check for rewiring
COST_DIST_WEIGHT = 1.0     # Penalty for traveling long physical distances
COST_JOINT_WEIGHT = 0.5    # Penalty for twisting servos unnecessarily
COST_PITCH_WEIGHT = 5.0    # Heavy penalty for climbing/descending rubble vertically
COST_OBS_WEIGHT = 10.0     # Penalty for navigating too close to debris walls
