import numpy as np

# --- ROBOT PHYSICAL DIMENSIONS ---
NUM_SEGMENTS = 5        # Head + 4 Links
NUM_JOINTS = 4          # J1, J2, J3, J4
SEGMENT_LENGTH = 3.0    
SNAKE_WIDTH = 4.0       
INFLATION_RADIUS = int(SNAKE_WIDTH / 2.0 + 1.0)

# --- CONSTRAINTS ---
JOINT_LIMIT = 50.0          # Degrees (+/-)
MAX_JOINT_CHANGE = 20.0     # Degrees per step
RRT_STEP_SIZE = 1.0         # Max euclidean step for base
MAX_TURN_ANGLE = np.deg2rad(30) # Max rotation for base per step

# --- RRT SETTINGS ---
MAX_ITER = 200000
GOAL_POS_TOLERANCE = 2.0    # Units (checked at HEAD)
GOAL_ANGLE_TOLERANCE = 15.0 # Degrees (checked at JOINTS)

# --- KD-TREE WEIGHTS ---
# State: [x_base, y_base, yaw_base, q1, q2, q3, q4]
# We weight position higher to guide expansion, yaw/joints lower.
W_POS = 1.0
W_YAW = 0.5
W_JOINT = 0.1