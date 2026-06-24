import numpy as np
import math
import config
from environment import line_segment_intersection

class SnakeRobotModel:
    @staticmethod
    def get_body_from_head_base(state):
        """
        Reconstructs the full 4-segment body from the Head (the kinematic anchor).
        
        State Vector (6D): [x_head, y_head, yaw_head, q1, q2, q3]
        
        Kinematic Chain (built BACKWARD from the Head tip):
        
          HeadEnd --q1--> J1 --q2--> J2 --q3--> J3 --> TailEnd
        
        Body heading reconstruction:
          yaw_seg2 = yaw_head - radians(q1)
          yaw_seg3 = yaw_seg2 - radians(q2)
          yaw_tail = yaw_seg3 + radians(q3)

        Joint Responsibilities (Head-anchored model):
          - q1: Primary steering joint. Rotates the Head relative to Segment 2.
          - q2: Support joint. Rotates Segment 2 relative to Segment 3.
          - q3: ONLY rotates the Tail segment backward from J3.
        
        Returns:
            List of 5 points in standard order:
            [(head_end), (j1), (j2), (j3), (tail_end)]
        """
        x_head, y_head, yaw_head = state[0], state[1], state[2]
        q1, q2, q3 = state[3], state[4], state[5]
        
        # The head tip IS the anchor — (x_head, y_head) is the nose.
        head_end_x, head_end_y = x_head, y_head
        
        # 1. Head (Seg 1): Walk backward from the nose to J1.
        j1_x = x_head - config.L_HEAD * math.cos(yaw_head)
        j1_y = y_head - config.L_HEAD * math.sin(yaw_head)
        
        # 2. Seg 2 (Link 1): Walk backward from J1 to J2. Heading changes by q1.
        yaw_seg2 = yaw_head - math.radians(q1)
        j2_x = j1_x - config.L_SEG2 * math.cos(yaw_seg2)
        j2_y = j1_y - config.L_SEG2 * math.sin(yaw_seg2)
        
        # 3. Seg 3 (Link 2): Walk backward from J2 to J3. Heading changes by q2.
        yaw_seg3 = yaw_seg2 - math.radians(q2)
        j3_x = j2_x - config.L_SEG3 * math.cos(yaw_seg3)
        j3_y = j2_y - config.L_SEG3 * math.sin(yaw_seg3)
        
        # 4. Tail (Seg 4): Walk backward from J3. q3 ONLY affects the tail.
        yaw_tail = yaw_seg3 + math.radians(q3)
        tail_end_x = j3_x - config.L_TAIL * math.cos(yaw_tail)
        tail_end_y = j3_y - config.L_TAIL * math.sin(yaw_tail)
        
        return [(head_end_x, head_end_y), (j1_x, j1_y), (j2_x, j2_y), (j3_x, j3_y), (tail_end_x, tail_end_y)]

    @staticmethod
    def get_body_from_seg3_base(state):
        """
        [DEPRECATED] Reconstructs the full 4-segment body from Segment 3 (the old anchor).
        Kept for reference only — the active planner now uses get_body_from_head_base().
        
        State Vector (6D): [x_j3, y_j3, yaw_seg3, q1, q2, q3]
        
        Kinematic Chain (built outward from J3):
        
          TailEnd <--q3-- [J3/Anchor] --seg3--> J2 --q2--> J1 --q1--> HeadEnd
        
        Joint Responsibilities:
          - q3: ONLY rotates the Tail segment backward from J3.
          - q2: Steers Segment 2 (Link 1), and everything forward (J1 + Head).
          - q1: ONLY rotates the Head segment forward from J1.
        
        Returns:
            List of 5 points in standard order:
            [(head_end), (j1), (j2), (j3), (tail_end)]
        """
        x_j3, y_j3, yaw_seg3 = state[0], state[1], state[2]
        q1, q2, q3 = state[3], state[4], state[5]
        
        # 1. Tail (Seg 4): Goes backward from J3. q3 ONLY affects this!
        yaw_tail = yaw_seg3 + math.radians(q3)
        tail_end_x = x_j3 - config.L_TAIL * math.cos(yaw_tail)
        tail_end_y = y_j3 - config.L_TAIL * math.sin(yaw_tail)
        
        # 2. Seg 3 (Link 2): Goes forward from J3 to J2. (This is the anchor segment)
        j2_x = x_j3 + config.L_SEG3 * math.cos(yaw_seg3)
        j2_y = y_j3 + config.L_SEG3 * math.sin(yaw_seg3)
        
        # 3. Seg 2 (Link 1): Goes forward from J2 to J1. Affected by q2.
        yaw_seg2 = yaw_seg3 + math.radians(q2)
        j1_x = j2_x + config.L_SEG2 * math.cos(yaw_seg2)
        j1_y = j2_y + config.L_SEG2 * math.sin(yaw_seg2)
        
        # 4. Seg 1 (Head): Goes forward from J1 to the front tip. Affected by q2 and q1.
        yaw_head = yaw_seg2 + math.radians(q1)
        head_end_x = j1_x + config.L_HEAD * math.cos(yaw_head)
        head_end_y = j1_y + config.L_HEAD * math.sin(yaw_head)
        
        return [(head_end_x, head_end_y), (j1_x, j1_y), (j2_x, j2_y), (x_j3, y_j3), (tail_end_x, tail_end_y)]

    @staticmethod
    def check_self_collision(body_points):
        """Checks if non-adjacent segments intersect."""
        n = len(body_points)
        for i in range(n - 2): # Segment i
            for j in range(i + 2, n - 1): # Segment j
                if line_segment_intersection(body_points[i], body_points[i+1], 
                                             body_points[j], body_points[j+1]):
                    return True
        return False

    @staticmethod
    def is_valid_state(state, env):
        # 1. Check Joint Limits
        if np.any(np.abs(state[3:]) > config.JOINT_LIMIT):
            return False

        # 2. Reconstruct Body (Head-anchored model)
        body = SnakeRobotModel.get_body_from_head_base(state)
        
        # 3. Check Map Boundaries
        for x, y in body:
            if not (0 <= x < env.width and 0 <= y < env.height):
                return False

        # 4. Check Environment Collision
        for i in range(len(body)-1):
            if env.check_line_collision(body[i], body[i+1]):
                return False
                
        # 5. Check Self Collision
        if SnakeRobotModel.check_self_collision(body):
            return False
            
        return True
