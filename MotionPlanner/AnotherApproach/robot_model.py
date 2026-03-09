import numpy as np
import math
import config
from environment import line_segment_intersection

class SnakeRobotModel:
    @staticmethod
    def get_body_from_tail_base(state):
        """
        Reconstructs the full 5-segment body starting from Joint 4 (Base).
        
        State Vector (7D): [x_j4, y_j4, yaw_link4, q1, q2, q3, q4]
        
        Robot Chain:
        [Head] --(q1)-- [Link1] --(q2)-- [Link2] --(q3)-- [Link3] --(q4)-- [Link4/Tail]
        
        Logic:
        1. We are given J4 position and Link4 (Tail) orientation.
        2. Calculate Tail_End by moving backwards from J4 along Link4.
        3. Calculate J3 by moving FORWARD from J4 (but we need Link3 angle).
           Link3 Angle = Link4 Angle + q4
        4. Continue up the chain to Head.
        
        Returns:
            List of points in STANDARD order: [Head_Start, Head_End, J2, J3, J4, Tail_End]
        """
        x_j4, y_j4, yaw_tail = state[0], state[1], state[2]
        joint_angles = state[3:] # [q1, q2, q3, q4]
        
        L = config.SEGMENT_LENGTH
        
        # 1. Calculate Tail End (Back of the robot)
        # J4 is the front of Link 4. Tail_End is L units behind along yaw_tail.
        tail_end_x = x_j4 - L * math.cos(yaw_tail)
        tail_end_y = y_j4 - L * math.sin(yaw_tail)
        
        # Points list. We will build it: [Tail_End, J4, J3, J2, J1, Head_Start]
        # Then reverse it to match standard visualization order.
        chain_points = [(tail_end_x, tail_end_y), (x_j4, y_j4)]
        
        current_x, current_y = x_j4, y_j4
        current_yaw = yaw_tail
        
        # 2. Walk up the chain (Link 4 -> Link 3 -> ... -> Head)
        # Order of angles reversed for walking up: q4 -> q3 -> q2 -> q1
        # Indices in joint_angles: 3, 2, 1, 0
        indices = [3, 2, 1, 0] 
        
        for i in indices:
            # Angle of next segment = Current Angle + Joint Angle
            # Note: Joint definition is usually relative deviation.
            # Assuming +angle means turn left.
            current_yaw += math.radians(joint_angles[i])
            
            # Calculate end of this segment
            next_x = current_x + L * math.cos(current_yaw)
            next_y = current_y + L * math.sin(current_yaw)
            
            chain_points.append((next_x, next_y))
            
            # Update current for next iteration
            current_x, current_y = next_x, next_y
            
        # chain_points is now: [Tail_End, J4, J3, J2, J1(Head_End), Head_Start]
        # Standard visualization expects: [Head_Start, Head_End, J2, J3, J4, Tail_End]
        return list(reversed(chain_points))

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

        # 2. Reconstruct Body
        body = SnakeRobotModel.get_body_from_tail_base(state)
        
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