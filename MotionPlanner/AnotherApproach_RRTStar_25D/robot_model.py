import numpy as np
import math
import config
from environment import line_segment_intersection

class SnakeRobotModel:
    @staticmethod
    def get_body_from_tail_base(state, env):
        """
        Reconstructs the full 4-segment body starting from Joint 3 (Base).
        
        State Vector (6D): [x_j3, y_j3, yaw_link3, q1, q2, q3]
        
        Robot Chain:
        [Head] --(q1)-- [Link1] --(q2)-- [Link2] --(q3)-- [Link3/Tail]
        
        Logic:
        1. We are given J3 position and Link3 (Tail) orientation.
        2. Calculate Tail_End by moving backwards from J3 along Link3.
        3. Calculate J2 by moving FORWARD from J3 (but we need Link2 angle).
           Link2 Angle = Link3 Angle + q3
        4. Continue up the chain to Head.
        
        Returns:
            List of points in STANDARD order: [Head_Start, Head_End, J2, J3, Tail_End]
            Returns empty list [] if elevation change is too steep for the segment length.
        """
        x_j3, y_j3, yaw_tail = state[0], state[1], state[2]
        joint_angles = state[3:] # [q1, q2, q3]
        
        L = config.SEGMENT_LENGTH
        
        # 1. Calculate Tail End (Back of the robot)
        theoretical_tail_x = x_j3 - L * math.cos(yaw_tail)
        theoretical_tail_y = y_j3 - L * math.sin(yaw_tail)
        dz_tail = env.get_elevation(theoretical_tail_x, theoretical_tail_y) - env.get_elevation(x_j3, y_j3)
        if abs(dz_tail) >= L:
            return []
            
        true_L_tail = math.sqrt(L**2 - dz_tail**2)
        tail_end_x = x_j3 - true_L_tail * math.cos(yaw_tail)
        tail_end_y = y_j3 - true_L_tail * math.sin(yaw_tail)
        
        # Points list. We will build it: [Tail_End, J3, J2, J1, Head_Start]
        # Then reverse it to match standard visualization order.
        chain_points = [(tail_end_x, tail_end_y), (x_j3, y_j3)]
        
        current_x, current_y = x_j3, y_j3
        current_yaw = yaw_tail
        
        # 2. Walk up the chain (Link 3 -> Link 2 -> ... -> Head)
        # Order of angles reversed for walking up: q3 -> q2 -> q1
        # Indices in joint_angles: 2, 1, 0
        indices = [2, 1, 0] 
        
        for i in indices:
            # Angle of next segment = Current Angle + Joint Angle
            # Note: Joint definition is usually relative deviation.
            # Assuming +angle means turn left.
            current_yaw += math.radians(joint_angles[i])
            
            # Calculate theoretical end of this segment
            theoretical_next_x = current_x + L * math.cos(current_yaw)
            theoretical_next_y = current_y + L * math.sin(current_yaw)
            
            dz = env.get_elevation(theoretical_next_x, theoretical_next_y) - env.get_elevation(current_x, current_y)
            if abs(dz) >= L:
                return []
                
            true_L = math.sqrt(L**2 - dz**2)
            
            next_x = current_x + true_L * math.cos(current_yaw)
            next_y = current_y + true_L * math.sin(current_yaw)
            
            chain_points.append((next_x, next_y))
            
            # Update current for next iteration
            current_x, current_y = next_x, next_y
            
        # chain_points is now: [Tail_End, J3, J2, J1(Head_End), Head_Start]
        # Standard visualization expects: [Head_Start, Head_End, J2, J3, Tail_End]
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
    def check_3d_segment_collision(p1, p2, env):
        """Checks if the rigid track body clips through the terrain between joints."""
        x1, y1 = p1
        x2, y2 = p2
        z1 = env.get_elevation(x1, y1)
        z2 = env.get_elevation(x2, y2)
        
        num_samples = 5
        for i in range(num_samples):
            t = i / (num_samples - 1) if num_samples > 1 else 0
            mid_x = x1 + t * (x2 - x1)
            mid_y = y1 + t * (y2 - y1)
            z_robot = z1 + t * (z2 - z1)
            z_terrain = env.get_elevation(mid_x, mid_y)
            if z_terrain > z_robot + 0.1:
                return True
        return False

    @staticmethod
    def is_valid_state(state, env):
        # 1. Check Joint Limits
        if np.any(np.abs(state[3:]) > config.JOINT_LIMIT):
            return False

        # 2. Reconstruct Body
        body = SnakeRobotModel.get_body_from_tail_base(state, env)
        if not body:
            return False
        
        # 3. Check Map Boundaries
        for x, y in body:
            if not (0 <= x < env.width and 0 <= y < env.height):
                return False

        # 4. Check Environment Collision
        for i in range(len(body)-1):
            if env.check_line_collision(body[i], body[i+1]):
                return False
            if SnakeRobotModel.check_3d_segment_collision(body[i], body[i+1], env):
                return False
                
        # 5. Check Self Collision
        if SnakeRobotModel.check_self_collision(body):
            return False
            
        # 6. Check 2.5D Pitch
        for i in range(len(body)-1):
            x1, y1 = body[i]
            x2, y2 = body[i+1]
            z1 = env.get_elevation(x1, y1)
            z2 = env.get_elevation(x2, y2)
            dist_xy = math.hypot(x2 - x1, y2 - y1)
            if dist_xy > 1e-5:
                pitch_deg = math.degrees(math.atan2(z2 - z1, dist_xy))
                if abs(pitch_deg) > config.JOINT_LIMIT:
                    return False
            
        return True
