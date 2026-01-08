"""
SELF-COLLISION DETECTION MODULE

This module adds self-collision checking to prevent the snake from
hitting itself when sampling configurations in C-space.
"""

import math
import numpy as np


def line_segment_intersection(p1, p2, p3, p4):
    """
    Check if line segment (p1, p2) intersects with line segment (p3, p4).
    
    Uses parametric line intersection test.
    Returns True if segments intersect (excluding endpoints touching).
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    
    # Direction vectors
    dx1 = x2 - x1
    dy1 = y2 - y1
    dx2 = x4 - x3
    dy2 = y4 - y3
    
    # Calculate determinant
    det = dx1 * dy2 - dy1 * dx2
    
    # Lines are parallel if determinant is zero
    if abs(det) < 1e-10:
        return False
    
    # Calculate parameters
    t = ((x3 - x1) * dy2 - (y3 - y1) * dx2) / det
    u = ((x3 - x1) * dy1 - (y3 - y1) * dx1) / det
    
    # Check if intersection is within both segments
    # Use small epsilon to avoid adjacent segments touching at joints
    epsilon = 0.01
    if (epsilon < t < 1.0 - epsilon) and (epsilon < u < 1.0 - epsilon):
        return True
    
    return False


def check_self_collision(body_points, min_distance=1.0):
    """
    Check if snake body collides with itself.
    
    Args:
        body_points: List of (x, y) tuples representing body positions
        min_distance: Minimum allowed distance between non-adjacent segments
    
    Returns:
        True if self-collision detected, False otherwise
    
    Method:
        1. Check line-line intersections between non-adjacent segments
        2. Check point-to-segment distances for near-miss detection
    """
    n = len(body_points)
    
    # Check all pairs of segments (skip adjacent segments)
    for i in range(n - 1):
        seg1_start = body_points[i]
        seg1_end = body_points[i + 1]
        
        # Only check against segments that are at least 2 segments away
        # (adjacent segments naturally touch at joints)
        for j in range(i + 2, n - 1):
            seg2_start = body_points[j]
            seg2_end = body_points[j + 1]
            
            # Check line-line intersection
            if line_segment_intersection(seg1_start, seg1_end, seg2_start, seg2_end):
                return True
            
            # Also check minimum distance (for near-miss detection)
            if min_distance > 0:
                dist = segment_to_segment_distance(
                    seg1_start, seg1_end, seg2_start, seg2_end
                )
                if dist < min_distance:
                    return True
    
    return False


def segment_to_segment_distance(p1, p2, p3, p4):
    """
    Calculate minimum distance between two line segments.
    
    Args:
        p1, p2: Endpoints of first segment
        p3, p4: Endpoints of second segment
    
    Returns:
        Minimum distance between the two segments
    """
    # Check all point-to-segment distances
    distances = [
        point_to_segment_distance(p1, p3, p4),
        point_to_segment_distance(p2, p3, p4),
        point_to_segment_distance(p3, p1, p2),
        point_to_segment_distance(p4, p1, p2),
    ]
    
    return min(distances)


def point_to_segment_distance(point, seg_start, seg_end):
    """
    Calculate minimum distance from point to line segment.
    
    Args:
        point: (x, y) tuple
        seg_start, seg_end: Segment endpoints
    
    Returns:
        Minimum distance from point to segment
    """
    px, py = point
    sx, sy = seg_start
    ex, ey = seg_end
    
    # Vector from segment start to end
    dx = ex - sx
    dy = ey - sy
    
    # If segment has zero length
    if dx == 0 and dy == 0:
        return math.hypot(px - sx, py - sy)
    
    # Parameter t of closest point on line
    t = ((px - sx) * dx + (py - sy) * dy) / (dx * dx + dy * dy)
    
    # Clamp t to [0, 1] to stay on segment
    t = max(0, min(1, t))
    
    # Closest point on segment
    closest_x = sx + t * dx
    closest_y = sy + t * dy
    
    # Distance from point to closest point
    return math.hypot(px - closest_x, py - closest_y)


def visualize_self_collision(body_points):
    """
    Debug function to visualize which segments are colliding.
    
    Returns list of colliding segment pairs.
    """
    n = len(body_points)
    collisions = []
    
    for i in range(n - 1):
        for j in range(i + 2, n - 1):
            seg1 = (body_points[i], body_points[i + 1])
            seg2 = (body_points[j], body_points[j + 1])
            
            if line_segment_intersection(*seg1, *seg2):
                collisions.append((i, j, "intersection"))
            
            dist = segment_to_segment_distance(*seg1, *seg2)
            if dist < 1.0:
                collisions.append((i, j, f"near-miss: {dist:.2f}"))
    
    return collisions


# ============================================================================
# INTEGRATION EXAMPLE
# ============================================================================

def is_valid_configuration_with_self_collision(node, env, snake_width=4.0):
    """
    Enhanced collision checking that includes self-collision detection.
    
    Args:
        node: RRT node with state and yaw
        env: Environment object with collision checking
        snake_width: Width of snake body for minimum distance check
    
    Returns:
        True if configuration is valid (no collisions), False otherwise
    """
    from ConfigurationSpace_5Segments import get_snake_body
    
    body = get_snake_body(node.state, yaw_override=node.yaw)
    
    # 1. Check boundary collision
    for (bx, by) in body:
        if not (0 <= bx < env.width and 0 <= by < env.height):
            return False
    
    # 2. Check environment collision (walls, obstacles)
    for i in range(len(body) - 1):
        if env.check_line_collision(body[i], body[i + 1]):
            return False
    
    # 3. NEW: Check self-collision
    min_clearance = snake_width / 2.0  # Half width as minimum distance
    if check_self_collision(body, min_distance=min_clearance):
        return False
    
    return True


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    print("Testing Self-Collision Detection")
    print("=" * 60)
    
    # Test Case 1: Straight snake (no collision)
    straight_snake = [
        (0, 0), (3, 0), (6, 0), (9, 0), (12, 0), (15, 0)
    ]
    result1 = check_self_collision(straight_snake)
    print(f"\n1. Straight snake: {'COLLISION ❌' if result1 else 'OK ✓'}")
    
    # Test Case 2: U-shaped snake (no collision, but close)
    u_snake = [
        (0, 0), (3, 0), (6, 0), (9, 0), (9, 3), (6, 3)
    ]
    result2 = check_self_collision(u_snake, min_distance=1.0)
    print(f"2. U-shaped snake: {'COLLISION ❌' if result2 else 'OK ✓'}")
    
    # Test Case 3: Self-intersecting snake (collision!)
    crossing_snake = [
        (0, 0), (6, 0), (6, 6), (0, 6), (3, 3), (3, 0)
    ]
    result3 = check_self_collision(crossing_snake)
    print(f"3. Crossing snake: {'COLLISION ❌' if result3 else 'OK ✓'}")
    if result3:
        collisions = visualize_self_collision(crossing_snake)
        print(f"   Detected collisions: {collisions}")
    
    # Test Case 4: Tight spiral (should detect near-miss)
    spiral_snake = [
        (5, 5), (8, 5), (8, 8), (5, 8), (5, 6), (7, 6)
    ]
    result4 = check_self_collision(spiral_snake, min_distance=1.5)
    print(f"4. Tight spiral: {'COLLISION ❌' if result4 else 'OK ✓'}")
    if result4:
        collisions = visualize_self_collision(spiral_snake)
        print(f"   Detected near-misses: {collisions}")
    
    # Test Case 5: Loop back (clear self-intersection)
    loop_snake = [
        (0, 0), (3, 0), (6, 0), (6, 3), (3, 3), (0, 3), (0, 1.5)
    ]
    result5 = check_self_collision(loop_snake)
    print(f"5. Loop-back snake: {'COLLISION ❌' if result5 else 'OK ✓'}")
    
    print("\n" + "=" * 60)
    print("Self-collision detection module ready for integration!")