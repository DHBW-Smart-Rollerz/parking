"""Classic geometric planners for the two parking maneuvers of the track.

Both planners only use straight lines and arcs with a constant radius R >= R_min,
which can be driven directly with a constant steering angle:

- Perpendicular spots (left side): straight - arc - straight. The arc is tangent to
  the line of the current heading and to the center line of the parking spot (the
  shortest path of Dubins 1957 for this case). The first straight may be driven in
  the opposite direction (cusp), e.g. to back up when the spot was passed.
- Parallel spots (right side): two arcs with the same radius and opposite curvature
  driven backwards (S-curve, Paromtchik & Laugier 1996; "circle-circle" planner in
  Vorobieva et al. 2015). The vehicle first drives straight to the start point of the
  S-curve. A small angle between vehicle and spot is compensated by different
  turning angles of the two arcs.
"""

from __future__ import annotations

import math

import numpy as np

from parking.perception.geometry import Pose2D, normalize_angle
from parking.planning.path import BACKWARD, FORWARD, Path, PathSegment

# Maximum lateral distance to a parallel spot that is driven straight (no S-curve)
STRAIGHT_TOLERANCE = 30.0


class PlanningError(Exception):
    """No path was found, the message contains the reason."""


def plan_straight_arc_straight(
    start: Pose2D, goal: Pose2D, radius: float, direction: int = FORWARD
) -> Path:
    """Plan straight - arc - straight from the start into the goal pose.

    The arc is driven in the given direction and ends tangent to the goal heading,
    then the vehicle drives straight into the goal.

    Args:
        start: Pose of the rear axle center.
        goal: Goal pose of the rear axle center.
        radius: Radius of the arc in mm.
        direction: FORWARD or BACKWARD driving during the arc and the last straight.

    Returns:
        The path.

    Raises:
        PlanningError: If the goal cannot be reached this way.
    """
    # Headings of the motion (opposite to the vehicle heading when driving backwards)
    offset = 0.0 if direction == FORWARD else math.pi
    start_motion = start.yaw + offset
    goal_motion = goal.yaw + offset
    turn_angle = normalize_angle(goal_motion - start_motion)
    if abs(turn_angle) < math.radians(1) or abs(turn_angle) > math.radians(179):
        raise PlanningError("goal heading parallel to the vehicle")
    turn = math.copysign(1.0, turn_angle)

    start_heading = np.array([math.cos(start_motion), math.sin(start_motion)])
    goal_heading = np.array([math.cos(goal_motion), math.sin(goal_motion)])
    start_normal = np.array([-start_heading[1], start_heading[0]])
    goal_normal = np.array([-goal_heading[1], goal_heading[0]])

    # The arc center has the distance R to both lines:
    # start + t * h0 + turn * R * n0 = goal - u * hg + turn * R * ng
    first, last = np.linalg.solve(
        np.column_stack([start_heading, goal_heading]),
        goal.position - start.position + turn * radius * (goal_normal - start_normal),
    )
    if last < 0:
        raise PlanningError(f"too close to the spot for R={radius:.0f}")

    return Path(
        start,
        [
            PathSegment(abs(first), 0.0, direction if first >= 0 else -direction),
            PathSegment(radius * abs(turn_angle), direction * turn / radius, direction),
            PathSegment(last, 0.0, direction),
        ],
        name="perpendicular " + ("forward" if direction == FORWARD else "backward"),
    )


def plan_parallel(
    start: Pose2D, goal: Pose2D, radius: float, max_heading_error: float
) -> Path:
    """Plan the backward S-curve into a parallel spot.

    The vehicle drives straight to the start point of the S-curve, then backwards on
    an arc that turns the rear towards the spot and on a second arc with the opposite
    curvature into the goal. The arcs are constructed backwards from the goal in the
    goal frame: the last arc lies on the circle with the center C1 = (0, s R) (s = +1
    for a spot on the right). The circle of the first arc touches it, so its center
    is C2 = C1 + 2R (sin(a), -s cos(a)) for the turning angle a of the last arc, and
    it must have the distance R to the line on which the vehicle approaches. This
    gives a in closed form. With the vehicle parallel to the spot, both arcs have
    the same angle a = acos(1 - dy / 2R) (Paromtchik & Laugier 1996).

    Args:
        start: Pose of the rear axle center.
        goal: Goal pose of the rear axle center (heading = driving direction).
        radius: Radius of the arcs in mm.
        max_heading_error: Maximum angle between vehicle and spot in rad.

    Returns:
        The path.

    Raises:
        PlanningError: If the goal cannot be reached this way.
    """
    local = goal.inverse().compose(start)
    heading = local.yaw
    if abs(heading) > max_heading_error:
        raise PlanningError(
            f"vehicle not parallel to the spot ({math.degrees(heading):.0f} deg)"
        )

    side = math.copysign(1.0, local.y)
    approach_normal = np.array([-math.sin(heading), math.cos(heading)])
    last_center = np.array([0.0, side * radius])
    # Center of the first arc on the right (s = +1) of the approach line:
    # n * (C2 - P0) = -s R  ->  cos(a - s * heading) = -s * k
    k = (-side * radius - approach_normal @ (last_center - local.position)) / (
        2 * radius
    )
    if -side * k < -1:
        raise PlanningError(f"lateral distance {abs(local.y):.0f} mm > 4R")
    if -side * k > 1:
        # The vehicle is already in line with the spot, no S-curve possible
        if abs(local.y) > STRAIGHT_TOLERANCE:
            raise PlanningError("lateral distance too small for an S-curve")
        distance = float(local.heading @ -local.position)
        return Path(
            start,
            [PathSegment(abs(distance), 0.0, FORWARD if distance >= 0 else BACKWARD)],
            name="parallel straight",
        )
    last_angle = side * heading + math.acos(-side * k)
    first_angle = last_angle - side * heading
    if first_angle < 0 or last_angle > math.pi / 2:
        raise PlanningError("no S-curve with this heading")

    # Start point of the S-curve: from the goal backwards along both arcs
    middle = PathSegment(radius * last_angle, side / radius).end_pose(Pose2D())
    curve_start = PathSegment(radius * first_angle, -side / radius).end_pose(middle)
    approach = float(local.heading @ (curve_start.position - local.position))
    return Path(
        start,
        [
            PathSegment(abs(approach), 0.0, FORWARD if approach >= 0 else BACKWARD),
            PathSegment(radius * first_angle, -side / radius, BACKWARD),
            PathSegment(radius * last_angle, side / radius, BACKWARD),
        ],
        name="parallel backward",
    )
