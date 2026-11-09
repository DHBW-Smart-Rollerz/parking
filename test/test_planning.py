import math
from pathlib import Path

import numpy as np
import pytest

from parking.perception.geometry import (
    Pose2D,
    fit_rectangle,
    min_distance_to_edges,
    rectangle_template,
)
from parking.perception.parking_spot import ParkingSpot
from parking.perception.tracker import Measurement, SpotTracker
from parking.pipeline import ParkingPipeline
from parking.planning.path import BACKWARD, FORWARD
from parking.planning.planner import (
    PlanningError,
    plan_parallel,
    plan_straight_arc_straight,
)
from parking.planning.vehicle import VehicleGeometry

CONFIG = Path(__file__).resolve().parents[1] / "config" / "planning.yaml"
VEHICLE = VehicleGeometry()
START = Pose2D(VEHICLE.rear_axle_x, 0, 0)


def assert_pose_close(actual: Pose2D, expected: Pose2D, tolerance=1e-6):
    assert actual.x == pytest.approx(expected.x, abs=tolerance)
    assert actual.y == pytest.approx(expected.y, abs=tolerance)
    assert math.cos(actual.yaw - expected.yaw) == pytest.approx(1, abs=tolerance)


# --- Geometry -------------------------------------------------------------------


def test_pose_compose_inverse():
    pose = Pose2D(100, -50, 0.7)
    other = Pose2D(30, 20, -0.2)
    assert_pose_close(pose.inverse().compose(pose.compose(other)), other)
    points = np.array([[1.0, 2.0], [-300.0, 40.0]])
    np.testing.assert_allclose(pose.to_local(pose.to_parent(points)), points)


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("shift", range(4))
def test_fit_rectangle_any_corner_order(shift, reverse):
    rng = np.random.default_rng(0)
    truth = Pose2D(500, 300, 1.2)
    corners = truth.to_parent(rectangle_template(490, 380))
    corners += rng.normal(0, 5, corners.shape)
    corners = np.roll(corners, shift, axis=0)
    if reverse:
        corners = corners[::-1]

    fit = fit_rectangle(corners, 490, 380)

    assert fit.pose.x == pytest.approx(truth.x, abs=10)
    assert fit.pose.y == pytest.approx(truth.y, abs=10)
    # The yaw of the long side is ambiguous by 180 deg
    assert abs(math.sin(fit.pose.yaw - truth.yaw)) < 0.03
    assert fit.residual < 15


def test_fit_rectangle_ignores_hidden_corner():
    truth = Pose2D(400, -350, 0.1)
    corners = truth.to_parent(rectangle_template(700, 300))
    corners[2] += [-200, 150]  # Guessed corner outside of the image
    visible = np.array([True, True, False, True])

    # Start with a wrong size: the size is measured from the two visible edges
    fit = fit_rectangle(corners, 750, 450, visible=visible)

    # The guessed corner keeps a small weight, which shifts the fit slightly
    assert_pose_close(fit.pose, truth, tolerance=2)
    assert fit.residual < 2
    assert fit.measured_length == pytest.approx(700)
    assert fit.measured_width == pytest.approx(300)
    # Without the visibility, the wrong corner shifts the rectangle
    assert fit_rectangle(corners, 750, 450).residual > 50


def test_fit_rectangle_refines_size():
    corners = Pose2D(0, 0, 0.3).to_parent(rectangle_template(600, 350))
    fit = fit_rectangle(corners, 490, 380, refine_iterations=2)
    assert fit.measured_length == pytest.approx(600, abs=1)
    assert fit.measured_width == pytest.approx(350, abs=1)
    assert fit.residual < 1


def test_min_distance_to_edges():
    square = rectangle_template(100, 100)
    assert min_distance_to_edges(np.array([[0.0, 0.0]]), square) == pytest.approx(50)
    assert min_distance_to_edges(np.array([[60.0, 0.0]]), square) == pytest.approx(-10)


# --- Vehicle and planner --------------------------------------------------------


def test_vehicle_geometry():
    assert VEHICLE.min_turning_radius == pytest.approx(260 / math.tan(math.radians(35)))
    # Minimum length of a parallel spot (Paromtchik & Laugier 1996)
    radius = VEHICLE.min_turning_radius
    expected = 70 + math.sqrt((radius + 100) ** 2 + 300**2 - (radius - 100) ** 2)
    assert VEHICLE.min_parallel_length() == pytest.approx(expected)
    footprint = VEHICLE.footprint(Pose2D())
    assert footprint[:, 0].max() == pytest.approx(300)
    assert footprint[:, 0].min() == pytest.approx(-70)


@pytest.mark.parametrize(
    ("goal", "direction"),
    [
        (Pose2D(700, 600, math.pi / 2), FORWARD),  # spot ahead on the left
        (Pose2D(200, 600, math.pi / 2), FORWARD),  # passed: back up first
        (Pose2D(800, 900, -math.pi / 2), BACKWARD),
        (Pose2D(900, 700, 1.2), FORWARD),  # not exactly perpendicular
    ],
)
def test_straight_arc_straight_reaches_goal(goal, direction):
    radius = VEHICLE.min_turning_radius
    path = plan_straight_arc_straight(START, goal, radius, direction)
    assert_pose_close(path.end_pose(), goal)
    assert all(segment.length >= 0 for segment in path.segments)
    assert abs(path.segments[1].curvature) == pytest.approx(1 / radius)
    # The sampled path ends in the goal as well
    np.testing.assert_allclose(path.sample(10.0)[-1, :2], goal.position, atol=1e-6)


def test_straight_arc_straight_too_close():
    with pytest.raises(PlanningError):
        plan_straight_arc_straight(
            START, Pose2D(500, 200, math.pi / 2), VEHICLE.min_turning_radius
        )


@pytest.mark.parametrize(
    "goal",
    [
        Pose2D(400, -350, 0),
        Pose2D(-200, -300, 0),
        Pose2D(400, -350, 0.1),  # vehicle not parallel to the spot
        Pose2D(600, -400, -0.1),
    ],
)
def test_parallel_reaches_goal(goal):
    path = plan_parallel(START, goal, VEHICLE.min_turning_radius, 0.3)
    assert_pose_close(path.end_pose(), goal)
    assert [s.direction for s in path.segments[1:]] == [BACKWARD, BACKWARD]
    # First steer to the right (towards the spot), then to the left
    assert path.segments[1].curvature < 0 < path.segments[2].curvature


def test_parallel_in_line_with_spot():
    # Vehicle already in line with the spot (small offset and angle): no S-curve
    # exists, the vehicle drives straight
    goal = Pose2D(300, 5, 0.05)
    path = plan_parallel(START, goal, VEHICLE.min_turning_radius, 0.3)
    assert len(path.segments) == 1
    assert np.linalg.norm(path.end_pose().position - goal.position) < 30
    with pytest.raises(PlanningError, match="> 4R"):
        plan_parallel(START, Pose2D(300, -1600, 0), VEHICLE.min_turning_radius, 0.3)


def test_parallel_rejects_heading_error():
    with pytest.raises(PlanningError):
        plan_parallel(START, Pose2D(400, -350, 0.5), VEHICLE.min_turning_radius, 0.2)


# --- Tracker --------------------------------------------------------------------


def measurement(x, y, status="free", *, valid=True):
    fit = fit_rectangle(
        Pose2D(x, y, 0).to_parent(rectangle_template(700, 300)), 700, 300
    )
    return Measurement("parallel", status, 0.9, fit, valid)


def test_tracker_associates_and_votes():
    tracker = SpotTracker(gate=200, min_hits=3, min_valid_hits=2, vote_window=5)
    statuses = ["free", "occupied", "free", "free"]
    for i, status in enumerate(statuses):
        tracks = tracker.update([measurement(1000 + 20 * i, -350, status)])
    assert len(tracks) == 1
    assert tracks[0].status == "free"
    assert tracks[0].hits == len(statuses)
    assert tracker.confirmed(tracks[0])


def test_tracker_keeps_pose_of_valid_measurements():
    tracker = SpotTracker(smoothing=0.5)
    tracker.update([measurement(1000, -350)])
    tracks = tracker.update([measurement(1100, -350, valid=False)])
    assert tracks[0].pose.x == pytest.approx(1000)
    assert tracks[0].hits == 2


@pytest.mark.parametrize("predict_motion", [True, False])
def test_tracker_predicts_common_motion(predict_motion):
    # Without a pose, the spots move through the vehicle frame
    tracker = SpotTracker(gate=250, smoothing=0.4, predict_motion=predict_motion)
    for i in range(8):
        tracks = tracker.update([measurement(2000 - 150 * i, -350)])
    # With the prediction the spot keeps its first track, without it is lost
    assert (tracks[-1].id == 0) == predict_motion


def test_target_hysteresis():
    pipeline = ParkingPipeline.from_file(CONFIG)
    first = Pose2D(900, -380, 0).to_parent(rectangle_template(700, 300))
    second = Pose2D(1700, -380, 0).to_parent(rectangle_template(700, 300))
    for _ in range(3):
        result = pipeline.update([ParkingSpot(second, 0.9, "Parking_Parallel_Frei")])
    target_id = result.target.track.id
    # A spot with a shorter path appears, the target is kept
    spots = [ParkingSpot(c, 0.9, "Parking_Parallel_Frei") for c in (first, second)]
    for _ in range(3):
        result = pipeline.update(spots)
    assert result.target.track.id == target_id


def test_tracker_deletes_lost_tracks():
    tracker = SpotTracker(max_misses=2)
    tracker.update([measurement(1000, -350)])
    for _ in range(3):
        tracks = tracker.update([])
    assert tracks == []


# --- Pipeline -------------------------------------------------------------------


def test_pipeline_plans_into_free_parallel_spot():
    pipeline = ParkingPipeline.from_file(CONFIG)
    spot = Pose2D(700, -380, 0.02)
    corners = spot.to_parent(rectangle_template(700, 300))
    for _ in range(3):
        result = pipeline.update([ParkingSpot(corners, 0.9, "Parking_Parallel_Frei")])

    assert result.target is not None
    assert result.target.track.kind == "parallel"
    assert result.target.clearance > 0
    assert_pose_close(result.target.path.end_pose(), result.target.goal)
    # The vehicle is centered in the spot
    footprint_center = result.target.footprint.mean(axis=0)
    np.testing.assert_allclose(footprint_center, spot.position, atol=1)


@pytest.mark.parametrize(("length", "valid"), [(450, True), (1100, True), (350, False)])
def test_pipeline_parallel_spots_have_variable_length(length, valid):
    # Regulations 5.4.1: parallel spots are 300 mm deep, the length varies, gaps
    # shorter than 400 mm are no spots
    pipeline = ParkingPipeline.from_file(CONFIG)
    corners = Pose2D(700, -380, 0).to_parent(rectangle_template(length, 300))
    measurement = pipeline.measure(ParkingSpot(corners, 0.9, "Parking_Parallel_Frei"))
    assert measurement.valid == valid
    assert measurement.fit.measured_length == pytest.approx(length)


def test_pipeline_rejects_wrong_perpendicular_size():
    pipeline = ParkingPipeline.from_file(CONFIG)
    corners = rectangle_template(800, 350)  # nominal 500 x 350 (regulations A.5.2)
    measurement = pipeline.measure(ParkingSpot(corners, 0.9, "Parking_90_free"))
    assert not measurement.valid
    assert measurement.reason.startswith("wrong size")


def test_pipeline_ignores_occupied_spot():
    pipeline = ParkingPipeline.from_file(CONFIG)
    corners = Pose2D(900, 700, math.pi / 2).to_parent(rectangle_template(500, 350))
    for _ in range(3):
        result = pipeline.update([ParkingSpot(corners, 0.9, "Parking_90_besetzt")])
    assert result.target is None
    assert result.candidates[0].reason == "occupied"


def test_pipeline_tracks_in_global_frame():
    pipeline = ParkingPipeline.from_file(CONFIG)
    spot_global = Pose2D(2000, 1000, math.pi / 2)
    for i in range(4):
        vehicle = Pose2D(1000 + 50 * i, 0, 0)
        spot = vehicle.inverse().compose(spot_global)
        corners = spot.to_parent(rectangle_template(500, 350))
        result = pipeline.update(
            [ParkingSpot(corners, 0.9, "Parking_90_free")], vehicle
        )

    assert result.frame == "global"
    assert len(result.candidates) == 1
    assert result.target is not None
    goal_global = vehicle.compose(result.target.goal)
    assert goal_global.x == pytest.approx(spot_global.x, abs=1)
