"""Unit tests for DirectionStateMachine and nearest_arm (pure Python)."""

from backend.pipeline.direction import (
    DirectionStateMachine,
    LineSeg,
    TrackState,
    nearest_arm,
)


# --- nearest_arm tests ---

# Junction border lines — drawn along each edge of a rectangular ROI.
# Proximity-based inference: the closest line to the trajectory start/end
# determines the entry/exit arm.
ROI_CENTROID = (500, 200)
ARMS = {
    "north": LineSeg(label="North", start=(400, 100), end=(600, 100)),  # top edge, y=100
    "south": LineSeg(label="South", start=(400, 300), end=(600, 300)),  # bottom edge, y=300
    "east": LineSeg(label="East", start=(600, 100), end=(600, 300)),    # right edge, x=600
    "west": LineSeg(label="West", start=(400, 100), end=(400, 300)),    # left edge, x=400
}


def test_nearest_arm_exit_south():
    """Trajectory ending near south edge → exit = south."""
    # Starts near north (y=110), ends near south (y=290)
    traj = [(500, 110 + i * 10, i) for i in range(20)]
    assert nearest_arm(traj, ARMS, use_start=False) == "south"


def test_nearest_arm_exit_north():
    """Trajectory ending near north edge → exit = north."""
    traj = [(500, 290 - i * 10, i) for i in range(20)]
    assert nearest_arm(traj, ARMS, use_start=False) == "north"


def test_nearest_arm_exit_east():
    """Trajectory ending near east edge → exit = east."""
    traj = [(410 + i * 10, 200, i) for i in range(20)]
    assert nearest_arm(traj, ARMS, use_start=False) == "east"


def test_nearest_arm_exit_west():
    """Trajectory ending near west edge → exit = west."""
    traj = [(590 - i * 10, 200, i) for i in range(20)]
    assert nearest_arm(traj, ARMS, use_start=False) == "west"


def test_nearest_arm_entry_and_exit():
    """use_start=True → entry (closest to start), use_start=False → exit (closest to end)."""
    # Starts near west edge (x=410), ends near south edge (y=290)
    traj = [(410 + i * 5, 200 + i * 5, i) for i in range(20)]
    assert nearest_arm(traj, ARMS, use_start=True) == "west"
    assert nearest_arm(traj, ARMS, use_start=False) == "south"


def test_nearest_arm_too_short():
    traj = [(500, 110, 0), (500, 120, 1)]
    assert nearest_arm(traj, ARMS, use_start=True) is None


def test_nearest_arm_oblique_entry():
    """Vehicle enters from north but heads diagonally — proximity still picks north."""
    # Starts near north edge at (450, 105), moves diagonally
    traj = [(450 + i * 8, 105 + i * 10, i) for i in range(20)]
    assert nearest_arm(traj, ARMS, use_start=True) == "north"


# --- DirectionStateMachine tests ---


def test_dsm_confirmed_transit():
    """Entry then exit produces confirmed transit alert."""
    dsm = DirectionStateMachine()
    assert dsm.state == TrackState.UNKNOWN

    result = dsm.on_crossing("north", "North", "entry")
    assert result is None
    assert dsm.state == TrackState.IN_TRANSIT
    assert dsm.entry_arm == "north"

    result = dsm.on_crossing("south", "South", "exit")
    assert result is not None
    assert result["entry_arm"] == "north"
    assert result["exit_arm"] == "south"
    assert result["method"] == "confirmed"
    assert result["was_stagnant"] is False


def test_dsm_inferred_exit_on_loss():
    """Entry crossed, track lost -> infer exit from trajectory."""
    dsm = DirectionStateMachine()
    dsm.on_crossing("north", "North", "entry")

    # Starts near north (y=110), ends near south (y=290) → exit = south
    traj = [(500, 110 + i * 10, i) for i in range(20)]
    result = dsm.on_track_lost(traj, ARMS, roi_centroid=ROI_CENTROID)
    assert result is not None
    assert result["entry_arm"] == "north"
    assert result["exit_arm"] == "south"
    assert result["method"] == "inferred"


def test_dsm_inferred_both_on_loss():
    """No crossings, track lost -> infer both entry and exit."""
    dsm = DirectionStateMachine()

    # Starts near north (y=110), ends near east (x=590)
    traj = [(450, 110 + i * 5, i) for i in range(10)]
    traj += [(450 + i * 15, 160, 10 + i) for i in range(10)]
    result = dsm.on_track_lost(traj, ARMS, roi_centroid=ROI_CENTROID)
    assert result is not None
    assert result["method"] == "inferred"
    assert result["entry_arm"] == "north"  # starts near north edge
    assert result["exit_arm"] == "east"    # ends near east edge


def test_dsm_no_inference_same_arm():
    """Trajectory starts and ends near the same arm -> no alert."""
    dsm = DirectionStateMachine()

    # Starts near north edge, loops back to north edge
    traj = [(500, 110 + i * 10, i) for i in range(10)]         # moves south
    traj += [(500, 210 - i * 10, 10 + i) for i in range(10)]   # back toward north
    result = dsm.on_track_lost(traj, ARMS, roi_centroid=ROI_CENTROID)
    # entry = north, exit = north -> same arm -> None
    assert result is None


def test_dsm_no_inference_short_trajectory():
    """Very short trajectory can't be inferred."""
    dsm = DirectionStateMachine()
    traj = [(500, 200, 0), (510, 200, 1)]
    result = dsm.on_track_lost(traj, ARMS)
    assert result is None


def test_dsm_exit_without_entry_ignored():
    """Exit crossing in UNKNOWN state is ignored (no entry yet)."""
    dsm = DirectionStateMachine()
    result = dsm.on_crossing("south", "South", "exit")
    assert result is None
    assert dsm.state == TrackState.UNKNOWN


def test_dsm_duplicate_entry_ignored():
    """Second entry crossing after already entered is ignored."""
    dsm = DirectionStateMachine()
    dsm.on_crossing("north", "North", "entry")
    assert dsm.state == TrackState.IN_TRANSIT

    # Second entry from a different arm — ignored (already in transit)
    result = dsm.on_crossing("east", "East", "entry")
    assert result is None
    assert dsm.entry_arm == "north"  # unchanged


def _simulate_stopped(dsm, num_frames, x=500, y=200, start_frame=0):
    """Helper: simulate a stopped vehicle, calling check_stagnant each frame."""
    window = dsm._motion_check_window
    for frame in range(num_frames):
        f = start_frame + frame
        start = max(0, f - window + 1)
        traj = [(x, y, t) for t in range(start, f + 1)]
        result = dsm.check_stagnant(traj)
        if result:
            return f
    return None


def test_dsm_stagnant_detection():
    """Vehicle stopped for stagnant_threshold triggers STAGNANT."""
    dsm = DirectionStateMachine(stagnant_threshold_sec=5.0, fps=10.0)
    # stagnant_window = 50 frames, motion_check_window = 30 frames
    # First detection at frame ~29, stagnant at frame ~79
    dsm.on_crossing("north", "North", "entry")

    triggered_frame = _simulate_stopped(dsm, 100)
    assert triggered_frame is not None
    assert dsm.state == TrackState.STAGNANT
    assert dsm.was_stagnant is True


def test_dsm_stagnant_then_exit():
    """Stagnant vehicle that eventually exits has was_stagnant=True."""
    dsm = DirectionStateMachine(stagnant_threshold_sec=5.0, fps=10.0)
    dsm.on_crossing("north", "North", "entry")

    _simulate_stopped(dsm, 100)
    assert dsm.state == TrackState.STAGNANT

    result = dsm.on_crossing("south", "South", "exit")
    assert result is not None
    assert result["was_stagnant"] is True
    assert result["method"] == "confirmed"


def test_dsm_not_stagnant_if_moving():
    """Moving vehicle doesn't trigger stagnant."""
    dsm = DirectionStateMachine(stagnant_threshold_sec=5.0, fps=10.0)
    dsm.on_crossing("north", "North", "entry")

    traj = [(500 + i * 5, 200, i) for i in range(31)]
    became_stagnant = dsm.check_stagnant(traj)
    assert became_stagnant is False
    assert dsm.state == TrackState.IN_TRANSIT


def test_dsm_stagnant_in_unknown_state():
    """Vehicle that never crossed entry can still become stagnant."""
    dsm = DirectionStateMachine(stagnant_threshold_sec=5.0, fps=10.0)
    assert dsm.state == TrackState.UNKNOWN

    triggered_frame = _simulate_stopped(dsm, 100)
    assert triggered_frame is not None
    assert dsm.state == TrackState.STAGNANT


def test_dsm_exit_without_entry_too_short():
    """Exit in UNKNOWN with trajectory too short to infer entry -> no alert."""
    dsm = DirectionStateMachine()

    # Only 3 points — nearest_arm needs ≥4 to infer
    traj = [(500, 280, 0), (500, 290, 1), (500, 300, 2)]
    result = dsm.on_crossing(
        "south", "South", "exit", trajectory=traj, arms=ARMS,
        roi_centroid=ROI_CENTROID,
    )
    assert result is None


def test_dsm_exit_without_entry_inferred():
    """Exit in UNKNOWN — infer entry from trajectory start proximity."""
    dsm = DirectionStateMachine()

    # Starts near north edge (y=110), exits south
    traj = [(500, 110 + i * 10, i) for i in range(10)]
    result = dsm.on_crossing(
        "south", "South", "exit", trajectory=traj, arms=ARMS,
        roi_centroid=ROI_CENTROID,
    )
    assert result is not None
    assert result["exit_arm"] == "south"
    assert result["entry_arm"] == "north"  # starts near north edge
    assert result["method"] == "inferred"


def test_dsm_stagnant_resumes_to_unknown():
    """Stagnant vehicle in UNKNOWN that starts moving goes back to UNKNOWN."""
    dsm = DirectionStateMachine(stagnant_threshold_sec=5.0, fps=10.0)

    _simulate_stopped(dsm, 100)
    assert dsm.state == TrackState.STAGNANT

    # Starts moving — displacement is large
    moving_traj = [(500 + i * 20, 200, 100 + i) for i in range(31)]
    dsm.check_stagnant(moving_traj)
    assert dsm.state == TrackState.UNKNOWN  # back to UNKNOWN, never had entry
