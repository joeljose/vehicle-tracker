"""Unit tests for DirectionStateMachine and nearest_arm (pure Python)."""

from backend.pipeline.direction import (
    DirectionStateMachine,
    LineSeg,
    TrackState,
    nearest_arm,
)


# --- nearest_arm tests ---

# Simple arms: lines pointing in cardinal directions from origin
ARMS = {
    "north": LineSeg(label="North", start=(500, 200), end=(500, 100)),  # points up (y decreasing)
    "south": LineSeg(label="South", start=(500, 200), end=(500, 300)),  # points down
    "east": LineSeg(label="East", start=(500, 200), end=(600, 200)),    # points right
    "west": LineSeg(label="West", start=(500, 200), end=(400, 200)),    # points left
}


def test_nearest_arm_heading_east():
    """Trajectory moving right should match east arm."""
    # Moving from left to right
    traj = [(100 + i * 10, 200, i) for i in range(20)]
    result = nearest_arm(traj, ARMS, use_start=False)
    assert result == "east"


def test_nearest_arm_heading_west():
    traj = [(500 - i * 10, 200, i) for i in range(20)]
    result = nearest_arm(traj, ARMS, use_start=False)
    assert result == "west"


def test_nearest_arm_heading_north():
    # Moving upward (y decreasing)
    traj = [(500, 300 - i * 10, i) for i in range(20)]
    result = nearest_arm(traj, ARMS, use_start=False)
    assert result == "north"


def test_nearest_arm_heading_south():
    traj = [(500, 100 + i * 10, i) for i in range(20)]
    result = nearest_arm(traj, ARMS, use_start=False)
    assert result == "south"


def test_nearest_arm_use_start():
    """use_start=True uses the first centroids for direction."""
    # First 10 points go east, last 10 go south
    traj = [(100 + i * 10, 200, i) for i in range(10)]
    traj += [(200, 200 + i * 10, 10 + i) for i in range(10)]
    assert nearest_arm(traj, ARMS, use_start=True) == "east"
    assert nearest_arm(traj, ARMS, use_start=False) == "south"


def test_nearest_arm_too_short():
    traj = [(100, 200, 0), (110, 200, 1)]
    assert nearest_arm(traj, ARMS, use_start=True) is None


def test_nearest_arm_stationary():
    """Stationary trajectory returns None."""
    traj = [(500, 200, i) for i in range(20)]
    assert nearest_arm(traj, ARMS, use_start=True) is None


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

    # Trajectory heading south (y increasing)
    traj = [(500, 100 + i * 10, i) for i in range(20)]
    result = dsm.on_track_lost(traj, ARMS)
    assert result is not None
    assert result["entry_arm"] == "north"
    assert result["exit_arm"] == "south"
    assert result["method"] == "inferred"


def test_dsm_inferred_both_on_loss():
    """No crossings, track lost -> infer both entry and exit."""
    dsm = DirectionStateMachine()

    # Trajectory: enters from north (heading south initially), exits east (heading east at end)
    # First 10 points go south, last 10 points go east
    traj = [(500, 100 + i * 10, i) for i in range(10)]
    traj += [(500 + i * 10, 200, 10 + i) for i in range(10)]
    result = dsm.on_track_lost(traj, ARMS)
    assert result is not None
    assert result["method"] == "inferred"
    assert result["entry_arm"] == "south"  # heading south at start
    assert result["exit_arm"] == "east"    # heading east at end


def test_dsm_no_inference_same_arm():
    """If inferred entry == exit, no alert (wouldn't make sense)."""
    dsm = DirectionStateMachine()

    # Trajectory barely moves east
    traj = [(500 + i, 200, i) for i in range(20)]
    result = dsm.on_track_lost(traj, ARMS)
    # Both start and end point east -> entry==exit -> no alert
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


def test_dsm_exit_without_entry_same_heading():
    """Exit in UNKNOWN where heading matches exit arm -> no alert (can't distinguish)."""
    dsm = DirectionStateMachine()

    # Heading south, exits south -> inferred entry = south = exit -> None
    traj = [(500, 100 + i * 10, i) for i in range(20)]
    result = dsm.on_crossing(
        "south", "South", "exit", trajectory=traj, arms=ARMS,
    )
    assert result is None


def test_dsm_exit_without_entry_curved_trajectory():
    """Exit in UNKNOWN with curved trajectory can infer different entry."""
    dsm = DirectionStateMachine()

    # Start heading east, exit south
    traj = [(100 + i * 10, 200, i) for i in range(10)]
    result = dsm.on_crossing(
        "south", "South", "exit", trajectory=traj, arms=ARMS,
    )
    assert result is not None
    assert result["exit_arm"] == "south"
    assert result["entry_arm"] == "east"  # heading east at start
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
