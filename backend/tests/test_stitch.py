"""Tests for TrackStitcher — proximity-based track stitching."""

from backend.pipeline.direction import DirectionStateMachine, TrackState
from backend.pipeline.stitch import TrackStitcher
from backend.pipeline.trajectory import TrajectoryBuffer


def _make_trajectory(points: list[tuple[int, int, int]]) -> TrajectoryBuffer:
    """Build a TrajectoryBuffer from a list of (x, y, frame) tuples."""
    buf = TrajectoryBuffer()
    for x, y, f in points:
        buf.append(x, y, f)
    return buf


class TestLostTrackBuffer:
    """Test buffering and expiry of lost tracks."""

    def test_buffer_lost_track(self):
        stitcher = TrackStitcher()
        traj = _make_trajectory([(100, 200, 1), (105, 200, 2)])
        dsm = DirectionStateMachine()
        stitcher.on_track_lost(track_id=1, trajectory=traj, dsm=dsm, frame_number=10)
        assert 1 in stitcher.lost_tracks

    def test_expire_old_tracks(self):
        stitcher = TrackStitcher(max_age_frames=60)
        traj = _make_trajectory([(100, 200, 1)])
        dsm = DirectionStateMachine()
        stitcher.on_track_lost(track_id=1, trajectory=traj, dsm=dsm, frame_number=10)
        # Not expired at frame 69 (59 frames later)
        expired = stitcher.expire(frame_number=69)
        assert 1 in stitcher.lost_tracks
        assert len(expired) == 0
        # Expired at frame 71 (61 frames later)
        expired = stitcher.expire(frame_number=71)
        assert 1 not in stitcher.lost_tracks
        assert len(expired) == 1
        assert expired[0][0] == 1  # track_id

    def test_expire_keeps_recent(self):
        stitcher = TrackStitcher(max_age_frames=60)
        traj1 = _make_trajectory([(100, 200, 1)])
        traj2 = _make_trajectory([(300, 400, 50)])
        dsm1 = DirectionStateMachine()
        dsm2 = DirectionStateMachine()
        stitcher.on_track_lost(track_id=1, trajectory=traj1, dsm=dsm1, frame_number=10)
        stitcher.on_track_lost(track_id=2, trajectory=traj2, dsm=dsm2, frame_number=50)
        expired = stitcher.expire(frame_number=71)
        assert 1 not in stitcher.lost_tracks
        assert 2 in stitcher.lost_tracks
        assert len(expired) == 1


class TestProximityMatching:
    """Test matching new tracks to lost tracks by predicted position."""

    def test_match_stationary(self):
        """New track at same position as recently lost stationary track matches."""
        stitcher = TrackStitcher(max_distance=100.0)
        traj = _make_trajectory([(200, 300, 8), (200, 300, 9), (200, 300, 10)])
        dsm = DirectionStateMachine()
        stitcher.on_track_lost(track_id=1, trajectory=traj, dsm=dsm, frame_number=10)
        match = stitcher.find_match(position=(205, 305), frame_number=12)
        assert match is not None
        assert match["lost_track_id"] == 1

    def test_match_moving(self):
        """Predicted position accounts for velocity."""
        stitcher = TrackStitcher(max_distance=100.0)
        # Moving 10px/frame to the right
        traj = _make_trajectory([(100, 200, 8), (110, 200, 9), (120, 200, 10)])
        dsm = DirectionStateMachine()
        stitcher.on_track_lost(track_id=1, trajectory=traj, dsm=dsm, frame_number=10)
        # 5 frames later: predicted position = (120 + 10*5, 200) = (170, 200)
        match = stitcher.find_match(position=(175, 200), frame_number=15)
        assert match is not None
        assert match["lost_track_id"] == 1

    def test_no_match_too_far(self):
        stitcher = TrackStitcher(max_distance=100.0)
        traj = _make_trajectory([(100, 200, 10)])
        dsm = DirectionStateMachine()
        stitcher.on_track_lost(track_id=1, trajectory=traj, dsm=dsm, frame_number=10)
        match = stitcher.find_match(position=(500, 500), frame_number=12)
        assert match is None

    def test_no_match_expired(self):
        stitcher = TrackStitcher(max_age_frames=60, max_distance=100.0)
        traj = _make_trajectory([(100, 200, 10)])
        dsm = DirectionStateMachine()
        stitcher.on_track_lost(track_id=1, trajectory=traj, dsm=dsm, frame_number=10)
        match = stitcher.find_match(position=(100, 200), frame_number=100)
        assert match is None

    def test_closest_match_wins(self):
        stitcher = TrackStitcher(max_distance=100.0)
        traj_far = _make_trajectory([(100, 200, 10)])
        traj_close = _make_trajectory([(150, 200, 10)])
        dsm1 = DirectionStateMachine()
        dsm2 = DirectionStateMachine()
        stitcher.on_track_lost(track_id=1, trajectory=traj_far, dsm=dsm1, frame_number=10)
        stitcher.on_track_lost(track_id=2, trajectory=traj_close, dsm=dsm2, frame_number=10)
        match = stitcher.find_match(position=(155, 200), frame_number=12)
        assert match is not None
        assert match["lost_track_id"] == 2

    def test_match_removes_from_buffer(self):
        """Matched lost track is consumed — can't match again."""
        stitcher = TrackStitcher(max_distance=100.0)
        traj = _make_trajectory([(100, 200, 10)])
        dsm = DirectionStateMachine()
        stitcher.on_track_lost(track_id=1, trajectory=traj, dsm=dsm, frame_number=10)
        match = stitcher.find_match(position=(105, 205), frame_number=12)
        assert match is not None
        match2 = stitcher.find_match(position=(105, 205), frame_number=12)
        assert match2 is None

    def test_match_returns_distance_and_gap(self):
        stitcher = TrackStitcher(max_distance=100.0)
        traj = _make_trajectory([(100, 200, 10)])
        dsm = DirectionStateMachine()
        stitcher.on_track_lost(track_id=1, trajectory=traj, dsm=dsm, frame_number=10)
        match = stitcher.find_match(position=(110, 200), frame_number=40)
        assert match is not None
        assert match["distance"] > 0
        assert match["gap_frames"] == 30


class TestStateInheritance:
    """Test that merged tracks inherit direction state and trajectory."""

    def test_match_includes_dsm(self):
        stitcher = TrackStitcher()
        traj = _make_trajectory([(100, 200, 10)])
        dsm = DirectionStateMachine()
        dsm.state = TrackState.IN_TRANSIT
        dsm.entry_arm = "north"
        dsm.entry_label = "741-N"
        stitcher.on_track_lost(track_id=1, trajectory=traj, dsm=dsm, frame_number=10)
        match = stitcher.find_match(position=(105, 205), frame_number=12)
        assert match["dsm"].state == TrackState.IN_TRANSIT
        assert match["dsm"].entry_arm == "north"

    def test_match_includes_trajectory(self):
        stitcher = TrackStitcher()
        traj = _make_trajectory([(100, 200, 5), (110, 200, 6), (120, 200, 7)])
        dsm = DirectionStateMachine()
        stitcher.on_track_lost(track_id=1, trajectory=traj, dsm=dsm, frame_number=7)
        match = stitcher.find_match(position=(130, 200), frame_number=9)
        assert match["trajectory"] is not None
        full = match["trajectory"].get_full()
        assert len(full) == 3
        assert full[0] == (100, 200, 5)

    def test_dsm_state_preserved_across_merge(self):
        stitcher = TrackStitcher()
        traj = _make_trajectory([(100, 200, 10)])
        dsm = DirectionStateMachine()
        dsm.state = TrackState.IN_TRANSIT
        dsm.entry_arm = "east"
        dsm.entry_label = "73-E"
        stitcher.on_track_lost(track_id=1, trajectory=traj, dsm=dsm, frame_number=10)
        match = stitcher.find_match(position=(100, 200), frame_number=12)
        assert match["dsm"].entry_arm == "east"
        assert match["dsm"].entry_label == "73-E"
        assert match["dsm"].state == TrackState.IN_TRANSIT


class TestVelocityPrediction:
    """Test velocity estimation and position prediction."""

    def test_velocity_from_last_two_points(self):
        stitcher = TrackStitcher(max_distance=50.0)
        # Moving 5px/frame right, 3px/frame down
        traj = _make_trajectory([(100, 200, 8), (105, 203, 9), (110, 206, 10)])
        dsm = DirectionStateMachine()
        stitcher.on_track_lost(track_id=1, trajectory=traj, dsm=dsm, frame_number=10)
        # 4 frames later: predicted = (110 + 5*4, 206 + 3*4) = (130, 218)
        match = stitcher.find_match(position=(130, 218), frame_number=14)
        assert match is not None

    def test_single_point_trajectory_zero_velocity(self):
        stitcher = TrackStitcher(max_distance=50.0)
        traj = _make_trajectory([(100, 200, 10)])
        dsm = DirectionStateMachine()
        stitcher.on_track_lost(track_id=1, trajectory=traj, dsm=dsm, frame_number=10)
        match = stitcher.find_match(position=(105, 205), frame_number=15)
        assert match is not None

    def test_prediction_clamped_to_max_prediction_frames(self):
        """Velocity extrapolation is clamped to avoid wild predictions."""
        stitcher = TrackStitcher(max_age_frames=60, max_distance=100.0)
        # Moving 5px/frame rightward
        traj = _make_trajectory([(100, 200, 9), (105, 200, 10)])
        dsm = DirectionStateMachine()
        stitcher.on_track_lost(track_id=1, trajectory=traj, dsm=dsm, frame_number=10)
        # 50 frames later: unclamped = (105 + 5*49, 200) = (350, 200)
        # Clamped to 15 frames: (105 + 5*15, 200) = (180, 200)
        # Match near clamped prediction should work
        match = stitcher.find_match(position=(180, 200), frame_number=59)
        assert match is not None
        # Match near unclamped prediction (350, 200) should NOT match
        # since it's > 100px from the clamped prediction (180, 200)
        stitcher2 = TrackStitcher(max_age_frames=60, max_distance=100.0)
        traj2 = _make_trajectory([(100, 200, 9), (105, 200, 10)])
        dsm2 = DirectionStateMachine()
        stitcher2.on_track_lost(track_id=1, trajectory=traj2, dsm=dsm2, frame_number=10)
        match2 = stitcher2.find_match(position=(350, 200), frame_number=59)
        assert match2 is None
