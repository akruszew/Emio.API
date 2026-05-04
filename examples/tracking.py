"""3D multi-target tracking utilities for Emio camera pose estimation.

This module implements a constant-velocity Kalman filter and a simple
tracker that performs data association using the Hungarian algorithm.
The tracker maintains tracks over time, updates them with new 3D
measurements, and removes stale tracks after a configurable number of misses.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment


class KalmanCV3D:
    """Constant Velocity Kalman filter for 3D point tracking.

    State vector: [x, y, z, vx, vy, vz]
    Measurement vector: [x, y, z]
    """

    def __init__(self, x0, P0=None, q=1e-2, r=1e-3):
        """Initialize the Kalman filter.

        Args:
            x0: Sequence[3] initial 3D position [x, y, z].
            P0: Optional[ndarray] initial covariance matrix (6x6).
            q: process noise variance.
            r: measurement noise variance.
        """
        self.x = np.zeros(6)
        self.x[:3] = x0

        self.P = np.eye(6) * 1.0 if P0 is None else P0.copy()

        # Process noise intensity.
        self.q = float(q)
        # Measurement noise intensity.
        self.r = float(r)

    def predict(self, dt):
        """Predict the next state after a time interval dt.

        Args:
            dt: Time interval in seconds.

        Returns:
            The predicted state vector as a copy.
        """
        F = np.eye(6)
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt

        # Simple process noise model.
        Q = np.eye(6) * self.q

        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        return self.x.copy()

    def update(self, z):
        """Correct the state estimate with a new measurement.

        Args:
            z: Sequence[3] measurement vector [x, y, z].

        Returns:
            The updated state vector as a copy.
        """
        z = np.asarray(z, dtype=float).reshape(3,)
        H = np.zeros((3, 6))
        H[0, 0] = 1.0
        H[1, 1] = 1.0
        H[2, 2] = 1.0

        R = np.eye(3) * self.r

        y = z - (H @ self.x)
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        I = np.eye(6)
        self.P = (I - K @ H) @ self.P
        return self.x.copy()


class Track3D:
    """A single tracked 3D target with a Kalman filter."""

    def __init__(self, track_id, x0, t, q=1e-2, r=1e-3):
        """Create a new track.

        Args:
            track_id: Unique identifier for the track.
            x0: Initial position of the track as a 3D vector.
            t: Initial timestamp in seconds.
            q: Process noise variance for the Kalman filter.
            r: Measurement noise variance for the Kalman filter.
        """
        self.id = track_id
        self.kf = KalmanCV3D(x0=x0, q=q, r=r)
        self.last_t = t
        self.hits = 1
        self.misses = 0
        self.age = 1

    def predict_to(self, t):
        """Advance the track state to the given timestamp.

        Args:
            t: New timestamp in seconds.
        """
        dt = float(t - self.last_t)
        if dt < 0:
            dt = 0.0
        self.kf.predict(dt)
        self.last_t = t
        self.age += 1

    @property
    def pos(self):
        """Return the current estimated 3D position of the track."""
        return self.kf.x[:3]


class MultiMarkerTracker3D:
    """Multi-target tracker for 3D marker measurements.

    This tracker uses a constant velocity Kalman filter for each target and
    associates measurements to tracks using the Hungarian algorithm. New tracks
    are created for unassigned measurements and stale tracks are removed after
    too many consecutive misses.
    """

    def __init__(self,
                 dist_gate=0.10,
                 max_misses=10,
                 q=1e-2,
                 r=1e-3):
        """Initialize the multi-target tracker.

        Args:
            dist_gate: Maximum allowed Euclidean distance (meters) for assigning a measurement to a track.
            max_misses: Number of consecutive frames without assignment before a track is deleted.
            q: Process noise variance for each track.
            r: Measurement noise variance for each track.
        """
        self.dist_gate = float(dist_gate)
        self.max_misses = int(max_misses)
        self.q = float(q)
        self.r = float(r)
        self.tracks = []
        self.next_id = 0

    def _cost_matrix(self, tracks, meas):
        """Compute the cost matrix between tracks and measurements.

        Cost is the Euclidean distance between predicted track position and measurement.
        """
        C = np.zeros((len(tracks), len(meas)), dtype=float)
        for i, tr in enumerate(tracks):
            for j, z in enumerate(meas):
                C[i, j] = np.linalg.norm(tr.pos - z)
        return C

    def update(self, measurements_3d, t):
        """Update tracks with a new set of 3D measurements.

        Args:
            measurements_3d: Array-like of shape (M, 3) containing 3D points.
            t: Timestamp of the current measurement frame in seconds.

        Returns:
            List[dict]: Current track state summaries.
        """
        meas = np.asarray(measurements_3d, dtype=float)
        if meas.ndim == 1 and meas.size == 0:
            meas = meas.reshape(0, 3)

        if meas.size == 0:
            # No measurements: predict every track and increase miss count.
            for tr in self.tracks:
                tr.predict_to(t)
                tr.misses += 1
            self.tracks = [tr for tr in self.tracks if tr.misses <= self.max_misses]
            return self.get_tracks()

        # 1) Predict all tracks to the current timestamp.
        for tr in self.tracks:
            tr.predict_to(t)

        # 2) If no tracks exist yet, create one for each measurement.
        if len(self.tracks) == 0:
            for z in meas:
                self._start_track(z, t)
            return self.get_tracks()

        # 3) Compute data association cost and solve assignment.
        C = self._cost_matrix(self.tracks, meas)
        row_ind, col_ind = linear_sum_assignment(C)

        assigned_tracks = set()
        assigned_meas = set()

        # 4) Update assigned tracks if the association is within gate.
        for r_i, c_j in zip(row_ind, col_ind):
            if C[r_i, c_j] <= self.dist_gate:
                tr = self.tracks[r_i]
                z = meas[c_j]
                tr.kf.update(z)
                tr.hits += 1
                tr.misses = 0
                assigned_tracks.add(r_i)
                assigned_meas.add(c_j)

        # 5) Increase miss count for unassigned tracks.
        for i, tr in enumerate(self.tracks):
            if i not in assigned_tracks:
                tr.misses += 1

        # 6) Start new tracks for measurements that were not assigned.
        for j, z in enumerate(meas):
            if j not in assigned_meas:
                self._start_track(z, t)

        # 7) Remove tracks that have missed too many frames.
        self.tracks = [tr for tr in self.tracks if tr.misses <= self.max_misses]
        return self.get_tracks()

    def _start_track(self, z, t):
        """Create a new track for an unassigned measurement."""
        tr = Track3D(self.next_id, x0=z, t=t, q=self.q, r=self.r)
        self.next_id += 1
        self.tracks.append(tr)

    def get_tracks(self):
        """Return the current set of active tracks as dictionaries."""
        return [
            {
                "id": tr.id,
                "pos": tr.pos.copy(),
                "vel": tr.kf.x[3:6].copy(),
                "hits": tr.hits,
                "misses": tr.misses,
                "age": tr.age,
            }
            for tr in self.tracks
        ]
