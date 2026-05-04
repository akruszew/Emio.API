import numpy as np
from scipy.optimize import linear_sum_assignment
class KalmanCV3D:
    """
    Constant Velocity Kalman filter in 3D.
    State: [x, y, z, vx, vy, vz]
    Measurement: [x, y, z]
    """
    def __init__(self, x0, P0=None, q=1e-2, r=1e-3):
        self.x = np.zeros(6)
        self.x[:3] = x0

        self.P = np.eye(6) * 1.0 if P0 is None else P0.copy()

        # process noise intensity (tuning)
        self.q = float(q)
        # measurement noise (tuning)
        self.r = float(r)

    def predict(self, dt):
        F = np.eye(6)
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt

        # Simple process noise model
        Q = np.eye(6) * self.q
        # (option: build Q from dt for CV model; this simple version often suffices)

        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        return self.x.copy()

    def update(self, z):
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
    def __init__(self, track_id, x0, t, q=1e-2, r=1e-3):
        self.id = track_id
        self.kf = KalmanCV3D(x0=x0, q=q, r=r)
        self.last_t = t
        self.hits = 1
        self.misses = 0
        self.age = 1

    def predict_to(self, t):
        dt = float(t - self.last_t)
        if dt < 0:
            dt = 0.0
        self.kf.predict(dt)
        self.last_t = t
        self.age += 1

    @property
    def pos(self):
        return self.kf.x[:3]

class MultiMarkerTracker3D:
    def __init__(self,
                 dist_gate=0.10,     # m: max distance track↔measurement to allow match
                 max_misses=10,      # frames before deleting a track
                 q=1e-2,
                 r=1e-3):
        self.dist_gate = float(dist_gate)
        self.max_misses = int(max_misses)
        self.q = float(q)
        self.r = float(r)
        self.tracks = []
        self.next_id = 0

    def _cost_matrix(self, tracks, meas):
        # cost = Euclidean distance (m)
        C = np.zeros((len(tracks), len(meas)), dtype=float)
        for i, tr in enumerate(tracks):
            for j, z in enumerate(meas):
                C[i, j] = np.linalg.norm(tr.pos - z)
        return C

    def update(self, measurements_3d, t):
        """
        measurements_3d: array-like (M,3) list of 3D points measured at time t
        t: timestamp (seconds)
        returns: list of dict with track states
        """
        meas = np.asarray(measurements_3d, dtype=float)
        if meas.ndim == 1 and meas.size == 0:
            meas = meas.reshape(0, 3)
        if meas.size == 0:
            # no measurements: just predict and count misses
            for tr in self.tracks:
                tr.predict_to(t)
                tr.misses += 1
            self.tracks = [tr for tr in self.tracks if tr.misses <= self.max_misses]
            return self.get_tracks()

        # 1) predict all tracks to current time
        for tr in self.tracks:
            tr.predict_to(t)

        # 2) if no existing tracks, start all
        if len(self.tracks) == 0:
            for z in meas:
                self._start_track(z, t)
            return self.get_tracks()

        # 3) build assignment cost and solve Hungarian
        C = self._cost_matrix(self.tracks, meas)
        row_ind, col_ind = linear_sum_assignment(C)

        assigned_tracks = set()
        assigned_meas = set()

        # 4) apply gated assignments
        for r_i, c_j in zip(row_ind, col_ind):
            if C[r_i, c_j] <= self.dist_gate:
                tr = self.tracks[r_i]
                z = meas[c_j]
                tr.kf.update(z)
                tr.hits += 1
                tr.misses = 0
                assigned_tracks.add(r_i)
                assigned_meas.add(c_j)

        # 5) tracks not assigned => misses++
        for i, tr in enumerate(self.tracks):
            if i not in assigned_tracks:
                tr.misses += 1

        # 6) measurements not assigned => start new tracks
        for j, z in enumerate(meas):
            if j not in assigned_meas:
                self._start_track(z, t)

        # 7) delete dead tracks
        self.tracks = [tr for tr in self.tracks if tr.misses <= self.max_misses]
        return self.get_tracks()

    def _start_track(self, z, t):
        tr = Track3D(self.next_id, x0=z, t=t, q=self.q, r=self.r)
        self.next_id += 1
        self.tracks.append(tr)

    def get_tracks(self):
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