"""
Triangulation Module for Multi-Camera Systems

This module provides functions for triangulating 3D points from multiple camera views
in computer vision applications. It includes methods for:

- Building projection matrices from camera intrinsics and extrinsics
- Computing fundamental matrices for epipolar geometry
- Triangulating points using Direct Linear Transform (DLT) and nonlinear refinement
- Finding closest points to rays using quadratic programming
- Matching detections across multiple cameras with epipolar constraints

The module is designed for use with the Emio API for camera calibration and pose estimation.

Dependencies:
- numpy: For numerical computations and matrix operations
- scipy.optimize: For nonlinear least squares refinement
- qpsolvers: For quadratic programming optimization
- itertools: For combinations generation
- emioapi._logging_config: For logging functionality

Author: Emio API Team
"""

import numpy as np
from scipy.optimize import least_squares
from qpsolvers import Problem, solve_problem, available_solvers
from itertools import combinations
from typing import List, Tuple, Dict, Optional, Union, Any
from emioapi._logging_config import logger

# Constants
DEFAULT_EPI_GATE_PX = 2.0
DEFAULT_REPROJ_GATE_PX = 3.0
DEFAULT_BEAM_WIDTH = 50
DEFAULT_MIN_VIEWS = 3
HIGH_COST = 1e9
EPSILON = 1e-12
HOMOGENEOUS_COORD = 1.0

def make_projection(K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Create a camera projection matrix from intrinsic and extrinsic parameters.

    This function combines the camera intrinsic matrix K with the extrinsic
    rotation R and translation t to form a 3x4 projection matrix P.

    Args:
        K: 3x3 camera intrinsic matrix
        R: 3x3 rotation matrix (world to camera coordinates)
        t: 3-element translation vector (world to camera coordinates)

    Returns:
        3x4 projection matrix P such that x = P @ [X, 1].T

    Raises:
        ValueError: If input matrices have incorrect shapes
    """
    if K.shape != (3, 3) or R.shape != (3, 3) or t.shape not in [(3,), (3, 1)]:
        raise ValueError("Invalid input shapes for projection matrix construction")

    t = t.reshape(3, 1)
    Rt = np.hstack([R, t])           # (3,4)
    P = K @ Rt
    return P

def make_projection_from_camera_pose(K: np.ndarray, R_wc: np.ndarray, C_wc: np.ndarray) -> np.ndarray:
    """
    Build a projection matrix from camera pose in world coordinates.

    This function creates a projection matrix when the camera pose is given
    as rotation from camera to world (R_wc) and camera center in world coordinates (C_wc).

    Args:
        K: 3x3 camera intrinsic matrix
        R_wc: 3x3 rotation matrix from camera to world coordinates
        C_wc: 3-element camera center position in world coordinates

    Returns:
        3x4 projection matrix

    Raises:
        ValueError: If input matrices have incorrect shapes
    """
    if K.shape != (3, 3) or R_wc.shape != (3, 3) or C_wc.shape not in [(3,), (3, 1)]:
        raise ValueError("Invalid input shapes for camera pose projection")

    R_cw = R_wc.T
    t_cw = -R_cw @ C_wc.reshape(3, 1)
    return make_projection(K, R_cw, t_cw)


def skew(t: np.ndarray) -> np.ndarray:
    """
    Compute the skew-symmetric matrix from a 3D vector.

    The skew-symmetric matrix [t]× satisfies [t]× @ v = t × v (cross product).

    Args:
        t: 3-element vector

    Returns:
        3x3 skew-symmetric matrix

    Raises:
        ValueError: If input is not a 3-element vector
    """
    if t.shape not in [(3,), (3, 1)]:
        raise ValueError("Input must be a 3-element vector")

    tx, ty, tz = t.reshape(3,)
    return np.array([[0, -tz,  ty],
                     [tz,  0, -tx],
                     [-ty, tx,  0]], dtype=float)

def relative_Rt_from_world(Ri: np.ndarray, ti: np.ndarray, Rj: np.ndarray, tj: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute relative rotation and translation between two camera poses.

    Given two camera poses in world coordinates, compute the transformation
    from camera i to camera j coordinate systems.

    Args:
        Ri, Rj: 3x3 rotation matrices (world to camera)
        ti, tj: 3-element translation vectors (world to camera)

    Returns:
        Tuple of (R_ij, t_ij) where R_ij transforms points from camera i to camera j,
        and t_ij is the corresponding translation

    Raises:
        ValueError: If input matrices have incorrect shapes
    """
    if any(m.shape != (3, 3) for m in [Ri, Rj]) or any(v.shape not in [(3,), (3, 1)] for v in [ti, tj]):
        raise ValueError("Invalid input shapes for relative pose computation")

    # X_ci = Ri X_w + ti ;  X_cj = Rj X_w + tj
    R_ij = Rj @ Ri.T
    t_ij = tj - R_ij @ ti
    return R_ij, t_ij

def fundamental_from_world(Ki: np.ndarray, Ri: np.ndarray, ti: np.ndarray,
                          Kj: np.ndarray, Rj: np.ndarray, tj: np.ndarray) -> np.ndarray:
    """
    Compute the fundamental matrix between two cameras from their world poses.

    The fundamental matrix F_ij relates points in camera i to epipolar lines in camera j.

    Args:
        Ki, Kj: 3x3 intrinsic matrices for cameras i and j
        Ri, Rj: 3x3 rotation matrices (world to camera)
        ti, tj: 3-element translation vectors (world to camera)

    Returns:
        3x3 fundamental matrix F_ij (normalized)

    Raises:
        ValueError: If input matrices have incorrect shapes
    """
    if any(m.shape != (3, 3) for m in [Ki, Kj, Ri, Rj]) or any(v.shape not in [(3,), (3, 1)] for v in [ti, tj]):
        raise ValueError("Invalid input shapes for fundamental matrix computation")

    R_ij, t_ij = relative_Rt_from_world(Ri, ti, Rj, tj)
    E = skew(t_ij) @ R_ij
    F = np.linalg.inv(Kj).T @ E @ np.linalg.inv(Ki)
    return F / np.linalg.norm(F)

def precompute_P_and_F(Ks, Rs, ts):
    """
    Precompute projection matrices and fundamental matrices for all camera pairs.

    This function efficiently computes all necessary matrices for multi-camera triangulation.

    Args:
        Ks (list): List of 3x3 intrinsic matrices
        Rs (list): List of 3x3 rotation matrices (world to camera)
        ts (list): List of 3-element translation vectors (world to camera)

    Returns:
        tuple: (Ps, F) where:
            - Ps: List of 3x4 projection matrices
            - F: Dictionary with keys (i,j) containing F_ij fundamental matrices
    """
    N = len(Ks)
    Ps = [make_projection(Ks[i], Rs[i], ts[i]) for i in range(N)]
    F = {}
    for i, j in combinations(range(N), 2):
        F[(i, j)] = fundamental_from_world(Ks[i], Rs[i], ts[i], Ks[j], Rs[j], ts[j])
    return Ps, F


def project_point(P: np.ndarray, X: np.ndarray) -> np.ndarray:
    """
    Project a 3D world point onto the image plane using a projection matrix.

    Args:
        P: 3x4 projection matrix
        X: 3-element 3D point in world coordinates

    Returns:
        2-element array [u, v] pixel coordinates

    Raises:
        ValueError: If input shapes are incorrect
    """
    if P.shape != (3, 4) or X.shape not in [(3,), (3, 1)]:
        raise ValueError("Invalid input shapes for point projection")

    Xh = np.hstack([X, HOMOGENEOUS_COORD])        # (4,)
    x = P @ Xh                      # (3,)
    return np.array([x[0]/x[2], x[1]/x[2]])

def refine_lm(Ps, uvs, X0, loss="linear"):
    """
    Refine 3D point estimate using nonlinear least squares optimization.

    Minimizes reprojection error across all camera views using Levenberg-Marquardt
    or Trust Region Reflective algorithm.

    Args:
        Ps (list): List of 3x4 projection matrices
        uvs (list): List of 2-element arrays [(u,v), ...] observed pixel coordinates
        X0 (np.ndarray): 3-element initial 3D point estimate
        loss (str): Loss function type - "linear", "huber", or "soft_l1"

    Returns:
        tuple: (X, sol) where X is the refined 3D point and sol is the optimization result
    """
    uvs = np.asarray(uvs, dtype=float)

    def residuals(X):
        res = []
        for P, (u_obs, v_obs) in zip(Ps, uvs):
            u_pred, v_pred = project_point(P, X)
            res.extend([u_pred - u_obs, v_pred - v_obs])
        return np.array(res)

    # method="lm" only supports loss="linear"
    if loss == "linear":
        sol = least_squares(residuals, X0, method="lm")
    else:
        sol = least_squares(residuals, X0, method="trf", loss=loss)

    return sol.x, sol

def positive_depth(R, t, X):
    """
    Check cheirality constraint: ensure 3D point is in front of the camera.

    Args:
        R (np.ndarray): 3x3 rotation matrix (world to camera)
        t (np.ndarray): 3-element translation vector (world to camera)
        X (np.ndarray): 3-element 3D point in world coordinates

    Returns:
        bool: True if Z_camera > 0 (point is in front of camera)
    """
    # X_cam = R @ X + t
    X_cam = R @ X + t.reshape(3,)
    return X_cam[2] > 0

# ----- Example usage for ONE 3D point seen in N cameras -----
# Ks, Rs, ts should be lists of length N
# uvs is list of N pixel observations [(u0,v0), (u1,v1), ...]
def triangulate_nview(Ks, Rs, ts, uvs, robust=True):
    """
    Triangulate a 3D point from multiple camera views using DLT and nonlinear refinement.

    This function implements a two-stage triangulation process:
    1. Initial estimate using Direct Linear Transform (DLT)
    2. Nonlinear refinement using Levenberg-Marquardt optimization

    Args:
        Ks (list): List of 3x3 camera intrinsic matrices
        Rs (list): List of 3x3 rotation matrices (world to camera)
        ts (list): List of 3-element translation vectors (world to camera)
        uvs (list): List of 2-tuples [(u,v), ...] pixel observations
        robust (bool): If True, use Huber loss for robustness to outliers

    Returns:
        tuple: (X, info) where:
            - X: 3-element triangulated 3D point in world coordinates (or None if invalid)
            - info: Dictionary with triangulation statistics or failure reason

    Note:
        The function checks cheirality (positive depth) for all cameras and rejects
        solutions where the point appears behind any camera.
    """
    Ps = [make_projection(K, R, t) for K, R, t in zip(Ks, Rs, ts)]

    X0 = triangulate_dlt(Ps, uvs)

    if robust:
        X, sol = refine_lm(Ps, uvs, X0, loss="huber")
    else:
        X, sol = refine_lm(Ps, uvs, X0, loss="linear")

    # Optional: reject if behind any camera
    for R, t in zip(Rs, ts):
        if not positive_depth(R, t, X):
            return None, {"reason": "behind_camera"}

    # Optional: compute RMS reprojection error (pixels)
    r = sol.fun
    rms = np.sqrt(np.mean(r**2))
    return X, {"rms_reproj_px": float(rms), "X0": X0}



class ray:
    """
    Represents a 3D ray defined by an origin point and direction vector.

    The ray is parameterized as: point = origin + t * direction for t >= 0
    """
    def __init__(self, origin, direction):
        """
        Initialize a ray.

        Args:
            origin (np.ndarray): 3-element origin point
            direction (np.ndarray): 3-element direction vector (will be normalized)
        """
        self.origin = origin
        self.direction = direction / np.linalg.norm(direction)

def point_closest_to_rays(rays: list[ray])-> Tuple[Optional[np.ndarray], np.ndarray]:
    """
    Find the 3D point that minimizes the sum of squared distances to multiple rays.

    This function solves a quadratic programming problem to find the point that
    is closest to all rays in a least-squares sense. The problem is formulated as:

    minimize (1/2) x^T P x + q^T x
    subject to: distance constraints for each ray

    Args:
        rays (list[ray]): List of ray objects, each with origin and normalized direction

    Returns:
        tuple: (point, errors) where:
            - point: 3-element array of the optimal 3D position (or None if solver fails)
            - errors: List of distances from the point to each ray (or None if solver fails)

    Note:
        Uses the Clarabel solver via qpsolvers. The formulation ensures the point
        lies on the line defined by each ray's origin and direction.
    """
    R = np.zeros((len(rays), 3, 3 + len(rays)))
    P = np.zeros((3 + len(rays), 3 + len(rays)))
    q = np.zeros((3 + len(rays),))
    for i in range(len(rays)):
        R[i][0:3, 0:3] = np.eye(3)
        R[i][0:3, 3 + i] = -rays[i].direction
        P += R[i].T @ R[i]
        q += -R[i].T @ rays[i].origin

    problem = Problem(P, q)
    solution = solve_problem(problem, solver="clarabel")
    # Use the OSQP solver as an example. You can choose any other solver available in qpsolvers.
    if (solution.x is None):
        logger.warning("QP solver failed to find a solution.")
        return None, np.array([float('inf')] * len(rays))

    errors = np.array([np.linalg.norm(R[i] @ solution.x - rays[i].origin) for i in range(len(rays))])
    return solution.x[0:3], errors



def epipolar_line(Fij, uv_i):
    """
    Compute the epipolar line in image j corresponding to a point in image i.

    Args:
        Fij (np.ndarray): 3x3 fundamental matrix from camera i to camera j
        uv_i (tuple): 2-element (u,v) pixel coordinates in image i

    Returns:
        np.ndarray: 3-element line coefficients [a,b,c] for ax + by + c = 0
    """
    u, v = uv_i
    x = np.array([u, v, 1.0], dtype=float)
    return Fij @ x  # (a,b,c) in image j

def point_line_distance(l: np.ndarray, uv: Tuple[float, float]) -> float:
    """
    Compute the perpendicular distance from a point to a line.

    Args:
        l: 3-element line coefficients [a,b,c] for ax + by + c = 0
        uv: 2-element (u,v) point coordinates

    Returns:
        Distance from point to line in pixels

    Raises:
        ValueError: If line coefficients have wrong shape
    """
    if l.shape != (3,):
        raise ValueError("Line coefficients must be 3-element array")

    a, b, c = l
    u, v = uv
    return abs(a*u + b*v + c) / np.sqrt(a*a + b*b + EPSILON)

def epi_compatible(F, i, uv_i, j, uv_j, epi_gate_px):
    """
    Check if two points are epipolar compatible within a distance threshold.

    Args:
        F (dict): Dictionary of fundamental matrices with keys (min_i,j)
        i, j (int): Camera indices
        uv_i, uv_j (tuple): Pixel coordinates in cameras i and j
        epi_gate_px (float): Maximum allowed epipolar distance in pixels

    Returns:
        bool: True if points are epipolar compatible
    """
    if i < j:
        Fij = F[(i, j)]
        l = epipolar_line(Fij, uv_i)
        return point_line_distance(l, uv_j) <= epi_gate_px
    else:
        # use transpose relation: l_i = F_ji x_j  (with F_ji stored as (j,i) ?)
        # easiest: compute with stored (j,i) and swap
        Fji = F[(j, i)]
        l = epipolar_line(Fji, uv_j)
        return point_line_distance(l, uv_i) <= epi_gate_px


def triangulate_dlt(Ps, uvs):
    """
    Triangulate a 3D point using Direct Linear Transform (DLT).

    This method solves the triangulation problem by finding the 3D point X that
    minimizes the algebraic error in the projection equations x = P @ [X; 1].

    Args:
        Ps (list): List of 3x4 projection matrices
        uvs (list): List of 2-element arrays [(u,v), ...] pixel observations

    Returns:
        np.ndarray: 3-element triangulated 3D point in world coordinates

    Note:
        The method sets up a system A @ x = 0 where x = [X; 1] and solves
        using SVD to find the null space. This gives a closed-form solution
        but may not be optimal in the presence of noise.
    """
    Ps = list(Ps)
    uvs = np.asarray(uvs, dtype=float)
    #assert len(Ps) == len(uvs)
    N = len(Ps)

    A = np.zeros((2*N, 4), dtype=float)
    for i, (P, (u, v)) in enumerate(zip(Ps, uvs)):
        A[2*i + 0, :] = u * P[2, :] - P[0, :]
        A[2*i + 1, :] = v * P[2, :] - P[1, :]

    # Solve A X = 0 with SVD
    _, _, Vt = np.linalg.svd(A)
    X_h = Vt[-1, :]                 # last row of Vt = smallest singular value
    X = X_h[:3] / X_h[3]
    return X

def rms_reproj(Ps: List[np.ndarray], uvs: List[Tuple[int, Tuple[float, float]]], X: np.ndarray) -> float:
    """
    Compute RMS reprojection error for a triangulated point.

    Args:
        Ps: List of 3x4 projection matrices
        uvs: List of tuples [(cam_idx, (u,v)), ...] observations
        X: 3-element 3D point

    Returns:
        Root mean square reprojection error in pixels

    Raises:
        ValueError: If camera index in uvs is out of range of Ps
    """
    
    errs = []
    for cam, uv_obs in uvs:
        if cam >= len(Ps):
            raise ValueError(f"Camera index {cam} out of range")
        uvp = project_point(Ps[cam], X)
        e = uvp - np.array(uv_obs, dtype=float)
        errs.extend([e[0], e[1]])
    errs = np.array(errs)
    return float(np.sqrt(np.mean(errs**2)))


def _validate_match_inputs(detections: List[List[Tuple[float, float]]],
                           Ks: List[np.ndarray],
                           Rs: List[np.ndarray],
                           Ts: List[np.ndarray],
                           ref: int) -> int:
    """
    Validate inputs for multi-camera matching.

    Args:
        detections: List of detections per camera
        Ks: List of intrinsic matrices
        Rs: List of rotation matrices
        Ts: List of translation vectors
        ref: Reference camera index

    Returns:
        Number of cameras N

    Raises:
        ValueError: If inputs are invalid
    """
    N = len(detections)
    if N == 0 or len(Ks) != N or len(Rs) != N or len(Ts) != N:
        raise ValueError("Inconsistent number of cameras in inputs")
    if ref < 0 or ref >= N:
        raise ValueError(f"Invalid reference camera index {ref}")
    return N

def _expand_hypotheses(hyps: List[Tuple[Dict[int, int], List[Tuple[int, Tuple[float, float]]]]],
                      cam: int,
                      det_cam: List[Tuple[float, float]],
                      F: Dict[Tuple[int, int], np.ndarray],
                      ref: int,
                      uv_ref: Tuple[float, float],
                      epi_gate_px: float) -> List[Tuple[Dict[int, int], List[Tuple[int, Tuple[float, float]]]]]:
    """
    Expand hypotheses to include detections from a new camera.

    Args:
        hyps: Current hypotheses (idx_map, observations)
        cam: Camera index to expand to
        det_cam: Detections in current camera
        F: Fundamental matrices dictionary
        ref: Reference camera index
        uv_ref: Reference detection coordinates
        epi_gate_px: Epipolar distance threshold

    Returns:
        Expanded list of hypotheses
    """
    new_hyps = []

    for idx_map, obs in hyps:
        # Find candidate detections that are epipolar compatible
        cand = []
        for j, uv_j in enumerate(det_cam):
            if epi_compatible(F, ref, uv_ref, cam, uv_j, epi_gate_px):
                cand.append((j, uv_j))

        # Allow missing view (skip this camera) to handle occlusions
        new_hyps.append((idx_map, obs))

        # Add hypotheses with each compatible detection
        for j, uv_j in cand:
            idx_map2 = dict(idx_map)
            idx_map2[cam] = j
            obs2 = obs + [(cam, uv_j)]
            new_hyps.append((idx_map2, obs2))

    return new_hyps

def _score_hypotheses(hyps: List[Tuple[Dict[int, int], List[Tuple[int, Tuple[float, float]]]]],
                     Ps: List[np.ndarray],
                     Rs: List[np.ndarray],
                     Ts: List[np.ndarray],
                     beam_width: int) -> List[Tuple[Dict[int, int], List[Tuple[int, Tuple[float, float]]]]]:
    """
    Score hypotheses and keep the best ones.

    Args:
        hyps: Hypotheses to score
        Ps: Projection matrices
        Rs: Rotation matrices
        Ts: Translation vectors
        beam_width: Maximum number of hypotheses to keep

    Returns:
        Scored and pruned hypotheses
    """
    scored = []
    for idx_map, obs in hyps:
        if len(obs) >= 2:
            uvs_only = [uv for cam, uv in obs]
            cams = [cam for cam, uv in obs]
            Ps_cams = [Ps[cam] for cam in cams]
            X = triangulate_dlt(Ps_cams, uvs_only)
            #X = triangulate_dlt(Ps, uvs_only)
            # Check cheirality (positive depth) for all cameras used
            ok = True
            for c, _ in obs:
                if not positive_depth(Rs[c], Ts[c], X):
                    ok = False
                    break
            if not ok:
                continue
            rms = rms_reproj(Ps, obs, X)
            scored.append((rms, idx_map, obs, X))
        else:
            # Single view: cannot score yet, assign high cost
            scored.append((HIGH_COST, idx_map, obs, None))

    # Keep only the best hypotheses
    scored.sort(key=lambda x: x[0])
    return [(idx_map, obs) for _, idx_map, obs, _ in scored[:beam_width]]

def _finalize_hypotheses(hyps: List[Tuple[Dict[int, int], List[Tuple[int, Tuple[float, float]]]]],
                        Ps: List[np.ndarray],
                        Rs: List[np.ndarray],
                        Ts: List[np.ndarray],
                        min_views: int,
                        reproj_gate_px: float,
                        iref: int) -> List[Dict[str, Any]]:
    """
    Finalize hypotheses by validating triangulation quality.

    Args:
        hyps: Hypotheses to finalize
        Ps: Projection matrices
        Rs: Rotation matrices
        Ts: Translation vectors
        min_views: Minimum number of views required
        reproj_gate_px: Maximum reprojection error threshold
        iref: Reference detection index

    Returns:
        List of valid proposals
    """
    
    proposals = []
    for idx_map, obs in hyps:
        if len(obs) < min_views:
            continue
        
        uvs_only = [uv for cam, uv in obs]
        cams = [cam for cam, uv in obs]
        Ps_cams = [Ps[cam] for cam in cams]
        X = triangulate_dlt(Ps_cams, uvs_only)
        #X = triangulate_dlt(Ps, uvs_only)
        if not all(positive_depth(Rs[c], Ts[c], X) for c, _ in obs):
            continue
        rms = rms_reproj(Ps, obs, X)
        if rms <= reproj_gate_px:
            proposals.append({
                "idx": idx_map,        # dict cam -> det index
                "X": X,
                "rms": rms,
                "views": len(obs),
                "ref_idx": iref
            })
    return proposals

def _select_matches(proposals: List[Dict[str, Any]], N: int) -> List[Dict[str, Any]]:
    """
    Perform global conflict resolution to select non-conflicting matches.

    Args:
        proposals: List of proposal dictionaries
        N: Number of cameras

    Returns:
        List of selected matches without conflicts
    """
    # Sort by (reprojection_error, -num_views) - prefer low error, then more views
    proposals.sort(key=lambda p: (p["rms"], -p["views"]))
    used = {c: set() for c in range(N)}  # Track used detections per camera
    matches = []
    for p in proposals:
        # Check for conflicts (same detection used in same camera)
        conflict = False
        for c, j in p["idx"].items():
            if j in used[c]:
                conflict = True
                break
        if conflict:
            continue
        # Mark detections as used and add to matches
        for c, j in p["idx"].items():
            used[c].add(j)
        matches.append(p)
    return matches
def match_ncams_one_frame(detections: List[List[Tuple[float, float]]],
                          Ks: List[np.ndarray],
                          Rs: List[np.ndarray],
                          Ts: List[np.ndarray],
                          ref: int = 0,
                          epi_gate_px: float = DEFAULT_EPI_GATE_PX,
                          reproj_gate_px: float = DEFAULT_REPROJ_GATE_PX,
                          beam_width: int = DEFAULT_BEAM_WIDTH,
                          min_views: int = DEFAULT_MIN_VIEWS) -> List[Dict[str, Any]]:
    """
    Match detections across multiple cameras using epipolar geometry and triangulation.

    This function implements a beam search algorithm to find consistent multi-camera
    matches for detected features. It uses epipolar constraints for fast pruning and
    reprojection error for final validation.

    Args:
        detections: List of length N, where detections[c] = [(u,v), ...] pixel detections
        Ks: List of 3x3 intrinsic matrices
        Rs: List of 3x3 rotation matrices (world to camera)
        Ts: List of 3-element translation vectors (world to camera)
        ref: Reference camera index for epipolar constraint evaluation
        epi_gate_px: Maximum epipolar distance for compatibility (pixels)
        reproj_gate_px: Maximum RMS reprojection error for valid matches (pixels)
        beam_width: Maximum number of hypotheses to keep at each expansion step
        min_views: Minimum number of cameras that must see the point

    Returns:
        List of match dictionaries, each containing:
            - "idx": dict cam->detection_index mapping
            - "X": 3-element triangulated 3D position
            - "rms": RMS reprojection error
            - "views": number of cameras observing the point
            - "ref_idx": detection index in reference camera

    Raises:
        ValueError: If input dimensions are inconsistent or invalid
    """
    N = _validate_match_inputs(detections, Ks, Rs, Ts, ref)
    Ps, F = precompute_P_and_F(Ks, Rs, Ts)

    det_ref = detections[ref]
    proposals = []
    for i in range(N):
        ref = i
        det_ref = detections[i]
        # Beam search: for each detection in reference camera, build hypotheses
        for iref, uv_ref in enumerate(det_ref):
            # Initialize hypothesis with reference detection
            hyps = [({ref: iref}, [(ref, uv_ref)])]
            # Expand hypotheses to other cameras using epipolar constraints
            for cam in range(N):
                if cam == ref:
                    continue
                det_cam = detections[cam]
                hyps = _expand_hypotheses(hyps, cam, det_cam, F, ref, uv_ref, epi_gate_px)
                hyps = _score_hypotheses(hyps, Ps, Rs, Ts, beam_width)
            # Finalize valid hypotheses
            proposals.extend(_finalize_hypotheses(hyps, Ps, Rs, Ts, min_views, reproj_gate_px, iref))
            
        # Global conflict resolution
    return _select_matches(proposals, N)
