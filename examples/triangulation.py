import numpy as np
from scipy.optimize import least_squares
from qpsolvers import Problem, solve_problem, available_solvers
from itertools import combinations
from emioapi._logging_config import logger

def make_projection(K, R, t):
    """
    K: (3,3)
    R: (3,3)
    t: (3,) or (3,1)   such that X_cam = R @ X_world + t
    returns P: (3,4)
    """
    t = t.reshape(3, 1)
    Rt = np.hstack([R, t])           # (3,4)
    P = K @ Rt
    return P

def make_projection_from_camera_pose(K, R_wc, C_wc):
    """
    Build a projection matrix from a camera pose expressed in world coordinates.

    R_wc: camera orientation from camera to world (world = R_wc * camera)
    C_wc: camera center in world coordinates
    """
    R_cw = R_wc.T
    t_cw = -R_cw @ C_wc.reshape(3, 1)
    return make_projection(K, R_cw, t_cw)


def skew(t):
    tx, ty, tz = t.reshape(3,)
    return np.array([[0, -tz,  ty],
                     [tz,  0, -tx],
                     [-ty, tx,  0]], dtype=float)

def relative_Rt_from_world(Ri, ti, Rj, tj):
    # X_ci = Ri X_w + ti ;  X_cj = Rj X_w + tj
    R_ij = Rj @ Ri.T
    t_ij = tj - R_ij @ ti
    return R_ij, t_ij

def fundamental_from_world(Ki, Ri, ti, Kj, Rj, tj):
    R_ij, t_ij = relative_Rt_from_world(Ri, ti, Rj, tj)
    E = skew(t_ij) @ R_ij
    F = np.linalg.inv(Kj).T @ E @ np.linalg.inv(Ki)
    return F / np.linalg.norm(F)

def precompute_P_and_F(Ks, Rs, ts):
    N = len(Ks)
    Ps = [make_projection(Ks[i], Rs[i], ts[i]) for i in range(N)]
    F = {}
    for i, j in combinations(range(N), 2):
        F[(i, j)] = fundamental_from_world(Ks[i], Rs[i], ts[i], Ks[j], Rs[j], ts[j])
    return Ps, F


def project_point(P, X):
    """
    P: (3,4), X: (3,)
    returns (u,v) in pixels
    """
    Xh = np.hstack([X, 1.0])        # (4,)
    x = P @ Xh                      # (3,)
    return np.array([x[0]/x[2], x[1]/x[2]])

def refine_lm(Ps, uvs, X0, loss="linear"):
    """
    Refine 3D point by minimizing reprojection error across all views.
    loss: "linear" (pure LS) or "huber"/"soft_l1" for robustness
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
    Check cheirality: Z_cam > 0
    X_cam = R @ X + t
    """
    X_cam = R @ X + t.reshape(3,)
    return X_cam[2] > 0

# ----- Example usage for ONE 3D point seen in N cameras -----
# Ks, Rs, ts should be lists of length N
# uvs is list of N pixel observations [(u0,v0), (u1,v1), ...]
def triangulate_nview(Ks, Rs, ts, uvs, robust=True):
    """
    Triangulate a 3D point from N camera views.
    
    Args:
        Ks: list of (3,3) intrinsic matrices
        Rs: list of (3,3) rotation matrices (extrinsic: world->camera)
        ts: list of (3,) or (3,1) translation vectors (extrinsic: world->camera)
             such that X_cam = R @ X_world + t
        uvs: list of (u,v) pixel observations
        robust: whether to use robust loss (huber) or linear
    
    Returns:
        X: (3,) triangulated position in world coordinates
        dict with "rms_reproj_px" and "X0" (DLT estimate)
    
    Note: If you have camera pose (R_wc: world->camera rotation, C_wc: camera center in world),
          convert to extrinsic parameters as:
          R_cw = R_wc.T
          t_cw = -R_cw @ C_wc
    """
    Ps = [make_projection(K, R, t) for K, R, t in zip(Ks, Rs, ts)]

    X0 = triangulate_dlt(Ps, uvs)

    if robust:
        X, sol = refine_lm(Ps, uvs, X0, loss="huber")
    else:
        X, sol = refine_lm(Ps, uvs, X0, loss="linear")

    # Optional: reject if behind any camera
    # for R, t in zip(Rs, ts):
    #     if not positive_depth(R, t, X):
    #         return None, {"reason": "behind_camera"}

    # Optional: compute RMS reprojection error (pixels)
    r = sol.fun
    rms = np.sqrt(np.mean(r**2))
    return X, {"rms_reproj_px": float(rms), "X0": X0}



class ray:
    def __init__(self,origin,direction):
        self.origin = origin
        self.direction = direction/np.linalg.norm(direction)

def point_closest_to_rays(rays: list[ray]):
    """
    Solve the quadratic programming problem:
        minimize (1/2) x^T P x + q^T x
        subject to A x = b
                   G x <= h
                   lb <= x <= ub    
    Args:        rays: list of rays, each ray is defined by an origin and a direction
    Returns:        x: Solution vector (3,) the triangulated position
                    errors: list of errors for each ray, the distance from the triangulated position to the ray
    """
    R = np.zeros((len(rays),3,3+len(rays)))
    P = np.zeros((3+len(rays),3+len(rays)))
    q = np.zeros((3+len(rays),))
    for i in range(len(rays)):
        R[i][0:3,0:3] = np.eye(3)
        R[i][0:3,3+i] = -rays[i].direction   
        P += R[i].T @ R[i]
        q += -R[i].T @ rays[i].origin

    problem = Problem(P, q)
    solution = solve_problem(problem, solver="clarabel")
    # Use the OSQP solver as an example. You can choose any other solver available in qpsolvers.
    if (solution.x is None):
        logger.warning("QP solver failed to find a solution.")
        return None,None   
    
    errors = []
    for i in range(len(rays)):
        error = np.linalg.norm(R[i] @ solution.x - rays[i].origin)
        errors.append(error)
  
    
    return solution.x[0:3],errors



def epipolar_line(Fij, uv_i):
    u, v = uv_i
    x = np.array([u, v, 1.0], dtype=float)
    return Fij @ x  # (a,b,c) in image j

def point_line_distance(l, uv):
    a, b, c = l
    u, v = uv
    return abs(a*u + b*v + c) / np.sqrt(a*a + b*b + 1e-12)

def epi_compatible(F, i, uv_i, j, uv_j, epi_gate_px):
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
    Ps: list of (3,4) projection matrices
    uvs: list/array of (u,v) pixel observations, same length as Ps
    returns X (3,) in world coordinates
    """
    Ps = list(Ps)
    uvs = np.asarray(uvs, dtype=float)
    assert len(Ps) == len(uvs)
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

def project(P, X):
    Xh = np.array([X[0], X[1], X[2], 1.0], dtype=float)
    x = P @ Xh
    return np.array([x[0]/x[2], x[1]/x[2]])

def rms_reproj(Ps, uvs, X):
    errs = []
    for cam, uv_obs in uvs:
        uvp = project(Ps[cam], X)
        e = uvp - np.array(uv_obs, dtype=float)
        errs.extend([e[0], e[1]])
    errs = np.array(errs)
    return float(np.sqrt(np.mean(errs**2)))


def match_ncams_one_frame(detections, Ks, Rs, Ts,
                          ref=0,
                          epi_gate_px=2.0,
                          reproj_gate_px=3.0,
                          beam_width=50,
                          min_views=3):
    """
    detections: list length N, detections[c] = [(u,v), ...]
    returns: list of matches: {"idx": dict cam->det_index, "X": X, "rms": rms, "views": m}
    """

    N = len(detections)
    Ps, F = precompute_P_and_F(Ks, Rs, Ts)

    det_ref = detections[ref]
    proposals = []

    # For each detection in reference cam, build multi-view hypothesis
    for iref, uv_ref in enumerate(det_ref):
        # hypothesis is dict cam->index, and list of observations
        hyps = [({ref: iref}, [(ref, uv_ref)])]
        
        # expand to other cameras
        for cam in range(N):
            if cam == ref:
                continue

            new_hyps = []
            det_cam = detections[cam]

            for idx_map, obs in hyps:
                # last ref obs (we always have ref)
                # candidate detections in cam consistent with epipolar wrt ref (fast)
                cand = []
                for j, uv_j in enumerate(det_cam):
                    if epi_compatible(F, ref, uv_ref, cam, uv_j, epi_gate_px):
                        cand.append((j, uv_j))

                # option: allow "missing" view (skip this camera)
                new_hyps.append((idx_map, obs))

                for j, uv_j in cand:
                    idx_map2 = dict(idx_map)
                    idx_map2[cam] = j
                    obs2 = obs + [(cam, uv_j)]
                    new_hyps.append((idx_map2, obs2))
            
            # prune with reprojection if enough views, keep top beam_width
            scored = []
            for idx_map, obs in new_hyps:
                if len(obs) >= 2:
                    uvs_only = [uv for cam, uv in obs]
                    X = triangulate_dlt(Ps, uvs_only)
                    # cheirality for cams used
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
                    # only one view -> cannot score yet
                    scored.append((1e9, idx_map, obs, None))
            
            scored.sort(key=lambda x: x[0])
            hyps = []
            for rms, idx_map, obs, X in scored[:beam_width]:
                hyps.append((idx_map, obs))

        # finalize: keep hypotheses with enough views + low reprojection
        for idx_map, obs in hyps:
            if len(obs) < min_views:
                continue
            uvs_only = [uv for cam, uv in obs]
            X = triangulate_dlt(Ps, uvs_only)
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

    # Global selection to avoid reusing same blob in same camera
    # Greedy by best (rms, -views) works well for ~10 points.
    proposals.sort(key=lambda p: (p["rms"], -p["views"]))
    print(f"Proposals: {len(proposals)}")
    used = {c: set() for c in range(N)}
    matches = []
    for p in proposals:
        conflict = False
        for c, j in p["idx"].items():
            if j in used[c]:
                conflict = True
                break
        if conflict:
            continue
        for c, j in p["idx"].items():
            used[c].add(j)
        matches.append(p)

    return matches
