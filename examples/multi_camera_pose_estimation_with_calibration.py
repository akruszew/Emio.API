#!/usr/bin/env -S uv run --script

"""Multi-camera pose estimation with calibration for Emio robots.

This script demonstrates how to use multiple Emio cameras for 3D pose estimation
of tracked markers. It performs camera calibration using ArUco markers, then
runs real-time triangulation and tracking of 3D points from multiple camera views.

The script uses threading for concurrent camera updates and visualization.
"""

from enum import Enum
import time
import logging
import os
import sys

import threading
import cv2
from numpy import ma
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/..')
from emioapi import EmioCamera
from emioapi._logging_config import logger
import numpy as np
import emioapi._triangulation as triangulation
import emioapi._tracking as tracking
import emioapi._camerafeedwindow as camerafeedwindow
import tkinter as tk

logger.setLevel(logging.WARNING)  # Set to INFO or DEBUG for more verbose output during development

parameter = {'hue_h': 80, 'hue_l': 62, 'sat_h': 255, 'sat_l': 113, 'value_h': 255, 'value_l': 44, 'erosion_size': 0, 'area': 2}


def create_marker_center_and_rotation_dictionaries():
    marker_centers = {
        672: np.array([0, 0, 0]),  # Marker ID 672 at the origin
        42:  np.array([84, 0, 0]),  # Marker ID 42 at (84mm, 0, 0)
        909: np.array([0, 0, 0]),  
        0: np.array([-90, 90, 0]),  
        1: np.array([90, 90, 0]),  
        2: np.array([-90, -90, 0]),
        3: np.array([90, -90, 0]), 
        # Add more markers here if needed
    }
    from scipy.spatial.transform import Rotation as R
    R_xy_plane = np.eye(3)
    marker_rotation = { 
        672: R_xy_plane,
        42:  R.from_euler('z', 90, degrees=True).as_matrix() @ R_xy_plane,  # Rotate marker 42 by 90 degrees around Z-axis
        909: R_xy_plane,  # No rotation for marker 909
        0: R_xy_plane,
        1: R_xy_plane,
        2: R_xy_plane,
        3: R_xy_plane,
    }
    return marker_centers, marker_rotation





def DetectArucoCorners(frame:np.ndarray):
    
    return frame

def estimate_transform(camera: EmioCamera, marker_corner_position_dic: dict[int, np.ndarray],gamma=80, threshold=80)-> tuple[float, np.ndarray, np.ndarray]:
    """Estimate the transform of a single camera using ArUco markers.

    Detects ArUco markers in the camera frame and solves for the camera's pose
    relative to the marker coordinate system. Updates the camera's transform
    with the computed rotation and translation.

    Args:
        camera: The EmioCamera instance to calibrate.
        marker_position_dic: Dictionary mapping marker IDs to their 3D positions.
        threshold: Threshold value for image binarization (default: 80).
        gamma: 100/Gamma correction value for image preprocessing (default: 80).

    returns:
        tuple: (max_error, color_frame, thresh_image) where max_error is the maximum reprojection error in pixels,
               color_frame is the original camera frame with detected markers drawn, and thresh_image is the preprocessed image used for detection.
    """
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    
    dist_coeffs = np.zeros((4, 1))
    camera_matrix = camera._camera.build_camera_intrinsics_matrix()
    _, color_frame,_,_ = camera._camera.get_frame()
    
    
    gray = cv2.cvtColor(color_frame, cv2.COLOR_BGR2GRAY)
    _,thresh_image = cv2.threshold(gray,threshold,255,cv2.THRESH_TOZERO)
    thresh_image = np.array(255 * (thresh_image / 255) ** gamma, dtype='uint8')
    
    # Detect ArUco markers
    corners, ids, rejected = detector.detectMarkers(thresh_image)
    if ids is None or len(ids) == 0:
        logger.warning(f"No ArUco markers detected for camera {camera.camera_serial}. Calibration failed.")
        return np.inf,color_frame.copy(), thresh_image.copy()
    else:
        logger.info(f"Camera {camera.camera_serial} detected ArUco markers with IDs: {ids.flatten()}")
    for i in range(len(ids)):
        id = int(ids[i][0])
        if id in marker_corner_position_dic.keys():
            logger.info(f"Camera {camera.camera_serial} detected marker {id}.")
            

    success = False
    rvec = None
    tvec = None

    obj_points = np.empty((0,3), dtype=np.float32)
    obj_corners = np.empty((0,2), dtype=np.float32)
    for i in range(len(ids)):
        id = int(ids[i][0])
        # 3D object points of marker corners
        if id in marker_corner_position_dic.keys():
            marker_position = marker_corner_position_dic[id]
            obj_points = np.vstack([obj_points, marker_position])
            obj_corners = np.vstack([obj_corners, [corners[i][0][j] for j in range(4)]])
            color_frame = cv2.aruco.drawDetectedMarkers(color_frame, [corners[i]], ids[i])

        
    if len(obj_points) > 0:
        success, rvec, tvec = cv2.solvePnP(
            obj_points,
            obj_corners,
            camera_matrix,
            dist_coeffs,
        )

    if success and rvec is not None and tvec is not None:
        rotation_matrix, _ = cv2.Rodrigues(rvec)
        # Location of the camera in the marker reference frame is given by -R^T * tvec
        camera_position = -rotation_matrix.T @ tvec.flatten()
        # Camera orientation in the marker reference frame is given by R^T
        camera_orientation = rotation_matrix.T


        camera._camera.position_estimator.R = camera_orientation
        camera._camera.position_estimator.t = camera_position
        logger.debug(f"Camera {camera.camera_serial} position estimator set with R:\n{rotation_matrix.T}\nt:\n{camera_position}")
    else:
        logger.warning(f"Camera {camera.camera_serial} pose estimation failed. solvePnP did not succeed.")

    # verify that the reprojection of the corners is correct
    max_error = np.inf
    if success and rvec is not None and tvec is not None:
        projected_points, _ = cv2.projectPoints(obj_points, rvec, tvec, camera_matrix, dist_coeffs)
        for i in range(len(obj_points)):
            u_proj = int(projected_points[i][0][0]) 
            v_proj = int(projected_points[i][0][1]) 
            cv2.circle(color_frame, (u_proj, v_proj), 5, (0, 255, 0), -1)
        error = np.linalg.norm(projected_points.squeeze() - obj_corners, axis=1)
        max_error = np.max(error)
        logger.info(f"Camera {camera.camera_serial} reprojection error: {np.round(error, decimals=2)} pixels")
            
    
    return max_error, color_frame.copy(), thresh_image.copy()





def camera_update_loop(camera: EmioCamera, synchronization_barrier: threading.Barrier):
    """Run camera update loop in a separate thread.

    Continuously updates the camera frame and trackers, synchronizing with other threads.

    Args:
        camera: The EmioCamera instance to run.
        synchronization_barrier: Threading barrier for synchronization.
    """
    while camera.is_running:
        camera.update()  # update the camera frame and trackers
        try:
            synchronization_barrier.wait() # Wait for all threads to reach this point before proceeding
        except threading.BrokenBarrierError:
            break  # Exit the loop if the barrier is broken (e.g., on shutdown)


def main(cameras: list[EmioCamera]):
    """Main function to run the multi-camera pose estimation pipeline.

    Calibrates cameras, starts tracking threads, and performs real-time triangulation and visualization.

    Args:
        cameras: List of EmioCamera instances to use.
    """
    MAX_ERROR_THRESHOLD = 5.0  # Maximum acceptable reprojection error in pixels for calibration
    start_time = time.time()

    marker_size = 90
    marker_position_dic, marker_rotation_dic = create_marker_center_and_rotation_dictionaries()
    arcuco_corner_3D_position_dictionnay = create_aruco_corners_3D_positions_dictionaty(marker_position_dic,marker_rotation_dic,marker_size)

    error = calibrate_cameras(cameras, arcuco_corner_3D_position_dictionnay) # Calibrate cameras and get the maximum reprojection error across all cameras, will update the camera transforms
    if error > MAX_ERROR_THRESHOLD:
        logger.warning(f"Calibration error {error:.2f} exceeds threshold of {MAX_ERROR_THRESHOLD}. Pose estimation may be inaccurate.")
        input("Press Enter to continue anyway, or Ctrl-C to exit.")
    

    logger.info(f"Calibration completed in {time.time() - start_time:.2f} seconds.")
    


    tracker = tracking.MultiMarkerTracker3D(
        dist_gate=150,    # Tune according to max speed and dt
        max_misses=5,      # Tolerates x frames of occlusion
        q=1e-0,
        r=1e-6
    )

    def log_tracker(tracker=tracker):
        while True:
            time.sleep(0.1)
            s = ""
            for track in tracker.tracks:
                s += f"ID {track.id}: pos {np.round(track.pos, decimals=2)}, misses {track.misses} | "
            if len(s) > 0:  
                logger.info("\n"+s)
            
    # Start the logger thread
    logger_thread = threading.Thread(target=log_tracker, args=(tracker,))
    logger_thread.daemon = True
    logger_thread.start()

    run_tracking_system(cameras,tracker)

    


def run_tracking_system(cameras,tracker):
    '''
    Run the main tracking system loop. 
    This function starts the camera update threads, then continuously performs triangulation and tracking, visualizing the results.
        
        Args:
            cameras: List of EmioCamera instances to use.
            tracker: MultiMarkerTracker3D instance to use for tracking.

    '''
    frame_synchronization_barrier = threading.Barrier(len(cameras)+1)
    run_camera_threads = []
    for camera in cameras:
        thread = threading.Thread(target=camera_update_loop, args=(camera, frame_synchronization_barrier))
        thread.daemon = True  # Set the thread as a daemon thread
        run_camera_threads.append(thread)
        thread.start()

    Rs, Ts, Ks = build_camera_matrices(cameras)
    
    t = 0.0
    dt = 1.0 / 30.0  # Assuming cameras run at 30 FPS, adjust as needed

    cv2.namedWindow("Tracks", cv2.WINDOW_FREERATIO)
    cv2.resizeWindow('Tracks', 1200, 800)
    

    class VisualizerMode(Enum):
        MATCHES = 'matches'
        TRACKS = 'tracks'
        RAW = 'raw'
    
    
    visu_mode = VisualizerMode.MATCHES
    def set_visu_mode(mode):
        nonlocal visu_mode
        visu_mode = mode
        logger.info(f"Visualizer mode set to {visu_mode.value}")

    # Replace Qt buttons (not available in some OpenCV builds) with keyboard controls.
    # Controls: 'm' = MATCHES, 't' = TRACKS, 'r' = RAW, 'q' or ESC = Exit
    print("Visualizer controls: 'm'=matches, 't'=tracks, 'r'=raw, 'q'/Esc=exit")
    
    for p in parameter.keys():
        cv2.createTrackbar(p, "Tracks", parameter[p], 255, lambda value, param=p: parameter.update({param: value}))

    while cameras[0].is_running:  # Assuming all cameras have the same running state
        try:
            frame_synchronization_barrier.wait()  # Wait for all camera threads to update before proceeding
            #matches = marker_position_estimation(cameras, Ks, Rs, Ts)
            current_time = time.time()
            # Collect detections from all cameras
            detections = [[] for _ in range(len(cameras))]  # List of lists of (u,v) in pixels for each camera
            for id_camera in range(len(cameras)):
                camera = cameras[id_camera]
                for tracker_pos in camera._trackers_pos_camera_image:
                    detections[id_camera].append(tracker_pos)    
            
            matches = triangulation.match_ncams_one_frame(detections, Ks, Rs, Ts,
                            ref=0,
                            epi_gate_px=30.0,
                            reproj_gate_px=30.0,
                            beam_width=50,
                            min_views=2)

            X_meas = []  # list of 3D points measured at this frame
            if matches is None:
                continue
            
            for m in matches:
                Xk = m["X"]
                Rms = m["rms"]

                if Rms < 3.0:  # Gate reprojection error
                    X_meas.append(Xk)
            
            tracks = tracker.update(X_meas, t)
            t += dt
            logger.debug(f"Frame time: {time.time() - current_time:.4f} seconds, Tracks: {len(tracks)}")
            if visu_mode is VisualizerMode.MATCHES:
                images = visualize_matches(cameras, matches, Ks, Rs, Ts)
            elif visu_mode is VisualizerMode.TRACKS:
                images = visualize_tracks(cameras, tracks, Ks, Rs, Ts)
            else:
                images = [cv2.resize(camera._camera.frame, (0, 0), fx=0.5, fy=0.5) for camera in cameras]

            concatenated_image = None
            for image in images:
                if image is not None:
                    if concatenated_image is None:
                        concatenated_image = image
                    else:
                        concatenated_image = np.hstack((concatenated_image, image))
            if concatenated_image is not None:
                cv2.imshow("Tracks", concatenated_image)
            for camera in cameras:
                camera._camera.parameter = parameter
            # Handle keyboard controls in lieu of Qt buttons
            key = cv2.waitKey(1) & 0xFF
            if key != 255:
                if key == ord('m'):
                    set_visu_mode(VisualizerMode.MATCHES)
                elif key == ord('t'):
                    set_visu_mode(VisualizerMode.TRACKS)
                elif key == ord('r'):
                    set_visu_mode(VisualizerMode.RAW)
                elif key == ord('q') or key == 27:
                    logger.info("Exit key pressed. Exiting tracking loop.")
                    break
            
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received.")
            break
        except Exception as e:
            logger.exception(f"Error during communication: {e}")
            break

    frame_synchronization_barrier.abort()

def calibrate_cameras(cameras,arcuco_corner_3D_position_dictionnay):
    '''
    Calibrate the cameras.  
    This function displays a calibration window with trackbars to adjust gamma and threshold for each camera. 
    It continuously estimates the camera pose using ArUco markers and updates the camera transforms until the user validates the calibration or closes the window. 
    It returns the maximum reprojection error across all cameras after calibration.

        Args:
            cameras: List of EmioCamera instances to calibrate.
    '''
    logger.info("Starting camera calibration...")
    calibration_parameters = dict()

    cv2.namedWindow("Camera calibration", cv2.WINDOW_FREERATIO)
    cv2.resizeWindow('Camera calibration', 1200, 800)
    for camera in cameras:
        calibration_parameters['gamma '+camera.camera_serial] = 1.0
        cv2.createTrackbar(f"Gamma {camera.camera_serial}", "Camera calibration", 100, 400, lambda value, cam=camera: calibration_parameters.update({f'gamma {cam.camera_serial}': (float(value)/100)}))
        cv2.setTrackbarPos(f"Gamma {camera.camera_serial}", "Camera calibration", 100)
        calibration_parameters['threshold '+camera.camera_serial] = 80
        cv2.createTrackbar(f"Threshold {camera.camera_serial}", "Camera calibration", 80, 255, lambda value, cam=camera: calibration_parameters.update({f'threshold {cam.camera_serial}': int(float(value))}))
        cv2.setTrackbarPos(f"Threshold {camera.camera_serial}", "Camera calibration", 80)
       
    
    
    validated = False
    def on_validate_calibration(*args):
        nonlocal validated
        validated = True
    # Qt buttons/overlays may not be available in this OpenCV build. Use keyboard control instead.
    print("Calibration: adjust gamma/threshold via trackbars; press 'v' to validate or close the window to proceed.")
    
    max_of_all_errors = np.inf
    while not validated:  # Wait for calibration to complete (or timeout)
        max_of_all_errors = 0.0
        calibration_images = np.empty((cameras[0]._camera.height*2,0 ,3), dtype=np.uint8)  # Initialize an empty array to hold the calibration images
        for camera in cameras:
            max_error, color_image,thresh_image = estimate_transform(camera, arcuco_corner_3D_position_dictionnay,gamma=calibration_parameters['gamma '+camera.camera_serial], threshold=calibration_parameters['threshold '+camera.camera_serial])
            max_of_all_errors = max(max_of_all_errors, max_error)
            if thresh_image is None:
                continue
            else:
                image = np.vstack((color_image, cv2.cvtColor(thresh_image, cv2.COLOR_GRAY2BGR)))
                cv2.putText(image, f"Camera {camera.camera_serial}, reprojection error = {max_error:.2f} pixels", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            if calibration_images is None:
                calibration_images = image
            else:
                calibration_images = np.hstack((calibration_images, image))
            
        calibration_images = cv2.resize(calibration_images, (0, 0), fx=0.5, fy=0.5)

        if cv2.getWindowProperty("Camera calibration", cv2.WND_PROP_VISIBLE) < 1:  # Check if the window has been closed
            logger.info("Calibration window closed by user. Exiting calibration.")
            validated = True
            break
        cv2.imshow(f"Camera calibration", calibration_images)
        # Use keyboard input to validate calibration ('v') or exit (Esc)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('v'):
            validated = True
            break
        if key == 27:
            validated = True
            break
       
    cv2.destroyWindow("Camera calibration")
    return max_of_all_errors

def create_aruco_corners_3D_positions_dictionaty(marker_position_dic, marker_rotation_dic,marker_size):
    '''
    Create a dictionary mapping ArUco marker IDs to their 3D corner positions in the marker reference frame.
    This function takes the marker center positions and rotations, and computes the 3D positions of the corners of each marker based on the marker size.   
    
        Args:
            marker_position_dic: Dictionary mapping marker IDs to their 3D center positions.
            marker_rotation_dic: Dictionary mapping marker IDs to their rotation matrices.
            marker_size: Size of the ArUco marker in millimeters.
        Returns:
            dict: Dictionary mapping marker IDs to their 3D corner positions as numpy arrays of shape (4, 3).
    '''

    arcuco_marker_positions = dict()
    marker_points = np.array([
                [-marker_size/2, marker_size/2, 0],
                [marker_size/2, marker_size/2, 0],
                [marker_size/2, -marker_size/2, 0],
                [-marker_size/2, -marker_size/2, 0]
            ], dtype=np.float32)
    for id in marker_position_dic.keys():
        arcuco_marker_positions[id]=( marker_rotation_dic[id] @ marker_points.T ).T + marker_position_dic[id]
    return arcuco_marker_positions

def build_camera_matrices(cameras):
    """Build camera matrices from calibrated cameras.

    Args:
        cameras: List of calibrated EmioCamera instances.

    Returns:
        tuple: (Rs, Ts, Ks) where Rs are rotation matrices, Ts are translation vectors, Ks are intrinsic matrices.
    """
    Ts = []
    Ks = []
    Rs = []
    for id_camera in range(len(cameras)):
        camera = cameras[id_camera]
                # Always add camera parameters, even if no trackers
        R_cw = camera._camera.position_estimator.R.T
        t_cw = -R_cw @ camera._camera.position_estimator.t.reshape(3, 1)
        Rs.append(R_cw)
        Ts.append(t_cw)
        intrinsics = camera._camera.intr    
        camera_matrix = np.array([
                    [intrinsics.fx, 0, intrinsics.ppx],
                    [0, intrinsics.fy, intrinsics.ppy],
                    [0, 0, 1]
                        ], dtype=np.float32)
        Ks.append(camera_matrix)
    return Rs,Ts,Ks

    

def marker_position_estimation(cameras, Ks, Rs, Ts):
    """Perform marker position estimation using triangulation.

    This function is currently not used in the main loop but demonstrates
    alternative triangulation methods.

    Args:
        cameras: List of EmioCamera instances.
        Ks: List of intrinsic matrices.
        Rs: List of rotation matrices.
        Ts: List of translation vectors.

    Returns:
        list: List of matched 3D points.
    """
    current_time = time.time()
    logger.info(f"Camera update time: {time.time() - current_time:.4f} seconds")

    info = f"Count tracker: "
    for camera in cameras:
        info += f"{len(camera.trackers_pos)},"
    logger.info(info)
    current_time = time.time()
    rays = []
    uvs = []
    
    detections = [[] for _ in range(len(cameras))] # list of list of (u,v) in pixels for each camera
    for id_camera in range(len(cameras)):
        camera = cameras[id_camera]
                # Always add camera parameters, even if no trackers
                
        if len(camera.trackers_pos) > 0:
            x = camera._trackers_pos_camera_image[0][0]
            y = camera._trackers_pos_camera_image[0][1]
            rays.append(triangulation.ray(camera._camera.position_estimator.t, camera._camera.position_estimator.camera_image_to_simulation(x, y, 1.0) - camera._camera.position_estimator.t))
            uvs.append((x, y))
            logger.info(f"Trackers positions Camera {camera.camera_serial}: {np.round(camera.trackers_pos[0], decimals=2)}")
        for tracker in camera._trackers_pos_camera_image:
            detections[id_camera].append((tracker[0],tracker[1]))    


    current_time = time.time()    
    estimation_nview, estimation_errors_nv = triangulation.triangulate_nview(
                Ks, 
                Rs, 
                Ts, 
                uvs, robust=True
            )
    
    logger.info(f"Triangulation with LM refinement time: {time.time() - current_time:.4f} seconds")
    if estimation_nview is not None:
        logger.info(f"Triangulated position with LM refinement: {np.round(estimation_nview.T, decimals=2)}")
        logger.info(f"Reprojection error with LM refinement: {np.round(np.array(estimation_errors_nv['rms_reproj_px']), decimals=2)} pixels")
    current_time = time.time()
    estimation, estimation_errors = triangulation.point_closest_to_rays(rays)
    logger.info(f"QP Triangulation time: {time.time() - current_time:.4f} seconds")
    if estimation is not None:
        logger.info(f"QP Triangulated position: {np.round(estimation.T, decimals=2)}")
        logger.info(f"QP Point to rays error: {np.round(estimation_errors, decimals=2)}") 


    # Detections[c] = list of (u,v) in pixels for camera c at current frame
    current_time = time.time()
    matches = triangulation.match_ncams_one_frame(detections, Ks, Rs, Ts,
                               ref=0,
                               epi_gate_px=200.0,
                               reproj_gate_px=200.0,
                               beam_width=50,
                               min_views=2)
    logger.info(f"Matches: {len(matches)}")
    logger.info(f"Matching time: {time.time() - current_time:.4f} seconds")


    return matches

def visualize_matches(cameras, matches, Ks, Rs, Ts):
    """Visualize matched 3D points on camera frames.

    Args:
        cameras: List of EmioCamera instances.
        matches: List of matched 3D points from triangulation.
        Ks: List of intrinsic matrices.
        Rs: List of rotation matrices.
        Ts: List of translation vectors.
    """
    final_images = []
    for id_camera in range(len(cameras)):
        camera = cameras[id_camera]
        image = camera._camera.frame.copy()
        for i in range(len(matches)):
            m = matches[i]
            logger.info(f"\nMatch {i}: views: {m['views']}, rms: {m['rms']}, idx_map: {m['idx']}, X: {np.round(m['X'], decimals=2)}")
            # Find in m['idx'] the index of the view corresponding to id_camera
            for idx_view, view in enumerate(m["idx"]):
                if view != 0:
                    Ps = triangulation.make_projection(Ks[id_camera], Rs[id_camera], Ts[id_camera])
                    x = triangulation.project_point(Ps, m["X"])
                    u_proj = int(x[0]) 
                    v_proj = int(x[1]) 
                    cv2.circle(image, (u_proj, v_proj), 20, (255, 0, 0), 4)
                    cv2.putText(image, f"ID {i}", (u_proj+20, v_proj), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
        final_images.append(image)
    return final_images

    
def visualize_tracks(cameras, tracks, Ks, Rs, Ts):
    """Visualize tracked 3D points on camera frames.

    Args:
        cameras: List of EmioCamera instances.
        tracks: List of track dictionaries from the tracker.
        Ks: List of intrinsic matrices.
        Rs: List of rotation matrices.
        Ts: List of translation vectors.
    """
    # for track in tracks:
    #     logger.info(f"Id {track['id']} Track position: {np.round(track['pos'], decimals=2)}")

    final_images = []
    for id_camera in range(len(cameras)):
        camera = cameras[id_camera]
        image = camera._camera.frame.copy()
        for track in tracks:
            Ps = triangulation.make_projection(Ks[id_camera], Rs[id_camera], Ts[id_camera])
            x = triangulation.project_point(Ps, track['pos'])
            u_proj = int(x[0]) 
            v_proj = int(x[1]) 
            cv2.circle(image, (u_proj, v_proj), 10, (0, 255, 0), 2)
            cv2.putText(image, f"ID {track['id']}", (u_proj+10, v_proj+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
        final_images.append(image)
    return final_images



if __name__ == "__main__":
    
    try:
        logger.info("List of available cameras\n"+str(EmioCamera.listCameras()))

        logger.info("Opening and configuring EMIO Camera...")

        camera = EmioCamera(show=False, track_markers=True, compute_point_cloud=True)
        
        cameras_sn = camera.listCameras()
        cameras = []
        for sn in cameras_sn:
            logger.info(f"Camera SN: {sn}")
            cameras.append(EmioCamera(sn, show=False, track_markers=True, compute_point_cloud=False, parameter=parameter))
    except Exception as e:
        logger.exception(f"An error occurred while initializing cameras: {e}")
        sys.exit(1)
    
    try:
        for camera in cameras:
            camera._camera.height = 720
            camera._camera.width = 1280
            camera.fps = 30  # Sets the fps to 30. Default is 60 and can only be one of 30, 60 or 90 fps
            camera.depth_max = 6000  # Sets the maximum depth to 600mm. Default is 430mm
            camera.depth_min = 0  # Sets the minimum depth to 0mm. Default is 2mm
            camera._camera.depth_channel_enabled = False  # Disable the depth channel to reduce CPU usage, since we only need the color channel for this example
            camera.open(camera.camera_serial)
            logger.info(f"Emio camera {camera.camera_serial} opened.")
        
        main(cameras)

        logger.info("Main function completed.")
        logger.info("Closing Emio API...")

        for camera in cameras:
            camera.close()
            logger.info(f"Emio camera {camera.camera_serial} closed.")
        logger.info("EMIO API closed.")
    except Exception as e:
        logger.exception(f"An error occurred: {e}")
        for camera in cameras:
            try:
                logger.exception(f"Attempting to close camera {camera.camera_serial} after error...")
                camera.close()
            except Exception as close_e:
                logger.exception(f"An error occurred while closing camera {camera.camera_serial}: {close_e}")
