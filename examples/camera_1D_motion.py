#!/usr/bin/env -S uv run --script

import time
import logging
import os
import sys
import csv
sys.path.append(os.path.dirname(os.path.realpath(__file__))+'/..')
from emioapi import EmioCamera, EmioMotors
from emioapi._logging_config import logger

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np

def main(camera: EmioCamera, emioMotors: EmioMotors):

    #camera.calibrate()  # calibrate the camera if needed
    emioMotors.max_velocity = [1] * 4
    initial_pos_pulse = [0] * 4
    logger.info(f"Initial position in rad: {initial_pos_pulse}")
    emioMotors.angles = initial_pos_pulse
    time.sleep(1)
    trackers_trajectory = []
    trackers_trajectory_camera_image = []
    motor_angle_trajectory = []
    for i in range(20):
        camera.update() # update the camera frame and trackers
    
    for i in range(20):
        if not camera.is_running:
            break
        try:
            camera.update() # update the camera frame and trackers
            motor_angle = -((2*3.14)*((i+1)%20)/100)
            emioMotors.angles = [motor_angle, -motor_angle] * 2
            time.sleep(2)
            print("-"*20)
            logger.info(f"Camera parameters: {camera.parameters}")
            logger.info(f"Camera show: {camera.show_frames}")
            logger.info(f"Camera tracking: {camera.track_markers}")
            logger.info(f"Camera compute point cloud: {camera.compute_point_cloud}")
            logger.info(f"Camera is running: {camera.is_running}")
            logger.info(f"Count tracker: {len(camera.trackers_pos)}")
            logger.info(f"Trackers positions: {camera.trackers_pos}")
            logger.info(f"Point cloud shape: {camera.point_cloud.shape}")
            logger.info(f"HSV Frame shape: {camera.hsv_frame.shape}")
            logger.info(f"Mask Frame shape: {camera.mask_frame.shape}")
            logger.info(f"Camera trackers positions in camera frame: {camera._trackers_pos_camera_image}")
            for i in range(1):
                camera.update() # update the camera frame and trackers
                trackers_trajectory.append(camera.trackers_pos)
                trackers_trajectory_camera_image.append(camera._trackers_pos_camera_image)
                motor_angle_trajectory.append(emioMotors.angles[3])
            
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received.")
            break
        except Exception as e:
            logger.exception(f"Error during communication: {e}")
            break
    logger.info(f"Trackers trajectory: {trackers_trajectory}")
    with open('trackers_trajectory.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(trackers_trajectory)   

    fig, ax = plt.subplots()
    for snap_shot in trackers_trajectory:
        x = [tracker[0] for tracker in filter(None, snap_shot)]
        y = [tracker[1] for tracker in filter(None, snap_shot)]
        z = [tracker[2] for tracker in filter(None, snap_shot)]
        ax.plot(z,y, '-o')

    ax.set_xlabel('Z')
    ax.set_ylabel('Y')
    
    fig, ax2 = plt.subplots()
    logger.info(f"Trackers trajectory in camera image: {trackers_trajectory_camera_image}")
    logger.info(f"Trackers trajectory in camera image shape: {len(trackers_trajectory_camera_image)}, {len(trackers_trajectory_camera_image[1])}, {len(trackers_trajectory_camera_image[1][0])}")
    
    angles = []
    for snap_shot in trackers_trajectory_camera_image:
        x = []
        y = []
        z = []
        for tracker in snap_shot:
            tracker_position = camera._camera.position_estimator.camera_image_to_simulation_plane_intersection(tracker[0], tracker[1], np.array([1,0,0]), -7)
            x.append(tracker_position[0])
            y.append(tracker_position[1])
            z.append(tracker_position[2])
        ax2.plot(z,y, '-o')
        ax2.set_xlim(-100, 100)
        ax2.set_ylim(-100, 100)
        
        if len(snap_shot) > 2:
            # calculate the angle between the first two trackers and the two last trackers
            v1 = np.array(snap_shot[0])-np.array(snap_shot[1])
            v2 = np.array(snap_shot[1])-np.array(snap_shot[2])
            angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
            logger.info(f"Angle between the two pairs of trackers: {angle}")
            angles.append(np.degrees(angle))
        
    fig, ax3 = plt.subplots()
    ax3.plot(motor_angle_trajectory,angles, '-o')
    ax3.set_xlabel('Motor angle (rad)')
    ax3.set_ylabel('Angle (deg)')   
    
    
    plt.show()

if __name__ == "__main__":
    try:
        logger.info("Starting EMIO Camera test...")

        logger.info("List of available cameras\n"+str(EmioCamera.listCameras()))

        logger.info("Opening and configuring EMIO Camera...")

        emioCamera = EmioCamera(show=True, track_markers=True, compute_point_cloud=True)
        emioCamera.fps = 30 # sets the fps to 30. Default is 60 and can only be one of 30. 60 or 90fps
        emioCamera.depth_max = 6000 # sets the maximum depth to 600mm. Default is 430mm
        emioCamera.depth_min = 0 # sets the minimum depth to 0mm. Default is 2mm
        
        emioMotors = EmioMotors()
        emioMotors.open()

        if emioCamera.open(): # This will open the first available Realsense camera

            logger.info(f"Emio camera {emioCamera.camera_serial} opened.")
            logger.info("Running main function...")
            main(emioCamera, emioMotors)

            logger.info("Main function completed.")
            logger.info("Closing Emio API...")

            emioCamera.close()

            logger.info("EMIO API closed.")
    except Exception as e:
        logger.exception(f"An error occurred: {e}")
