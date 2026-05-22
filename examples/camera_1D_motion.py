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
import argparse

SAVE_FILE = "camera_1D_motion_data.csv"

logger.setLevel(logging.WARNING) # set the logging level to WARNING to reduce the amount of logs printed to the console. Change to INFO or DEBUG for more detailed logs.

def main(camera: EmioCamera, emioMotors: EmioMotors, angles_deg: list):

    #camera.calibrate()  # calibrate the camera if needed
    initial_pos_pulse = [0] * 4
    
    emioMotors.angles = initial_pos_pulse
    print("\n"*5)
    print("-"*20)
    input("Wait for the stabilization then press Enter to start the test...")

    emioMotors.max_velocity = [1] * 4
    initial_pos_pulse = [0] * 4

    emioMotors.angles = initial_pos_pulse
    time.sleep(1)
    trackers_trajectory = []
    trackers_trajectory_camera_image = []
    motor_angle_trajectory = []
    for i in range(20):
        camera.update() # update the camera frame and trackers
    
    # angles_deg = [0,5,10,15,20]
    time.sleep(0.5) # wait for the motor to start moving
    for angle in angles_deg:
        if not camera.is_running:
            break
        try:
            camera.update() # update the camera frame and trackers
            pos = np.asarray(camera._trackers_pos_camera_image)
            motor_angle = -angle*np.pi/180
            emioMotors.angles = [motor_angle, -motor_angle] * 2
            print("-"*20)
            print(f"Set motor angle to {angle} degrees. Waiting for the trackers to stabilize...")
            frame_without_moving = 0
            while frame_without_moving < 20: # wait until the trackers have stabilized
                camera.update() # update the camera frame and trackers
                new_pos = np.asarray(camera._trackers_pos_camera_image)
                delta = np.linalg.norm(new_pos - pos)
                if delta < 1: # if the trackers have stabilized
                    frame_without_moving += 1
                else:
                    time.sleep(0.1) # wait for the motor to start moving
                pos = new_pos
                
            
            print(f"Trackers stabilized at angle {angle} degrees. Recording coordinates")
            camera.update() # update the camera frame and trackers
            trackers_trajectory.append(camera.trackers_pos)
            trackers_trajectory_camera_image.append(camera._trackers_pos_camera_image)
            motor_angle_trajectory.append(angle)

        
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received.")
            break
        except Exception as e:
            logger.exception(f"Error during communication: {e}")
            break
    
    
    fig, ax2 = plt.subplots()
    
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
        ax2.set_xlim(-150, 50)
        ax2.set_ylim(-250, -50)
        
        if len(snap_shot) > 2:
            # calculate the angle between the first two trackers and the two last trackers
            v1 = np.array(snap_shot[0])-np.array(snap_shot[1])
            v2 = np.array(snap_shot[1])-np.array(snap_shot[2])
            angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
            angles.append(np.degrees(angle))
        
    fig, ax3 = plt.subplots()
    ax3.plot(motor_angle_trajectory,angles, '-o')
    ax3.set_xlabel('Motor angle (deg)')
    ax3.set_ylabel('Angle (deg)')   
    
    # save the data (motor angle, x,y,z) to a csv file
    with open(SAVE_FILE, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Motor angle (deg)', 'Tracker 1 x (mm)', 'Tracker 1 y (mm)', 'Tracker 1 z (mm)', 'Tracker 2 x (mm)', 'Tracker 2 y (mm)', 'Tracker 2 z (mm)', 'Tracker 3 x (mm)', 'Tracker 3 y (mm)', 'Tracker 3 z (mm)'])
        for i in range(len(motor_angle_trajectory)):
            row = [motor_angle_trajectory[i]]
            for tracker in trackers_trajectory_camera_image[i]:
                tracker_position = camera._camera.position_estimator.camera_image_to_simulation_plane_intersection(tracker[0], tracker[1], np.array([1,0,0]), -7)
                row.extend(tracker_position)
            writer.writerow(row)

    print("\n"*5)
    print("-"*20)
    print(f"recorded data saved to {SAVE_FILE}")
    print("Plotting completed. Close the plot window to finish.")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--angles", nargs="+", type=float, default=[0,5,10,15,20], help="List of angles to test")
    parser.add_argument("--show", action="store_true", help="Whether to show the camera image during the test") 
    args = parser.parse_args()
    angles_deg = args.angles
    show = args.show

    # if no arguments are provided, use the default values and tell the user
    if len(sys.argv) == 1:
        print("\n"*2)
        print("No arguments provided. Using default values: angles = [0,5,10,15,20]\nUse --help to see available options.")
    else:
        print("\n"*2)
        print(f"Using provided arguments: angles = {angles_deg}, show = {show}")


    try:
        logger.info("Opening and configuring EMIO Camera and motors...")

        emioCamera = EmioCamera(show=show, track_markers=True, compute_point_cloud=False)
        emioCamera.fps = 30 # sets the fps to 30. Default is 60 and can only be one of 30. 60 or 90fps
        emioCamera.depth_max = 6000 # sets the maximum depth to 600mm. Default is 430mm
        emioCamera.depth_min = 0 # sets the minimum depth to 0mm. Default is 2mm
        
        emioMotors = EmioMotors()
        emioMotors.open()

        if emioCamera.open(): # This will open the first available Realsense camera

            main(emioCamera, emioMotors, angles_deg=angles_deg)

            logger.info("Main function completed.")
            logger.info("Closing Emio API...")

            emioCamera.close()

            logger.info("EMIO API closed.")
    except Exception as e:
        logger.exception(f"An error occurred: {e}")
