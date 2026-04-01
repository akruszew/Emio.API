#!/usr/bin/env -S uv run --script

import time
import logging
import os
import sys

sys.path.append(os.path.dirname(os.path.realpath(__file__))+'/..')
from emioapi import EmioCamera
from emioapi._logging_config import logger
import numpy as np
from emioapi._positionestimation import *
def main(emio: EmioCamera):
    plane_d = -7
    plane_n = np.array([1,0,0])
    # emio.calibrate()  # calibrate the camera if needed           
    while emio.is_running:
        try:
            emio.update() # update the camera frame and trackers

            print("-"*20)
            logger.info(f"Camera parameters: {emio.parameters}")
            logger.info(f"Camera show: {emio.show_frames}")
            logger.info(f"Camera tracking: {emio.track_markers}")
            logger.info(f"Camera compute point cloud: {emio.compute_point_cloud}")
            logger.info(f"Camera is running: {emio.is_running}")
            logger.info(f"Count tracker: {len(emio.trackers_pos)}")
            logger.info(f"Trackers positions: {emio.trackers_pos}")
            logger.info(f"Point cloud shape: {emio.point_cloud.shape}")
            logger.info(f"HSV Frame shape: {emio.hsv_frame.shape}")
            logger.info(f"Mask Frame shape: {emio.mask_frame.shape}")

            logger.info(f"Camera trackers positions in camera : {emio._trackers_pos_camera_image}")
            
            trackers_projected = []
            for pixel_pos in emio._trackers_pos_camera_image:
                if pixel_pos is None:
                    continue
               
                result = emio._camera.position_estimator.camera_image_to_simulation_plane_intersection(
                    pixel_pos[0],pixel_pos[1],plane_n, plane_d)
                trackers_projected.append(result)
            
            for i in range(len(trackers_projected)):
                error = np.linalg.norm(np.array(trackers_projected[i]) - np.array(emio.trackers_pos[i]))
                logger.info(f"Tracker {i} projected position: {trackers_projected[i]}, actual position: {emio.trackers_pos[i]}")
                logger.info(f"Tracker {i} error: {error}")
        

            time.sleep(.2)
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received.")
            break
        except Exception as e:
            logger.exception(f"Error during communication: {e}")
            break


if __name__ == "__main__":
    try:
        logger.info("Starting EMIO Camera test...")

        logger.info("List of available cameras\n"+str(EmioCamera.listCameras()))

        logger.info("Opening and configuring EMIO Camera...")

        emio = EmioCamera(show=True, track_markers=True, compute_point_cloud=True)
        emio.fps = 30 # sets the fps to 30. Default is 60 and can only be one of 30. 60 or 90fps
        emio.depth_max = 600 # sets the maximum depth to 600mm. Default is 430mm
        emio.depth_min = 0 # sets the minimum depth to 0mm. Default is 2mm

        if emio.open(): # This will open the first available Realsense camera

            logger.info(f"Emio camera {emio.camera_serial} opened.")
            logger.info("Running main function...")
            main(emio)

            logger.info("Main function completed.")
            logger.info("Closing Emio API...")

            emio.close()

            logger.info("EMIO API closed.")
    except Exception as e:
        logger.exception(f"An error occurred: {e}")
