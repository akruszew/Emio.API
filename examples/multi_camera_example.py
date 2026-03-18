#!/usr/bin/env -S uv run --script

import time
import logging
import os
import sys

sys.path.append(os.path.dirname(os.path.realpath(__file__))+'/..')
from emioapi import EmioCamera
from emioapi._logging_config import logger


def main(emio: EmioCamera,emio2: EmioCamera):

    # emio.calibrate()  # calibrate the camera if needed

    while emio.is_running:
        try:
            emio.update() # update the camera frame and trackers
            emio2.update() # update the camera frame and trackers

            print("-"*20)
            logger.info(f"Count tracker: {len(emio.trackers_pos)},{len(emio2.trackers_pos)}")
            logger.info(f"Trackers positions: {emio.trackers_pos},{emio2.trackers_pos}")
            
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


        emio2 = EmioCamera(show=False, track_markers=True, compute_point_cloud=True)
        emio2.fps = 30 # sets the fps to 30. Default is 60 and can only be one of 30. 60 or 90fps
        emio2.depth_max = 600 # sets the maximum depth to 600mm. Default is 430mm
        emio2.depth_min = 0 # sets the minimum depth to 0mm. Default is 2mm
        cameras = emio.listCameras()


        if emio.open(cameras[1]): # This will open the first available Realsense camera
            logger.info(f"Emio camera {emio.camera_serial} opened.")
            logger.info("Running main function...")
            emio2.open(cameras[0])
            logger.info(f"Emio camera {emio2.camera_serial} opened.")
            logger.info("Running main function...")
            
            main(emio,emio2)

            logger.info("Main function completed.")
            logger.info("Closing Emio API...")

            emio.close()

            logger.info("EMIO API closed.")
    except Exception as e:
        logger.exception(f"An error occurred: {e}")
