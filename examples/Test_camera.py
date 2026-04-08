#!/usr/bin/env -S uv run --script

import time
import logging
import os
import sys
import matplotlib
from matplotlib.widgets import Cursor
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.dirname(os.path.realpath(__file__))+'/..')
from emioapi import EmioCamera
from emioapi._logging_config import logger


def main(emio: EmioCamera):

    # emio.calibrate()  # calibrate the camera if needed

    while emio.is_running:
        try:

            length = []
            for i in range(100):
                emio.update() # update the camera frame and trackers
                if len(emio.trackers_pos) >= 2:
                    distance = ((emio.trackers_pos[0][0] - emio.trackers_pos[1][0])**2 + (emio.trackers_pos[0][1] - emio.trackers_pos[1][1])**2 + (emio.trackers_pos[0][2] - emio.trackers_pos[1][2])**2)**0.5
                    length.append(distance)
            # mean length over the 100 frames
            mean_length = np.mean(length)
            logger.info(f"Mean length of the object: {mean_length} mm")
            # std deviation of the length over the 100 frames
            std_length = np.std(length)
            logger.info(f"Standard deviation of the length: {std_length} mm")
            # max deviation of the length over the 100 frames
            max_length = np.max(length)
            min_length = np.min(length)
            logger.info(f"Max length of the object: {max_length-min_length} mm")
            # median length of the object         median_length = np.median(length)
            logger.info(f"Median length of the object: {np.median(length)} mm")
            # plot the length over time
            fig, ax = plt.subplots()
            ax.plot(length)
            plt.xlabel("Time (s)")
            plt.ylabel("Length (mm)")
            plt.title("Length of the object over time")
            cursor = Cursor(ax, useblit=True, color='red', linewidth=2)
            print(cursor)
            plt.show()

        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received.")
            break
        except Exception as e:
            logger.exception(f"Error during communication: {e}")
            break
            # logger.info(f"Camera show: {emio.show_frames}")
            # logger.info(f"Camera tracking: {emio.track_markers}")
            # logger.info(f"Camera compute point cloud: {emio.compute_point_cloud}")
            # logger.info(f"Camera is running: {emio.is_running}")
            # logger.info(f"Count tracker: {len(emio.trackers_pos)}")
            # logger.info(f"Trackers positions: {emio.trackers_pos}")
            # logger.info(f"Point cloud shape: {emio.point_cloud.shape}")
            # logger.info(f"HSV Frame shape: {emio.hsv_frame.shape}")
            # logger.info(f"Mask Frame shape: {emio.mask_frame.shape}")
            # distance between the first two trackers
            if len(emio.trackers_pos) >= 2:
                distance = ((emio.trackers_pos[0][0] - emio.trackers_pos[1][0])**2 + (emio.trackers_pos[0][1] - emio.trackers_pos[1][1])**2 + (emio.trackers_pos[0][2] - emio.trackers_pos[1][2])**2)**0.5
                logger.info(f"Distance between the first two trackers: {distance} mm")

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

        emio = EmioCamera(show=True, track_markers=True, compute_point_cloud=True, configuration="compact")
        emio.fps = 60 # sets the fps to 30. Default is 60 and can only be one of 30. 60 or 90fps
        emio.depth_max = 80000 # sets the maximum depth to 600mm. Default is 430mm
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
