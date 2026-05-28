import cv2
import numpy as np
import pyrealsense2 as rs
import time 
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

def main():
    # Initialize RealSense pipeline
    pipeline = rs.pipeline()
    config = rs.config()

    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    depth_sensor = device.first_depth_sensor()
    depth_sensor.set_option(rs.option.depth_units, 0.001)


    
    
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    
    cfg = pipeline.start(config)
    profile = cfg.get_stream(rs.stream.depth)
    intrinsics = profile.as_video_stream_profile().get_intrinsics()

    # ArUco marker detection setup
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    


    # Camera calibration matrix (replace with your calibration)
    camera_matrix = np.array([
        [intrinsics.fx, 0, intrinsics.ppx],
        [0, intrinsics.fy, intrinsics.ppy],
        [0, 0, 1]
    ], dtype=np.float32)
    dist_coeffs = np.zeros((4, 1))

    # Marker size in mm
    marker_size = 70

    # Data collection
    times = []
    depth_x = []
    depth_y = []
    depth_z = []
    pnp_x = []
    pnp_y = []
    pnp_z = []
    start_time = time.time()

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame:
                continue

            image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _,thresh_image = cv2.threshold(gray,95,255,cv2.THRESH_TOZERO)
            image = thresh_image
            # Detect ArUco markers
            corners, ids, rejected = detector.detectMarkers(thresh_image)

            if ids is not None:
                # Draw detected markers
                image = cv2.aruco.drawDetectedMarkers(image, corners, ids)

                # Estimate pose for each marker
                for i in range(len(ids)):
                    # 3D object points of marker corners
                    obj_points = np.array([
                        [-marker_size/2, marker_size/2, 0],
                        [marker_size/2, marker_size/2, 0],
                        [marker_size/2, -marker_size/2, 0],
                        [-marker_size/2, -marker_size/2, 0]
                    ], dtype=np.float32)

                    # Solve PnP
                    success, rvec, tvec = cv2.solvePnP(
                        obj_points,
                        corners[i],
                        camera_matrix,
                        dist_coeffs
                    )
                    # compute the intersection of the two diagonals of the marker to get a more stable depth measurement
                    intersections = np.array([
                        (corners[i][0][0] + corners[i][0][2]) / 2,
                        (corners[i][0][1] + corners[i][0][3]) / 2
                    ], dtype=np.float32)

                    marker_center = intersections
                    depth = 10.0#depth_image[int(marker_center[1])][int(marker_center[0])]
                    x = (marker_center[0] - intrinsics.ppx) * depth / intrinsics.fx
                    y = (marker_center[1] - intrinsics.ppy) * depth / intrinsics.fy
                    z = depth
                    print(f"Marker {ids[i][0]}: depth={depth:.3f}, x={x:.3f}, y={y:.3f}, z={z:.3f}")

                    if success:
                        # Draw axis
                        image = cv2.drawFrameAxes(
                            image, camera_matrix, dist_coeffs,
                            rvec, tvec, marker_size/2
                        )
                        print(f"Marker {ids[i][0]}: tvec={tvec.flatten()}")

                        # Collect data (assuming marker id 0 for simplicity)
                        if(ids[i][0] == 672):
                            current_time = time.time() - start_time
                            times.append(current_time)
                            depth_x.append(x)
                            depth_y.append(y)
                            depth_z.append(z)
                            pnp_x.append(tvec[0][0])
                            pnp_y.append(tvec[1][0])
                            pnp_z.append(tvec[2][0])

            cv2.imshow("ArUco Detection", image)
      
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Plot the data
        if times:
            # print mean values for debugging
            print(f"Mean Depth X: {np.mean(depth_x):.3f}, Mean Depth Y: {np.mean(depth_y):.3f}, Mean Depth Z: {np.mean(depth_z):.3f}")
            print(f"Mean PnP X: {np.mean(pnp_x):.3f}, Mean PnP Y: {np.mean(pnp_y):.3f}, Mean PnP Z: {np.mean(pnp_z):.3f}")

            print("Plotting results...")

            fig, axs = plt.subplots(3, 1, figsize=(10, 8))
            axs[0].plot(times, depth_x, label='Depth Sensor X')
            axs[0].plot(times, pnp_x, label='PnP X')
            axs[0].set_ylabel('X Position (mm)')
            axs[0].legend()
            axs[0].set_title('Marker Position X over Time')

            axs[1].plot(times, depth_y, label='Depth Sensor Y')
            axs[1].plot(times, pnp_y, label='PnP Y')
            axs[1].set_ylabel('Y Position (mm)')
            axs[1].legend()
            axs[1].set_title('Marker Position Y over Time')

            axs[2].plot(times, depth_z, label='Depth Sensor Z')
            axs[2].plot(times, pnp_z, label='PnP Z')
            axs[2].set_ylabel('Z Position (mm)')
            axs[2].set_xlabel('Time (s)')
            axs[2].legend()
            axs[2].set_title('Marker Position Z over Time')

            plt.tight_layout()
            plt.show()

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()