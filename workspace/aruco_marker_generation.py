import cv2
import numpy as np

size_of_marker = 400  # size of marker.

white_square = np.full((size_of_marker, size_of_marker), 255, dtype=np.uint8)

# create the dictionary for markers type
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)

marker_id_list = [[0,None, 1],[None,None,None] ,[2,None, 3]]  # list of marker IDs to generate
# generating IDs with for loop
final_image = None
for row in marker_id_list:
    row_image = None
    for marker_id in row:
        # generating the marker
        if marker_id is None:
            img = white_square
        else:
            img = cv2.aruco.generateImageMarker(dictionary, marker_id, size_of_marker)
        row_image = img if row_image is None else cv2.hconcat([row_image, img])
    
    final_image = row_image if final_image is None else cv2.vconcat([final_image, row_image])

# save/write the image
    
    
cv2.imwrite("marker_image.png", final_image)

# display the image(marker) on windows
cv2.imshow("Marker", final_image)
cv2.waitKey(0)