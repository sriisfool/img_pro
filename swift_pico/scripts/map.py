import cv2
import numpy as np
from cv2 import aruco

# Load the uploaded image
image_path = '/home/rudy/pico_ws/src/swift_pico/scripts/task1c_image.jpg'
image = cv2.imread(image_path)

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Load the predefined dictionary that was used to generate the Aruco markers
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters()

# Detect the markers in the image
corners, ids, rejected_img_points = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

# Draw the markers on the image
image_with_markers = aruco.drawDetectedMarkers(image.copy(), corners, ids)

# Save the image with detected markers
cv2.imwrite('marked_task1c_image.jpg', image_with_markers)

# Display the image with markers
cv2.imshow("Aruco Markers Detected", image_with_markers)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Perspective transform points
if ids is not None:
    # Define destination points for perspective transform (as a square)
    destination_points = np.array([[0, 0], [400, 0], [400, 400], [0, 400]], dtype="float32")

    # Get the Aruco marker points in the same order
    marker_corners = [corner[0][0] for corner in corners]

    # Compute the perspective transform matrix
    M = cv2.getPerspectiveTransform(np.array(marker_corners, dtype="float32"), destination_points)

    # Apply perspective transformation
    warped_image = cv2.warpPerspective(image, M, (400, 400))

    # Save and show the warped image
    cv2.imwrite('warped_task1c_image.jpg', warped_image)
    cv2.imshow("Warped Image", warped_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Obstacle detection in the warped image using contours
gray_warped = cv2.cvtColor(warped_image, cv2.COLOR_BGR2GRAY)
_, thresholded = cv2.threshold(gray_warped, 200, 255, cv2.THRESH_BINARY_INV)

# Find contours
contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours on the warped image
warped_image_with_contours = warped_image.copy()
cv2.drawContours(warped_image_with_contours, contours, -1, (0, 255, 0), 3)

# Save and display the warped image with detected obstacles
cv2.imwrite('warped_with_obstacles.jpg', warped_image_with_contours)
cv2.imshow("Obstacles Detected", warped_image_with_contours)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Count the number of obstacles and calculate the area covered by obstacles
num_obstacles = len(contours)
total_area = sum(cv2.contourArea(c) for c in contours)

# Write the results to a .txt file
with open("obstacle_detection_results.txt", "w") as file:
    file.write(f"Detected Aruco IDs: {ids.flatten()}\n")
    file.write(f"Number of Obstacles: {num_obstacles}\n")
    file.write(f"Total Area Covered by Obstacles: {total_area}\n")
