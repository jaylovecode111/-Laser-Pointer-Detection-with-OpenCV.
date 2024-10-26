import cv2
import numpy as np

# Define the distance calculation function
def calculate_distance(area, calibration_constant):
    # Ensure the area is not zero to avoid division by zero
    if area == 0:
        return None
    # Calculate distance, assuming it is inversely proportional to the square root of the area
    distance = calibration_constant / np.sqrt(area)
    return distance

def main():
    # Open the webcam
    camera = cv2.VideoCapture(0)

    # Check if the webcam is opened correctly
    if not camera.isOpened():
        print("Error: Could not open the webcam.")
        return

    # Calibration constant
    calibration_constant = 500

    while True:
        # Capture each frame
        success, frame = camera.read()
        if not success:
            print("Error: Could not capture the frame.")
            break

        # Convert the frame to HSV color space for color segmentation
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Define the red color range in HSV space
        red_lower_bound1 = np.array([0, 100, 100])
        red_upper_bound1 = np.array([10, 255, 255])

        red_lower_bound2 = np.array([160, 100, 100])
        red_upper_bound2 = np.array([179, 255, 255])

        # Create masks for the red color range (handling the circular nature of HSV hue)
        mask_part1 = cv2.inRange(hsv_frame, red_lower_bound1, red_upper_bound1)
        mask_part2 = cv2.inRange(hsv_frame, red_lower_bound2, red_upper_bound2)
        combined_mask = cv2.bitwise_or(mask_part1, mask_part2)

        # Apply Gaussian blur to reduce noise
        blurred_mask = cv2.GaussianBlur(combined_mask, (9, 9), 0)

        # Find contours in the mask
        contours, _ = cv2.findContours(blurred_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # If any contours are found, proceed
        if contours:
            # Assume the largest contour is the laser spot
            max_contour = max(contours, key=cv2.contourArea)
            contour_area = cv2.contourArea(max_contour)

            # Filter out small areas to reduce false positives
            if contour_area > 50:
                # Calculate the moments of the contour to get the centroid
                moments = cv2.moments(max_contour)
                if moments["m00"] != 0:
                    centerX = int(moments["m10"] / moments["m00"])
                    centerY = int(moments["m01"] / moments["m00"])

                    # Draw a circle around the detected laser spot
                    cv2.circle(frame, (centerX, centerY), 15, (0, 255, 0), 2)

                    # Estimate distance
                    distance = calculate_distance(contour_area, calibration_constant)

                    if distance is not None:
                        # Check if distance is within 2 meters ±10 cm
                        if 1.9 <= distance <= 2.1:
                            distance_info = f"Distance: {distance:.2f} m"
                            cv2.putText(frame, distance_info, (centerX - 50, centerY - 40),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

                    # Display detection info
                    cv2.putText(frame, "Laser Pointer", (centerX - 50, centerY - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Display the processed frame
        cv2.imshow('Laser Pointer Detection', frame)

        # Press 'q' key to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close all windows
    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
