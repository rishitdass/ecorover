import cv2
import numpy as np
from ultralytics import YOLO
from shapely.geometry import box

# Load the YOLO model
model = YOLO('saved-model.pt')

# Capture video from the webcam
cap = cv2.VideoCapture(0)


# Function to outline each tilled soil pattern within bounding boxes
def outline_soil_pattern(frame, xywh, xyxy):
    # Unpack bounding box information
    x_min, y_min, x_max, y_max = map(int, xyxy[0])

    # Define the region of interest (ROI) within the bounding box
    roi = frame[y_min:y_max, x_min:x_max]

    # Convert the ROI to grayscale for processing
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Use GaussianBlur to smooth the image and reduce noise
    blurred_roi = cv2.GaussianBlur(gray_roi, (5, 5), 0)

    # Apply Canny edge detection to detect potential boundaries
    edges = cv2.Canny(blurred_roi, threshold1=50, threshold2=150)

    # Find contours in the edge-detected ROI
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Offset contours back to original frame coordinates and draw
    for contour in contours:
        # Offset the contour coordinates to fit the original frame position
        contour += [x_min, y_min]

        # Draw each contour on the frame with a thinner line
        cv2.drawContours(frame, [contour], -1, (0, 255, 0), 1)  # Thickness set to 1 for narrower line


# Main loop to process each frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally (left to right inversion)
    frame = cv2.flip(frame, 1)  # '1' indicates horizontal flip

    # Run detection on the flipped frame
    results = model.predict(frame)

    # Iterate over each detected result
    for result in results:
        boxes = result.boxes

        # Process each box detection
        for box in boxes:
            # Extract bounding box information
            xywh = box.xywh.numpy()  # Center x, y, width, height
            xyxy = box.xyxy.numpy()  # Bounding box x_min, y_min, x_max, y_max

            # Outline each tilled soil pattern using contours
            outline_soil_pattern(frame, xywh, xyxy)

    # Display the frame with the outlined boundaries
    cv2.imshow("Field Pattern Detection", frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()
