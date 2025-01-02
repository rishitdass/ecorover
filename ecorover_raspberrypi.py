import cv2
import numpy as np
import time
from ultralytics import YOLO
from picamera2 import Picamera2

# Initialize Picamera2
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (512, 384), "format": "RGB888"})  # Reduced resolution by 20%
picam2.configure(config)
picam2.start()

# Load the YOLO model
model = YOLO('/home/rishitdass/ecorover/ecorover/ecorover-model5n.pt')

# Set YOLO confidence threshold to reduce sensitivity by 40% (further reduction)
confidence_threshold = 0.6  # Even lower confidence threshold

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
while True:
    # Capture a frame from the Picamera2
    start_time = time.time()  # Track the start time of the frame processing
    frame = picam2.capture_array()

    # Flip the frame horizontally (left to right inversion)
    frame = cv2.flip(frame, 1)  # '1' indicates horizontal flip

    # Run detection on the flipped frame with reduced resolution
    results = model.predict(frame, conf=confidence_threshold)  # Set confidence threshold

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

    # Calculate FPS
    end_time = time.time()
    fps = 1 / (end_time - start_time)  # FPS = 1 / Time taken for the frame

    # Display the frame with the outlined boundaries and FPS
    if frame is not None:
        # Convert FPS to a string for display
        fps_text = f"FPS: {fps:.2f}"
        cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("Field Pattern Detection", frame)  # Display after processing
    else:
        print("Error: Frame is None")

    # Exit the loop if the 'Esc' key is pressed
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # 27 is the ASCII value for 'Esc'
        break

# Release resources
picam2.stop()
cv2.destroyAllWindows()