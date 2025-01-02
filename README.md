
# EcoRover Raspberry Pi Soil Detection

This repository contains the code and models to detect tilled soil patterns in real-time using a Raspberry Pi and YOLOv5. The script `ecorover-raspberrypi.py` utilizes the **Picamera2** library for capturing images and the **YOLOv5** models for detecting soil patterns. The project is designed to be used with an EcoRover, where the Raspberry Pi is mounted for on-field soil detection and analysis.

## Project Overview

The goal of this project is to detect tilled soil patterns in real-time using a Raspberry Pi running a lightweight YOLOv5 model. The detection process is optimized for low-resource devices and provides real-time feedback for field analysis. The project utilizes:

- **YOLOv5** (v5n and v5s) for object detection.
- **Picamera2** for capturing live video.
- **OpenCV** for image processing and displaying results.

## Requirements

- **Hardware:**
  - Raspberry Pi 4 Model B (or compatible)
  - Raspberry Pi Camera Module (Picamera2)
  - A heat sink case for Raspberry Pi (optional, for thermal management)

- **Software:**
  - Raspberry Pi OS (latest version)
  - Python 3.x
  - Required Python libraries:
    - `ultralytics` (for YOLOv5)
    - `picamera2` (for camera interface)
    - `opencv-python` (for image processing)
    - `numpy` (for numerical operations)

## Installation

1. **Clone this repository:**

   ```bash
   git clone https://github.com/rishitdass/ecorover.git
   ```

2. **Install required Python packages:**

   ```bash
   pip install -r requirements.txt
   ```


3. **Download the YOLOv5 models:**

   - `ecorover-model5n.pt` (YOLOv5 Nano model)
   - `ecorover-model5s.pt` (YOLOv5 Small model)

   Place the models in the root directory of this repository or modify the path in the code accordingly.

## Usage

### Running the Script

To run the detection script on your Raspberry Pi, execute the following command:

```bash
python3 ecorover-raspberrypi.py
```

This will start capturing frames from the Raspberry Pi camera, process the images using YOLOv5, and display the detected tilled soil patterns with bounding boxes on the screen. The script will also show the **FPS (Frames Per Second)** in the top left corner.

### Switching Models

By default, the script loads the **YOLOv5 Nano** model (`ecorover-model5n.pt`). If you want to use the **YOLOv5 Small** model (`ecorover5s.pt`), you can change the model loading line in the script:

```python
model = YOLO('/path/to/your/model/ecorover-model5n.pt')  # For Nano model
```

To use the small model:

```python
model = YOLO('/path/to/your/model/ecorover5s.pt')  # For Small model
```

### Camera Configuration

The script uses **Picamera2** to capture video. You can adjust the camera configuration to change resolution or other settings:

```python
config = picam2.create_preview_configuration(main={"size": (512, 384), "format": "RGB888"})  # Adjust resolution
```

## Key Features

- **Real-time detection:** Detects tilled soil patterns in real-time with reduced latency.
- **Frame processing:** The script processes frames from the camera and performs YOLOv5-based object detection.
- **FPS Display:** The Frames Per Second (FPS) is displayed on the live stream to indicate the performance.
- **Resource optimization:** Utilizes a low-confidence threshold to reduce sensitivity and improve performance on resource-constrained devices like the Raspberry Pi.

## Project Setup

This repository is set up for easy integration with a Raspberry Pi running the EcoRover. Simply mount the Raspberry Pi with the camera to the rover and start the detection script. The detection results can be used for soil pattern analysis in real-time.

## Contributing

Feel free to fork this repository and submit pull requests. Contributions are welcome!

### Future Work

- **Improve accuracy:** Enhance detection models and fine-tune them for better results.
- **Frame interpolation:** Implement techniques like frame interpolation to improve FPS and performance.
- **Cloud Integration:** Consider offloading processing tasks to the cloud for heavier computations.

