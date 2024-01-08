PoseYOLO: Real-time Pose Detection Application

PoseYOLO is a Python application for real-time human pose detection using the YOLOv8 model. It captures poses from a webcam stream, visualizes keypoints and skeletal connections, and provides a GUI for user interaction.

Features:
- Real-time Webcam Pose Tracking: Utilizes a webcam to capture and process frames in real-time, identifying human poses.
- Model Switching Capability: Offers the flexibility to switch between different YOLOv8 model variants (Nano, Medium, XLarge) for pose detection.
- Graphical User Interface: Provides a simple and interactive GUI for starting the webcam, switching models, and exiting the application.

Requirements:
- Python 3.x
- OpenCV (cv2)
- PyTorch
- Ultralytics YOLO package
- Tkinter for GUI

Installation:
1. Ensure Python 3.x is installed.
2. Install the required packages:
   pip install torch opencv-python tkinter
3. Clone the Ultralytics YOLO repository or install it via pip.
4. Download the desired YOLOv8 model weights (e.g., yolov8x-pose.pt) from the official YOLO website.

How it Works:
- Model Loading: Loads the specified YOLO model for pose detection.
- Webcam Capture and Processing: Captures frames from the webcam and processes them using the YOLO model to detect human poses.
- Pose Visualization: Visualizes the detected poses by drawing keypoints and skeletal connections on the frames.
- GUI Interaction: Allows users to interact with the application, switch models, and control the webcam stream.

Note:
- Ensure the model paths are correctly set in the script for model switching functionality.
- The application requires a functional webcam connected to the computer.

Contributing:
Contributions, issues, and feature requests are welcome. Feel free to check the issues page for open issues or to open a new one.

License:
This project is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License - see the LICENSE.md file for details.

