ClassifyYOLO: Real-time Object Classification Application

ClassifyYOLO is a Python-based application designed for real-time object classification using the powerful YOLOv8 model. It captures video frames from a webcam, classifies objects in each frame, and displays the results in a user-friendly GUI.

Features:
- Real-time Webcam Object Classification: Leverages a webcam to capture live video frames and classifies objects in real time using YOLOv8.
- Model Switching Capability: Users can switch between various YOLOv8 classification model variants like Nano, Medium, and XLarge, according to their requirements.
- Graphical User Interface: Includes an intuitive GUI for initiating webcam capture, toggling between different models, and exiting the application.

Requirements:
- Python 3.x
- OpenCV (cv2)
- PyTorch
- Ultralytics YOLO package
- Tkinter for GUI

Installation:
1. Ensure Python 3.x is installed on your system.
2. Install the necessary dependencies:
   pip install torch opencv-python tkinter
3. Clone or install the Ultralytics YOLO package via pip.
4. Download the desired YOLOv8 classification model weights (e.g., yolov8n-cls.pt, yolov8m-cls.pt, yolov8x-cls.pt) from the official Ultralytics website or GitHub repository.

How It Works:
- Model Loading: The application loads the specified YOLOv8 classification model.
- Webcam Capture and Object Classification: Captures video frames from the webcam and utilizes the YOLOv8 model to classify objects in each frame.
- Displaying Classification Results: The classifications, including object names and confidence scores, are displayed in real time in the application's GUI.
- User Interaction through GUI: Provides the functionality to start/stop the webcam, switch between different YOLOv8 models, and exit the application.

Usage Notes:
- Make sure to specify the correct paths to the YOLOv8 model files within the application script.
- A functional webcam is required for the application to operate correctly.
- The application's output, including class names and confidence scores, is displayed in the GUI.

Contributing:
We welcome contributions, issues, and feature requests. Feel free to check our issues page if you'd like to contribute or report a problem.

License:
ClassifyYOLO is distributed under the Creative Commons Attribution-NonCommercial 4.0 International License. See the LICENSE.md file for more details.