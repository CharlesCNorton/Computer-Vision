
# YoloWorld Object Detection App

## Overview
The YoloWorld Object Detection App is a powerful, easy-to-use desktop application that leverages the YOLO (You Only Look Once) deep learning model for real-time object detection. This application is designed to be user-friendly, allowing users to customize detection settings and classes, adjust confidence thresholds, and even capture screenshots of detected objects.

## Features
- Real-time object detection using the YOLO model
- Customizable object classes for detection
- Adjustable confidence threshold for detection precision
- Screenshot capture functionality for detected objects
- Simple and intuitive graphical user interface (GUI)

## Installation
To use the YoloWorld Object Detection App, you need to have Python installed on your computer (Python 3.6 or newer is recommended). Follow these steps to set up the app:

1. Clone the repository or download the source code.
2. Install the required dependencies by running `pip install -r requirements.txt` in your terminal or command prompt.
3. Run the app by executing `python YoloWorld.py`.

## Usage
After launching the app, you can configure the object detection settings through the GUI. The main options include:

- **Set YOLO Model Path**: Select the path to your YOLO model file (.pt format).
- **Start Webcam Capture**: Begin real-time object detection using your webcam.
- **Set Custom Classes**: Specify which object classes you want the model to detect.
- **Set Confidence Threshold**: Adjust the confidence level required for an object to be detected.
- **Toggle Screenshot Feature**: Enable or disable the ability to take screenshots of detected objects.

## Customizing Object Classes
To customize which objects are detected, use the "Set Custom Classes" button in the GUI and enter the class names separated by commas. The app will then reload the model to detect only the specified classes.

## Taking Screenshots
If the screenshot feature is enabled, the app will save screenshots of detected objects to a predefined folder. You can change the screenshot save location by modifying the `screenshot_path` variable in the script.

## License
This project is licensed under the MIT License. See the LICENSE.txt file for details.

## Contributing
Contributions to the YoloWorld Object Detection App are welcome! Please refer to the contribution guidelines for more information on how to contribute to this project.

## Acknowledgements
This project utilizes the YOLO object detection system. We acknowledge the YOLO developers and the open-source community for making such powerful tools accessible.

