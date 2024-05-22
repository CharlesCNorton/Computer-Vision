
# EdgeDetect

**AI-powered security for low-resource edge devices!**

## Overview

EdgeDetect is a proof-of-concept application demonstrating the feasibility of running YOLOv8 object detection on a Raspberry Pi 5 with 8GB of RAM. This project showcases how low-resource edge devices can be utilized for AI-powered security purposes.

## Features

- Real-time object detection using YOLOv8.
- User-friendly GUI for easy interaction.
- Configurable confidence threshold for detection.
- Alarm feature with customizable classes.
- Option to download or select YOLOv8 model.
- Comprehensive error handling and modular code structure.
- Integrated sound alarm for specified object classes using `simpleaudio`.

## Requirements

- Raspberry Pi 5 with 8GB of RAM.
- Python 3.6 or higher.
- Internet connection (for downloading model if not selecting manually).

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/PortfolioAI/Computer-Vision/tree/main/EdgeDetect.git
   cd EdgeDetect
   ```

2. **Run the installation script:**
   ```bash
   python edgedetect.py
   ```

   This script will:
   - Install the required Python packages.
   - Download the YOLOv8 model (if chosen).

## Usage

1. **Run the application:**
   ```bash
   python edgedetect.py
   ```

2. **Model Selection:**
   - You will be prompted to download the YOLOv8 model or select an existing model file from your system.
   
3. **GUI Features:**
   - **Start Webcam Capture**: Begin real-time object detection using your webcam.
   - **Stop Webcam Capture**: Stop the webcam capture.
   - **Toggle Alarm Feature**: Enable or disable the alarm sound.
   - **Set Confidence Threshold**: Adjust the confidence threshold for detections.
   - **Toggle Class Display**: Show or hide specific object classes.
   - **Toggle Alarm Classes**: Specify which classes should trigger the alarm.
   - **Toggle Video Display**: Enable or disable the real-time video display.
   - **Exit**: Close the application.

## Implementation Details

EdgeDetect utilizes the following key components:
- **YOLOv8 Model**: The smallest YOLOv8 model (`yolov8n.pt`) is used for efficient real-time object detection.
- **tkinter**: Provides a graphical user interface for user interactions.
- **simpleaudio**: Plays alarm sounds when specified objects are detected.
- **scipy**: Generates sawtooth waveforms for the alarm sound.

## Model and File Paths

- The YOLOv8 model will be downloaded to `~/Downloads/yolov8n.pt` if not selected manually.

## Notes

- This project is a proof of concept to demonstrate the capabilities of running YOLOv8 on a Raspberry Pi 5 with 8GB of RAM.
- Performance may vary based on the complexity of the YOLOv8 model and the available system resources.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [YOLO by Ultralytics](https://github.com/ultralytics/yolov5)
- [simpleaudio](https://simpleaudio.readthedocs.io/en/latest/)
- [tkinter](https://docs.python.org/3/library/tkinter.html)
- [scipy](https://www.scipy.org/)

## Contact

For any questions or suggestions, please contact [yourname@example.com](mailto:yourname@example.com).
