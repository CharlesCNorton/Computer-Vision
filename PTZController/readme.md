# PTZ Controller

PTZ Controller is a Python application that allows you to control a PTZ (Pan-Tilt-Zoom) camera using a graphical user interface (GUI). The application supports camera movement (pan, tilt, zoom), taking snapshots, and recording video.

## Features

- Pan, Tilt, and Zoom Control Use the GUI buttons or keyboard keys to control the camera's movements.
- Snapshot Capture a snapshot from the camera feed and save it as a PNG file with a timestamp.
- Video Recording Start and stop video recording, saving the video with a timestamp in the filename.

## Requirements

- Python 3.x
- OpenCV
- Tkinter
- Pillow (PIL Fork)
- hidapi

## Installation

1. Install Python 3.x Make sure Python 3.x is installed on your system.
2. Install Required Packages Use pip to install the required packages.
    ```bash
    pip install opencv-python-headless tkinter Pillow hidapi
    ```

## Usage

1. Run the Application
    ```bash
    python ptz_controller.py
    ```

2. Control the Camera
    - Pan Left Click the left arrow button or press the Left arrow key.
    - Pan Right Click the right arrow button or press the Right arrow key.
    - Tilt Up Click the up arrow button or press the Up arrow key.
    - Tilt Down Click the down arrow button or press the Down arrow key.
    - Zoom In Click the `+` button or press the `+` key.
    - Zoom Out Click the `-` button or press the `-` key.

3. Take a Snapshot Click the Snapshot button to capture a snapshot and save it as a PNG file with a timestamp.

4. Record Video
    - Click the Start Recording button to start recording video.
    - Click the Stop Recording button to stop recording. The video will be saved with a timestamp in the filename.

## File Structure

- `ptz_controller.py` The main application script.
- `snapshots` Directory where snapshots are saved.
- `recordings` Directory where video recordings are saved.

## Example

After running the application, you should see a GUI window with buttons to control the camera and capture snapshots or record videos.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue.

## Acknowledgements

- [OpenCV](httpsopencv.org)
- [Tkinter](httpsdocs.python.org3librarytkinter.html)
- [Pillow](httpspython-pillow.org)
- [hidapi](httpsgithub.comtrezorcython-hidapi)

