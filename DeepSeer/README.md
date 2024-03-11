
# DeepSeer

DeepSeer is a versatile tool designed to bridge the gap between visual content and natural language processing. Powered by the advanced capabilities of the DeepSeeker model, DeepSeer offers an intuitive interface for image analysis and text-based interactions, making it an excellent tool for exploring visual language understanding. The name "DeepSeer" reflects the model's ability to "see" and understand visual inputs deeply, providing users with insights and interactions based on the visual content.

## Features

- Model Initialization and Selection
- Image Capture via Webcam
- Image Analysis with Detailed Descriptions
- Interactive Chat with Support for Image Inputs
- User-friendly CLI and optional GUI interactions

## Setup

### Prerequisites

- Python 3.8 or newer
- pip and virtualenv
- Access to a GPU is recommended for optimal performance

### Installation

1. Clone the repository or download the script files to your local machine.
2. Navigate to the script's directory.
3. Create a virtual environment:

    ```bash
    python -m virtualenv venv
    ```

4. Activate the virtual environment:

    - On Windows:

        ```bash
        .\venv\Scripts\activate
        ```

    - On macOS and Linux:

        ```bash
        source venv/bin/activate
        ```

5. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. **Model Initialization**

    First, you need to select the directory containing the DeepSeeker model files. This will be prompted when you run the script.

2. **Running DeepSeer**

    To start the application, ensure you're in the script's directory and your virtual environment is activated. Then, run:

    ```bash
    python deepseer.py
    ```

3. **Main Menu**

    Follow the on-screen instructions to interact with DeepSeer. The main menu offers options to chat, analyze images, or exit the application.

### Chat with DeepSeer

Type your text input directly or select 'upload image' to analyze an image. You can also capture an image from your webcam by choosing the 'webcam' option.

### Analyze Image

Select 'Analyze Image' from the main menu to upload and analyze an image directly.

## Additional Notes

- For best performance, use images with clear subjects and minimal background clutter.
- Ensure you have the necessary permissions for webcam access if you plan to use the webcam feature.
- This tool is intended for educational and research purposes. Please respect privacy and copyright laws when analyzing images.

## Troubleshooting

- If you encounter issues with tkinter dialogs appearing in the background, ensure that your Python and tkinter installations are up to date.
- For issues related to model loading, verify the model directory path and ensure it contains the correct model files.
