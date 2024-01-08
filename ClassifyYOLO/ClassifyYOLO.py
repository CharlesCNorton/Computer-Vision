import io
import sys
import contextlib
import torch
import cv2
import threading
import tkinter as tk
from ultralytics import YOLO

class StreamRedirector:
    def __init__(self):
        self.old_stdout = sys.stdout
        self.stream = io.StringIO()

    def __enter__(self):
        sys.stdout = self.stream
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.old_stdout

class ClassificationApp:
    def __init__(self):
        self.model = None
        self.webcam_thread_running = False
        self.latest_inference = ""
        self.load_model('SET PATH HERE')

    def load_model(self, model_path):
        try:
            self.model = YOLO(model_path)
            print("YOLOv8 Classification Model loaded successfully.")
        except Exception as e:
            print(f"Failed to load YOLOv8 Classification Model: {e}")

    def run_inference(self, img):
        with StreamRedirector() as redirector:
            self.model.predict(source=img)
            output = redirector.stream.getvalue()

        self.latest_inference = output.split('\n')[0]

    def webcam_capture(self):
        self.webcam_thread_running = True
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        try:
            if not cap.isOpened():
                print("Error: Could not open webcam.")
                return
            while self.webcam_thread_running:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Unable to read from webcam.")
                    break
                self.run_inference(frame)
                cv2.imshow("Classification", frame)
                if cv2.waitKey(1) == 27:
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.webcam_thread_running = False

    def create_model_switch_buttons(self, root):
        yolov8_classification_model_variants = {
            'Nano': 'SET PATH HERE',
            'Medium': 'SET PATH HERE',
            'XLarge': 'SET PATH HERE'
        }
        for model_name, model_path in yolov8_classification_model_variants.items():
            button = tk.Button(root, text=f"Switch to {model_name} Classification Model",
                               command=lambda mp=model_path: self.load_model(mp))
            button.pack()

    def create_gui(self):
        self.root = tk.Tk()
        self.root.title("Image Classification Settings")
        tk.Button(self.root, text="Start Webcam Capture",
                  command=lambda: threading.Thread(target=self.webcam_capture).start()).pack()

        self.create_model_switch_buttons(self.root)

        self.confidence_label = tk.Label(self.root, text="Confidences will appear here")
        self.confidence_label.pack()

        self.root.after(100, self.update_gui)
        self.root.mainloop()

    def update_gui(self):
        self.confidence_label.config(text=self.latest_inference)
        self.root.after(100, self.update_gui)

def main():
    app = ClassificationApp()
    app.create_gui()

if __name__ == "__main__":
    main()
