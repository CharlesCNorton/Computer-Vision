import os
import sys
import subprocess
import urllib.request
import tkinter as tk
from tkinter import filedialog, messagebox
import pygame

def install_packages():
    packages = [
        'opencv-python', 'numpy', 'tk', 'ultralytics', 'scipy', 'pygame'
    ]
    for package in packages:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])

def download_model():
    model_url = 'https://huggingface.co/Ultralytics/YOLOv8/resolve/main/yolov8n.pt'
    download_path = os.path.join(os.path.expanduser('~'), 'Downloads', 'yolov8n.pt')
    if not os.path.exists(download_path):
        print(f"Downloading model from {model_url}")
        urllib.request.urlretrieve(model_url, download_path)
        print(f"Model downloaded to {download_path}")
    else:
        print(f"Model already exists at {download_path}")
    return download_path

def select_model():
    root = tk.Tk()
    root.withdraw()
    model_path = filedialog.askopenfilename(title="Select YOLOv8 Model File", filetypes=(("PyTorch files", "*.pt"), ("All files", "*.*")))
    return model_path

def main():
    install_packages()
    if messagebox.askyesno("Model Download", "Do you want to download the YOLOv8 model?"):
        model_path = download_model()
    else:
        model_path = select_model()
        if not model_path:
            print("No model selected. Exiting.")
            return
    run_application(model_path)

def run_application(model_path):
    import cv2
    import time
    import tkinter as tk
    from tkinter import simpledialog, messagebox
    import threading
    import numpy as np
    from scipy import signal
    from ultralytics import YOLO

    class EdgeDetectApp:
        def __init__(self, model_path):
            self.class_names = [
                'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
                'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
                'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
                'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                'teddy bear', 'hair drier', 'toothbrush'
            ]
            self.model = YOLO(model_path)
            self.webcam_thread_running = False
            self.confidence_threshold = 0.65
            self.current_frame = None
            self.alarm_classes = set()
            self.allow_alarm = False
            self.display_classes = set(self.class_names)
            self.object_counts = {class_name: 0 for class_name in self.class_names}
            self.thread = None
            self.display_video = False
            pygame.mixer.init()

        def run_inference(self, img):
            if not hasattr(self.model, 'predict'):
                print("Model does not have 'predict' method.")
                return []
            try:
                results = self.model.predict(source=img)
            except Exception as e:
                print(f"Error during inference: {e}")
                return []
            return self.process_results(results)

        def process_results(self, results):
            detections = []
            try:
                for result in results:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confidences = result.boxes.conf.cpu().numpy()
                    class_ids = result.boxes.cls.cpu().numpy()
                    for i, box in enumerate(boxes):
                        x1, y1, x2, y2 = box
                        conf = confidences[i]
                        class_id = class_ids[i]
                        detections.append((x1, y1, x2, y2, conf, class_id))
            except Exception as e:
                print(f"Error processing results: {e}")
            return detections

        def postprocess_and_visualize(self, img, detections):
            self.object_counts = {class_name: 0 for class_name in self.class_names}
            self.current_frame = img.copy()
            try:
                for det in detections:
                    x1, y1, x2, y2, conf, cls_id = det
                    if conf < self.confidence_threshold:
                        continue
                    cls_id = int(cls_id)
                    class_name = self.class_names[cls_id]
                    self.object_counts[class_name] += 1
                    if class_name in self.display_classes:
                        label = f'{class_name} {conf:.2f}'
                        img = cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        img = cv2.putText(img, label, (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    if self.allow_alarm and class_name in self.alarm_classes:
                        self.play_alarm()
            except Exception as e:
                print(f"Error during postprocessing: {e}")
            if self.display_video:
                cv2.imshow('Object Detection', img)

        def play_alarm(self):
            try:
                fs = 44100
                duration = 1.0
                frequency = 1000
                t = np.linspace(0, duration, int(fs * duration), False)
                sawtooth_wave = 0.5 * (1 + signal.sawtooth(2 * np.pi * frequency * t))
                pulse_wave = np.zeros_like(t)
                pulse_wave[::int(fs / 2)] = 1
                alarm_wave = sawtooth_wave * pulse_wave
                audio = np.hstack([alarm_wave, alarm_wave])
                audio *= 32767 / np.max(np.abs(audio))
                audio = audio.astype(np.int16)
                pygame.mixer.init(frequency=fs, size=-16, channels=2)
                sound = pygame.sndarray.make_sound(audio)
                sound.play()
            except Exception as e:
                print(f"Failed to play alarm sound: {e}")

        def start_webcam_capture_thread(self):
            if self.thread is None or not self.thread.is_alive():
                self.webcam_thread_running = True
                self.thread = threading.Thread(target=self.webcam_capture, args=())
                self.thread.start()
            else:
                messagebox.showinfo("Info", "Webcam capture is already running.")

        def webcam_capture(self):
            cap = cv2.VideoCapture(0)
            try:
                if not cap.isOpened():
                    print("Error: Could not open webcam.")
                    return
                while self.webcam_thread_running:
                    ret, frame = cap.read()
                    if not ret:
                        print("Error: Unable to read from webcam.")
                        break
                    detections = self.run_inference(frame)
                    self.postprocess_and_visualize(frame, detections)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            except Exception as e:
                print(f"Error during webcam capture: {e}")
            finally:
                cap.release()
                cv2.destroyAllWindows()
                self.webcam_thread_running = False

        def stop_webcam_capture(self):
            if self.webcam_thread_running:
                self.webcam_thread_running = False
                if self.thread:
                    self.thread.join()
                messagebox.showinfo("Info", "Webcam capture stopped.")
            else:
                messagebox.showinfo("Info", "Webcam capture is not running.")

        def toggle_video_display(self):
            self.display_video = not self.display_video
            print(f"Video display {'enabled' if self.display_video else 'disabled'}.")

        def create_gui(self):
            root = tk.Tk()
            root.title("EdgeDetect - AI-powered security for low-resource edge devices!")
            tk.Button(root, text="Start Webcam Capture", command=self.start_webcam_capture_thread).pack()
            tk.Button(root, text="Stop Webcam Capture", command=self.stop_webcam_capture).pack()
            tk.Button(root, text="Toggle Alarm Feature", command=self.toggle_alarm_feature).pack()
            tk.Button(root, text="Set Confidence Threshold", command=self.set_confidence_threshold).pack()
            tk.Button(root, text="Toggle Class Display", command=self.toggle_class_display).pack()
            tk.Button(root, text="Toggle Alarm Classes", command=self.toggle_alarm_class).pack()
            tk.Button(root, text="Toggle Video Display", command=self.toggle_video_display).pack()
            tk.Button(root, text="Exit", command=root.destroy).pack()
            root.mainloop()

        def toggle_alarm_feature(self):
            self.allow_alarm = not self.allow_alarm
            print(f"Alarm feature {'enabled' if self.allow_alarm else 'disabled'}.")

        def set_confidence_threshold(self):
            new_threshold = simpledialog.askfloat("Confidence Threshold", "Enter new confidence threshold (0.0 to 1.0):", minvalue=0.0, maxvalue=1.0)
            if new_threshold is not None:
                self.confidence_threshold = new_threshold
                print(f"Confidence threshold set to {self.confidence_threshold}")

        def toggle_class_display(self):
            class_input = simpledialog.askstring("Input", "Enter class name to toggle on/off:")
            if class_input:
                class_input = class_input.strip()
                if class_input in self.class_names:
                    if class_input in self.display_classes:
                        self.display_classes.remove(class_input)
                        print(f"Class '{class_input}' is now hidden.")
                    else:
                        self.display_classes.add(class_input)
                        print(f"Class '{class_input}' is now visible.")
                else:
                    print("Invalid class name. Please enter a valid class name from the list.")

        def toggle_alarm_class(self):
            class_input = simpledialog.askstring("Input", "Enter class name to toggle on/off for alarm:")
            if class_input:
                class_input = class_input.strip()
                if class_input in self.class_names:
                    if class_input in self.alarm_classes:
                        self.alarm_classes.remove(class_input)
                        print(f"{class_input} removed from alarm classes.")
                    else:
                        self.alarm_classes.add(class_input)
                        print(f"{class_input} added to alarm classes.")
                else:
                    print("Invalid class name.")

    app = EdgeDetectApp(model_path)
    app.create_gui()

if __name__ == "__main__":
    main()
