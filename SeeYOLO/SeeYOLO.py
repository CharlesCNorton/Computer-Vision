import torch
import cv2
import threading
import os
import time
import tkinter as tk
from tkinter import simpledialog
from ultralytics import YOLO

class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
               'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
               'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
               'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
               'scissors', 'teddy bear', 'hair drier', 'toothbrush']
yolov8_model_variants = {'Nano': 'enter path to yolov8n.pt', 'Medium': 'enter path to yolov8m.pt', 'XLarge': ' enter path to yolov8x.pt'}

class ObjectDetectionApp:
    def __init__(self):
        self.model = None
        self.webcam_thread_running = False
        self.confidence_threshold = 0.45
        self.current_frame = None
        self.screenshot_objects = set()
        self.allow_screenshots = False
        self.screenshot_path = 'enter path to screenshots folder'
        self.display_classes = set(class_names)
        self.object_counts = {class_name: 0 for class_name in class_names}
        self.load_model('enter path to yolov8x.pt')

    def load_model(self, model_path):
        try:
            self.model = YOLO(model_path)
            print("YOLO Model loaded successfully.")
        except Exception as e:
            print(f"Failed to load YOLO Model: {e}")

    def run_inference(self, img):
        if not hasattr(self.model, 'predict'):
            return []
        results = self.model.predict(source=img)
        detections = []
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy()
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box
                conf = confidences[i]
                class_id = class_ids[i]
                detections.append((x1, y1, x2, y2, conf, class_id))
        return detections

    def postprocess_and_visualize(self, img, detections):
        self.object_counts = {class_name: 0 for class_name in class_names}
        self.current_frame = img.copy()
        for det in detections:
            x1, y1, x2, y2, conf, cls_id = det
            if conf < self.confidence_threshold:
                continue
            cls_id = int(cls_id)
            class_name = class_names[cls_id]
            self.object_counts[class_name] += 1
            if class_name in self.display_classes:
                label = f'{class_name} {conf:.2f}'
                img = cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                img = cv2.putText(img, label, (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            if self.allow_screenshots and class_name in self.screenshot_objects:
                self.save_screenshot()
        return img

    def save_screenshot(self):
        if self.current_frame is not None:
            filename = f"{self.screenshot_path}screenshot_{int(time.time())}.png"
            cv2.imwrite(filename, self.current_frame)
            print(f"Screenshot saved: {filename}")

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
                detections = self.run_inference(frame)
                frame_with_detections = self.postprocess_and_visualize(frame, detections)
                self.current_frame = frame_with_detections.copy()
                cv2.imshow("Object Detection", frame_with_detections)
                if cv2.waitKey(1) == 27:
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.webcam_thread_running = False
    def print_object_counts(self):
        print("\nCurrent Object Counts:")
        for class_name, count in self.object_counts.items():
            print(f"{class_name}: {count}")

    def print_class_menu(self):
        print("\nClasses:")
        for class_name in class_names:
            status = "On" if class_name in self.display_classes else "Off"
            print(f"{class_name}: {status}")

    def toggle_class_display(self):
        self.print_class_menu()
        class_input = simpledialog.askstring("Input", "Enter class name to toggle on/off:")
        if class_input:
            class_input = class_input.strip()
            matched_class = next((name for name in class_names if name.lower() == class_input.lower()), None)
            if matched_class is None:
                print("Invalid class name. Please enter a valid class name from the list.")
                return
            if matched_class in self.display_classes:
                self.display_classes.remove(matched_class)
                print(f"Class '{matched_class}' is now hidden.")
            else:
                self.display_classes.add(matched_class)
                print(f"Class '{matched_class}' is now visible.")

    def set_confidence_threshold(self):
        new_threshold = simpledialog.askfloat("Confidence Threshold", "Enter new confidence threshold (0.0 to 1.0):", minvalue=0.0, maxvalue=1.0)
        if new_threshold is not None:
            self.confidence_threshold = new_threshold
            print(f"Confidence threshold set to {self.confidence_threshold}")

    def toggle_screenshot_object(self):
        print("\nScreenshot Classes:")
        for class_name in class_names:
            status = "On" if class_name in self.screenshot_objects else "Off"
            print(f"{class_name}: {status}")
        class_input = simpledialog.askstring("Input", "Enter class name to toggle on/off for screenshots:")
        if class_input:
            class_input = class_input.strip()
            matched_class = next((name for name in class_names if name.lower() == class_input.lower()), None)
            if matched_class is None:
                print("Invalid class name.")
                return
            if matched_class in self.screenshot_objects:
                self.screenshot_objects.remove(matched_class)
                print(f"{matched_class} removed from screenshot classes.")
            else:
                self.screenshot_objects.add(matched_class)
                print(f"{matched_class} added to screenshot classes.")

    def toggle_screenshot_feature(self):
        self.allow_screenshots = not self.allow_screenshots
        print(f"Screenshot feature {'enabled' if self.allow_screenshots else 'disabled'}.")

    def create_model_switch_buttons(self, root):
        for model_name, model_path in yolov8_model_variants.items():
            button = tk.Button(root, text=f"Switch to {model_name}", command=lambda mp=model_path: self.load_model(mp))
            button.pack()

    def create_gui(self):
        root = tk.Tk()
        root.title("Object Detection Settings")
        tk.Button(root, text="Start Webcam Capture", command=lambda: threading.Thread(target=self.webcam_capture).start()).pack()
        tk.Button(root, text="Toggle Class Display", command=self.toggle_class_display).pack()
        tk.Button(root, text="Set Confidence Threshold", command=self.set_confidence_threshold).pack()
        tk.Button(root, text="Toggle Screenshot Objects", command=self.toggle_screenshot_object).pack()
        tk.Button(root, text="Toggle Screenshot Feature", command=self.toggle_screenshot_feature).pack()
        self.create_model_switch_buttons(root)
        tk.Button(root, text="Exit", command=lambda: root.destroy()).pack()
        root.mainloop()

def main():
    app = ObjectDetectionApp()
    if not os.path.exists(app.screenshot_path):
        os.makedirs(app.screenshot_path)
    app.create_gui()

if __name__ == "__main__":
    main()
