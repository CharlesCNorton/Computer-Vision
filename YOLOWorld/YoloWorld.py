import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
import threading
import os
import time
import cv2
from ultralytics import YOLO
import numpy as np

class ObjectDetectionApp:
    def __init__(self):
        self.model = None
        self.webcam_thread_running = False
        self.confidence_threshold = 0.65
        self.current_frame = None
        self.allow_screenshots = False
        self.screenshot_path = 'D:\\Screenshots\\'
        self.display_classes = set()
        self.model_path = ''
        self.initialize_model()

    def initialize_model(self):
        if self.model_path:
            try:
                self.model = YOLO(self.model_path)
                if self.display_classes:
                    self.model.set_classes(list(self.display_classes))
                print("YOLO Model loaded successfully.")
            except Exception as e:
                messagebox.showerror("Error", "Failed to load the YOLO model: " + str(e))
        else:
            messagebox.showinfo("Information", "Model path is not set.")

    def set_model_path(self):
        path = filedialog.askopenfilename(title="Select YOLO Model File", filetypes=[("Model Files", "*.pt")])
        if path:
            self.model_path = path
            self.initialize_model()

    def set_custom_classes(self):
        custom_classes = simpledialog.askstring("Custom Classes", "Enter classes for detection (comma-separated):")
        if custom_classes:
            new_display_classes = set(cls.strip() for cls in custom_classes.split(','))
            if new_display_classes != self.display_classes:
                self.display_classes = new_display_classes
                self.initialize_model()
                messagebox.showinfo("Information", "Detection classes updated and model reloaded.")
            else:
                messagebox.showinfo("Information", "Detection classes remain unchanged.")

    def set_confidence_threshold(self):
        new_threshold = simpledialog.askfloat("Confidence Threshold", "Enter new confidence threshold (0.0 - 1.0):", minvalue=0.0, maxvalue=1.0)
        if new_threshold is not None:
            self.confidence_threshold = new_threshold

    def toggle_screenshot_feature(self):
        self.allow_screenshots = not self.allow_screenshots

    def run_inference(self, img):
        if not self.model:
            messagebox.showerror("Error", "Model is not loaded.")
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
        if not detections:
            return img
        for det in detections:
            x1, y1, x2, y2, conf, cls_id = det
            if conf < self.confidence_threshold:
                continue
            class_name = self.model.names[cls_id]
            if class_name in self.display_classes:
                label = f'{class_name} {conf:.2f}'
                img = cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                img = cv2.putText(img, label, (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return img

    def create_gui(self):
        root = tk.Tk()
        root.title("Object Detection Settings")
        tk.Button(root, text="Set YOLO Model Path", command=self.set_model_path).pack()
        tk.Button(root, text="Start Webcam Capture", command=lambda: threading.Thread(target=self.webcam_capture).start()).pack()
        tk.Button(root, text="Set Custom Classes", command=self.set_custom_classes).pack()
        tk.Button(root, text="Set Confidence Threshold", command=self.set_confidence_threshold).pack()
        tk.Button(root, text="Toggle Screenshot Feature", command=self.toggle_screenshot_feature).pack()
        tk.Button(root, text="Exit", command=lambda: root.destroy()).pack()
        root.mainloop()

    def webcam_capture(self):
        if not self.model_path:
            messagebox.showerror("Error", "Model path is not set.")
            return
        self.webcam_thread_running = True
        cap = cv2.VideoCapture(0)
        try:
            while self.webcam_thread_running:
                ret, frame = cap.read()
                if not ret:
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

def main():
    app = ObjectDetectionApp()
    if not os.path.exists(app.screenshot_path):
        os.makedirs(app.screenshot_path)
    app.create_gui()

if __name__ == "__main__":
    main()