import torch
import cv2
import threading
import os
import tkinter as tk
from ultralytics import YOLO

class PoseDetectionApp:
    def __init__(self):
        self.model = None
        self.webcam_thread_running = False
        self.load_model('ENTER MODEL PATH')

    def load_model(self, model_path):
        try:
            self.model = YOLO(model_path)
            print("YOLO Pose Model loaded successfully.")
        except Exception as e:
            print(f"Failed to load YOLO Pose Model: {e}")

    def run_inference(self, img):
        if not hasattr(self.model, 'predict'):
            return []
        results = self.model.predict(source=img)
        poses = []
        for result in results:
            if result.keypoints is not None and result.keypoints.conf is not None:
                keypoints = result.keypoints.xy.cpu().numpy()
                for keypoint_set in keypoints:
                    for keypoint in keypoint_set:
                        x, y = keypoint
                        if x != 0 or y != 0:
                            poses.append((x, y))
        return poses

    def postprocess_and_visualize(self, img, poses):
        connections = [
            (0, 1), (1, 3),
            (0, 2), (2, 4),
            (5, 6),
            (5, 7), (7, 9),
            (6, 8), (8, 10),
            (5, 11), (6, 12),
            (11, 13), (13, 15),
            (12, 14), (14, 16)
        ]
        for point in poses:
            x, y = point
            cv2.circle(img, (int(x), int(y)), 5, (0, 255, 0), -1)
        for start, end in connections:
            if start < len(poses) and end < len(poses):
                start_point = (int(poses[start][0]), int(poses[start][1]))
                end_point = (int(poses[end][0]), int(poses[end][1]))
                cv2.line(img, start_point, end_point, (255, 0, 0), 2)

        return img

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
                poses = self.run_inference(frame)
                frame_with_poses = self.postprocess_and_visualize(frame, poses)
                cv2.imshow("Pose Detection", frame_with_poses)
                if cv2.waitKey(1) == 27:
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.webcam_thread_running = False

    def create_model_switch_buttons(self, root):
        yolov8_pose_model_variants = {'Nano': 'ENTER MODEL PATH', 'Medium': 'ENTER MODEL PATH', 'XLarge': 'ENTER MODEL PATH'}
        for model_name, model_path in yolov8_pose_model_variants.items():
            button = tk.Button(root, text=f"Switch to {model_name} Pose Model", command=lambda mp=model_path: self.load_model(mp))
            button.pack()

    def create_gui(self):
        root = tk.Tk()
        root.title("Pose Detection Settings")
        tk.Button(root, text="Start Webcam Capture", command=lambda: threading.Thread(target=self.webcam_capture).start()).pack()
        self.create_model_switch_buttons(root)
        tk.Button(root, text="Exit", command=lambda: root.destroy()).pack()
        root.mainloop()

def main():
    app = PoseDetectionApp()
    app.create_gui()

if __name__ == "__main__":
    main()
