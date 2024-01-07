import torch
import cv2
import threading
import os
import tkinter as tk
from tkinter import filedialog, simpledialog
from ultralytics import YOLO

yolov5_repo_path = 'enter path here'
yolov8_model_path = 'enter path here'

model = None
device = 'cuda' if torch.cuda.is_available() else 'cpu'

webcam_thread_running = False
confidence_threshold = 0.45
current_frame = None
screenshot_objects = set()
allow_screenshots = False
screenshot_path = 'enter path here'

class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
               'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
               'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
               'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
               'scissors', 'teddy bear', 'hair drier', 'toothbrush']

object_counts = {class_name: 0 for class_name in class_names}
display_classes = set(class_names)

def run_inference(model, img):
    if hasattr(model, 'predict'):
        results = model.predict(source=img)
        detections = []

        if isinstance(results, list):
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

    else:
        results = model(img)
        detections = []
        boxes = results.xyxy[0].cpu().numpy()

        for box in boxes:
            if box[4] >= confidence_threshold:  
                x1, y1, x2, y2, conf, class_id = box[:6]
                detections.append((x1, y1, x2, y2, conf, int(class_id)))
        return detections


def postprocess_and_visualize(img, detections):
    global confidence_threshold, object_counts, current_frame
    object_counts = {class_name: 0 for class_name in class_names}
    current_frame = img.copy()
    for det in detections:
        x1, y1, x2, y2, conf, cls_id = det
        if conf < confidence_threshold:
            continue
        cls_id = int(cls_id)
        class_name = class_names[cls_id]
        object_counts[class_name] += 1
        if class_name in display_classes:
            label = f'{class_name} {conf:.2f}'
            img = cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            img = cv2.putText(img, label, (int(x1), int(y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
        if allow_screenshots and class_name in screenshot_objects:
            save_screenshot()
    return img

def save_screenshot():
    if current_frame is not None:
        filename = f"{screenshot_path}screenshot_{int(time.time())}.png"
        cv2.imwrite(filename, current_frame)
        print(f"Screenshot saved: {filename}")

def webcam_capture(model):
    global webcam_thread_running, current_frame
    webcam_thread_running = True
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    try:
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return
        while webcam_thread_running:
            ret, frame = cap.read()
            if not ret:
                print("Error: Unable to read from webcam.")
                break
            detections = run_inference(model, frame)
            frame_with_detections = postprocess_and_visualize(frame, detections)
            current_frame = frame_with_detections.copy()
            cv2.imshow("Object Detection", frame_with_detections)
            key = cv2.waitKey(1)
            if key == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        webcam_thread_running = False
def print_object_counts():
    global object_counts
    print("\nCurrent Object Counts:")
    for class_name, count in object_counts.items():
        print(f"{class_name}: {count}")

def print_class_menu():
    global display_classes
    print("\nClasses:")
    for class_name in class_names:
        status = "On" if class_name in display_classes else "Off"
        print(f"{class_name}: {status}")

def toggle_class_display():
    global display_classes
    print_class_menu()
    class_input = simpledialog.askstring("Input", "Enter class name to toggle on/off:")
    if class_input:
        class_input = class_input.strip()
        matched_class = next((name for name in class_names if name.lower() == class_input.lower()), None)
        if matched_class is None:
            print("Invalid class name. Please enter a valid class name from the list.")
            return
        if matched_class in display_classes:
            display_classes.remove(matched_class)
            print(f"Class '{matched_class}' is now hidden.")
        else:
            display_classes.add(matched_class)
            print(f"Class '{matched_class}' is now visible.")

def load_model(model_choice):
    global model
    try:
        if model_choice == 'yolov5':
            model = torch.hub.load(yolov5_repo_path, 'custom', path='yolov5s.pt', source='local')
        elif model_choice == 'yolov8':
            model = YOLO(yolov8_model_path)
        print(f"Model {model_choice} loaded successfully.")
    except Exception as e:
        print(f"Failed to load model {model_choice}: {e}")

def set_confidence_threshold():
    global confidence_threshold
    new_threshold = simpledialog.askfloat("Confidence Threshold", "Enter new confidence threshold (0.0 to 1.0):", minvalue=0.0, maxvalue=1.0)
    if new_threshold is not None:
        confidence_threshold = new_threshold
        print(f"Confidence threshold set to {confidence_threshold}")

def toggle_screenshot_object():
    global screenshot_objects
    print("\nScreenshot Classes:")
    for class_name in class_names:
        status = "On" if class_name in screenshot_objects else "Off"
        print(f"{class_name}: {status}")
    class_input = simpledialog.askstring("Input", "Enter class name to toggle on/off for screenshots:")
    if class_input:
        class_input = class_input.strip()
        matched_class = next((name for name in class_names if name.lower() == class_input.lower()), None)
        if matched_class is None:
            print("Invalid class name.")
            return
        if matched_class in screenshot_objects:
            screenshot_objects.remove(matched_class)
            print(f"{matched_class} removed from screenshot classes.")
        else:
            screenshot_objects.add(matched_class)
            print(f"{matched_class} added to screenshot classes.")

def toggle_screenshot_feature():
    global allow_screenshots
    allow_screenshots = not allow_screenshots
    print(f"Screenshot feature {'enabled' if allow_screenshots else 'disabled'}.")

def select_model():
    global model
    model_choice = simpledialog.askstring("Select Model", "Enter model to use (yolov5, yolov8):")
    if model_choice and model_choice.lower() in ['yolov5', 'yolov8']:
        load_model(model_choice.lower())
    else:
        print("Invalid model choice. Please enter 'yolov5' or 'yolov8'.")

def create_gui():
    root = tk.Tk()
    root.title("Object Detection Settings")

    model_button = tk.Button(root, text="Select Model", command=select_model)
    model_button.pack()

    start_button = tk.Button(root, text="Start Webcam Capture", command=lambda: threading.Thread(target=webcam_capture, args=(model,)).start())
    start_button.pack()

    toggle_button = tk.Button(root, text="Toggle Class Display", command=toggle_class_display)
    toggle_button.pack()

    threshold_button = tk.Button(root, text="Set Confidence Threshold", command=set_confidence_threshold)
    threshold_button.pack()

    screenshot_object_button = tk.Button(root, text="Toggle Screenshot Objects", command=toggle_screenshot_object)
    screenshot_object_button.pack()

    screenshot_toggle_button = tk.Button(root, text="Toggle Screenshot Feature", command=toggle_screenshot_feature)
    screenshot_toggle_button.pack()

    exit_button = tk.Button(root, text="Exit", command=lambda: root.destroy())
    exit_button.pack()

    root.mainloop()

if __name__ == "__main__":
    if not os.path.exists(screenshot_path):
        os.makedirs(screenshot_path)
    create_gui()