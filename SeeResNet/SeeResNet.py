import tensorflow as tf
import cv2
import numpy as np
import threading
import time
import logging as py_logging
import tkinter as tk
from tkinter import filedialog
import os

tf.get_logger().setLevel('ERROR')

print("TensorFlow will use CPU.")

logging_enabled = False
display_detections = False
webcam_thread_running = False
take_screenshot_enabled = False
screenshot_class = None
previous_detections = set()
excluded_objects = set()
current_resolution = 'native'

MODEL_PATH = "ENTER RESNET MODEL PATH"

try:
    model = tf.saved_model.load(MODEL_PATH)
except Exception as e:
    print(f"Error loading model from {MODEL_PATH}: {e}")
    exit()

class_names = [
    'background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'placeholder1',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'placeholder2',
    'backpack', 'umbrella', 'placeholder3', 'placeholder4', 'handbag', 'tie',
    'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'placeholder5', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
    'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
    'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
    'placeholder6', 'dining table', 'placeholder7', 'placeholder8', 'toilet',
    'placeholder9', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'placeholder10',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

def toggle_tensorflow_logging():
    global logging_enabled
    if logging_enabled:
        tf.get_logger().setLevel('INFO')
    else:
        tf.get_logger().setLevel('ERROR')

def update_detections(detections, previous_detections):
    current_detections = set()
    for detection in detections:
        class_name, score, _ = detection
        if score >= 0.6:
            current_detections.add(class_name)

    new_objects = current_detections - previous_detections
    disappeared_objects = previous_detections - current_detections

    return current_detections, new_objects, disappeared_objects

def preprocess_image(img):
    try:
        img = cv2.resize(img, (1024, 1024))
        img = img.astype(np.uint8)
        return tf.convert_to_tensor([img], dtype=tf.uint8)
    except Exception as e:
        print(f"Error during image preprocessing: {e}")
        return None
def run_inference(model, img):
    input_tensor = preprocess_image(img)
    if input_tensor is None:
        return None
    try:
        detections = model(input_tensor)
        return detections
    except Exception as e:
        print(f"Error during model inference: {e}")
        return None

def postprocess_and_visualize(img, detections):
    if detections is None:
        return img, set(), set(), set()

    global display_detections, previous_detections, excluded_objects

    try:
        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
        detections['num_detections'] = num_detections
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

        detection_info = []

        for i in range(num_detections):
            class_id = detections['detection_classes'][i]
            class_name = class_names[class_id]
            if class_name in excluded_objects:
                continue

            box = detections['detection_boxes'][i]
            score = detections['detection_scores'][i]

            if score < 0.5:
                continue

            h, w, _ = img.shape
            box = [int(v) for v in (box[0] * h, box[1] * w, box[2] * h, box[3] * w)]
            img = cv2.rectangle(img, (box[1], box[0]), (box[3], box[2]), (0, 255, 0), 2)
            label = f'{class_name}, Score: {score:.2f}'
            img = cv2.putText(img, label, (box[1], box[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

            detection_info.append((class_name, score, time.time()))

        current_detections, new_objects, disappeared_objects = update_detections(detection_info, previous_detections)
        previous_detections = current_detections

        if display_detections:
            for obj in new_objects:
                print(f"Appeared: {obj}, Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            for obj in disappeared_objects:
                print(f"Disappeared: {obj}, Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

        return img, current_detections, new_objects, disappeared_objects
    except Exception as e:
        print(f"Error during post-processing: {e}")
        return img, set(), set(), set()

def load_new_model():
    global model
    root = tk.Tk()
    root.withdraw()
    model_path = filedialog.askdirectory(title="Select Model Directory")
    if model_path:
        try:
            model = tf.saved_model.load(model_path)
            print(f"Model loaded from {model_path}")
        except Exception as e:
            print(f"Failed to load model from {model_path}: {e}")
    else:
        print("Model loading cancelled.")

def change_resolution():
    global current_resolution
    print("\nAvailable Resolutions: 'native', '1024', '640', '320'")
    resolution = input("Enter desired resolution: ")
    if resolution in ['native', '1024', '640', '320']:
        current_resolution = resolution
        print(f"Resolution changed to {resolution}")
    else:
        print("Invalid resolution. Please choose from 'native', '1024', '640', '320'.")

def take_screenshot(frame):
    save_path = "D:\\AI Screenshots"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"{save_path}\\screenshot_{timestamp}.png"
    cv2.imwrite(filename, frame)
    print(f"Screenshot taken: {filename}")

def webcam_capture(model):
    global webcam_thread_running, current_resolution, take_screenshot_enabled, previous_detections, screenshot_class
    webcam_thread_running = True
    cap = cv2.VideoCapture(0)

    resolution_settings = {
        'native': (cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        '1024': (1024, 768),
        '640': (640, 480),
        '320': (320, 240)
    }

    try:
        width, height = resolution_settings[current_resolution]
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return

        while webcam_thread_running:
            ret, frame = cap.read()
            if not ret:
                print("Error: Unable to read from webcam.")
                break

            detections = run_inference(model, frame)
            if detections is None:
                continue

            frame_with_detections, current_detections, new_objects, _ = postprocess_and_visualize(frame, detections)

            if take_screenshot_enabled and screenshot_class in new_objects:
                cap.release()
                take_screenshot(frame)
                cap = cv2.VideoCapture(0)

            previous_detections = current_detections

            cv2.imshow("Object Detection", frame_with_detections)
            key = cv2.waitKey(1)
            if key == 27:
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        webcam_thread_running = False

def select_all_objects():
    global excluded_objects
    excluded_objects.clear()
    print("All objects selected for detection.")

def unselect_all_objects():
    global excluded_objects
    excluded_objects = set(class_names)
    excluded_objects.remove('background')
    print("All objects unselected for detection.")

def menu():
    global logging_enabled, display_detections, webcam_thread_running, excluded_objects, take_screenshot_enabled, screenshot_class

    print("\n--- Menu ---")
    print("1. Start Webcam Capture")
    print("2. Toggle TensorFlow Logging")
    print("3. Toggle Display of Detections")
    print("4. Add/Remove Excluded Object")
    print("5. Select All Objects")
    print("6. Unselect All Objects")
    print("7. Load New Model")
    print("8. Change Resolution")
    print("9. Toggle Screenshot on Detection")
    print("10. Set Screenshot Detection Class")
    print("11. Exit")
    while True:
        choice = input("Enter your choice: ")

        if choice == '1':
            if not webcam_thread_running:
                threading.Thread(target=webcam_capture, args=(model,)).start()
            else:
                print("Webcam capture is already running.")

        elif choice == '2':
            logging_enabled = not logging_enabled
            toggle_tensorflow_logging()
            print(f"TensorFlow Logging {'enabled' if logging_enabled else 'disabled'}.")

        elif choice == '3':
            display_detections = not display_detections
            print(f"Display of Detections {'enabled' if display_detections else 'disabled'}.")

        elif choice == '4':
            obj = input("Enter object name to add/remove from exclusion (e.g., 'person'): ")
            if obj in excluded_objects:
                excluded_objects.remove(obj)
                print(f"Removed '{obj}' from exclusions.")
            else:
                excluded_objects.add(obj)
                print(f"Added '{obj}' to exclusions.")

        elif choice == '5':
            select_all_objects()

        elif choice == '6':
            unselect_all_objects()

        elif choice == '7':
            load_new_model()

        elif choice == '8':
            change_resolution()

        elif choice == '9':
            take_screenshot_enabled = not take_screenshot_enabled
            print(f"Screenshot on Detection {'enabled' if take_screenshot_enabled else 'disabled'}.")

        elif choice == '10':
            print("Available Classes for Screenshot Detection: ")
            for name in class_names:
                print(name)
            selected_class = input("Enter the class name for screenshot detection: ")
            if selected_class in class_names:
                screenshot_class = selected_class
                print(f"Screenshot will be taken for: {screenshot_class}")
            else:
                print("Invalid class name. Please choose from the available classes.")

        elif choice == '11':
            webcam_thread_running = False
            break

        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    menu()
