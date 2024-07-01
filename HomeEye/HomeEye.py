import torch
import cv2
import threading
import os
import time
import tkinter as tk
from tkinter import simpledialog, filedialog
from ultralytics import YOLO
import pyttsx3
from queue import Queue
from transformers import AutoModelForCausalLM
from deepseek_vl.models import VLChatProcessor, MultiModalityCausalLM
from deepseek_vl.utils.io import load_pil_images
from colorama import init, Fore, Style
init()

class_names = [

    'person', 'bicycle', 'car', 'motorcycle', 'airplane',

    'bus', 'train', 'truck', 'boat', 'traffic light',

    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',

    'cat', 'dog', 'horse', 'sheep', 'cow',

    'elephant', 'bear', 'zebra', 'giraffe', 'backpack',

    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',

    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',

    'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',

    'wine glass', 'cup', 'fork', 'knife', 'spoon',

    'bowl', 'banana', 'apple', 'sandwich', 'orange',

    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',

    'cake', 'chair', 'couch', 'potted plant', 'bed',

    'dining table', 'toilet', 'tv', 'laptop', 'mouse',

    'remote', 'keyboard', 'cell phone', 'microwave', 'oven',

    'toaster', 'sink', 'refrigerator', 'book', 'clock',

    'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'

]

yolov8_model_variants = {'Nano': 'D:\\Yolov8\\yolov8n.pt', 'Medium': 'D:\\Yolov8\\yolov8l.pt', 'XLarge': 'D:\\Yolov8\\yolov8x.pt'}

def select_model_directory():
    try:
        root = tk.Tk()
        root.withdraw()
        model_dir = filedialog.askdirectory(title="Select Model Directory")
        return model_dir
    except Exception as e:
        print(Fore.RED + f"Error selecting model directory: {str(e)}" + Style.RESET_ALL)
        return None

def initialize_model():
    try:
        model_path = select_model_directory()
        if not model_path:
            print(Fore.RED + "Model directory not selected. Exiting." + Style.RESET_ALL)
            exit()
        print(Fore.GREEN + "Initializing VLChatProcessor..." + Style.RESET_ALL)
        vl_chat_processor = VLChatProcessor.from_pretrained(model_path)
        print(Fore.GREEN + "VLChatProcessor initialized." + Style.RESET_ALL)
        tokenizer = vl_chat_processor.tokenizer
        print(Fore.GREEN + "Loading AutoModelForCausalLM..." + Style.RESET_ALL)
        vl_gpt = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).to(torch.bfloat16).cuda().eval()
        print(Fore.GREEN + "AutoModelForCausalLM loaded." + Style.RESET_ALL)
        return vl_chat_processor, tokenizer, vl_gpt
    except Exception as e:
        print(Fore.RED + f"Error initializing model: {str(e)}" + Style.RESET_ALL)
        exit()

class ObjectDetectionApp:
    def __init__(self):
        print(Fore.GREEN + "Initializing ObjectDetectionApp..." + Style.RESET_ALL)
        self.model = None
        self.webcam_thread_running = False
        self.confidence_threshold = 0.65
        self.current_frame = None
        self.screenshot_objects = set()
        self.allow_screenshots = False
        self.screenshot_path = 'enter path to screenshots folder'
        self.display_classes = set()
        self.object_counts = {class_name: 0 for class_name in class_names}
        self.last_announced_detections = set()
        self.voice_enabled = False
        self.speech_engine = pyttsx3.init()
        self.speech_queue = Queue()
        self.speech_thread = threading.Thread(target=self.process_speech_queue, daemon=True)
        self.speech_thread.start()
        print(Fore.GREEN + "Loading YOLO model..." + Style.RESET_ALL)
        self.load_model('D:\\Yolov8\\yolov8x.pt')
        print(Fore.GREEN + "Initializing language model..." + Style.RESET_ALL)
        self.initialize_language_model()
        self.language_model_enabled = False
        self.language_model_classes = set()
        print(Fore.GREEN + "ObjectDetectionApp initialized." + Style.RESET_ALL)

    def load_model(self, model_path):
        try:
            self.model = YOLO(model_path)
            print("YOLO Model loaded successfully.")
        except Exception as e:
            print(f"Failed to load YOLO Model: {e}")

    def initialize_language_model(self):
        try:
            self.vl_chat_processor, self.tokenizer, self.vl_gpt = initialize_model()
        except Exception as e:
            print(Fore.RED + f"Error initializing language model: {str(e)}" + Style.RESET_ALL)

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
        current_detection_set = set()
        detection_descriptions = []
        for det in detections:
            x1, y1, x2, y2, conf, cls_id = det
            if conf < self.confidence_threshold:
                continue
            cls_id = int(cls_id)
            if cls_id >= len(class_names) or cls_id < 0:
                print(f"Warning: Detected class ID {cls_id} is out of range. Ignoring this detection.")
                continue
            class_name = class_names[cls_id]
            self.object_counts[class_name] += 1
            if class_name in self.display_classes:
                detection_description = f"{class_name} detected"
                current_detection_set.add(detection_description)
                label = f'{class_name} {conf:.2f}'
                img = cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                img = cv2.putText(img, label, (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            if self.allow_screenshots and class_name in self.screenshot_objects:
                self.save_screenshot(class_name)
            if self.language_model_enabled and class_name in self.language_model_classes:
                self.analyze_image(self.current_frame, class_name)
        new_detections = current_detection_set - self.last_announced_detections
        if new_detections and self.voice_enabled:
            self.speech_queue.put(', '.join(new_detections))
            self.last_announced_detections = current_detection_set
        return img


    def process_speech_queue(self):
        while True:
            text = self.speech_queue.get()
            if text is None:
                break
            self.speech_engine.say(text)
            self.speech_engine.runAndWait()

    def save_screenshot(self, class_name):
        if self.current_frame is not None:
            filename = f"{self.screenshot_path}screenshot_{class_name}_{int(time.time())}.png"
            cv2.imwrite(filename, self.current_frame)
            print(f"Screenshot saved: {filename}")

    def analyze_image(self, image, class_name):
        try:
            cv2.imwrite("temp_image.jpg", image)
            conversation_image = ["temp_image.jpg"]
            user_input = f"<image_placeholder>Describe the {class_name} in this image extensively."
            conversation = [{
                "role": "User",
                "content": user_input,
                "images": conversation_image
            }, {
                "role": "Assistant",
                "content": ""
            }]
            pil_images = load_pil_images(conversation)
            prepare_inputs = self.vl_chat_processor(conversations=conversation, images=pil_images, force_batchify=True).to(self.vl_gpt.device)
            inputs_embeds = self.vl_gpt.prepare_inputs_embeds(**prepare_inputs)
            outputs = self.vl_gpt.language_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=prepare_inputs.attention_mask,
                pad_token_id=self.tokenizer.eos_token_id,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=512,
                do_sample=False,
                use_cache=True
            )
            answer = self.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
            print(Fore.GREEN + f"Image Analysis for {class_name}: {answer}" + Style.RESET_ALL)
            os.remove("temp_image.jpg")
        except Exception as e:
            print(Fore.RED + f"Error analyzing image: {str(e)}" + Style.RESET_ALL)

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

    def toggle_voice_feature(self):
        self.voice_enabled = not self.voice_enabled
        print(f"Voice feature {'enabled' if self.voice_enabled else 'disabled'}.")

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

    def toggle_language_model(self):
        self.language_model_enabled = not self.language_model_enabled
        print(f"Language model {'enabled' if self.language_model_enabled else 'disabled'}.")

    def set_language_model_classes(self):
        print("\nLanguage Model Classes:")
        for class_name in class_names:
            status = "On" if class_name in self.language_model_classes else "Off"
            print(f"{class_name}: {status}")
        class_input = simpledialog.askstring("Input", "Enter class names to toggle on/off for language model (comma-separated):")
        if class_input:
            class_input = class_input.strip()
            input_classes = [name.strip() for name in class_input.split(',')]
            for class_name in input_classes:
                matched_class = next((name for name in class_names if name.lower() == class_name.lower()), None)
                if matched_class is None:
                    print(f"Invalid class name: {class_name}")
                    continue
                if matched_class in self.language_model_classes:
                    self.language_model_classes.remove(matched_class)
                    print(f"{matched_class} removed from language model classes.")
                else:
                    self.language_model_classes.add(matched_class)
                    print(f"{matched_class} added to language model classes.")

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
        tk.Button(root, text="Toggle Voice Feature", command=self.toggle_voice_feature).pack()
        tk.Button(root, text="Toggle Language Model", command=self.toggle_language_model).pack()
        tk.Button(root, text="Set Language Model Classes", command=self.set_language_model_classes).pack()
        self.create_model_switch_buttons(root)
        tk.Button(root, text="Exit", command=lambda: root.destroy()).pack()
        root.mainloop()

def main():
    print(Fore.GREEN + "Starting main function..." + Style.RESET_ALL)
    app = ObjectDetectionApp()
    if not os.path.exists(app.screenshot_path):
        os.makedirs(app.screenshot_path)
    print(Fore.GREEN + "Creating GUI..." + Style.RESET_ALL)
    app.create_gui()
    print(Fore.GREEN + "Program finished." + Style.RESET_ALL)

if __name__ == "__main__":
    print(Fore.GREEN + "Program started." + Style.RESET_ALL)
    main()
