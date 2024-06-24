import cv2
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import numpy as np
import tkinter as tk
from tkinter import filedialog, simpledialog
from colorama import Fore, Style, init
import threading
import os
import time
from datetime import datetime
from huggingface_hub import hf_hub_download

init(autoreset=True)

class YO_FLO:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.processor = None
        self.class_name = None
        self.detections = []
        self.beep_active = False
        self.screenshot_active = False
        self.target_detected = False
        self.last_beep_time = 0
        self.stop_webcam_flag = threading.Event()
        self.model_path = None
        self.phrase = None
        self.debug = False
        self.caption_window = None
        self.caption_label = None
        self.object_detection_active = False
        self.phrase_grounding_active = False
        self.webcam_thread = None

    def init_model(self, model_path):
        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).eval().to(self.device)
            self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
            self.model_path = model_path
            print(f"{Fore.GREEN}{Style.BRIGHT}Model loaded successfully from {model_path}{Style.RESET_ALL}")
        except FileNotFoundError:
            print(f"{Fore.RED}{Style.BRIGHT}Model path not found: {model_path}{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}{Style.BRIGHT}Error loading model: {e}{Style.RESET_ALL}")

    def run_object_detection(self, image):
        try:
            if not self.model or not self.processor:
                raise ValueError("Model or processor is not initialized.")

            task_prompt = '<OD>'
            if self.debug: print(f"Running object detection with task prompt: {task_prompt}")
            inputs = self.processor(text=task_prompt, images=image, return_tensors="pt").to(self.device)
            if self.debug: print(f"Inputs: {inputs}")
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs.get("pixel_values"),
                max_new_tokens=1024,
                early_stopping=False,
                do_sample=False,
                num_beams=1,
            )
            if self.debug: print(f"Generated IDs: {generated_ids}")
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            if self.debug: print(f"Generated text: {generated_text}")
            parsed_answer = self.processor.post_process_generation(
                generated_text,
                task=task_prompt,
                image_size=image.size
            )
            if self.debug: print(f"Parsed answer: {parsed_answer}")
            return parsed_answer
        except AttributeError as e:
            print(f"{Fore.RED}{Style.BRIGHT}Model or processor not initialized properly: {e}{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}{Style.BRIGHT}Error running object detection: {e}{Style.RESET_ALL}")

    def run_phrase_grounding(self, image, phrase):
        try:
            task_prompt = '<CAPTION_TO_PHRASE_GROUNDING>'
            if self.debug: print(f"Running phrase grounding with task prompt: {task_prompt} and phrase: {phrase}")
            inputs = self.processor(text=task_prompt, images=image, return_tensors="pt").to(self.device)
            inputs["input_ids"] = self.processor.tokenizer(phrase, return_tensors="pt").input_ids.to(self.device)
            if self.debug: print(f"Inputs: {inputs}")
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs.get("pixel_values"),
                max_new_tokens=1024,
                early_stopping=False,
                do_sample=False,
                num_beams=1,
            )
            if self.debug: print(f"Generated IDs: {generated_ids}")
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            if self.debug: print(f"Generated text: {generated_text}")
            return generated_text
        except Exception as e:
            print(f"{Fore.RED}{Style.BRIGHT}Error running phrase grounding: {e}{Style.RESET_ALL}")

    def plot_bbox(self, image):
        try:
            if not self.detections:
                return image
            for bbox, label in self.detections:
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            return image
        except Exception as e:
            print(f"{Fore.RED}{Style.BRIGHT}Error plotting bounding boxes: {e}{Style.RESET_ALL}")

    def select_model_path(self):
        try:
            root = tk.Tk()
            root.withdraw()
            model_path = filedialog.askdirectory()
            if model_path:
                self.init_model(model_path)
            else:
                print(f"{Fore.YELLOW}{Style.BRIGHT}Model path selection cancelled.{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}{Style.BRIGHT}Error selecting model path: {e}{Style.RESET_ALL}")

    def download_model(self):
        try:
            model_name = "microsoft/Florence-2-base-ft"
            model_path = hf_hub_download(repo_id=model_name, filename="pytorch_model.bin")
            processor_path = hf_hub_download(repo_id=model_name, filename="preprocessor_config.json")
            local_model_dir = os.path.dirname(model_path)
            self.init_model(local_model_dir)
            print(f"{Fore.GREEN}{Style.BRIGHT}Model downloaded and initialized from {local_model_dir}{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}{Style.BRIGHT}Error downloading model: {e}{Style.RESET_ALL}")

    def set_class_name(self):
        try:
            class_name = simpledialog.askstring("Set Class Name", "Enter the class name you want to detect (leave blank to show all detections, e.g., 'cat', 'dog'):")
            self.class_name = class_name if class_name else None
            if self.class_name:
                print(f"{Fore.GREEN}{Style.BRIGHT}Set to detect: {self.class_name}{Style.RESET_ALL}")
            else:
                print(f"{Fore.GREEN}{Style.BRIGHT}Showing all detections{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}{Style.BRIGHT}Error setting class name: {e}{Style.RESET_ALL}")

    def set_phrase(self):
        try:
            phrase = simpledialog.askstring("Set Phrase", "Enter the phrase you want to ground (e.g., 'a red car', 'a woman with a hat'):")
            self.phrase = phrase if phrase else None
            if self.phrase:
                print(f"{Fore.GREEN}{Style.BRIGHT}Set to ground: {self.phrase}{Style.RESET_ALL}")
            else:
                print(f"{Fore.GREEN}{Style.BRIGHT}No phrase set for grounding{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}{Style.BRIGHT}Error setting phrase: {e}{Style.RESET_ALL}")

    def toggle_beep(self):
        try:
            self.beep_active = not self.beep_active
            status = "active" if self.beep_active else "inactive"
            print(f"{Fore.GREEN}{Style.BRIGHT}Beep is now {status}{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}{Style.BRIGHT}Error toggling beep: {e}{Style.RESET_ALL}")

    def toggle_screenshot(self):
        try:
            self.screenshot_active = not self.screenshot_active
            status = "active" if self.screenshot_active else "inactive"
            print(f"{Fore.GREEN}{Style.BRIGHT}Screenshot on detection is now {status}{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}{Style.BRIGHT}Error toggling screenshot: {e}{Style.RESET_ALL}")

    def toggle_debug(self):
        try:
            self.debug = not self.debug
            status = "enabled" if self.debug else "disabled"
            print(f"{Fore.GREEN}{Style.BRIGHT}Debug mode is now {status}{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}{Style.BRIGHT}Error toggling debug mode: {e}{Style.RESET_ALL}")

    def toggle_object_detection(self):
        try:
            self.object_detection_active = not self.object_detection_active
            if not self.object_detection_active:
                self.detections.clear()
                self.update_display()
            status = "enabled" if self.object_detection_active else "disabled"
            print(f"{Fore.GREEN}{Style.BRIGHT}Object detection is now {status}{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}{Style.BRIGHT}Error toggling object detection: {e}{Style.RESET_ALL}")

    def toggle_phrase_grounding(self):
        try:
            self.phrase_grounding_active = not self.phrase_grounding_active
            if not self.phrase_grounding_active and self.caption_window:
                self.caption_window.destroy()
                self.caption_window = None
            status = "enabled" if self.phrase_grounding_active else "disabled"
            print(f"{Fore.GREEN}{Style.BRIGHT}Phrase grounding is now {status}{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}{Style.BRIGHT}Error toggling phrase grounding: {e}{Style.RESET_ALL}")

    def update_caption_window(self, caption):
        if not self.caption_window:
            self.caption_window = tk.Tk()
            self.caption_window.title("Phrase Grounding Caption")
            self.caption_label = tk.Label(self.caption_window, text=caption)
            self.caption_label.pack()
        else:
            self.caption_label.config(text=caption)
        self.caption_window.update_idletasks()

    def beep_sound(self):
        try:
            if os.name == 'nt':
                os.system('echo \a')
            else:
                print('\a')
        except Exception as e:
            print(f"{Fore.RED}{Style.BRIGHT}Error playing beep sound: {e}{Style.RESET_ALL}")

    def save_screenshot(self, frame):
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"screenshot_{timestamp}.png"
            cv2.imwrite(filename, frame)
            print(f"{Fore.GREEN}{Style.BRIGHT}Screenshot saved: {filename}{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}{Style.BRIGHT}Error saving screenshot: {e}{Style.RESET_ALL}")

    def start_webcam_detection(self):
        if self.webcam_thread and self.webcam_thread.is_alive():
            print(f"{Fore.RED}{Style.BRIGHT}Webcam detection is already running.{Style.RESET_ALL}")
            return
        self.stop_webcam_flag.clear()
        self.webcam_thread = threading.Thread(target=self._webcam_detection_thread)
        self.webcam_thread.start()

    def _webcam_detection_thread(self):
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print(f"{Fore.RED}{Style.BRIGHT}Error: Could not open webcam.{Style.RESET_ALL}")
                return

            while not self.stop_webcam_flag.is_set():
                ret, frame = cap.read()
                if not ret:
                    print(f"{Fore.RED}{Style.BRIGHT}Error: Failed to capture image from webcam.{Style.RESET_ALL}")
                    break

                try:
                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image_pil = Image.fromarray(image)
                    if self.debug: print(f"Captured frame from webcam")

                    if self.phrase_grounding_active and self.phrase:
                        if self.debug: print(f"Phrase grounding enabled with phrase: {self.phrase}")
                        results = self.run_phrase_grounding(image_pil, self.phrase)
                        if results:
                            caption = "Yes" if "yes" in results else "No"
                            self.update_caption_window(caption)

                    if self.object_detection_active:
                        if self.debug: print(f"Running object detection")
                        results = self.run_object_detection(image_pil)
                        if results and '<OD>' in results:
                            self.target_detected = False
                            self.detections = []
                            for bbox, label in zip(results['<OD>']['bboxes'], results['<OD>']['labels']):
                                if self.class_name is None or label.lower() == self.class_name.lower():
                                    self.detections.append((bbox, label))
                                    if self.class_name and label.lower() == self.class_name.lower():
                                        self.target_detected = True

                    bbox_image = self.plot_bbox(frame.copy())
                    cv2.imshow('Object Detection', bbox_image)

                    current_time = time.time()
                    if self.beep_active and self.target_detected and current_time - self.last_beep_time > 1:
                        threading.Thread(target=self.beep_sound).start()
                        if self.debug: print(f"{Fore.GREEN}{Style.BRIGHT}Target detected: {self.class_name}{Style.RESET_ALL}")
                        self.last_beep_time = current_time

                    if self.screenshot_active and self.target_detected:
                        self.save_screenshot(bbox_image)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                except Exception as e:
                    print(f"{Fore.RED}{Style.BRIGHT}Error during frame processing: {e}{Style.RESET_ALL}")

                # Ensure the Tkinter window is updated
                if self.caption_window:
                    self.caption_window.update_idletasks()

            cap.release()
            cv2.destroyAllWindows()
        except cv2.error as e:
            print(f"{Fore.RED}{Style.BRIGHT}OpenCV error: {e}{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}{Style.BRIGHT}Error during webcam detection: {e}{Style.RESET_ALL}")

    def stop_webcam_detection(self):
        if not self.webcam_thread or not self.webcam_thread.is_alive():
            print(f"{Fore.RED}{Style.BRIGHT}Webcam detection is not running.{Style.RESET_ALL}")
            return
        self.stop_webcam_flag.set()
        self.webcam_thread.join()

    def update_display(self):
        if not self.object_detection_active:
            empty_frame = np.zeros((480, 640, 3), np.uint8)
            cv2.imshow('Object Detection', empty_frame)
            cv2.waitKey(1)

    def main_menu(self):
        root = tk.Tk()
        root.title("YO-FLO Menu")

        try:
            tk.Button(root, text="Select Model Path", command=self.select_model_path).pack(fill='x')
            tk.Button(root, text="Download Model from HuggingFace", command=self.download_model).pack(fill='x')
            tk.Button(root, text="Set Class Name", command=self.set_class_name).pack(fill='x')
            tk.Button(root, text="Set Phrase", command=self.set_phrase).pack(fill='x')
            tk.Button(root, text="Toggle Beep on Detection", command=self.toggle_beep).pack(fill='x')
            tk.Button(root, text="Toggle Screenshot on Detection", command=self.toggle_screenshot).pack(fill='x')
            tk.Button(root, text="Toggle Debug Mode", command=self.toggle_debug).pack(fill='x')
            tk.Button(root, text="Toggle Object Detection", command=self.toggle_object_detection).pack(fill='x')
            tk.Button(root, text="Toggle Phrase Grounding", command=self.toggle_phrase_grounding).pack(fill='x')
            tk.Button(root, text="Start Webcam Detection", command=self.start_webcam_detection).pack(fill='x')
            tk.Button(root, text="Stop Webcam Detection", command=self.stop_webcam_detection).pack(fill='x')
            tk.Button(root, text="Exit", command=root.quit).pack(fill='x')
        except Exception as e:
            print(f"{Fore.RED}{Style.BRIGHT}Error creating menu: {e}{Style.RESET_ALL}")

        root.mainloop()

if __name__ == "__main__":
    try:
        yo_flo = YO_FLO()
        print(f"{Fore.BLUE}{Style.BRIGHT}Discover YO-FLO: A proof-of-concept in using advanced vision models as a YOLO alternative.{Style.RESET_ALL}")
        yo_flo.main_menu()
    except Exception as e:
        print(f"{Fore.RED}{Style.BRIGHT}Error initializing YO-FLO: {e}{Style.RESET_ALL}")
