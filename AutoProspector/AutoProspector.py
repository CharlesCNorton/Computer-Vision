import os
import tkinter as tk
from tkinter import filedialog, simpledialog
from PIL import Image, ImageDraw, ImageShow
import base64
import requests
from io import BytesIO
from tqdm import tqdm
from colorama import init, Fore
from concurrent.futures import ThreadPoolExecutor, as_completed

init(autoreset=True)

debug_enabled = True
subdivision_base_number = 2
deep_scan_enabled = False
consensus_enabled = True
consensus_mode = 5
min_size = 50
output_dir = os.getcwd()
max_workers = 10
API_KEY = None
batch_processing_enabled = True

def debug_print(message):
    if debug_enabled:
        print(Fore.CYAN + f"[DEBUG]: {message}")

def encode_image(image_input):
    try:
        if isinstance(image_input, str):
            with open(image_input, "rb") as image_file:
                encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
        elif isinstance(image_input, Image.Image):
            buffered = BytesIO()
            image_input.save(buffered, format="JPEG")
            encoded_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
        else:
            raise ValueError("Invalid image input. Expected a file path or PIL.Image object.")

        debug_print("Image successfully encoded to base64.")
        return encoded_image
    except Exception as e:
        debug_print(Fore.RED + f"Error encoding image: {e}")
        return None

def detect_gold_in_image(image_base64):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }

    payload = {
        "model": "chatgpt-4o-latest",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an advanced AI model specialized in detecting "
                    "the precious metal gold placer and gold dust within images. Your task is to analyze "
                    "each provided image or image section and respond with "
                    "'yes' only if you are certain that the precious metal gold"
                    "is present based on its visual appearance. Similar or "
                    "look-alike substances must be rejected. If you are unsure, "
                    "respond with 'no'. Only respond yes if you are certain."
                )
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Do you see the precious metal gold in this image? The gold panner is relying on you to help them find gold ore."},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}",
                            "detail": "high"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 50,
        "temperature": 0.8
    }

    try:
        debug_print("Sending request to OpenAI...")
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        debug_print(f"Received response from OpenAI: {result}")

        if "choices" in result and result["choices"]:
            content = result["choices"][0]["message"]["content"].lower().strip()
            if "yes" in content:
                debug_print(Fore.YELLOW + "Model decision: Yes")
                return "yes"
            else:
                debug_print("Model decision: No")
                return "no"
        else:
            debug_print(Fore.RED + "Error: No valid response received from GPT-4o.")
            return "no"
    except requests.exceptions.RequestException as e:
        debug_print(Fore.RED + f"API request error: {e}")
        return "no"
    except Exception as e:
        debug_print(Fore.RED + f"Error processing API response: {e}")
        return "no"

def merge_bounding_boxes(bounding_boxes):
    if not bounding_boxes:
        return []

    merged_boxes = []
    bounding_boxes = sorted(bounding_boxes, key=lambda box: (box[0], box[1]))

    current_box = bounding_boxes[0]

    for box in bounding_boxes[1:]:
        if (box[0] <= current_box[2] and box[1] <= current_box[3]):
            current_box = (min(current_box[0], box[0]), min(current_box[1], box[1]),
                           max(current_box[2], box[2]), max(current_box[3], box[3]))
        else:
            merged_boxes.append(current_box)
            current_box = box

    merged_boxes.append(current_box)
    return merged_boxes

def should_draw_border(box, yes_boxes):
    for yes_box in yes_boxes:
        if (box[0] < yes_box[2] and box[2] > yes_box[0] and box[1] < yes_box[3] and box[3] > yes_box[1]):
            return True
    return False

def process_section(section, image):
    cropped_image = image.crop(section)
    section_base64 = encode_image(cropped_image)
    if section_base64:
        return detect_gold_in_image(section_base64), section
    return "no", section

def process_grid_parallel(image, draw, x0, y0, x1, y1, base_num, min_size=50, deep_scan_enabled=False, consensus_enabled=True, level=0):
    width, height = x1 - x0
    step_x = width // base_num
    step_y = height // base_num

    bounding_boxes = []
    yes_boxes = []
    no_boxes = []
    sections = [(x0 + i * step_x, y0 + j * step_y, x0 + (i + 1) * step_x, y0 + (j + 1) * step_y)
                for i in range(base_num) for j in range(base_num)]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_section, section, image): section for section in sections}
        for future in as_completed(futures):
            result, section = future.result()
            if result == "yes":
                yes_boxes.append(section)
                if deep_scan_enabled and step_x > min_size and step_y > min_size and level < 1:
                    boxes = process_grid_parallel(image, draw, *section, base_num, min_size=min_size, deep_scan_enabled=deep_scan_enabled, consensus_enabled=consensus_enabled, level=level + 1)
                    bounding_boxes.extend(boxes)
                else:
                    bounding_boxes.append(section)
            else:
                no_boxes.append(section)

    for box in bounding_boxes:
        draw.rectangle(box, outline="gold", width=2)
        draw.text((box[0] + 5, box[1] + 5), "gold", fill="gold")

    for no_box in no_boxes:
        if should_draw_border(no_box, yes_boxes):
            draw.rectangle(no_box, outline="gold", width=2)

    return merge_bounding_boxes(bounding_boxes)

def process_grid_serial(image, draw, x0, y0, x1, y1, base_num, min_size=50, deep_scan_enabled=False, consensus_enabled=True, level=0):
    width, height = x1 - x0
    step_x = width // base_num
    step_y = height // base_num

    bounding_boxes = []
    yes_boxes = []
    no_boxes = []
    sections = [(x0 + i * step_x, y0 + j * step_y, x0 + (i + 1) * step_x, y0 + (j + 1) * step_y)
                for i in range(base_num) for j in range(base_num)]

    for section in tqdm(sections, desc="Processing sections serially", unit="section"):
        result, section = process_section(section, image)
        if result == "yes":
            yes_boxes.append(section)
            if deep_scan_enabled and step_x > min_size and step_y > min_size and level < 1:
                boxes = process_grid_serial(image, draw, *section, base_num, min_size=min_size, deep_scan_enabled=deep_scan_enabled, consensus_enabled=consensus_enabled, level=level + 1)
                bounding_boxes.extend(boxes)
            else:
                bounding_boxes.append(section)
        else:
            no_boxes.append(section)

    for box in bounding_boxes:
        draw.rectangle(box, outline="gold", width=2)
        draw.text((box[0] + 5, box[1] + 5), "gold", fill="gold")

    for no_box in no_boxes:
        if should_draw_border(no_box, yes_boxes):
            draw.rectangle(no_box, outline="gold", width=2)

    return merge_bounding_boxes(bounding_boxes)

def main():
    global debug_enabled, subdivision_base_number, deep_scan_enabled, consensus_enabled, consensus_mode, min_size, output_dir, max_workers, API_KEY, batch_processing_enabled

    root = tk.Tk()
    root.withdraw()

    while True:
        print("Welcome to AutoProspector!")
        print("--------------------------------")
        print(f"1. Enter API Key (Currently {'Set' if API_KEY else 'Not Set'})")
        print(f"2. Select Image for Analysis")
        print(f"3. Set Output Directory")
        print(f"4. Toggle Subdivision Base Number (Current: {Fore.GREEN + str(subdivision_base_number)})")
        print(f"5. Set Minimum Subdivision Size (Current: {Fore.GREEN + str(min_size)} pixels)")
        print(f"6. Toggle Deep Scan (Currently {Fore.GREEN if deep_scan_enabled else Fore.RED}{'Enabled' if deep_scan_enabled else 'Disabled'})")
        print(f"7. Toggle Consensus Mode (Currently {Fore.GREEN if consensus_enabled else Fore.RED}{'Enabled' if consensus_enabled else 'Disabled'}, Best of {consensus_mode})")
        print(f"8. Toggle Consensus Detail (Current Mode: {Fore.GREEN if consensus_mode == 5 else Fore.YELLOW}{'Best of 4/5' if consensus_mode == 5 else 'Best of 2/3'})")
        print(f"9. Toggle Debug Mode (Currently {Fore.GREEN if debug_enabled else Fore.RED}{'Enabled' if debug_enabled else 'Disabled'})")
        print(f"10. Set Maximum Threads for Parallel Processing (Current: {Fore.GREEN + str(max_workers)})")
        print(f"11. Toggle Processing Mode (Currently {'Batch' if batch_processing_enabled else 'Serial'})")
        print("12. Start Analysis")
        print("13. Exit")
        print("--------------------------------")

        choice = input("Please select an option: ")

        if choice == '1':
            API_KEY = simpledialog.askstring("API Key", "Enter your OpenAI API Key:")
            if API_KEY:
                print(Fore.GREEN + "API Key set successfully.")
                debug_print(f"API Key set: {API_KEY[:6]}...{API_KEY[-4:]}")
            else:
                print(Fore.RED + "API Key not set.")

        elif choice == '2':
            if not API_KEY:
                print(Fore.RED + "API Key is not set. Please enter your API Key first.")
                continue

            file_path = filedialog.askopenfilename()
            if not file_path:
                print(Fore.RED + "No file selected.")
            else:
                print(f"Selected file: {file_path}")
                debug_print(f"File selected: {file_path}")

        elif choice == '3':
            output_dir = filedialog.askdirectory()
            if not output_dir:
                print(Fore.RED + "No directory selected. Using default directory.")
                output_dir = os.getcwd()
            else:
                print(f"Output directory set to: {output_dir}")
                debug_print(f"Output directory set to: {output_dir}")

        elif choice == '4':
            subdivision_base_number += 1
            if subdivision_base_number > 5:
                subdivision_base_number = 1
            print(f"Subdivision base number set to: {Fore.GREEN + str(subdivision_base_number)}")
            debug_print(f"Subdivision base number set to: {subdivision_base_number}")

        elif choice == '5':
            min_size = simpledialog.askinteger("Minimum Subdivision Size", "Enter the minimum size (in pixels) for subdivisions:")
            if not min_size or min_size < 10:
                print(Fore.RED + "Invalid minimum size. Using default 50 pixels.")
                min_size = 50
            else:
                print(f"Minimum subdivision size set to: {Fore.GREEN + str(min_size)} pixels")
                debug_print(f"Minimum subdivision size set to: {min_size} pixels")

        elif choice == '6':
            deep_scan_enabled = not deep_scan_enabled
            status = "enabled" if deep_scan_enabled else "disabled"
            print(f"Deep Scan has been {Fore.GREEN if deep_scan_enabled else Fore.RED}{status}.")
            debug_print(f"Deep Scan {status} by the user.")

        elif choice == '7':
            consensus_enabled = not consensus_enabled
            status = "enabled" if consensus_enabled else "disabled"
            print(f"Consensus Mode has been {Fore.GREEN if consensus_enabled else Fore.RED}{status}.")
            debug_print(f"Consensus Mode {status} by the user.")

        elif choice == '8':
            if consensus_mode == 5:
                consensus_mode = 3
            else:
                consensus_mode = 5
            mode_name = "Best of 4/5" if consensus_mode == 5 else "Best of 2/3"
            print(f"Consensus Detail switched to: {Fore.GREEN if consensus_mode == 5 else Fore.YELLOW}{mode_name}")
            debug_print(f"Consensus Detail switched to: {mode_name}")

        elif choice == '9':
            debug_enabled = not debug_enabled
            status = "enabled" if debug_enabled else "disabled"
            print(f"Debug Mode has been {Fore.GREEN if debug_enabled else Fore.RED}{status}.")
            debug_print(f"Debug Mode {status} by the user.")

        elif choice == '10':
            max_workers = simpledialog.askinteger("Maximum Threads", "Enter the maximum number of threads for parallel processing:")
            if not max_workers or max_workers < 1:
                print(Fore.RED + "Invalid number of threads. Using default 10.")
                max_workers = 10
            else:
                print(f"Maximum threads set to: {Fore.GREEN + str(max_workers)}")
                debug_print(f"Maximum threads set to: {max_workers}")

        elif choice == '11':
            batch_processing_enabled = not batch_processing_enabled
            status = "Batch" if batch_processing_enabled else "Serial"
            print(f"Processing Mode set to: {Fore.GREEN if batch_processing_enabled else Fore.YELLOW}{status}")
            debug_print(f"Processing Mode set to: {status}")

        elif choice == '12':
            if not file_path:
                print(Fore.RED + "No image file selected. Please select an image first.")
                continue

            try:
                image = Image.open(file_path)
                debug_print(f"Image opened: {file_path}")

                if image.mode not in ('RGB', 'L'):
                    image = image.convert('RGB')
                    debug_print("Image converted to RGB mode.")

                draw = ImageDraw.Draw(image)

                image_base64 = encode_image(image)

                if image_base64 and detect_gold_in_image(image_base64) == "yes":
                    print(Fore.YELLOW + "Gold detected in the entire image! Proceeding with localization...")
                    width, height = image.size

                    if batch_processing_enabled:
                        bounding_boxes = process_grid_parallel(image, draw, 0, 0, width, height, subdivision_base_number, min_size=min_size, deep_scan_enabled=deep_scan_enabled, consensus_enabled=consensus_enabled)
                    else:
                        bounding_boxes = process_grid_serial(image, draw, 0, 0, width, height, subdivision_base_number, min_size=min_size, deep_scan_enabled=deep_scan_enabled, consensus_enabled=consensus_enabled)

                    for box in bounding_boxes:
                        draw.rectangle(box, outline="gold", width=2)
                        draw.text((box[0] + 5, box[1] + 5), "gold", fill="gold")

                    output_path = os.path.join(output_dir, "output_with_bounding_boxes.jpg")
                    image.save(output_path)
                    print(Fore.GREEN + f"Output saved as {output_path}")
                    debug_print(f"Output saved at: {output_path}")

                    ImageShow.show(image)
                    debug_print("Resulting image displayed.")
                else:
                    print(Fore.RED + "No gold detected in the entire image. Skipping further processing.")

            except Exception as e:
                debug_print(Fore.RED + f"Error processing image: {e}")

        elif choice == '13':
            print(Fore.YELLOW + "Exiting AutoProspector. Goodbye!")
            debug_print("Exiting the tool.")
            break

        else:
            print(Fore.RED + "Invalid option. Please try again.")
            debug_print("Invalid menu option selected.")

if __name__ == "__main__":
    main()
