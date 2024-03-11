import tkinter as tk
from tkinter import filedialog
import torch
from transformers import AutoModelForCausalLM
from deepseek_vl.models import VLChatProcessor, MultiModalityCausalLM
from deepseek_vl.utils.io import load_pil_images
import cv2
import tempfile
import os
from colorama import init, Fore, Style

init()

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
        vl_chat_processor = VLChatProcessor.from_pretrained(model_path)
        tokenizer = vl_chat_processor.tokenizer
        vl_gpt = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).to(torch.bfloat16).cuda().eval()
        return vl_chat_processor, tokenizer, vl_gpt
    except Exception as e:
        print(Fore.RED + f"Error initializing model: {str(e)}" + Style.RESET_ALL)
        exit()

def capture_webcam_image():
    try:
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()
        if ret:
            temp_dir = tempfile.gettempdir()
            image_path = os.path.join(temp_dir, "webcam_image.jpg")
            cv2.imwrite(image_path, frame)
            return image_path
        else:
            print(Fore.RED + "Failed to capture webcam image." + Style.RESET_ALL)
            return None
    except Exception as e:
        print(Fore.RED + f"Error capturing webcam image: {str(e)}" + Style.RESET_ALL)
        return None

def analyze_image(vl_chat_processor, tokenizer, vl_gpt):
    try:
        root = tk.Tk()
        root.withdraw()
        image_path = filedialog.askopenfilename(title="Select an image", filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
        if not image_path:
            print(Fore.YELLOW + "No image selected." + Style.RESET_ALL)
            return
        conversation_image = [image_path]
        print(Fore.GREEN + "Image uploaded successfully." + Style.RESET_ALL)
        user_input = "<image_placeholder>Describe this image extensively."

        conversation = [{
            "role": "User",
            "content": user_input,
            "images": conversation_image
        }, {
            "role": "Assistant",
            "content": ""
        }]

        pil_images = load_pil_images(conversation)
        prepare_inputs = vl_chat_processor(conversations=conversation, images=pil_images, force_batchify=True).to(vl_gpt.device)
        inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

        outputs = vl_gpt.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=512,
            do_sample=False,
            use_cache=True
        )

        answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
        print(Fore.GREEN + f"Image Analysis: {answer}" + Style.RESET_ALL)
    except Exception as e:
        print(Fore.RED + f"Error analyzing image: {str(e)}" + Style.RESET_ALL)

def chat_with_user(vl_chat_processor, tokenizer, vl_gpt):
    print(Fore.GREEN + "Welcome to DeepSeek-VL Chat! Type 'quit', 'bye', or 'exit' to exit. Type 'upload image' to review a picture. Type 'webcam' to capture an image from the webcam." + Style.RESET_ALL)
    conversation_history = []
    while True:
        user_input = input(Fore.BLUE + "You: " + Style.RESET_ALL)
        if user_input.lower() in ['quit', 'bye', 'exit']:
            print(Fore.GREEN + "Thank you for using DeepSeek-VL Chat. Goodbye!" + Style.RESET_ALL)
            break
        elif user_input.lower() == 'upload image':
            try:
                root = tk.Tk()
                root.withdraw()
                image_path = filedialog.askopenfilename(title="Select an image", filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
                if not image_path:
                    print(Fore.YELLOW + "No image selected." + Style.RESET_ALL)
                    continue
                conversation_image = [image_path]
                print(Fore.GREEN + "Image uploaded successfully." + Style.RESET_ALL)
                user_input = "<image_placeholder>Describe this image extensively."
            except Exception as e:
                print(Fore.RED + f"Error uploading image: {str(e)}" + Style.RESET_ALL)
                continue
        elif user_input.lower() == 'webcam':
            image_path = capture_webcam_image()
            if not image_path:
                continue
            conversation_image = [image_path]
            print(Fore.GREEN + "Webcam image captured successfully." + Style.RESET_ALL)
            user_input = "<image_placeholder>Describe this image."
        else:
            conversation_image = []

        conversation_history.append({
            "role": "User",
            "content": user_input,
            "images": conversation_image
        })

        conversation = conversation_history + [{
            "role": "Assistant",
            "content": ""
        }]

        try:
            pil_images = load_pil_images(conversation)
            prepare_inputs = vl_chat_processor(conversations=conversation, images=pil_images, force_batchify=True).to(vl_gpt.device)
            inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

            outputs = vl_gpt.language_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=prepare_inputs.attention_mask,
                pad_token_id=tokenizer.eos_token_id,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                max_new_tokens=512,
                do_sample=False,
                use_cache=True
            )

            answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
            print(Fore.GREEN + f"DeepSeek-VL: {answer}" + Style.RESET_ALL)

            conversation_history.append({
                "role": "Assistant",
                "content": answer
            })
        except Exception as e:
            print(Fore.RED + f"Error generating response: {str(e)}" + Style.RESET_ALL)

def main_menu(vl_chat_processor, tokenizer, vl_gpt):
    while True:
        print(Fore.GREEN + "\nMain Menu:" + Style.RESET_ALL)
        print("1. Chat with DeepSeek-VL")
        print("2. Analyze Image")
        print("3. Quit")

        choice = input(Fore.BLUE + "Enter your choice (1-3): " + Style.RESET_ALL)

        if choice == '1':
            chat_with_user(vl_chat_processor, tokenizer, vl_gpt)
        elif choice == '2':
            analyze_image(vl_chat_processor, tokenizer, vl_gpt)
        elif choice == '3':
            print(Fore.GREEN + "Thank you for using DeepSeek-VL. Goodbye!" + Style.RESET_ALL)
            break
        else:
            print(Fore.RED + "Invalid choice. Please try again." + Style.RESET_ALL)

if __name__ == "__main__":
    try:
        vl_chat_processor, tokenizer, vl_gpt = initialize_model()
        main_menu(vl_chat_processor, tokenizer, vl_gpt)
    except KeyboardInterrupt:
        print(Fore.GREEN + "\nThank you for using DeepSeek-VL. Goodbye!" + Style.RESET_ALL)
