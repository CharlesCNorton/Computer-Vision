import cv2
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import hid
import threading
import datetime
import time
import sys

class PTZController:
    """Class to control PTZ camera movements via HID commands."""

    def __init__(self, vendor_id=0x046D, product_id=0x085F, usage_page=65280, usage=1):
        """
        Initialize the PTZ Controller by auto-discovering the correct HID interface.
        Allows customization of vendor_id, product_id, usage_page, and usage if needed.
        """
        self.device = None
        try:
            # Find a HID interface matching the usage_page/usage.
            ptz_path = None
            for d in hid.enumerate(vendor_id, product_id):
                if d['usage_page'] == usage_page and d['usage'] == usage:
                    ptz_path = d['path']
                    break

            if ptz_path:
                self.device = hid.device()
                self.device.open_path(ptz_path)
                print("PTZ HID interface opened successfully.")
            else:
                print("No suitable PTZ HID interface found. PTZ commands may not work.")
        except IOError as e:
            print(f"Error opening device: {e}")
        except Exception as e:
            print(f"Unexpected error during device initialization: {e}")

    def send_command(self, report_id, value):
        """Send a command to the PTZ camera."""
        if not self.device:
            print("Device not initialized.")
            return
        command = [report_id & 0xFF, value] + [0x00]*30
        try:
            self.device.write(command)
            print(f"Command sent: report_id={report_id}, value={value}")
            time.sleep(0.2)  # small delay to ensure the camera processes the command
        except IOError as e:
            print(f"Error sending command: {e}")
        except Exception as e:
            print(f"Unexpected error sending command: {e}")

    def pan_right(self):
        self.send_command(0x0B, 0x02)

    def pan_left(self):
        self.send_command(0x0B, 0x03)

    def tilt_up(self):
        self.send_command(0x0B, 0x00)

    def tilt_down(self):
        self.send_command(0x0B, 0x01)

    def zoom_in(self):
        self.send_command(0x0B, 0x04)

    def zoom_out(self):
        self.send_command(0x0B, 0x05)

    def close(self):
        if self.device:
            try:
                self.device.close()
                print("Device closed successfully.")
            except Exception as e:
                print(f"Error closing device: {e}")

class PTZControllerApp:
    """GUI Application to control the PTZ camera."""

    def __init__(self, root, cap, camera):
        self.root = root
        self.cap = cap
        self.camera = camera

        self.recording = False
        self.out = None

        self.root.title("PTZ Controller")
        self.lbl_video = ttk.Label(self.root)
        self.lbl_video.grid(row=0, column=0, columnspan=3)

        self.create_buttons()
        self.bind_keys()
        self.update_frame()

        self.panning = False
        self.tilting = False

    def create_buttons(self):
        """Create control buttons for the GUI."""
        try:
            ttk.Button(self.root, text="←", command=self.camera.pan_left).grid(row=1, column=0)
            ttk.Button(self.root, text="↑", command=self.camera.tilt_up).grid(row=1, column=1)
            ttk.Button(self.root, text="→", command=self.camera.pan_right).grid(row=1, column=2)
            ttk.Button(self.root, text="↓", command=self.camera.tilt_down).grid(row=2, column=1)
            ttk.Button(self.root, text="+", command=self.camera.zoom_in).grid(row=2, column=0)
            ttk.Button(self.root, text="-", command=self.camera.zoom_out).grid(row=2, column=2)
            ttk.Button(self.root, text="Snapshot", command=self.take_snapshot).grid(row=3, column=0, columnspan=1)
            self.btn_record = ttk.Button(self.root, text="Start Recording", command=self.toggle_recording)
            self.btn_record.grid(row=3, column=2, columnspan=1)
        except Exception as e:
            print(f"Error creating buttons: {e}")

    def bind_keys(self):
        """Bind keyboard keys to PTZ functions."""
        try:
            self.root.bind('<Left>', self.start_pan_left)
            self.root.bind('<Right>', self.start_pan_right)
            self.root.bind('<Up>', self.start_tilt_up)
            self.root.bind('<Down>', self.start_tilt_down)
            self.root.bind('<KeyPress-+>', self.camera.zoom_in)
            self.root.bind('<KeyPress-equal>', self.camera.zoom_in)
            self.root.bind('<KeyPress-minus>', self.camera.zoom_out)

            self.root.bind('<KeyRelease-Left>', self.stop_pan)
            self.root.bind('<KeyRelease-Right>', self.stop_pan)
            self.root.bind('<KeyRelease-Up>', self.stop_tilt)
            self.root.bind('<KeyRelease-Down>', self.stop_tilt)
        except Exception as e:
            print(f"Error binding keys: {e}")

    def update_frame(self):
        """Update the video frame in the GUI."""
        try:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
                imgtk = ImageTk.PhotoImage(image=img)
                self.lbl_video.imgtk = imgtk
                self.lbl_video.configure(image=imgtk)

                if self.recording:
                    if self.out is None:
                        fourcc = cv2.VideoWriter_fourcc(*'XVID')
                        self.out = cv2.VideoWriter(
                            f'recording_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.avi',
                            fourcc, 20.0, (frame.shape[1], frame.shape[0])
                        )
                    self.out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            else:
                print("Failed to read frame from camera.")
        except cv2.error as e:
            print(f"OpenCV error: {e}")
        except Exception as e:
            print(f"Unexpected error updating frame: {e}")
        finally:
            self.lbl_video.after(10, self.update_frame)

    def take_snapshot(self):
        """Take a snapshot from the camera feed."""
        ret, frame = self.cap.read()
        if ret:
            filename = f'snapshot_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
            cv2.imwrite(filename, frame)
            print(f"Snapshot saved as {filename}")
        else:
            print("Failed to take snapshot.")

    def toggle_recording(self):
        """Toggle video recording on and off."""
        self.recording = not self.recording
        if self.recording:
            self.btn_record.config(text="Stop Recording")
            print("Started recording.")
        else:
            self.btn_record.config(text="Start Recording")
            if self.out:
                self.out.release()
                self.out = None
            print("Stopped recording.")

    def start_pan_left(self, event):
        if not self.panning:
            self.panning = True
            threading.Thread(target=self.pan_left_thread).start()

    def start_pan_right(self, event):
        if not self.panning:
            self.panning = True
            threading.Thread(target=self.pan_right_thread).start()

    def start_tilt_up(self, event):
        if not self.tilting:
            self.tilting = True
            threading.Thread(target=self.tilt_up_thread).start()

    def start_tilt_down(self, event):
        if not self.tilting:
            self.tilting = True
            threading.Thread(target=self.tilt_down_thread).start()

    def stop_pan(self, event):
        self.panning = False

    def stop_tilt(self, event):
        self.tilting = False

    def pan_left_thread(self):
        while self.panning:
            self.camera.pan_left()
            self.root.update()

    def pan_right_thread(self):
        while self.panning:
            self.camera.pan_right()
            self.root.update()

    def tilt_up_thread(self):
        while self.tilting:
            self.camera.tilt_up()
            self.root.update()

    def tilt_down_thread(self):
        while self.tilting:
            self.camera.tilt_down()
            self.root.update()

def select_camera_index():
    """
    Prompt the user to select which camera index to use.
    We'll try indices 0 to 10 and list which ones are available.
    The user can then input the desired index.
    """
    available_indices = []
    for i in range(11):
        cap_test = cv2.VideoCapture(i)
        if cap_test.isOpened():
            available_indices.append(i)
            cap_test.release()

    if not available_indices:
        print("No webcams found.")
        sys.exit(1)

    print("Available webcams:")
    for idx in available_indices:
        print(f"[{idx}] Camera index")

    choice = input("Select a camera index: ")
    try:
        choice = int(choice)
        if choice in available_indices:
            return choice
        else:
            print("Invalid choice. Using default camera index 0.")
            return 0
    except ValueError:
        print("Invalid input. Using default camera index 0.")
        return 0

if __name__ == "__main__":
    # Let the user choose the camera index at runtime
    camera_index = select_camera_index()

    try:
        camera = PTZController()
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise IOError("Could not open camera.")

        root = tk.Tk()
        app = PTZControllerApp(root, cap, camera)
        root.mainloop()

    except IOError as e:
        print(f"I/O error occurred: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if 'cap' in locals() and cap:
            try:
                cap.release()
                print("Camera released successfully.")
            except Exception as e:
                print(f"Error releasing camera: {e}")

        if 'camera' in locals() and camera:
            try:
                camera.close()
            except Exception as e:
                print(f"Error closing camera: {e}")

        try:
            cv2.destroyAllWindows()
            print("Windows destroyed successfully.")
        except Exception as e:
            print(f"Error destroying windows: {e}")
