import cv2
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import hid
import threading
import datetime

class PTZController:
    """Class to control PTZ camera movements via HID commands."""

    def __init__(self, vendor_id=1133, product_id=2143):
        """
        Initialize the PTZ Controller with specified vendor and product IDs.

        Args:
            vendor_id (int): USB vendor ID of the device.
            product_id (int): USB product ID of the device.
        """
        self.device = None
        try:
            self.device = hid.device()
            self.device.open(vendor_id, product_id)
            print("Device opened successfully.")
        except IOError as e:
            print(f"Error opening device: {e}")
        except Exception as e:
            print(f"Unexpected error during device initialization: {e}")

    def send_command(self, report_id, value):
        """Send a command to the PTZ camera."""
        if not self.device:
            print("Device not initialized.")
            return
        command = [0x00] * 32
        command[0] = report_id & 0xff
        command[1] = value
        try:
            self.device.write(command)
            print(f"Command sent: report_id={report_id}, value={value}")
        except IOError as e:
            print(f"Error sending command: {e}")
        except Exception as e:
            print(f"Unexpected error sending command: {e}")

    def pan_right(self):
        """Pan the camera to the right."""
        self.send_command(0x0b, 0x02)

    def pan_left(self):
        """Pan the camera to the left."""
        self.send_command(0x0b, 0x03)

    def tilt_up(self):
        """Tilt the camera up."""
        self.send_command(0x0b, 0x00)

    def tilt_down(self):
        """Tilt the camera down."""
        self.send_command(0x0b, 0x01)

    def zoom_in(self):
        """Zoom the camera in."""
        self.send_command(0x0b, 0x04)

    def zoom_out(self):
        """Zoom the camera out."""
        self.send_command(0x0b, 0x05)

    def close(self):
        """Close the HID device."""
        if self.device:
            try:
                self.device.close()
                print("Device closed successfully.")
            except Exception as e:
                print(f"Error closing device: {e}")

class PTZControllerApp:
    """GUI Application to control the PTZ camera."""

    def __init__(self, root, cap, camera):
        """
        Initialize the PTZ Controller App with Tkinter root, video capture, and camera control.

        Args:
            root (tk.Tk): The root window of the Tkinter application.
            cap (cv2.VideoCapture): The video capture object.
            camera (PTZController): The PTZController object.
        """
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
            ttk.Button(self.root, text="←", command=self.pan_left).grid(row=1, column=0)
            ttk.Button(self.root, text="↑", command=self.tilt_up).grid(row=1, column=1)
            ttk.Button(self.root, text="→", command=self.pan_right).grid(row=1, column=2)
            ttk.Button(self.root, text="↓", command=self.tilt_down).grid(row=2, column=1)
            ttk.Button(self.root, text="+", command=self.zoom_in).grid(row=2, column=0)
            ttk.Button(self.root, text="-", command=self.zoom_out).grid(row=2, column=2)
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
            self.root.bind('<KeyPress-+>', self.zoom_in)
            self.root.bind('<KeyPress-equal>', self.zoom_in)
            self.root.bind('<KeyPress-minus>', self.zoom_out)

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
                        self.out = cv2.VideoWriter(f'recording_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.avi', fourcc, 20.0, (frame.shape[1], frame.shape[0]))
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
        """Start panning the camera to the left."""
        if not self.panning:
            self.panning = True
            threading.Thread(target=self.pan_left).start()

    def start_pan_right(self, event):
        """Start panning the camera to the right."""
        if not self.panning:
            self.panning = True
            threading.Thread(target=self.pan_right).start()

    def start_tilt_up(self, event):
        """Start tilting the camera up."""
        if not self.tilting:
            self.tilting = True
            threading.Thread(target=self.tilt_up).start()

    def start_tilt_down(self, event):
        """Start tilting the camera down."""
        if not self.tilting:
            self.tilting = True
            threading.Thread(target=self.tilt_down).start()

    def stop_pan(self, event):
        """Stop panning the camera."""
        self.panning = False

    def stop_tilt(self, event):
        """Stop tilting the camera."""
        self.tilting = False

    def pan_left(self):
        """Pan the camera to the left continuously."""
        while self.panning:
            self.camera.pan_left()
            self.root.update()
            print("Panning left")

    def pan_right(self):
        """Pan the camera to the right continuously."""
        while self.panning:
            self.camera.pan_right()
            self.root.update()
            print("Panning right")

    def tilt_up(self):
        """Tilt the camera up continuously."""
        while self.tilting:
            self.camera.tilt_up()
            self.root.update()
            print("Tilting up")

    def tilt_down(self):
        """Tilt the camera down continuously."""
        while self.tilting:
            self.camera.tilt_down()
            self.root.update()
            print("Tilting down")

    def zoom_in(self, event=None):
        """Zoom the camera in."""
        try:
            self.camera.zoom_in()
            print("Zooming in")
        except Exception as e:
            print(f"Error zooming in: {e}")

    def zoom_out(self, event=None):
        """Zoom the camera out."""
        try:
            self.camera.zoom_out()
            print("Zooming out")
        except Exception as e:
            print(f"Error zooming out: {e}")

if __name__ == "__main__":
    try:
        camera = PTZController()
        cap = cv2.VideoCapture(0)
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
        if cap:
            try:
                cap.release()
                print("Camera released successfully.")
            except Exception as e:
                print(f"Error releasing camera: {e}")

        if camera:
            try:
                camera.close()
            except Exception as e:
                print(f"Error closing camera: {e}")

        try:
            cv2.destroyAllWindows()
            print("Windows destroyed successfully.")
        except Exception as e:
            print(f"Error destroying windows: {e}")
