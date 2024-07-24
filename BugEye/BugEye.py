import numpy as np
import cv2
from math import sqrt, floor
import tkinter as tk
from tkinter import simpledialog

class HexagonalGrid:
    def __init__(self, hex_size):
        self.hex_size = hex_size
        self.hex_sizes = {'low': 80, 'medium': 40, 'high': 20}
        self.current_resolution = 'medium'
        self.selected_hex = None

    def draw_hexagon(self, center):
        """Draw a single hexagon given a center and a size."""
        points = []
        for i in range(6):
            angle_deg = 60 * i + 30
            angle_rad = np.radians(angle_deg)
            point = (int(center[0] + self.hex_size * np.cos(angle_rad)), int(center[1] + self.hex_size * np.sin(angle_rad)))
            points.append(point)
        return np.array(points, np.int32)

    def draw_hexagon_highlighted(self, center, img):
        """Draw a highlighted hexagon."""
        hexagon = self.draw_hexagon(center)
        cv2.polylines(img, [hexagon], isClosed=True, color=(0, 255, 0), thickness=2)  # Highlight with a green outline
        return img

    def draw_numbered_hexagonal_grid(self, img):
        """Draw a hexagonal grid on the image and number each complete hexagon."""
        height, width = img.shape[:2]
        w = sqrt(3) * self.hex_size
        h = 2 * self.hex_size
        vert_dist = 3/4 * h
        horiz_count = int(floor(width / w)) + 1
        vert_count = int(floor(height / vert_dist)) + 1
        horiz_margin = (width - (horiz_count * w - w / 2)) / 2
        vert_margin = (height - (vert_count * vert_dist - vert_dist / 4)) / 2
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = self.hex_size / 100
        font_thickness = 1
        text_color = (0, 0, 0)
        outline_color = (255, 255, 255)
        number = 1

        for row in range(vert_count + 1):
            for col in range(horiz_count + 1):
                center_x = int(col * w - w / 2 + horiz_margin)
                center_y = int(row * vert_dist - vert_dist / 4 + vert_margin)
                if row % 2 == 0:
                    center_x -= int(w / 2)
                if (0 <= center_x < width) and (0 <= center_y < height):
                    hexagon = self.draw_hexagon((center_x, center_y))
                    cv2.polylines(img, [hexagon], isClosed=True, color=(255, 255, 255), thickness=1)

                    text_size = cv2.getTextSize(str(number), font, font_scale, font_thickness)[0]
                    text_x = center_x - text_size[0] // 2
                    text_y = center_y + text_size[1] // 4
                    cv2.putText(img, str(number), (text_x - 1, text_y + 1), font, font_scale, outline_color, font_thickness + 2, cv2.LINE_AA)
                    cv2.putText(img, str(number), (text_x, text_y), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
                    number += 1

        return img

    def get_hexagon_centers(self, width, height):
        """Get the centers of all hexagons in the grid."""
        w = sqrt(3) * self.hex_size
        h = 2 * self.hex_size
        vert_dist = 3/4 * h
        horiz_count = int(floor(width / w)) + 1
        vert_count = int(floor(height / vert_dist)) + 1
        horiz_margin = (width - (horiz_count * w - w / 2)) / 2
        vert_margin = (height - (vert_count * vert_dist - vert_dist / 4)) / 2

        centers = {}
        number = 1
        for row in range(vert_count + 1):
            for col in range(horiz_count + 1):
                center_x = int(col * w - w / 2 + horiz_margin)
                center_y = int(row * vert_dist - vert_dist / 4 + vert_margin)
                if row % 2 == 0:
                    center_x -= int(w / 2)
                if (0 <= center_x < width) and (0 <= center_y < height):
                    centers[number] = (center_x, center_y)
                    number += 1
        return centers

    def zoom_into_hexagon(self, frame, center):
        """Zoom into the selected hexagon."""
        zoom_factor = 3
        hex_width = int(sqrt(3) * self.hex_size)
        hex_height = 2 * self.hex_size
        x, y = center
        x1, y1 = max(x - hex_width // 2, 0), max(y - hex_height // 2, 0)
        x2, y2 = min(x + hex_width // 2, frame.shape[1]), min(y + hex_height // 2, frame.shape[0])
        cropped_img = frame[y1:y2, x1:x2]
        if cropped_img.size == 0:
            return None
        return cv2.resize(cropped_img, None, fx=zoom_factor, fy=zoom_factor, interpolation=cv2.INTER_LINEAR)

    def ask_hexagon_number(self):
        """Creates a Tkinter popup to ask for the hexagon number."""
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        try:
            num = simpledialog.askinteger("Input", "Enter hexagon number:", parent=root, minvalue=1)
        except Exception as e:
            print(f"Error: {e}")
            num = None
        return num

    def change_resolution(self, key):
        if key == ord('l'):
            self.hex_size = self.hex_sizes['low']
            self.current_resolution = 'low'
        elif key == ord('m'):
            self.hex_size = self.hex_sizes['medium']
            self.current_resolution = 'medium'
        elif key == ord('h'):
            self.hex_size = self.hex_sizes['high']
            self.current_resolution = 'high'

    def display_info(self, frame):
        info_text = f"Resolution: {self.current_resolution} (Press 'l' for low, 'm' for medium, 'h' for high)"
        cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

class HexagonalGridApp:
    def __init__(self):
        self.hex_grid = HexagonalGrid(hex_size=40)
        self.cap = cv2.VideoCapture(0)

    def run(self):
        if not self.cap.isOpened():
            raise IOError("Cannot open webcam")

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame_with_grid = self.hex_grid.draw_numbered_hexagonal_grid(frame)
            centers = self.hex_grid.get_hexagon_centers(frame.shape[1], frame.shape[0])

            if self.hex_grid.selected_hex and self.hex_grid.selected_hex in centers:
                center = centers[self.hex_grid.selected_hex]
                frame_with_grid = self.hex_grid.draw_hexagon_highlighted(center, frame_with_grid)
                zoomed_hex = self.hex_grid.zoom_into_hexagon(frame, center)
                if zoomed_hex is not None:
                    cv2.imshow('Zoomed Hexagon', zoomed_hex)

            cv2.imshow('Hexagonal Grid Overlay', frame_with_grid)
            self.hex_grid.display_info(frame_with_grid)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                new_selection = self.hex_grid.ask_hexagon_number()
                if new_selection is not None:
                    self.hex_grid.selected_hex = new_selection

            self.hex_grid.change_resolution(key)

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    app = HexagonalGridApp()
    app.run()
