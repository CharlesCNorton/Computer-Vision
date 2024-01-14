import numpy as np
import cv2
from math import sqrt, floor

def draw_hexagon(center, size):
    """Draw a single hexagon given a center and a size."""
    points = []
    for i in range(6):
        angle_deg = 60 * i + 30
        angle_rad = np.radians(angle_deg)
        point = (int(center[0] + size * np.cos(angle_rad)), int(center[1] + size * np.sin(angle_rad)))
        points.append(point)
    return np.array(points, np.int32)

def draw_numbered_hexagonal_grid(img, hex_size):
    """Draw a hexagonal grid on the image and number each complete hexagon."""
    height, width = img.shape[:2]
    w = sqrt(3) * hex_size
    h = 2 * hex_size
    vert_dist = 3/4 * h
    horiz_count = int(floor(width / w)) + 1
    vert_count = int(floor(height / vert_dist)) + 1
    horiz_margin = (width - (horiz_count * w - w / 2)) / 2
    vert_margin = (height - (vert_count * vert_dist - vert_dist / 4)) / 2
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = hex_size / 100
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
                hexagon = draw_hexagon((center_x, center_y), hex_size)
                cv2.polylines(img, [hexagon], isClosed=True, color=(255, 255, 255), thickness=1)

                text_size = cv2.getTextSize(str(number), font, font_scale, font_thickness)[0]
                text_x = center_x - text_size[0] // 2
                text_y = center_y + text_size[1] // 4
                cv2.putText(img, str(number), (text_x - 1, text_y + 1), font, font_scale, outline_color, font_thickness + 2, cv2.LINE_AA)
                cv2.putText(img, str(number), (text_x, text_y), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
                number += 1

    return img

def main():
    """Main function to capture video feed and overlay the hexagonal grid."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    hex_size = 40
    hex_sizes = {
        'low': 80,
        'medium': 40,
        'high': 20,
    }
    current_resolution = 'medium'

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_with_grid = draw_numbered_hexagonal_grid(frame, hex_size)
        cv2.imshow('Hexagonal Grid Overlay', frame_with_grid)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('l'):
            hex_size = hex_sizes['low']
            current_resolution = 'low'
        elif key == ord('m'):
            hex_size = hex_sizes['medium']
            current_resolution = 'medium'
        elif key == ord('h'):
            hex_size = hex_sizes['high']
            current_resolution = 'high'

        info_text = f"Resolution: {current_resolution} (Press 'l' for low, 'm' for medium, 'h' for high)"
        cv2.putText(frame_with_grid, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
