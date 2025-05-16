#!/usr/bin/env python3
"""
BugEye – hex-grid overlay & zoom for live camera frames.
Author: Charles Norton  •  Updated: 2025-05-16
"""
import logging
from math import sqrt, floor
from pathlib import Path
import tkinter as tk
from tkinter import simpledialog
from dataclasses import dataclass

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")


# --------------------------------------------------------------------------- #
#                           Geometry & Rendering                              #
# --------------------------------------------------------------------------- #
@dataclass
class GridSettings:
    hex_sizes: dict = None
    default_key: str = "medium"

    def __post_init__(self):
        if self.hex_sizes is None:
            self.hex_sizes = {"low": 80, "medium": 40, "high": 20}

    @property
    def default_size(self) -> int:
        return self.hex_sizes[self.default_key]


class HexagonalGrid:
    """Handles geometry, overlay caching, and zoom."""

    def __init__(self, settings: GridSettings):
        self.settings = settings
        self.hex_size = settings.default_size
        self.current_resolution = settings.default_key

        # --- cache ---
        self._overlay = None            # BGR image with grid & numbers
        self._centers = {}              # {hex_number: (x, y)}
        self._overlay_shape = None
        self._dirty = True

        # --- state ---
        self.selected_hex: int | None = None

    # --------------------  resolution helper -------------------- #
    def change_resolution(self, key: int) -> None:
        mapping = {ord("l"): "low", ord("m"): "medium", ord("h"): "high"}
        new_level = mapping.get(key)
        if new_level and new_level != self.current_resolution:
            self.current_resolution = new_level
            self.hex_size = self.settings.hex_sizes[new_level]
            self._dirty = True
            logging.info("Resolution → %s", new_level)

    # --------------------  public draw API ---------------------- #
    def compose_frame(self, frame: np.ndarray) -> np.ndarray:
        """Return a copy of frame with grid & optional highlight."""
        if self._needs_refresh(frame):
            self._build_overlay(frame)

        out = cv2.add(frame, self._overlay)

        if self.selected_hex and self.selected_hex in self._centers:
            center = self._centers[self.selected_hex]
            out = self._draw_hexagon(out, center, color=(0, 255, 0), thick=2)

        return out

    def zoom_selected(self, frame: np.ndarray) -> np.ndarray | None:
        """Return a masked & enlarged view of the selected hex (or None)."""
        if not (self.selected_hex and self.selected_hex in self._centers):
            return None
        cx, cy = self._centers[self.selected_hex]

        R = self.hex_size
        w = int(sqrt(3) * R)
        h = 2 * R
        x1, y1 = max(cx - w // 2, 0), max(cy - h // 2, 0)
        x2, y2 = min(cx + w // 2, frame.shape[1]), min(cy + h // 2,
                                                       frame.shape[0])
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return None

        # hex mask in ROI coordinates
        pts = self._hexagon_vertices((cx - x1, cy - y1))
        mask = np.zeros(roi.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 255)
        roi_masked = cv2.bitwise_and(roi, roi, mask=mask)

        return cv2.resize(roi_masked, None, fx=3, fy=3,
                          interpolation=cv2.INTER_LINEAR)

    # ------------------------------------------------------------------ #
    #                        Internal helpers                             #
    # ------------------------------------------------------------------ #
    def _needs_refresh(self, frame: np.ndarray) -> bool:
        return self._dirty or self._overlay is None or \
            self._overlay_shape != frame.shape[:2]

    def _build_overlay(self, frame: np.ndarray) -> None:
        """Cache the grid overlay (lines + numbers) and centers."""
        logging.debug("Rebuilding grid overlay …")
        h_frame, w_frame = frame.shape[:2]
        overlay = np.zeros_like(frame)
        self._centers.clear()

        R = self.hex_size
        w_hex = sqrt(3) * R
        h_hex = 2 * R
        vert_stride = 0.75 * h_hex

        # how many rows / cols *really* fit?
        n_rows = int((h_frame + h_hex / 2) // vert_stride) + 1
        n_cols = int((w_frame + w_hex) // w_hex) + 1

        # compute bounding box of grid to center it
        grid_w = n_cols * w_hex + w_hex / 2
        grid_h = n_rows * vert_stride + h_hex / 4
        margin_x = (w_frame - grid_w) / 2
        margin_y = (h_frame - grid_h) / 2

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = R / 110.0
        font_th = 1
        n = 1
        for row in range(n_rows):
            y = row * vert_stride + margin_y + h_hex / 2
            for col in range(n_cols):
                x = col * w_hex + margin_x + w_hex
                # even rows shifted right by half-width
                if row % 2 == 0:
                    x += w_hex / 2

                if not (0 <= x < w_frame and 0 <= y < h_frame):
                    continue

                center = (int(x), int(y))
                self._centers[n] = center

                # hex outline
                overlay = self._draw_hexagon(overlay, center)
                # number
                label = str(n)
                text_sz = cv2.getTextSize(label, font, font_scale, font_th)[0]
                cv2.putText(
                    overlay, label,
                    (center[0] - text_sz[0] // 2, center[1] + text_sz[1] // 2),
                    font, font_scale, (255, 255, 255), font_th + 2,
                    cv2.LINE_AA)
                cv2.putText(
                    overlay, label,
                    (center[0] - text_sz[0] // 2, center[1] + text_sz[1] // 2),
                    font, font_scale, (0, 0, 0), font_th, cv2.LINE_AA)
                n += 1

        self._overlay = overlay
        self._overlay_shape = frame.shape[:2]
        self._dirty = False
        logging.info("Grid overlay ready – %d hexes", len(self._centers))

    def _hexagon_vertices(self, center: tuple[int, int]) -> np.ndarray:
        cx, cy = center
        R = self.hex_size
        pts = []
        for i in range(6):
            ang = np.radians(60 * i + 30)
            pts.append([int(cx + R * np.cos(ang)),
                        int(cy + R * np.sin(ang))])
        return np.asarray(pts, dtype=np.int32)

    def _draw_hexagon(
        self, img: np.ndarray, center: tuple[int, int], *,
        color=(255, 255, 255), thick: int = 1
    ) -> np.ndarray:
        pts = self._hexagon_vertices(center)
        return cv2.polylines(img, [pts], isClosed=True, color=color,
                             thickness=thick)

    # ---------------------  selection dialog -------------------- #
    def ask_hexagon_number(self, max_n: int) -> int | None:
        root = tk.Tk()
        root.withdraw()
        try:
            num = simpledialog.askinteger(
                "Hexagon #",
                f"Enter a number (1–{max_n}):",
                parent=root,
                minvalue=1, maxvalue=max_n)
        except Exception as exc:
            logging.error("Dialog error: %s", exc)
            num = None
        return num


# --------------------------------------------------------------------------- #
#                               Main App                                      #
# --------------------------------------------------------------------------- #
class HexagonalGridApp:
    def __init__(self):
        self.grid = HexagonalGrid(GridSettings())
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise IOError("Cannot open webcam")

    # ---------------------------  loop  ----------------------------------- #
    def run(self) -> None:
        logging.info("BugEye – press l/m/h to change resolution, "
                     "s to select, q to quit")

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)   # mirror for friendlier UX

            out = self.grid.compose_frame(frame)
            cv2.putText(
                out,
                f"Resolution: {self.grid.current_resolution} "
                "(l/m/h)  –  s: select  q: quit",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow("Hexagonal Grid Overlay", out)

            # show zoom window if available
            zoom = self.grid.zoom_selected(frame)
            if zoom is not None:
                cv2.imshow("Zoomed Hexagon", zoom)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key in (ord("l"), ord("m"), ord("h")):
                self.grid.change_resolution(key)
            elif key == ord("s"):
                if not self.grid._centers:
                    continue
                sel = self.grid.ask_hexagon_number(max(self.grid._centers))
                if sel:
                    self.grid.selected_hex = sel

        self._cleanup()

    def _cleanup(self) -> None:
        self.cap.release()
        cv2.destroyAllWindows()


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    HexagonalGridApp().run()
