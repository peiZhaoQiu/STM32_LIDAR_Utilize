import numpy as np
import cv2

class HistogramGridViewer:
    def __init__(self, H=8, W=8, NUM_BIN=18, hist_height=80, hist_width=80, win_name="Histograms"):
        self.H = H
        self.W = W
        self.NUM_BIN = NUM_BIN
        self.HIST_HEIGHT = hist_height
        self.HIST_WIDTH = hist_width
        self.BAR_WIDTH = hist_width // NUM_BIN
        self.canvas_height = H * hist_height
        self.canvas_width = W * hist_width
        self.win_name = win_name

        # Create named window
        cv2.namedWindow(self.win_name, cv2.WINDOW_NORMAL)

    def update(self, income_matrix):
        # Normalize values to histogram height
        max_val = np.max(income_matrix)
        norm_matrix = (income_matrix / (max_val + 1e-5)) * self.HIST_HEIGHT
        norm_matrix = norm_matrix.astype(np.int32)

        # Create light gray canvas
        canvas = np.ones((self.canvas_height, self.canvas_width), dtype=np.uint8) * 230

        for i in range(self.H):
            for j in range(self.W):
                hist = norm_matrix[i, j]
                x0 = j * self.HIST_WIDTH
                y0 = i * self.HIST_HEIGHT

                for b in range(self.NUM_BIN):
                    bar_h = hist[b]
                    x1 = x0 + b * self.BAR_WIDTH
                    x2 = x1 + self.BAR_WIDTH - 1
                    y1 = y0 + self.HIST_HEIGHT
                    y2 = y1 - bar_h
                    cv2.rectangle(canvas, (x1, y2), (x2, y1), 0, -1)

        # Show image
        cv2.imshow(self.win_name, canvas)

    def close(self):
        cv2.destroyWindow(self.win_name)