import numpy as np
import cv2

class DepthImageViewer:
    def __init__(self, H, W, NUM_BIN, window_name="Depth Image"):
        self.H = H
        self.W = W
        self.NUM_BIN = NUM_BIN
        self.window_name = window_name
        cv2.namedWindow(self.window_name)

    def update(self, distance_array):

        # Normalize to 0-255 for display
        depth_normalized = (distance_array / (self.NUM_BIN - 1) * 255).astype(np.uint8)
        # Resize for better visibility
        zoom = 20
        depth_resized = cv2.resize(depth_normalized, (self.W * zoom, self.H * zoom), interpolation=cv2.INTER_NEAREST)
        # Apply colormap
        colored_depth = cv2.applyColorMap(depth_resized, cv2.COLORMAP_JET)

        cv2.imshow(self.window_name, colored_depth)

    def close(self):
        cv2.destroyWindow(self.window_name)