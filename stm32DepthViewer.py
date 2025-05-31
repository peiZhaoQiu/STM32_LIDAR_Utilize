import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

depthmap_colors = np.array([
    [0.0, 0.0, 0.5, 1.0],  # Dark Blue
    [0.0, 0.0, 1.0, 1.0],  # Blue
    [0.0, 0.5, 1.0, 1.0],  # Sky Blue
    [0.0, 1.0, 1.0, 1.0],  # Cyan
    [0.0, 0.75, 0.5, 1.0], # Turquoise
    [0.0, 1.0, 0.5, 1.0],  # Spring Green
    [0.0, 1.0, 0.0, 1.0],  # Green
    [0.5, 1.0, 0.0, 1.0],  # Light Green
    [0.8, 1.0, 0.2, 1.0],  # Yellow-Green
    [1.0, 1.0, 0.0, 1.0],  # Yellow
    [1.0, 0.84, 0.0, 1.0], # Gold
    [1.0, 0.65, 0.0, 1.0], # Orange
    [1.0, 0.55, 0.0, 1.0], # Dark Orange
    [1.0, 0.3, 0.0, 1.0],  # Vermilion
    [1.0, 0.0, 0.0, 1.0],  # Red
    [0.85, 0.0, 0.0, 1.0], # Crimson
    [0.55, 0.0, 0.0, 1.0]  # Dark Red
], dtype=np.float32)

# Normalize the colors for LinearSegmentedColormap
positions = np.linspace(0, 1, depthmap_colors.shape[0])
colors = [(pos, color) for pos, color in zip(positions, depthmap_colors)]
class DepthImageViewer:
    def __init__(self, H, W, NUM_BIN, window_name="Depth Image",range_max = 900+ 18*75,range_min = 900):
        self.H = H
        self.W = W
        self.NUM_BIN = NUM_BIN
        self.window_name = window_name
        self.range_min = range_min
        self.range_max = range_max
        cv2.namedWindow(self.window_name)

    def update(self, distance_array):
        # Step 1: Convert bin index map to physical distances
        depthImage = (distance_array + 0.5) * ((self.range_max - self.range_min) / self.NUM_BIN) + self.range_min

        # Step 2: Mask zero values (no return)
        masked_depth = np.ma.masked_where(distance_array == -1, depthImage)
        
        depthmap = mcolors.LinearSegmentedColormap.from_list('depth_cmap', colors, N=256)
        norm = mcolors.Normalize(vmin=self.range_min, vmax=self.range_max)
        rgba_image = depthmap(norm(masked_depth))  # Returns RGBA float (0-1)
        
        # Step 4: Convert RGBA [0-1] → BGR [0-255] for OpenCV
        rgb_image = (rgba_image[..., :3] * 255).astype(np.uint8)
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

        # Step 5: Resize and display
        zoom = 20
        resized = cv2.resize(bgr_image, (self.W * zoom, self.H * zoom), interpolation=cv2.INTER_NEAREST)
        cv2.imshow(self.window_name, resized)


    def close(self):
        cv2.destroyWindow(self.window_name)