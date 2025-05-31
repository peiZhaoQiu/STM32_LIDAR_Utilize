import time
import os
import numpy as np
import serial
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import h5py

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


def save_income_histogram(matrix, save_dir="snapshots", input_file_name = None):
    os.makedirs(save_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    if input_file_name == None:
        filename = os.path.join(save_dir, f"income_matrix_{timestamp}.npy")
    else:
        filename =  os.path.join(save_dir, input_file_name)
        
    np.save(filename, matrix)
    print(f"[INFO] income_matrix saved to {filename}")

def combine_histogram(file_list):
    summed_matrix = None

    for file_path in file_list:
        matrix = np.load(file_path)

        if summed_matrix is None:
            summed_matrix = np.zeros_like(matrix)
        
        if matrix.shape != summed_matrix.shape:
            raise ValueError(f"Shape mismatch: {file_path} has shape {matrix.shape}, expected {summed_matrix.shape}")

        summed_matrix += matrix

    if summed_matrix is None:
        raise ValueError("File list is empty or contains no valid .npy files.")

    print(f"[INFO] Summed {len(file_list)} histograms with shape {summed_matrix.shape}")
    total_sum = np.sum(summed_matrix)
    # print(f"[INFO] Total sum of all elements: {total_sum}")
    return summed_matrix


def read_serial_histogram(serial_port, income_matrix):
    H, W, NUM_BIN = income_matrix.shape
    expected_lines = H * W

    # Wait for START_OF_FRAME
    while True:
        line = serial_port.readline().decode('utf-8').strip()
        if line == "START_OF_FRAME":
            break

    # Read histogram data
    buffer = []
    for _ in range(expected_lines):
        line = serial_port.readline().decode('utf-8').strip()
        if line == "END_OF_FRAME":
            print("[ERROR] Unexpected END_OF_FRAME before full data read")
            return None        
        try:
            hist_values = [int(value) for value in line.split()]
        except ValueError:
            print(f"[ERROR] Non-integer value in histogram line: {line}")
            return None
        if len(hist_values) != NUM_BIN:
            print(f"[ERROR] Invalid histogram line: expected {NUM_BIN} bins, got {len(hist_values)} → {line}")
            return None
        buffer.append(hist_values)

    # Confirm END_OF_FRAME
    end_marker = serial_port.readline().decode('utf-8').strip()
    if end_marker != "END_OF_FRAME":
        print("[ERROR] END_OF_FRAME not found after full histogram read")
        return None

    # Reshape and store into income_matrix
    for idx, hist in enumerate(buffer):
        i = idx // W
        j = idx % W
        income_matrix[i, j, :] = hist

    return income_matrix

# HDF5 save function
def save_histogram_to_h5(filename, stamped_histogram, 
                          range_min, range_max, image_width = 8, image_height = 8, bin_number = 18):
    with h5py.File(filename, 'w') as h5f:
        h5f.create_dataset("stamped_histogram", data=stamped_histogram, compression="gzip")
        h5f.attrs["range_min"] = range_min
        h5f.attrs["range_max"] = range_max
        h5f.attrs["image_width"] = image_width
        h5f.attrs["image_height"] = image_height
        h5f.attrs["bin_number"] = bin_number




def display_image_from_h5(filename):

    with h5py.File(filename, 'r') as h5f:
        histogram = h5f["stamped_histogram"][:]
        range_min = h5f.attrs["range_min"]
        range_max = h5f.attrs["range_max"]
        bin_number = h5f.attrs["bin_number"]
        bin_width= (range_max - range_min) / bin_number
        height, width, _ = histogram.shape       
        depthImage= np.zeros((height, width), dtype=np.float32)
        for i in range(height):
            for j in range(width):
                max_bin_index = np.argmax(histogram[i, j])
                if histogram[i, j, max_bin_index] > 0:
                    depthImage[i, j] = range_min + (max_bin_index + 0.5) * bin_width

        depthmap = mcolors.LinearSegmentedColormap.from_list('depth_cmap', colors, N=256)
        cmap = depthmap
        cmap.set_bad(color='black')
        displayImage = np.ma.masked_where((depthImage == 0), depthImage)
        plt.imshow(displayImage,cmap=cmap ,vmin = range_min,vmax=range_max ,interpolation='nearest')
        plt.colorbar()
        plt.show()