import time
import os
import numpy as np
import serial

def save_income_histogram(matrix, save_dir="snapshots", input_file_name = None):
    os.makedirs(save_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    if input_file_name == None:
        filename = os.path.join(save_dir, f"income_matrix_{timestamp}.npy")
    else:
        filename =  os.path.join(save_dir, input_file_name)
        
    np.save(filename, matrix)
    print(f"[INFO] income_matrix saved to {filename}")


def read_serial_histogram(serial_port, income_matrix, NUM_BIN):
    H, W, _ = income_matrix.shape

    for i in range(H):
        for j in range(W):
            hist = serial_port.readline().decode('utf-8').strip()
            hist_values = [int(value) for value in hist.split()]
            if len(hist_values) != NUM_BIN:
                print(f"[ERROR] Invalid histogram at ({i}, {j}): Expected {NUM_BIN} values, got {len(hist_values)}")
                return None
            income_matrix[i, j, :] = hist_values

    return income_matrix