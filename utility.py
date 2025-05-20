import time
import os
import numpy as np

def save_income_histogram(matrix, save_dir="snapshots", input_file_name = None):
    os.makedirs(save_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    if input_file_name == None:
        filename = os.path.join(save_dir, f"income_matrix_{timestamp}.npy")
    else:
        filename =  os.path.join(save_dir, input_file_name)
        
    np.save(filename, matrix)
    print(f"[INFO] income_matrix saved to {filename}")
