import serial
import numpy as np
import matplotlib.pyplot as plt
import cv2
import matplotlib
from stm32HistogramViewer import *
from stm32DepthViewer import *
from utility import save_income_histogram, read_serial_histogram


H = 8
W = 8
NUM_BIN = 18
income_matrix = np.zeros((H, W, NUM_BIN), dtype=int)

serial_port = serial.Serial('COM5', baudrate=921600, timeout=1)
H, W, NUM_BIN = 8, 8, 18
# HIST_HEIGHT = 80
# HIST_WIDTH = 80
# BAR_WIDTH = HIST_WIDTH // NUM_BIN
histogramViewer = HistogramGridViewer(H, W, NUM_BIN)
depthImageViewer = DepthImageViewer(H,W,NUM_BIN) 
recordMode = False
recordLimit = 100
recordNum = 0
recordFilenamePrefix = 'record'
iterationCount = 0

while True:
    
    serialResult = read_serial_histogram(serial_port, income_matrix, NUM_BIN=18)
    if serialResult is None:
        continue

    distance_array_input = np.argmax(income_matrix, axis=2)

    histogramViewer.update(income_matrix)
    depthImageViewer.update(distance_array_input)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        save_income_histogram(income_matrix)
    elif key == ord('r'):
        if recordMode == False:
            recordNum = 0
            recordMode = True


    if recordMode == True and iterationCount%10:
        if recordNum < recordLimit:
            save_income_histogram(income_matrix, save_dir='recording',input_file_name= recordFilenamePrefix + " " + str(recordNum))
            recordNum = recordNum + 1
        else:
            recordMode = False
        
    iterationCount = iterationCount + 1

depthImageViewer.close()    
histogramViewer.close()
serial_port.close()
