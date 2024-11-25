import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pytesseract
plt.style.use('dark_background')
pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'

"""
- 특정 folder 내에 있는 "가장 최근에 생성된" 파일을 리턴하는 방법 
"""
folder_path = 'runs/detect/result/crops/0/'

# each_file_path_and_gen_time: 각 file의 경로와, 생성 시간을 저장함
each_file_path_and_gen_time = []
for each_file_name in os.listdir(folder_path):
    # getctime: 입력받은 경로에 대한 생성 시간을 리턴
    each_file_path = folder_path + each_file_name
    each_file_gen_time = os.path.getctime(each_file_path)
    each_file_path_and_gen_time.append(
        (each_file_path, each_file_gen_time)
    )

# 가장 생성시각이 큰(가장 최근인) 파일을 리턴 
most_recent_file = max(each_file_path_and_gen_time, key=lambda x: x[1])[0]
print(most_recent_file)
img_ori = cv2.imread(most_recent_file)      ##most_recent_file


height, width, channel = img_ori.shape
