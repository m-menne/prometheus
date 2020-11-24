''' this script generates video from a folder contaning several images
    python video.py --source stuttgart
    the result is saved in videos/stuttgart.avi'''

import numpy as np
import glob
import os
import sys
import argparse
import cv2
import re
numbers = re.compile(r'(\d+)')
def numericalSort(value):
    # otherwise it would read randomly from the folder
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


def create_video(image_folder):
    img_array = []
    path = os.path.join(os.getcwd(), image_folder)
    path = os.path.join(path, '*.png')
    for filename in sorted(glob.glob(path), key=numericalSort):
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)
 
    ouputpath = path = os.path.join(os.getcwd(), 'videos')
    ouputpath = os.path.join(ouputpath, image_folder+'.avi')
    out = cv2.VideoWriter(ouputpath,cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
    
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str)  # file/folder, 0 for webcam
    opt = parser.parse_args()
    create_video(opt.source)
