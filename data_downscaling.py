import os
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import skimage.io
import skimage.transform
import time
import numpy as np
from tqdm import tqdm

DATA_PATH = "D:\\Random\\Cityscapes\\leftImg8bit_demoVideo\\leftImg8bit\demoVideo\\"
FOLDERS = ["stuttgart_00", "stuttgart_01", "stuttgart_02"]

TARGET_PATH = "D:\\Random\\Cityscapes\\leftImg8bit_demoVideo\\leftImg8bit\demoVideo\\downsized\\"

start = time.time()

for folder in FOLDERS:
    images = glob.glob(DATA_PATH + folder + "\\*.png")
    for image_path in tqdm(images):
        img = skimage.io.imread(image_path)
        img_shape = np.shape(img)
        out_shape = (int(img_shape[0]*0.3), int(img_shape[1]*0.3), 3)

        downscaled = skimage.transform.resize(img, out_shape)
        downscaled = skimage.util.img_as_ubyte(downscaled)
        fname = TARGET_PATH + folder + "\\" + os.path.basename(image_path)
        skimage.io.imsave(fname, downscaled)

print(time.time() - start)