import os
import glob
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import skimage.io
import skimage.transform
import numpy as np
from tqdm import tqdm
import pandas as pd


_COLORS = list(mcolors.CSS4_COLORS)

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


def plot_detections(img, boxes, labels, scores, categories, score_thresh=0.5):
    """"""

    for i, _ in enumerate(scores):
        if scores[i] > score_thresh and labels[i] in categories:
            x1, y1, x2, y2 = boxes[i]

            col = _COLORS[labels[i]]
            plt.plot([x1, x1], [y1, y2], color=col)
            plt.plot([x2, x2], [y1, y2], color=col)
            plt.plot([x1, x2], [y1, y1], color=col)
            plt.plot([x2, x1], [y2, y2], color=col)

    plt.imshow(img)
    plt.show()


class CityScapesNumpy:
    """Class to store data as numpy array."""

    FOLDERS = ["stuttgart_00"] #, "stuttgart_01", "stuttgart_02"]

    def __init__(self, base_dir="D:/Random/Cityscapes/leftImg8bit_demoVideo/leftImg8bit/demoVideo/downsized/",
                 detections_path="D:/Random/Cityscapes/leftImg8bit_demoVideo/leftImg8bit/demoVideo/point_rcnn.csv",
                 downscale_factor=1.0):
        """"""
        self.detections = pd.read_csv(detections_path)

        self.dir = base_dir
        self.downscale_factor = downscale_factor

        self.data = None
        self.frames = None

        self._load()

    def _load(self):
        """"""
        data = []
        frames = []

        print("Loading data into array...")
        for folder in self.FOLDERS:
            images = glob.glob(self.dir + folder + "\\*.png")
            for image_path in tqdm(images):
                img = skimage.io.imread(image_path)
                img_shape = np.shape(img)
                out_shape = (int(img_shape[0] * self.downscale_factor), int(img_shape[1] * self.downscale_factor), 3)

                downscaled = skimage.transform.resize(img, out_shape)
                # downscaled = skimage.util.img_as_ubyte(downscaled)

                data.append(downscaled)
                frames.append(os.path.basename(image_path))

        self.data = np.array(data, dtype=np.double)
        self.frames = frames

        print("Size of data: ", self.data.nbytes)

    def get(self, frame: str):
        """Returns image and bounding box annotations for frame."""
        assert frame in self.frames, "Frame not known..."

        idx = self.frames.index(frame)
        img_np = self.data[idx]
        detections = self.detections.loc[self.detections['Frame name'] == frame]
        boxes, labels, scores = detections['Box'], detections['Label'], detections['Score']

        # Hotfix for wrong data classes...
        boxes = boxes.to_numpy()
        boxes_clipped = [bx.replace("[", "").replace("]", "") for bx in boxes]
        boxes_np = [np.fromstring(bx, dtype=float, sep=' ') for bx in boxes_clipped]

        # Maybe downscale bounding boxes
        boxes_np = np.array([bx * 0.3 for bx in boxes_np])

        return img_np, boxes_np, labels.to_numpy(), scores.to_numpy()

    def get_idx(self, idx):
        """Get data from index."""
        assert idx <= len(self.frames), "Idx is larger than number of frames"
        frame_name = self.frames[idx]
        return self.get(frame_name)

    def __len__(self):
        return len(self.frames)

    def __get_item__(self, idx):
        return self.get_idx(idx)


if __name__ == "__main__":
    data_set = CityScapesNumpy()

    for i, _ in enumerate(data_set):
        img, bx, lbl, scrs = data_set.get_idx(i)
        plot_detections(img, bx, lbl, scrs, categories=[1, 2, 3, 4, 5, 6, 7, 8, 8, 10])