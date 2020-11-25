"""Export PointRCNN detections to csv."""
import os
import glob
import skimage.io
import time
from tqdm import tqdm
import torchvision
import pandas as pd


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

DATA_PATH = "D:\\Random\\Cityscapes\\leftImg8bit_demoVideo\\leftImg8bit\demoVideo\\"
FOLDERS = ["stuttgart_00", "stuttgart_01", "stuttgart_02"]

# Folder to export images into
TARGET_PATH = "D:\\Random\\Cityscapes\\leftImg8bit_demoVideo\\leftImg8bit\demoVideo\\point_rcnn.csv"

MODEL = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
MODEL.eval()
MODEL.float()

start = time.time()

frames = []
bboxes = []
labels = []
scores = []

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

for folder in FOLDERS:
    images = glob.glob(DATA_PATH + folder + "\\*.png")

    for image_path in tqdm(images):

        img = skimage.io.imread(image_path)
        tensor = transform(img)
        frame_predictions = MODEL([tensor])[0]  # only one image per batch ...

        # print(frame_predictions)
  
        for i, _ in enumerate(frame_predictions['labels']):
            frames.append(os.path.basename(image_path))
            bboxes.append(frame_predictions['boxes'][i].detach().numpy())
            labels.append(frame_predictions['labels'][i].detach().numpy())
            scores.append(frame_predictions['scores'][i].detach().numpy())

df = pd.DataFrame({'Frame name': frames, 'Box': bboxes, 'Label': labels, 'Score': scores})
df.to_csv(TARGET_PATH)

print(time.time() - start)
