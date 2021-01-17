# Daimler SpeedHack - Team Prometheus
Second place winning implementation in Speed-Hack: Autonomous Driving @ Daimler.
Object detection, instance segmentation and tracking on [Cityscapes](https://www.cityscapes-dataset.com/).

## Approach
For [object detection and instance segmentation](https://github.com/m-menne/prometheus/blob/main/mask_RCNN/maskrcnn.ipynb), we use a pre-trained Mask R-CNN network with a ResNet50 backbone. Based on the detected bounding boxes, a simple [object tracking](https://github.com/m-menne/prometheus/blob/tracking/tracking.ipynb) was implemented using the intersection over union.

## Example
![Alt Text](final.gif)

## Requirements
To install dependencies use: 

`pip install -r requirements.txt`
