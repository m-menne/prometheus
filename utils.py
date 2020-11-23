import os
import torch
from PIL import Image
from torch.utils.data import Dataset


class DataReaderPlainImg:
    def __init__(self, root, transform=None):
        self.transform = transform
        self.root = root
        self.image_files = [f for f in os.listdir(root) if f.endswith(".png")]

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.root, self.image_files[idx])).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.image_files)
