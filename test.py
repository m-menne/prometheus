import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from utils import DataReaderPlainImg

path = ["./leftImg8bit/demoVideo/stuttgart_00", "./leftImg8bit/demoVideo/stuttgart_01", "./leftImg8bit/demoVideo/stuttgart_02"]
# size=(16, 16)
data_augmentations = transforms.Compose([
        # transforms.Resize(size),
        # transforms.RandomCrop(size, pad_if_needed=True),
        transforms.ToTensor()
    ])
data = DataReaderPlainImg(path[0], transform=data_augmentations)
loader = DataLoader(data, batch_size=1, shuffle=False)
t = tqdm(loader)
for image in t:
    print(image.shape)
    # Run Here for every image 


