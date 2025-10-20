# datasets.py - region 실험 및 baseline 호환 모두 지원
import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import cv2

class StegDataset(Dataset):
    def __init__(self, dataframe, transform=None, return_path=False):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform
        self.return_path = return_path

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_path = self.df.iloc[idx]['path']
        label = self.df.iloc[idx]['label']
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image=image)['image']

        if self.return_path:
            return image, label, image_path
        else:
            return image, label
