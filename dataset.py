
import os
import torch
import numpy as np
import random
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader

random.seed(0)

GOOD = os.listdir(f'data/good/')
DEFECT = os.listdir(f'data/defect/')
random.shuffle(GOOD)
random.shuffle(DEFECT)


class TestData(Dataset):
    def __init__(self):
        ts = pd.read_csv('data/submission_sample.csv')
        self.data = [i + '.bmp' for i in ts['id'].to_list()]  # os.listdir('data/test/')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        tensor = np.array(Image.open(f'data/test/{self.data[index]}').resize(size=(275, 267)))[np.newaxis, :, :]
        return torch.FloatTensor(tensor) / 255


class SemiCondData(Dataset):
    """
    use bootstrap to balance the good/defect data
    """
    def __init__(self, frac=1.0, train=False):
        if train:
            self.len_good = int(len(GOOD) * frac)
            self.len_defect = int(len(DEFECT) * frac)
            self.good = os.listdir(f'data/good/')[:self.len_good]
            self.defect = os.listdir(f'data/defect/')[:self.len_defect]

        else:
            self.len_good = len(GOOD) - int(len(GOOD) * frac)
            self.len_defect = len(DEFECT) - int(len(DEFECT) * frac)
            self.good = os.listdir(f'data/good/')[-self.len_good:]
            self.defect = os.listdir(f'data/defect/')[-self.len_defect:]
        self.total_len = self.len_good + self.len_defect

    def __len__(self):
        return len(self.good)*2

    def __getitem__(self, index):
        """
        0: good, 1: defect
        """
        class_idx = 1
        if index < self.len_good:
            tensor = np.array(Image.open(f'data/good/{self.good[index]}').resize(size=(275, 267)))[np.newaxis, :, :]
            class_idx = 0
        elif index < self.total_len:
            tensor = np.array(Image.open(f'data/defect/{self.defect[index-self.len_good]}').resize(size=(275, 267)))[np.newaxis, :, :]
        else:
            tensor = np.array(Image.open(f'data/defect/{random.choice(self.defect)}').resize(size=(275, 267)))[np.newaxis, :, :]
        return torch.FloatTensor(tensor) / 255, class_idx


train_loader = DataLoader(SemiCondData(frac=0.8, train=True), batch_size=256, shuffle=True)
validate_loader = DataLoader(SemiCondData(frac=0.8, train=False), batch_size=256, shuffle=True)
test_loader = DataLoader(TestData(), batch_size=256, shuffle=True)

defect_areas = pd.read_csv('data/defect_area.csv')

if __name__ == '__main__':
    ts = TestData()
    tr = SemiCondData(frac=0.8, train=True)
    val = SemiCondData(frac=0.8, train=False)

