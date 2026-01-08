import numpy as np
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from utils import MyCustomImageFolderDatset

if __name__ == "__main__":
    train_dataset = MyCustomImageFolderDatset(
        images_path="./data/images/train", includes_labels=True
    )
    test_dataset = MyCustomImageFolderDatset(
        images_path="./data/images/test", includes_labels=False
    )
