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
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=4)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"Using {device} device: {torch.cuda.get_device_name(device) if torch.cuda.is_available() else ''}"
    )
