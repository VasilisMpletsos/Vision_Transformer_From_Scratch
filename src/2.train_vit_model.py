import numpy as np
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import MyViT
from utils import MyCustomImageFolderDatset

if __name__ == "__main__":
    train_dataset = MyCustomImageFolderDatset(
        images_path="./data/images/train", includes_labels=True
    )
    test_dataset = MyCustomImageFolderDatset(
        images_path="./data/images/test", includes_labels=False
    )
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=64)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"Using {device} device: {torch.cuda.get_device_name(device) if torch.cuda.is_available() else ''}"
    )

    vit_model = MyViT(4, (1, 28, 28), 16, 2, 10)
    vit_model = vit_model.to("cuda")

    EPOCHS = 10
    # As described in the paper
    # "We train all models, using Adam (Kingma & Ba,2015) with β1 = 0.9, β2 = 0.999, a batch size of 4096 and apply a high weight decay of 0.1
    # And also in Table 3 the best learning rate for ViT Base is 8*10e-4
    optimizer = Adam(
        vit_model.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=0.01
    )
    loss_func = CrossEntropyLoss(label_smoothing=0)

    for epoch in range(EPOCHS):
        vit_model.train()
        epoch_train_loss = 0
        for step, (images, targets) in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            images = images.to(device)
            targets = targets.to(device)

            predictions = vit_model(images)

            loss = loss_func(predictions, targets)
            epoch_train_loss += loss.item()
            loss.backward()
            optimizer.step()

            if (step + 1) % 50 == 0:
                print(f"Step {step + 1}: {loss.item():.2f}")
                predictions = torch.argmax(predictions, dim=-1)
                accuracy = (torch.sum(predictions == targets) / targets.shape[0]) * 100
                print(f"Accuracy is {accuracy:.2f}%")

        print(f"Epoch {epoch + 1}: {epoch_train_loss:.2f}")
