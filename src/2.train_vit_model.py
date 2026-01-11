import numpy as np
import torch
import torch.nn as nn
from regex import W
from torch.nn import CrossEntropyLoss
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import v2
from tqdm import tqdm

from model import MyViT
from utils import MyCustomImageFolderDatset

if __name__ == "__main__":
    writer = SummaryWriter("./logs/vit_model/")
    # Here i create transforms that i can pass to my custom image dataset and perform random ajdjustments during training to boost results
    training_transforms = v2.Compose(
        [
            v2.RandomRotation(45),
            v2.RandomAffine(15),
            v2.RandomHorizontalFlip(p=0.2),
            v2.RandomResizedCrop(size=(28, 28), antialias=True, scale=(0.7, 1.0)),
        ]
    )

    train_dataset = MyCustomImageFolderDatset(
        images_path="./data/images/train",
        includes_labels=True,
        # transforms=training_transforms,
    )
    validation_dataset = MyCustomImageFolderDatset(
        images_path="./data/images/val", includes_labels=True
    )
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=64)
    validation_loader = DataLoader(validation_dataset, shuffle=False, batch_size=64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"Using {device} device: {torch.cuda.get_device_name(device) if torch.cuda.is_available() else ''}"
    )

    vit_model = MyViT(4, (1, 28, 28), 16, 2, 10)
    vit_model = vit_model.to("cuda")

    EPOCHS = 50
    # As described in the paper
    # "We train all models, using Adam (Kingma & Ba,2015) with β1 = 0.9, β2 = 0.999, a batch size of 4096 and apply a high weight decay of 0.1 (i am lowering it to 0.01)
    # And also in Table 3 the best learning rate for ViT Base is 8*10e-4
    # optimizer = Adam(
    #     vit_model.parameters(), lr=0.01, betas=(0.9, 0.999), weight_decay=0.01
    # )
    optimizer = AdamW(vit_model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-4)
    loss_func = CrossEntropyLoss(label_smoothing=0)

    train_size = len(train_dataset)
    validation_size = len(validation_dataset)
    for epoch in range(EPOCHS):
        vit_model.train()
        epoch_train_loss = 0
        correct = 0
        for step, (images, targets) in enumerate(
            tqdm(train_loader, desc="Training Loop")
        ):
            optimizer.zero_grad()
            images = images.to(device)
            targets = targets.to(device)

            predictions = vit_model(images)

            loss = loss_func(predictions, targets)
            loss_item = loss.item()
            epoch_train_loss += loss_item
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                predictions = torch.argmax(predictions, dim=-1)
                batch_correct = torch.sum(predictions == targets)
                correct += batch_correct.item()

        train_accuracy = (correct / train_size) * 100
        print(
            f"Epoch {epoch + 1} - Train Loss: {epoch_train_loss:.2f} & Train Accuracy: {train_accuracy:.3f}%"
        )
        writer.add_scalar("Loss/train", epoch_train_loss, epoch + 1)
        writer.add_scalar("Accuracy/train", train_accuracy, epoch + 1)
        scheduler.step()

        vit_model.eval()
        epoch_validation_loss = 0
        correct = 0
        for images, targets in tqdm(validation_loader, desc="Validation Loop"):
            images = images.to(device)
            targets = targets.to(device)
            predictions = vit_model(images)
            loss = loss_func(predictions, targets)
            epoch_validation_loss += loss.item()
            predictions = torch.argmax(predictions, dim=-1)
            batch_correct = torch.sum(predictions == targets)
            correct += batch_correct.item()

        validation_accuracy = (correct / validation_size) * 100
        print(
            f"Epoch {epoch + 1} - Val Loss: {epoch_validation_loss:.2f} & Val Accuracy: {validation_accuracy:.3f}%"
        )
        writer.add_scalar("Loss/validation", epoch_validation_loss, epoch + 1)
        writer.add_scalar("Accuracy/validation", validation_accuracy, epoch + 1)
