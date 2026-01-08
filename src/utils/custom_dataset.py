from pathlib import Path
from typing import Tuple, Union

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.io import decode_image
from torchvision.transforms import ToTensor


class MyCustomImageFolderDatset(Dataset):
    def __init__(self, images_path: Path, includes_labels: bool = True):
        super().__init__()
        file_path = Path(images_path)
        self.images = []
        self.labels = []
        for ext in ("*.png", "*.jpg"):
            for file in file_path.rglob(ext):
                self.images.append(file)
                if includes_labels:
                    self.labels.append(file.__str__().split("\\")[-2])

        if includes_labels:
            assert len(self.images) == len(self.labels), (
                "Number of images and labels must agree"
            )
        print(f"Succesfully created a dataset with {len(self.images)} images")

    def __len__(self):
        return len(self.images)

    def __getitem__(
        self, index
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        numpy_image = decode_image(self.images[index])
        if self.labels:
            return numpy_image, torch.tensor(int(self.labels[index]))
        else:
            return numpy_image
