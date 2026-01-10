import torch
import torch.nn as nn

from .encoder import MyEncoderBlock


class MyViT(nn.Module):
    def __init__(
        self,
        n_patches: int = 7,
        img_size: tuple = (1, 28, 28),
        hidden_dim=16,
        n_heads=2,
        n_classes=10,
    ):
        super().__init__()
        self.n_patches = n_patches
        assert img_size[-1] % n_patches == 0, (
            "Image input size must be divisible by n patches size"
        )
        projection_input = (img_size[-1] ** 2) * img_size[0]
        self.projection = nn.Linear(projection_input // (self.n_patches**2), hidden_dim)
        # Other option with lazy linear but i am not a big fan
        # self.projection = nn.LazyLinear(hidden_dim)

        self.class_token = nn.Parameter((torch.rand(1, hidden_dim)))
        self.pos_embed = nn.Parameter(
            self._get_positional_encoding(self.n_patches**2 + 1, hidden_dim),
            requires_grad=False,
        )
        self.encoder_blocks = nn.ModuleList(
            [
                MyEncoderBlock(
                    dimension=hidden_dim,
                    n_heads=n_heads,
                )
            ]
        )
        self.classification_head = nn.Sequential(
            nn.Linear(hidden_dim, n_classes), nn.Softmax(dim=-1)
        )

    def _patchify(self, x):
        n, c, h, w = x.shape
        assert h == w, "Height and Widht must be save for patches to work"

        patches = torch.zeros(n, self.n_patches**2, h * w * c // (self.n_patches**2))
        size = h // self.n_patches
        for idx, image in enumerate(x):
            for pos_x in range(self.n_patches):
                for pos_y in range(self.n_patches):
                    start_x = pos_x * size
                    end_x = start_x + size
                    start_y = pos_y * size
                    end_y = start_y + size
                    patch_image = image[:, start_x:end_x, start_y:end_y]
                    patches[idx, pos_x * self.n_patches + pos_y] = patch_image.flatten()

        return patches

    def _get_positional_encoding(self, sequence_length, dimension):
        result = torch.ones(sequence_length, dimension)
        for i in range(sequence_length):
            for j in range(dimension):
                result[i, j] = (
                    torch.sin(
                        torch.tensor(i)
                        / (10_000 ** (torch.tensor(j) / torch.tensor(dimension)))
                    )
                    if j % 2 == 0
                    else torch.cos(
                        torch.tensor(i)
                        / (10_000 ** (torch.tensor(j - 1) / torch.tensor(dimension)))
                    )
                )
        return result

    def forward(self, x):
        n = x.shape[0]
        x = self._patchify(x)
        x = self.projection(x)
        x = torch.stack([torch.vstack((self.class_token, x[i])) for i in range(n)])
        x = x + self.pos_embed
        # Until here we are at z0 = [xclass; x_1^{pE}; ...; x_N^{pE}] + Epos
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x)
        return x
