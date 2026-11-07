"""ResNet compacta para CIFAR-10 (variante educativa de He et al., 2015).

Una ResNet "mini" con bloques básicos y 3 etapas de anchos configurables.
Con ``widths=(32, 64, 128)`` tiene ~2.8M parámetros y alcanza ~80% de
precisión en CIFAR-10 con 10-15 épocas — un buen equilibrio para entrenar
en CPU sin GPU y aun así tener un modelo "de verdad" al que atacar.
"""

import torch
from torch import Tensor, nn

from adversarial_playground.models.layers import Normalize

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


class BasicBlock(nn.Module):
    """Bloque residual: ``conv-bn-relu-conv-bn + skip``, con ReLU pos-suma.

    Si el stride reduce la resolución o cambia el número de canales, el
    camino corto usa una conv 1x1 para casar dimensiones (proyección).
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x: Tensor) -> Tensor:
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return torch.relu(out + self.shortcut(x))


class CifarResNet(nn.Module):
    """ResNet ligera para imágenes 32x32 (CIFAR-10).

    Args:
        widths: canales de cada una de las 3 etapas.
        num_blocks: bloques por etapa (la 1ª etapa entra con stride 1,
            las otras con stride 2).
        num_classes: 10 para CIFAR-10.

    Example:
        >>> model = CifarResNet()
        >>> model(torch.rand(4, 3, 32, 32)).shape
        torch.Size([4, 10])
    """

    num_classes = 10

    def __init__(
        self,
        widths: tuple[int, int, int] = (32, 64, 128),
        num_blocks: tuple[int, int, int] = (2, 2, 2),
        num_classes: int = 10,
    ) -> None:
        super().__init__()
        self.norm = Normalize(CIFAR10_MEAN, CIFAR10_STD)
        self.stem = nn.Sequential(
            nn.Conv2d(3, widths[0], 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(widths[0]),
            nn.ReLU(inplace=True),
        )
        layers: list[nn.Module] = []
        in_ch = widths[0]
        for stage, (out_ch, n_blocks) in enumerate(zip(widths, num_blocks, strict=True)):
            stride = 1 if stage == 0 else 2
            for i in range(n_blocks):
                layers.append(BasicBlock(in_ch, out_ch, stride=stride if i == 0 else 1))
                in_ch = out_ch
        self.layers = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(widths[-1], num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm(x)
        x = self.stem(x)
        x = self.layers(x)
        return self.head(x)
