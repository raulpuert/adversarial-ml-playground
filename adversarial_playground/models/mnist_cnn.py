"""CNN compacta para MNIST (99% de precisión en 3 épocas de entrenamiento).

Arquitectura clásica del tutorial de PyTorch, elegida por ser pequeña
(~700k parámetros ⇒ entrenamiento en CPU en minutos) pero suficientemente
profunda como para que los ataques adversariales tengan efecto.
"""

import torch
from torch import Tensor, nn

from adversarial_playground.models.layers import Normalize

MNIST_MEAN = (0.1307,)
MNIST_STD = (0.3081,)


class MnistCNN(nn.Module):
    """Red convolucional 32-64 para MNIST.

    Flujo: ``[B, 1, 28, 28] → conv32 → conv64 → pool → fc128 → fc10``.

    Example:
        >>> model = MnistCNN()
        >>> model(torch.rand(4, 1, 28, 28)).shape
        torch.Size([4, 10])
    """

    num_classes = 10

    def __init__(self, dropout: float = 0.25) -> None:
        super().__init__()
        self.norm = Normalize(MNIST_MEAN, MNIST_STD)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.dropout1 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(64 * 14 * 14, 128)
        self.dropout2 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, self.num_classes)

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm(x)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x)  # [B, 64, 14, 14]
        x = self.dropout1(torch.flatten(x, 1))
        x = torch.relu(self.fc1(x))
        return self.fc2(self.dropout2(x))
