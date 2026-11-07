"""Capas auxiliares usadas por los modelos del paquete."""

import torch
from torch import Tensor, nn


class Normalize(nn.Module):
    """Normaliza la entrada con media y desviación típica fijas.

    Los modelos de este paquete esperan imágenes **en [0, 1]** (píxeles crudos
    tal cual salen de ``transforms.ToTensor``). En lugar de normalizar en el
    dataset — que es lo habitual —, la normalización vive **dentro del modelo**
    como primera capa. ¿Por qué?

    1. Los ataques adversariales deben trabajar en el **espacio de la imagen**
       (píxeles en [0, 1]), porque ahí es donde tienen sentido las restricciones
       de un atacante (un píxel no puede valer -2 ni 7). Si el dataset
       normalizara, ``eps=0.1`` ya no significaría nada interpretable.
    2. Al ir la normalización dentro de ``model.forward``, cualquier ataque que
       use ``model(x)`` es automáticamente correcto, sin态 tener que acordarse
       correcto, sin tener que acordarse de normalizar dentro de cada ataque.

    Args:
        mean: Media por canal, p. ej. ``(0.1307,)`` para MNIST.
        std: Desviación típica por canal, p. ej. ``(0.3081,)`` para MNIST.

    Example:
        >>> norm = Normalize(mean=(0.1307,), std=(0.3081,))
        >>> x = torch.rand(1, 1, 28, 28)
        >>> norm(x).shape
        torch.Size([1, 1, 28, 28])
    """

    def __init__(self, mean: tuple[float, ...], std: tuple[float, ...]) -> None:
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean).view(1, -1, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, -1, 1, 1))

    def forward(self, x: Tensor) -> Tensor:
        return (x - self.mean) / self.std

    def extra_repr(self) -> str:
        return f"mean={self.mean.flatten().tolist()}, std={self.std.flatten().tolist()}"
