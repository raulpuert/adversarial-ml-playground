"""Arquitecturas de modelos disponibles en el playground.

Todos los modelos:

* aceptan imágenes en ``[0, 1]`` (la normalización interna va en ``layers.Normalize``);
* exponen ``num_classes`` para saber cuántas clases predicen;
* se registran en :data:`MODEL_REGISTRY` para poder instanciarse por nombre
  desde la CLI (``--model mnist_cnn``).
"""

from torch import nn

from adversarial_playground.models.cifar_resnet import CifarResNet
from adversarial_playground.models.layers import Normalize
from adversarial_playground.models.mnist_cnn import MnistCNN

MODEL_REGISTRY: dict[str, type[nn.Module]] = {
    "mnist_cnn": MnistCNN,
    "cifar_resnet": CifarResNet,
}

__all__ = ["MnistCNN", "CifarResNet", "Normalize", "MODEL_REGISTRY"]
