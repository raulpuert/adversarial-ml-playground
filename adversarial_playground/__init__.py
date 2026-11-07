"""adversarial-ml-playground.

Playground educativo de *Machine Learning Adversarial* implementado desde cero
en PyTorch: ataques de evasión (FGSM, BIM, PGD, DeepFool, C&W), defensas
(entrenamiento adversarial, transformaciones de entrada, randomized smoothing)
y evaluación de robustez.

Ejemplo rápido::

    from adversarial_playground.attacks import FGSM
    from adversarial_playground.models import MnistCNN
    from adversarial_playground.utils.data import load_checkpoint

    model, meta = load_checkpoint("checkpoints/mnist_cnn.pt")
    attack = FGSM(model, eps=0.15)
    result = attack(images, labels)
    print(result.success.float().mean())  # tasa de éxito del ataque
"""

from adversarial_playground import attacks, defenses, evaluation, models, utils, visualization

__version__ = "1.0.0"
__all__ = [
    "__version__",
    "attacks",
    "defenses",
    "evaluation",
    "models",
    "utils",
    "visualization",
]
