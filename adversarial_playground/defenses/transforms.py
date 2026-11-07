"""Defensas por transformación de entrada (*input transformations*).

Idea: antes de pasar la imagen al modelo, "limpiarla" con una transformación
no diferenciable o estocástica que destruya la estructura del gradiente que
el atacante explotó. Son defensas baratas y compatibles con cualquier modelo,
pero **solas son débiles** (Athalye et al., 2018 las rompieron con EOT /
backwards pass diferenciable). En este playground sirven para *medir* ese
efecto con datos, no como recomendación de despliegue.

Referencia clave: "Obfuscated Gradients Give a False Sense of Security"
https://arxiv.org/abs/1802.00420
"""

from __future__ import annotations

from typing import Protocol

import torch
from torch import Tensor


class InputDefense(Protocol):
    """Contrato mínimo de una defensa de entrada: ``__call__(x) -> x``."""

    def __call__(self, images: Tensor) -> Tensor: ...


class BitDepthReduction:
    """Cuantiza los píxeles a ``bits`` niveles por canal.

    Un adversarial suele explotar decimales imperceptibles (p. ej. un píxel
    0.507 en vez de 0.500). Cuantizar a 3-5 bits "aplasta" esos matices con
    un coste casi nulo, a cambio de perder algo de precisión limpia.

    Args:
        bits: bits de profundidad (1 = binario, 8 = imagen original).

    Example:
        >>> defense = BitDepthReduction(bits=3)
        >>> out = defense(torch.rand(2, 1, 28, 28))
        >>> torch.unique(out).numel() <= 2**3
        True
    """

    def __init__(self, bits: int = 4) -> None:
        if not 1 <= bits <= 8:
            raise ValueError(f"bits debe estar en [1, 8], llega {bits}")
        self.bits = bits

    def __call__(self, images: Tensor) -> Tensor:
        levels = 2**self.bits - 1
        return (images * levels).round() / levels


class JPEGCompression:
    """Comprime cada imagen del lote como JPEG y la vuelve a decodificar.

    La cuantización DCT del JPEG "rompe" perturbaciones de alta frecuencia.
    Es no diferenciable (por eso confunde a los ataques de gradiente) y en
    la práctica casi no daña la precisión limpia si la calidad es alta.

    Args:
        quality: calidad JPEG 1-95 (menor = más destrucción).
    """

    def __init__(self, quality: int = 75) -> None:
        if not 1 <= quality <= 95:
            raise ValueError(f"quality debe estar en [1, 95], llega {quality}")
        self.quality = quality

    def __call__(self, images: Tensor) -> Tensor:
        import io

        from PIL import Image
        from torchvision.transforms import functional as tvf

        output = images.clone()
        for i in range(images.shape[0]):
            img_uint8 = (images[i].detach().clamp(0, 1) * 255).round().to(torch.uint8)
            pil_img = tvf.to_pil_image(img_uint8)  # modo "L" (MNIST) o "RGB" (CIFAR)
            buf = io.BytesIO()
            pil_img.save(buf, format="JPEG", quality=self.quality)
            buf.seek(0)
            tensor_back = tvf.to_tensor(Image.open(buf).convert(pil_img.mode))
            output[i] = tensor_back.to(images.device, images.dtype)
        return output


class GaussianNoise:
    """Añade ruido gaussiano N(0, sigma²) recortado a [0, 1].

    Usado como base de *randomized smoothing*; como defensa aislada inunda
    la señal del gradiente, pero también degrada la precisión limpia.

    Args:
        sigma: desviación típica del ruido (en unidades de píxel 0-1).
    """

    def __init__(self, sigma: float = 0.1) -> None:
        if sigma < 0:
            raise ValueError(f"sigma debe ser >= 0, llega {sigma}")
        self.sigma = sigma

    def __call__(self, images: Tensor) -> Tensor:
        noise = torch.randn_like(images) * self.sigma
        return (images + noise).clamp(0, 1)


class TotalVarianceMinimization:
    """Minimiza la variación total de la imagen suavizándola contra el original.

    Resuelve ``min_u ||u - x||² + lambda · TV(u)`` con descenso por gradiente,
    donde TV(u) = suma de normas L2 de las diferencias horizontales/verticales.
    La regularización TV elimina el "grano" de alta frecuencia de la
    perturbación adversarial manteniendo los bordes grandes de la imagen.

    Args:
        lam: peso de la regularización TV (mayor = más suavizado).
        steps: iteraciones de descenso por gradiente (con imagen en cuda,
            bajar a 10-20 por coste).
        lr: paso del descenso.

    Note:
        Implementación educativa: lenta (varias backward passes) y con
        hiperparámetros delicados. Incluida para experimentar.
    """

    def __init__(self, lam: float = 0.03, steps: int = 20, lr: float = 0.1) -> None:
        self.lam = lam
        self.steps = steps
        self.lr = lr

    def __call__(self, images: Tensor) -> Tensor:
        u = images.clone().detach().requires_grad_(True)
        for _ in range(self.steps):
            dh = u[:, :, :, 1:] - u[:, :, :, :-1]
            dv = u[:, :, 1:, :] - u[:, :, :-1, :]
            tv = dh.pow(2).sum() + dv.pow(2).sum()
            data = (u - images.detach()).pow(2).sum()
            obj = data + self.lam * tv
            grad = torch.autograd.grad(obj, u)[0]
            u = (u - self.lr * grad).detach().clamp(0, 1).requires_grad_(True)
        return u.detach()
