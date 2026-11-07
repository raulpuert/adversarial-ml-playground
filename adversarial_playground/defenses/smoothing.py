"""Randomized Smoothing (Cohen, Rosenfeld & Kolter, 2019).

La defensa con **garantía de robustez certificable** más usada. En lugar de
predecir con el modelo a secas, se predice con el **voto mayoritario** sobre
``n`` copias de la imagen perturbadas con ruido gaussiano N(0, σ²):

    ŷ(x) = argmax_c  P( f(x + δ) = c ),   δ ~ N(0, σ²I)

Si el clasificador suavizado elige la clase ``A`` con probabilidad suficientemente
alta (acotada con una binomial de confianza 1−α, *Clopper-Pearson*), se puede
**certificar** que ningún ataque L2 con radio ``r = σ·Φ⁻¹(p_A)`` puede cambiar
la predicción — demostrablemente, no empíricamente.

A cambio: la precisión limpia baja un poco y certificar es caro (miles de
pasadas por imagen). Aquí implementamos predicción suavizada y el radio
certificado del paper.

Referencia: https://arxiv.org/abs/1902.02918
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.distributions.normal import Normal


class RandomizedSmoothing:
    """Clasificador suavizado con ruido gaussiano.

    Args:
        model: modelo base (logits, imágenes en [0, 1]). Debe estar en ``eval()``.
        sigma: desviación típica del ruido (0.25 es el valor típico del paper).
        n: número de muestras por predicción (500-1000 para predecir,
            10 000-100 000 para certificar).
        batch_size: cuántas copias ruidosas se evalúan por pasada.

    Example:
        >>> smooth = RandomizedSmoothing(model, sigma=0.25, n=500)
        >>> preds = smooth.predict(images)          # etiquetas suavizadas
        >>> radii = smooth.certify(images, n=10000) # radio L2 certificado
    """

    def __init__(
        self,
        model: nn.Module,
        sigma: float = 0.25,
        n: int = 500,
        batch_size: int = 1024,
    ) -> None:
        self.model = model
        self.sigma = sigma
        self.n = n
        self.batch_size = batch_size
        self._normal = Normal(torch.tensor(0.0), torch.tensor(1.0))

    # ------------------------------------------------------------------ #
    # Predicción suavizada
    # ------------------------------------------------------------------ #
    def _vote_counts(self, images: Tensor, n: int, num_classes: int) -> Tensor:
        """Votos por clase: ``(B, num_classes)`` contando predicciones ruidosas."""
        batch = images.shape[0]
        counts = torch.zeros(batch, num_classes, device=images.device)
        done = 0
        while done < n:
            size = min(self.batch_size, n - done)
            # `size` copias ruidosas de todo el batch: (size*B, C, H, W)
            noisy = images.repeat(size, 1, 1, 1) + self.sigma * torch.randn(
                size * batch, *images.shape[1:], device=images.device
            )
            preds = self.model(noisy).argmax(dim=1)  # (size*B,)
            image_idx = torch.arange(batch, device=images.device).repeat(size)
            flat_votes = image_idx * num_classes + preds
            counts.view(-1).index_add_(
                0, flat_votes, torch.ones_like(preds, dtype=counts.dtype)
            )
            done += size
        return counts

    def predict(self, images: Tensor, n: int | None = None) -> Tensor:
        """Etiqueta suavizada (voto mayoritario) por imagen: ``(B,)``.

        Empates resueltos con la clase de menor índice (determinista).
        """
        n = n or self.n
        counts = self._vote_counts(images, n, _num_classes(self.model))
        return counts.argmax(dim=1)

    # ------------------------------------------------------------------ #
    # Certificación
    # ------------------------------------------------------------------ #
    def certify(
        self,
        images: Tensor,
        n: int = 10000,
        alpha: float = 0.001,
    ) -> Tensor:
        """Radio L2 certificado por imagen: ``(B,)`` (0 = no certificado).

        Devuelve ``r = σ · Φ⁻¹(p̄_A)`` donde ``p̄_A`` es la cota inferior
        *Clopper-Pearson* (nivel 1−α) de la probabilidad de la clase
        mayoritaria. Si la cota no supera 0.5, el radio es 0: la predicción
        no es certificable con esa confianza.

        Args:
            images: imágenes en [0, 1].
            n: número de muestras de Monte Carlo (más n ⇒ cotas más finas).
            alpha: nivel de significación (1−α de confianza).

        Raises:
            ImportError: si no está instalado ``scipy`` (necesario para la
                binomial de Clopper-Pearson: ``pip install scipy``).
        """
        try:
            from scipy.stats import beta as beta_dist
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "certify() necesita scipy (pip install scipy) para la cota "
                "Clopper-Pearson de la binomial."
            ) from exc

        counts = self._vote_counts(images, n, _num_classes(self.model)).cpu()
        top2 = counts.topk(2, dim=1)
        n_a = top2.values[:, 0]
        n_b = top2.values[:, 1]

        # Cota inferior de p_A con confianza 1-alpha (Clopper-Pearson):
        # p_lower = Beta^{-1}(alpha; nA, n - nA + 1)
        n_total = float(n)
        p_lower = torch.tensor(
            beta_dist.ppf(alpha, n_a.numpy(), n_total - n_a.numpy() + 1),
            dtype=torch.float64,
        )

        radius = torch.zeros(images.shape[0])
        certifiable = (n_a > n_b) & (p_lower > 0.5)
        if certifiable.any():
            z = self._normal.icdf(p_lower[certifiable].float())
            radius[certifiable] = self.sigma * z
        return radius


def _num_classes(model: nn.Module) -> int:
    """Intenta deducir el número de clases del modelo."""
    if hasattr(model, "num_classes"):
        return int(model.num_classes)
    # último Linear como último recurso
    last_linear = [m for m in model.modules() if isinstance(m, nn.Linear)][-1]
    return last_linear.out_features
