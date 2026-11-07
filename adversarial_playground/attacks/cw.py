"""C&W — ataque de Carlini & Wagner (2017), variante L2.

El ataque de optimización más conocido: en lugar de seguir gradientes de la
pérdida, **optimiza directamente** la perturbación con Adam para minimizar

    min_δ  ||δ||₂²  +  c · f(x + δ)

donde ``f`` es una función de margen que vale 0 cuando el ataque ya funciona
(con margen ``kappa`` de confianza). El hiperparámetro ``c`` se ajusta por
búsqueda binaria: pequeño ⇒ perturbaciones más sutiles; grande ⇒ más éxito.

Es el ataque más fuerte del zoo clásico (y el más lento): sirve como
referencia de "cuánta robustez real tiene un modelo". La variante L2 es la
más citada; C&W también definió L0 y L∞.

Referencia: https://arxiv.org/abs/1608.04644
"""

import torch
from torch import Tensor, nn

from adversarial_playground.attacks.base import AttackResult, BaseAttack


class CarliniWagnerL2(BaseAttack):
    """C&W L2 con búsqueda binaria sobre ``c``.

    Args:
        model: modelo objetivo (logits, imágenes en [0, 1]).
        confidence: margen ``kappa`` exigido (0 = basta con cambiar la
            predicción; valores > 0 fuerzan un margen de confianza).
        targeted: si ``True``, ``labels`` son las clases objetivo.
        learning_rate: paso de Adam sobre la perturbación reparametrizada.
        binary_search_steps: cuántas veces se ajusta ``c`` (3-9 típico).
        max_iter: iteraciones de Adam por búsqueda (200-1000 según presupuesto).
        initial_c: valor inicial de ``c``.
        clip_min, clip_max: rango válido de píxeles.

    Note:
        Es un ataque **costoso**: ``binary_search_steps × max_iter`` pasadas
        forward+backward. Para evaluación usa subconjuntos (500-1000 imágenes).

    Example:
        >>> attack = CarliniWagnerL2(model, binary_search_steps=3, max_iter=200)
        >>> result = attack(images, labels)
        >>> result.success_rate  # doctest: +SKIP
        1.0
    """

    def __init__(
        self,
        model: nn.Module,
        confidence: float = 0.0,
        targeted: bool = False,
        learning_rate: float = 0.01,
        binary_search_steps: int = 5,
        max_iter: int = 200,
        initial_c: float = 1e-3,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
    ) -> None:
        super().__init__(model, clip_min, clip_max)
        self.confidence = confidence
        self.targeted = targeted
        self.learning_rate = learning_rate
        self.binary_search_steps = binary_search_steps
        self.max_iter = max_iter
        self.initial_c = initial_c

    # ------------------------------------------------------------------ #
    # Función de margen f (paper, sección II, f6)
    # ------------------------------------------------------------------ #
    def _margin(self, logits: Tensor, labels: Tensor) -> tuple[Tensor, Tensor]:
        """Margen crudo y su versión con suelo ``-kappa``.

        No dirigido:  ``f = Z_y − max_{i≠y} Z_i``   (queremos que sea negativo)
        Dirigido:     ``f = max_{i≠t} Z_i − Z_t``   (queremos que Z_t domine)

        Devuelve ``(f_raw, f)``, donde ``f_raw`` determina el éxito (``< 0``)
        y ``f`` (con suelo en ``-kappa``) entra en el objetivo. Con
        ``kappa=0``, ``f`` vale exactamente 0 al tener éxito: por eso el
        éxito se comprueba sobre ``f_raw`` y nunca sobre ``f``.
        """
        z_label = logits.gather(1, labels.unsqueeze(1)).squeeze(1)
        z_masked = logits.clone()
        z_masked.scatter_(1, labels.unsqueeze(1), -1e10)
        z_other_max = z_masked.max(dim=1).values
        if self.targeted:
            f_raw = z_other_max - z_label
        else:
            f_raw = z_label - z_other_max
        return f_raw, f_raw.clamp_min(-self.confidence)

    def forward(self, images: Tensor, labels: Tensor) -> AttackResult:
        self._validate_batch(images, labels)
        batch = images.shape[0]
        device = images.device

        # Reparametrización tanh: optimizamos w sin restricciones y
        # x_adv = mid + half·tanh(w) vive siempre dentro del rango válido.
        mid = (self.clip_max + self.clip_min) / 2.0
        half = (self.clip_max - self.clip_min) / 2.0
        tanh_input = ((images - mid) / half).clamp(1e-7 - 1, 1 - 1e-7)
        w_init = torch.atanh(tanh_input)

        # Búsqueda binaria de c por muestra
        c = torch.full((batch,), self.initial_c, device=device)
        lower = torch.zeros(batch, device=device)
        upper = torch.full((batch,), 1e10, device=device)

        best_adv = images.clone()
        best_l2 = torch.full((batch,), float("inf"), device=device)

        for _ in range(self.binary_search_steps):
            w = w_init.clone().detach().requires_grad_(True)
            optimizer = torch.optim.Adam([w], lr=self.learning_rate)
            found_this_round = torch.zeros(batch, dtype=torch.bool, device=device)

            for _ in range(self.max_iter):
                optimizer.zero_grad()
                adv = mid + half * torch.tanh(w)
                logits = self.model(adv)
                f_raw, f = self._margin(logits, labels)
                l2_sq = (adv - images).flatten(1).pow(2).sum(dim=1)

                # Objetivo C&W por muestra; media para el paso de Adam.
                loss_per_sample = l2_sq + c * f
                loss_per_sample.mean().backward()
                optimizer.step()

                # Éxito: margen crudo negativo (independiente del clamp -kappa).
                success = f_raw < 0
                with torch.no_grad():
                    improved = success & (l2_sq < best_l2**2)
                    if improved.any():
                        best_l2[improved] = l2_sq[improved].sqrt()
                        best_adv[improved] = adv.detach()[improved]
                        found_this_round |= improved

            # Actualización de la búsqueda binaria (estilo torchattacks):
            # si tuvo éxito bajamos c; si no, lo subimos x10 hasta acotar.
            for i in range(batch):
                if found_this_round[i]:
                    upper[i] = min(upper[i], c[i])
                    if upper[i] < 1e9:
                        c[i] = (lower[i] + upper[i]) / 2
                else:
                    lower[i] = max(lower[i], c[i])
                    c[i] = (
                        (lower[i] + upper[i]) / 2 if upper[i] < 1e9 else c[i] * 10
                    )

        result = self.compute_result(images, best_adv, labels, targeted=self.targeted)
        result.info["final_c"] = c.detach()
        return result
