# 05 · Referencias y lecturas recomendadas

## Papers fundacionales

| Paper | Año | Qué aportó |
|---|---|---|
| [Intriguing properties of neural networks](https://arxiv.org/abs/1312.6199) — Szegedy et al. | 2013 | El descubrimiento: perturbaciones casi invisibles rompen redes con precisión ~100%. |
| [Explaining and Harnessing Adversarial Examples](https://arxiv.org/abs/1412.6572) — Goodfellow, Shlens & Szegedy | 2014 | La hipótesis lineal, FGSM y la demostración de que "demasiada linealidad" explica la fragilidad. |
| [Adversarial Examples in the Physical World](https://arxiv.org/abs/1607.02533) — Kurakin et al. | 2016 | BIM/I-FGSM: FGSM iterado; primeros resultados con imágenes físicas. |
| [DeepFool](https://arxiv.org/abs/1511.04599) — Moosavi-Dezfooli et al. | 2016 | Perturbación mínima por linearización iterativa; medida de márgenes. |
| [Towards Evaluating the Robustness of Neural Networks](https://arxiv.org/abs/1608.04644) — Carlini & Wagner | 2017 | C&W: el ataque de optimización que rompió defensas por destilación y sigue siendo el juez de referencia. |
| [Towards Deep Learning Models Resistant to Adversarial Attacks](https://arxiv.org/abs/1706.06083) — Madry et al. | 2017 | PGD como adversario universal de primer orden y el entrenamiento adversarial minimax. |

## Defensas y su evaluación

| Paper | Año | Qué aportó |
|---|---|---|
| [Obfuscated Gradients Give a False Sense of Security](https://arxiv.org/abs/1802.00420) — Athalye, Carlini & Wagner | 2018 | Rompe 7/9 defensas de ICLR'18 (EOT, backward pass diferenciable, SPSA). Lectura obligatoria antes de proponer una defensa. |
| [Certified Adversarial Robustness via Randomized Smoothing](https://arxiv.org/abs/1902.02918) — Cohen et al. | 2019 | Primera robustez certificable escalable (garantía L2 con voto mayoritario gaussiano). |
| [Theoretically Principled Trade-off (TRADES)](https://arxiv.org/abs/1901.08573) — Zhang et al. | 2019 | Marco minimax exactitud/robustez; ganador de la competición de robustez NeurIPS 2018. |
| [Adversarial Examples Are Not Easily Detected](https://arxiv.org/abs/1705.07263) — Carlini & Wagner | 2017 | Bypasa 10 métodos de detección: detectar adversariales es (casi) tan difícil como ser robusto. |
| [Ensemble Adversarial Training](https://arxiv.org/abs/1705.07204) — Tramèr et al. | 2017 | AT con adversariales transferidos de modelos sustitutos. |
| [Fast is better than free](https://arxiv.org/abs/2001.03994) — Wong et al. | 2020 | FGSM-restart: entrenamiento adversarial 10× más barato. |

## Contexto y panorama

| Recurso | Qué es |
|---|---|
| [Adversarial ML reading list de Nicholas Carlini](https://nicholas.carlini.com/writing/2019/chadversarial.html) | La lista del autor de C&W, con comentarios. |
| [CleverHans](https://github.com/cleverhans-lab/cleverhans) | La librería histórica de benchmarking adversarial (Goodfellow/Papernot). |
| [Foolbox](https://github.com/bethgelab/foolbox) | Librería de ataques bien diseñada (referencia de ingeniería). |
| [torchattacks](https://github.com/Harry24k/adversarial-attacks-pytorch) | Ataques PyTorch listos para usar — el paquete de este repo es la versión "hecha a mano para aprender". |
| [Adversarial Robustness Toolbox (IBM)](https://github.com/Trusted-AI/adversarial-robustness-toolbox) | El toolbox industrial: ataques, defensas, métricas, múltiples frameworks. |
| [RobustBench](https://github.com/RobustBench/RobustBench) | Benchmark estándar de modelos robustos (los números a batir). |

## Para ir más allá

- **Robustez certificada avanzada**: Columnar/DiffSmooth, Yin et al. 2025;
  semi-definite relaxations; redes 1-Lipschitz con certificados exactos.
- **Ataques físicos**: [Fooling automated surveillance cameras](https://arxiv.org/abs/1905.03808)
  (parches en camisetas), Eykholt et al. 2017 (señales de tráfico).
- **LLMs y seguridad**: prompt injection, jailbreaks, data extraction —
  la continuación natural de este campo al espacio textual.
- **Teoría**: separaciones informativas/estadísticas del learning con
  adversariales; transporte óptimo y robustez.

## Cómo citar este repo

```bibtex
@software{raul_2026_adversarial_ml_playground,
  author  = {Raúl},
  title   = {adversarial-ml-playground: ataques y defensas adversariales implementados desde cero en PyTorch},
  year    = {2026},
  url     = {https://github.com/raulpuert/adversarial-ml-playground},
}
```
