# %% [markdown]
# # 04 · Defensas: ¿qué funciona de verdad?
#
# Contra los ataques adversariales se han propuesto cientos de defensas.
# Casi todas han caído. Este notebook recorre las **tres familias** con
# historial real y te deja medirlas con datos:
#
# 1. **Entrenamiento adversarial** (Madry 2017) — el estándar de oro empírico.
# 2. **Transformaciones de entrada** — baratas, útiles, pero débiles solas.
# 3. **Randomized smoothing** (Cohen 2019) — la única con garantía *certificable*.
#
# Y sobre todo: entendemos **por qué** la mayoría de defensas fallan (la
# lección de Athalye et al. 2018).

# %%
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

import torch

from adversarial_playground.attacks import PGD
from adversarial_playground.evaluation.metrics import evaluate_attack
from adversarial_playground.utils.data import get_loaders, load_checkpoint
from adversarial_playground.utils.seeding import set_seed

set_seed(42)
clean_model, meta_c = load_checkpoint("checkpoints/mnist_cnn.pt")
clean_model.eval()

_, test_loader = get_loaders("mnist", batch_size=128)

# %% [markdown]
# ## 1 · El punto de partida: un modelo limpio se deshace
#
# Medimos la precisión del modelo limpio bajo PGD con presupuesto creciente:

# %%
for eps in [0.05, 0.1, 0.2, 0.3]:
    attack = PGD(clean_model, eps=eps, alpha=eps / 4, steps=40)
    ev = evaluate_attack(clean_model, attack, test_loader, max_samples=1024, show_progress=False)
    print(f"eps={eps:<5} → precisión robusta {ev.robust_acc:6.2%} (éxito del ataque {ev.attack_success:.2%})")

# %% [markdown]
# ## 2 · Familia 1: entrenamiento adversarial (PGD-AT)
#
# La idea de Madry es casi obviana pero fue revolucionaria: **entrena con
# adversariales generados al vuelo** en cada batch (PGD de 7 pasos, mismo
# `eps` que luego usarás). El modelo aprende la función correcta en *toda*
# la bola, no solo en los puntos del dataset.
#
# El checkpoint `mnist_cnn_adv.pt` del repo se entrenó así (Madry puro,
# eps=0.3, 7 pasos PGD, mismas épocas). Comparémoslos:

# %%
try:
    at_model, meta_a = load_checkpoint("checkpoints/mnist_cnn_adv.pt")
    at_model.eval()
except FileNotFoundError:
    at_model = None
    print("No tienes checkpoints/mnist_cnn_adv.pt — entrénalo con:")
    print("  python scripts/train.py --dataset mnist --adversarial")

if at_model is not None:
    print(f"Modelo limpio: {meta_c['clean_acc']:.2%} limpia")
    print(f"Modelo AT:     {meta_a['clean_acc']:.2%} limpia (el precio de la robustez)")

    eps = 0.3
    ev_clean = evaluate_attack(
        clean_model, PGD(clean_model, eps=eps, alpha=eps/4, steps=40),
        test_loader, max_samples=1024, show_progress=False,
    )
    ev_at = evaluate_attack(
        at_model, PGD(at_model, eps=eps, alpha=eps/4, steps=40),
        test_loader, max_samples=1024, show_progress=False,
    )
    print(f"\nBajo PGD eps={eps} (40 pasos):")
    print(f"  modelo limpio → {ev_clean.robust_acc:6.2%}")
    print(f"  modelo AT     → {ev_at.robust_acc:6.2%}")

# %% [markdown]
# ## 3 · Familia 2: transformaciones de entrada
#
# Cuantizar los píxeles (menos bits) o recomprimir como JPEG destruye las
# perturbaciones de alta frecuencia que viven en los decimales. Son defensas
# *baratas* y *compatibles con cualquier modelo*… veamos su precio:

# %%
from adversarial_playground.attacks import FGSM
from adversarial_playground.defenses.transforms import (
    BitDepthReduction,
    JPEGCompression,
)
from adversarial_playground.evaluation.metrics import evaluate_transform_defense

eps = 0.2
attack = PGD(clean_model, eps=eps, alpha=eps / 4, steps=40)

for nombre, defensa in [
    ("sin defensa", None),
    ("bit-depth 4 bits", BitDepthReduction(bits=4)),
    ("JPEG q=75", JPEGCompression(quality=75)),
]:
    if defensa is None:
        ev = evaluate_attack(clean_model, attack, test_loader, max_samples=512, show_progress=False)
    else:
        ev = evaluate_transform_defense(
            clean_model, defensa, attack, test_loader, max_samples=512
        )
    print(f"{nombre:<20} limpia {ev.clean_acc:6.2%} · robusta {ev.robust_acc:6.2%}")

# %% [markdown]
# Observa el patrón: la precisión **limpia** baja poco, pero la **robusta**
# apenas mejora (o baja). ¿Por qué?
#
# ## 4 · La trampa de las defensas falsas: *gradient masking*
#
# Una transformación no diferenciable (JPEG, cuantización) hace que el
# gradiente que ve el atacante **no represente la dirección real de ataque**
# — parece que el modelo es robusto, pero solo hemos cegado al ataque de
# gradiente que usamos. Athalye et al. (2018) formalizaron tres mecanismos
# de ofuscación (gradientes *shattered*, *stochastic* y *vanishing*) y
# **rompieron 7 de las 9 defensas aceptadas en ICLR 2018**, incluyendo
# defensas con JPEG y cuantización, usando:
#
# - **EOT** (*Expectation over Transformation*): atacar esperando sobre la
#   aleatoriedad/transformación de la defensa.
# - **Backward pass diferenciable**: sustituir el gradiente roto por uno
#   aproximado (p. ej., el de la identidad para cuantizaciones).
#
# **Regla práctica**: si tu defensa solo resiste FGSM/PGD "de fábrica",
# no has medido robustez — has medido la debilidad de tu evaluador. Toda
# defensa debe evaluarse contra ataques *adaptados*.
#
# El entrenamiento adversarial sobrevive a este escrutinio: no esconde el
# gradiente, lo **usa** en el entrenamiento.

# %% [markdown]
# ## 5 · Familia 3: randomized smoothing (robustez certificable)
#
# En vez de arreglar el modelo, cambiamos el clasificador: predecimos con el
# **voto mayoritario** sobre `n` copias ruidosas de la imagen:
#
# $$\hat y(x) = \mathrm{argmax}_c \; P\big(f(x + \delta) = c\big), \quad \delta \sim \mathcal N(0, \sigma^2 I)$$
#
# Cohen et al. (2019) demostraron que si la clase mayoritaria se mantiene con
# probabilidad suficientemente alta (acotada con una binomial de confianza),
# entonces **ningún ataque L2 con radio `r = σ·Φ⁻¹(p)` puede cambiar la
# predicción**. Una garantía matemática, no una observación empírica.

# %%
from adversarial_playground.defenses.smoothing import RandomizedSmoothing

smooth = RandomizedSmoothing(clean_model, sigma=0.25, n=200, batch_size=512)

# Precisión del clasificador suavizado sobre imágenes limpias
correct = total = 0
for x, y in test_loader:
    preds = smooth.predict(x)
    correct += int((preds == y).sum())
    total += len(y)
    if total >= 1024:
        break
print(f"Precisión limpia del clasificador suavizado (σ=0.25): {correct / total:.2%}")
print("(el modelo base tenía ~99% — el ruido cuesta puntos)")

# %%
# Certificación de radios L2 sobre un subconjunto
radii = smooth.certify(images[:32], n=1000, alpha=0.001)
print(f"Radios L2 certificados (σ=0.25, n=1000, α=0.001):")
print(f"  certificados: {int((radii > 0).sum())}/32")
print(f"  radio medio (certificados): {float(radii[radii > 0].mean()):.3f}")
print(f"  radio máximo: {float(radii.max()):.3f}")

# %% [markdown]
# ## 6 · Resumen del arsenal
#
# | Defensa | Coste entrenamiento | Coste inferencia | Precisión limpia | ¿Certifica? | ¿Aguanta ataques adaptados? |
# |---|---|---|---|---|---|
# | Entrenamiento adversarial | alto (×3-10) | nulo | baja 1-5 pts | no | sí (empíricamente) |
# | Transformaciones de entrada | nulo | bajo | baja algo | no | **no** (gradient masking) |
# | Randomized smoothing | no (reusa modelo) | alto (n pasadas) | baja 1-10 pts | **sí (L2)** | sí (con garantía) |
#
# La conclusión honesta del campo (2024-2026): **entrenamiento adversarial
# para robustez empírica + smoothing cuando necesites garantías**; todo lo
# demás, salvo evaluación adaptada, es decoración.

# %% [markdown]
# ## Ejercicios
#
# 1. Entrena tu propio modelo AT con `scripts/train.py --dataset mnist
#    --adversarial --adv-eps 0.1` y compara su robustez a eps=0.1 y a
#    eps=0.3 con el AT eps=0.3 del repo. (Pista: robustez *local* vs *global*.)
# 2. Implementa el ataque adaptado a `BitDepthReduction` (pista: usa el
#    "backward pass de la identidad": calcula el gradiente sobre la imagen
#    original pero aplica el paso a la versión cuantizada — EOT simplificado).
# 3. Sube `sigma` del smoothing a 0.5. ¿Qué pasa con la precisión limpia y
#    con los radios certificados?
