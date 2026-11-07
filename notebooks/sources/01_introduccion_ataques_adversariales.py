# %% [markdown]
# # 01 · ¿Qué es un ataque adversarial?
#
# En 2013, Szegedy et al. descubrieron algo inquietante: se puede tomar una
# imagen que una red neuronal clasifica perfectamente, cambiar **un puñado de
# píxeles imperceptibles**, y la red pasa a predecir —con confianza del 99%—
# una clase completamente distinta. A estas imágenes manipuladas se les llama
# **ejemplos adversariales** (*adversarial examples*).
#
# La ilustración más famosa es la de Goodfellow et al. (2014): un panda que
# la red etiqueta como "panda" con 57.7% de confianza; tras añadirle una
# perturbación amplificada ×10 que a simple vista es ruido, la misma imagen
# se clasifica como "gibón" con 99.3% de confianza.
#
# **¿Por qué funciona?** La explicación de Goodfellow: las redes modernas son
# **demasiado lineales** localmente. Cada capa es una transformación casi
# lineal (afín + ReLU), así que la salida responde de forma aproximadamente
# lineal a la entrada. Si `w` es el gradiente de la pérdida respecto a la
# imagen, empujar cada píxel en la dirección de `sign(w)` acumula el efecto
# a lo largo de cientos de dimensiones: un cambio invisible por píxel, un
# cambio enorme en el logit.
#
# En este notebook vamos a verlo con nuestros propios ojos sobre un clasificador
# de dígitos MNIST entrenado (99% de precisión limpia).

# %%
# Preparación: imports y carga del modelo
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))  # ejecuta el notebook desde la raíz del repo

import torch

from adversarial_playground.attacks import FGSM
from adversarial_playground.utils.data import CLASS_NAMES, get_loaders, load_checkpoint
from adversarial_playground.utils.seeding import set_seed

set_seed(42)
device = "cpu"  # este notebook está pensado para correr en CPU

model, meta = load_checkpoint("checkpoints/mnist_cnn.pt")
model.to(device).eval()
print(f"Modelo: {meta['arch']} · dataset: {meta['dataset']}")
print(f"Precisión limpia reportada: {meta['clean_acc']:.2%}")

# %%
# Cargamos las primeras imágenes del test y comprobamos que el modelo acierta.
_, test_loader = get_loaders("mnist", batch_size=64)
images, labels = next(iter(test_loader))
images, labels = images.to(device), labels.to(device)

with torch.no_grad():
    probs = model(images).softmax(dim=1)
    preds = probs.argmax(dim=1)

acc = float((preds == labels).float().mean())
print(f"Precisión en este batch: {acc:.2%} ({int((preds == labels).sum())}/{len(labels)})")

# %%
# Visualicemos algunas imágenes con su predicción y confianza.
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 6, figsize=(11, 3.2))
for i, ax in enumerate(axes.flat):
    ax.imshow(images[i, 0], cmap="gray")
    ok = int(preds[i]) == int(labels[i])
    ax.set_title(
        f"real {CLASS_NAMES['mnist'][int(labels[i])]} · pred {CLASS_NAMES['mnist'][int(preds[i])]} ({float(probs[i, preds[i]]):.0%})",
        color="#1b5e20" if ok else "#c62828",
        fontsize=9,
    )
    ax.axis("off")
fig.suptitle("El modelo limpia: 99% de acierto", y=1.04)
plt.show()

# %% [markdown]
# ## El primer ataque: FGSM
#
# La idea del ataque FGSM (*Fast Gradient Sign Method*) cabe en una línea:
#
# $$x_{adv} = \mathrm{clip}\big(x + \epsilon \cdot \mathrm{sign}(\nabla_x L(f(x), y))\big)$$
#
# Es decir: *sube la pérdida respecto a la imagen*, con el paso más grande
# permitido ($\epsilon$) en cada píxel. Es un **ascenso por gradiente de un
# solo paso** en la bola $L_\infty$ de radio $\epsilon$.
#
# La restricción $L_\infty$ ("cada píxel puede cambiar como mucho $\epsilon$")
# es el modelo de amenaza más usado: captura "cualquier cosa imperceptible"
# con un solo número.

# %%
# Atacamos el mismo batch con FGSM y eps = 0.15 (15% de intensidad por píxel).
attack = FGSM(model, eps=0.15)
result = attack(images, labels)

print(f"Tasa de éxito del ataque: {result.success_rate:.2%}")
print(f"Perturbación media:  L∞={result.mean_linf:.3f}  L2={result.mean_l2:.3f}")

# %%
# Visualicemos: original | perturbación (amplificada) | adversarial
delta = (result.adv_images[0] - images[0]) * 10  # amplificado ×10 para verlo
delta_viz = (delta - delta.min()) / (delta.max() - delta.min())

adv_probs = model(result.adv_images).softmax(dim=1)

fig, axes = plt.subplots(1, 3, figsize=(9, 3))
axes[0].imshow(images[0, 0], cmap="gray")
axes[0].set_title(f"original: {CLASS_NAMES['mnist'][int(labels[0])]} ({float(probs[0, labels[0]]):.0%})")
axes[1].imshow(delta_viz[0], cmap="viridis")
axes[1].set_title("perturbación ×10")
axes[2].imshow(result.adv_images[0, 0], cmap="gray")
pred_a = int(adv_probs[0].argmax())
axes[2].set_title(
    f"adversarial: {CLASS_NAMES['mnist'][pred_a]} ({float(adv_probs[0, pred_a]):.0%})",
    color="#c62828",
)
for ax in axes:
    ax.axis("off")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## ¿Qué está "viendo" el ataque? El gradiente de la entrada
#
# El signo del gradiente $\nabla_x L$ nos dice **qué píxel, al subir un
# pelín, más aumenta la pérdida**. Visualizarlo es muy revelador: la
# estructura que usa el ataque no es un dígito — es un patrón fino que
# explota los "razonamientos" internos de la red.

# %%
# Calculamos el gradiente de la pérdida respecto a la entrada de cada imagen.
x = images[:8].clone().requires_grad_(True)
loss = torch.nn.functional.cross_entropy(model(x), labels[:8])
grad, = torch.autograd.grad(loss, x)

fig, axes = plt.subplots(2, 8, figsize=(13, 3))
for i, ax in enumerate(axes[0]):
    ax.imshow(images[i, 0], cmap="gray")
    ax.axis("off")
for i, ax in enumerate(axes[1]):
    ax.imshow(grad[i, 0], cmap="RdBu_r", vmin=-0.02, vmax=0.02)
    ax.axis("off")
axes[0][0].set_ylabel("imagen", fontsize=9)
axes[1][0].set_ylabel("∇ₓL", fontsize=9)
fig.suptitle("Arriba: imágenes · Abajo: gradiente de la pérdida respecto a la entrada", y=1.05)
plt.show()

# %% [markdown]
# ## La trampa: la perturbación no es ruido, es una señal
#
# Si el ataque funcionara porque "introduce ruido", al promediar perturbaciones
# aleatorias el efecto se anularía. No es el caso: la perturbación adversarial
# es **consistente y transferible** entre arquitecturas distintas — como si
# explotara características reales (frágiles) que la red usa para decidir.
#
# En los siguientes notebooks:
#
# - **02** — derivamos e implementamos FGSM desde cero y estudiamos el
#   papel de $\epsilon$.
# - **03** — FGSM es un paso; PGD son muchos con proyección. El zoo completo.
# - **04** — ¿qué se puede hacer? Entrenamiento adversarial, smoothing y
#   por qué casi todas las demás defensas son espejismos.
#
# ### Referencias
#
# - Szegedy et al., *Intriguing properties of neural networks* (2013) — [arXiv:1312.6199](https://arxiv.org/abs/1312.6199)
# - Goodfellow et al., *Explaining and Harnessing Adversarial Examples* (2014) — [arXiv:1412.6572](https://arxiv.org/abs/1412.6572)
