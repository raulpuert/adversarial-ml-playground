# %% [markdown]
# # 02 · FGSM desde cero: la matemática del ataque mínimo
#
# En este notebook implementamos FGSM **sin usar el paquete** — solo PyTorch —
# y luego comprobamos que coincide con la implementación de
# `adversarial_playground`. La derivación es corta y merece la pena seguirla
# con calma: es la base de *todos* los ataques de gradiente.

# %%
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

import matplotlib.pyplot as plt
import torch

from adversarial_playground.attacks import FGSM
from adversarial_playground.utils.data import CLASS_NAMES, get_loaders, load_checkpoint
from adversarial_playground.utils.seeding import set_seed

set_seed(42)
model, meta = load_checkpoint("checkpoints/mnist_cnn.pt")
model.eval()
_, test_loader = get_loaders("mnist", batch_size=64)
images, labels = next(iter(test_loader))

# %% [markdown]
# ## 1 · El problema del atacante
#
# Un atacante quiere maximizar la pérdida del modelo **cambiando lo mínimo
# posible la imagen**. Formalmente:
#
# $$\max_{\delta} \; L(f(x + \delta), y) \quad \text{sujeto a} \quad \|\delta\|_\infty \le \epsilon,\; x+\delta \in [0,1]^d$$
#
# - $\|\delta\|_\infty \le \epsilon$ es la restricción de imperceptibilidad:
#   *ningún* píxel puede cambiar más de $\epsilon$.
# - Resolver este problema **exactamente** es intratable (es un problema no
#   convexo en $\delta$ para redes profundas). La solución práctica: una
#   **linealización de primer orden**.

# %% [markdown]
# ## 2 · La linealización
#
# Cerca de $x$, la pérdida se comporta como su desarrollo de Taylor:
#
# $$L(f(x + \delta), y) \approx L(f(x), y) + \nabla_x L^\top \delta$$
#
# Maximizar el término lineal $\nabla_x L^\top \delta$ sobre la bola
# $\|\delta\|_\infty \le \epsilon$ tiene solución cerrada y conocida:
#
# $$\delta^\star = \epsilon \cdot \mathrm{sign}(\nabla_x L)$$
#
# ¿Por qué el `sign`? El producto $\nabla L^\top \delta = \sum_i g_i \delta_i$
# se maximiza poniendo cada $\delta_i$ en su valor extremo permitido
# ($\pm\epsilon$) con el signo de $g_i$. Cada píxel "aporta lo máximo que
# puede" — la bola $L_\infty$ hace que la dirección óptima sea por componentes,
# no por norma global.
#
# Resultado:
#
# $$\boxed{\;x_{adv} = \mathrm{clip}_{[0,1]}\big(x + \epsilon\cdot \mathrm{sign}(\nabla_x L(f(x), y))\big)\;}$$
#
# **Coste: una pasada forward + una backward.** Eso es todo el ataque.

# %% [markdown]
# ## 3 · Implementación desde cero (12 líneas)

# %%
def fgsm_desde_cero(model, images, labels, eps: float) -> torch.Tensor:
    """FGSM tal cual la ecuación de arriba.

    La clave de la mecánica en PyTorch:

    1. ``images.requires_grad_(True)`` — queremos gradiente *respecto a la
       entrada*, no respecto a los parámetros.
    2. ``autograd.grad(loss, images)`` — el gradiente de la pérdida frente
       a la imagen. ``autograd.grad`` no acumula en ``.grad``, ideal aquí.
    3. ``sign`` + paso ``eps`` + clip al dominio válido de píxeles.
    """
    x = images.clone().detach().requires_grad_(True)
    loss = torch.nn.functional.cross_entropy(model(x), labels)
    (grad,) = torch.autograd.grad(loss, x)
    x_adv = x.detach() + eps * grad.sign()
    return x_adv.clamp(0.0, 1.0)


# %%
# Probamos sobre el batch del test
eps = 0.15
x_adv = fgsm_desde_cero(model, images, labels, eps)

with torch.no_grad():
    adv_preds = model(x_adv).argmax(dim=1)

success = adv_preds != labels
print(f"Tasa de éxito (FGSM casero, eps={eps}): {float(success.float().mean()):.2%}")

# %% [markdown]
# ## 4 · Verificación: ¿coincide con la implementación del paquete?

# %%
result = FGSM(model, eps=eps)(images, labels)
mismo_resultado = torch.allclose(x_adv, result.adv_images, atol=1e-6)
print(f"¿Idéntico al FGSM del paquete? {mismo_resultado}")

# %% [markdown]
# ## 5 · ¿Cuánto presupuesto necesita un atacante?
#
# La elección de $\epsilon$ es la negociación central del adversarial ML:
#
# | eps | visible a ojo humano | uso típico |
# |-----|----------------------|------------|
# | 0.01 | invisible | ataques "quirúrgicos" |
# | 0.05–0.15 | apenas perceptible | el rango más estudiado |
# | 0.3 (MNIST) | claramente ruidoso | estándar de robustez de Madry |
#
# Veamos cómo colapsa la confianza del modelo en la clase verdadera según
# crece $\epsilon$ para una imagen concreta.

# %%
# Una imagen bien clasificada con alta confianza
with torch.no_grad():
    probs = model(images).softmax(dim=1)
    conf, preds = probs.max(dim=1)
idx = int(torch.where((preds == labels) & (conf > 0.99))[0][0])
image, label = images[idx : idx + 1], int(labels[idx])

eps_list = [0.02 * i for i in range(16)]
true_conf, adv_conf = [], []
for e in eps_list:
    xa = fgsm_desde_cero(model, image, torch.tensor([label]), e)
    with torch.no_grad():
        p = model(xa).softmax(dim=1)[0]
    true_conf.append(float(p[label]))
    adv_conf.append(float(p.max()))

plt.figure(figsize=(6.4, 3.6))
plt.plot(eps_list, true_conf, marker="o", label=f"confianza en la clase real ({label})")
plt.plot(eps_list, adv_conf, marker="s", color="#c62828", label="confianza en la predicción")
plt.axhline(0.5, color="gray", lw=0.7, ls="--")
plt.xlabel("eps (presupuesto L∞ del atacante)")
plt.ylabel("probabilidad softmax")
plt.title("El colapso de la confianza según crece eps")
plt.legend()
plt.show()

# %% [markdown]
# ## 6 · ¿Dirigido o no dirigido?
#
# El FGSM de arriba es **no dirigido**: solo busca que la predicción cambie.
# La versión **dirigida** busca además una clase objetivo concreta $t$:
#
# $$x_{adv} = \mathrm{clip}\big(x - \epsilon\cdot\mathrm{sign}(\nabla_x L(f(x), t))\big)$$
#
# Fíjate en el **menos**: en vez de *subir* la pérdida hacia la etiqueta
# verdadera, *baja* la pérdida hacia el objetivo. Veamos que funciona.

# %%
targets = (labels + 1) % 10  # objetivo: "la siguiente clase"

x = images.clone().detach().requires_grad_(True)
loss_t = torch.nn.functional.cross_entropy(model(x), targets)
(grad,) = torch.autograd.grad(loss_t, x)
x_adv_t = (x.detach() - eps * grad.sign()).clamp(0, 1)  # ojo: MENOS, no más

with torch.no_grad():
    preds_t = model(x_adv_t).argmax(dim=1)
print(f"Éxito del ataque dirigido (pred == target): {float((preds_t == targets).float().mean()):.2%}")

# %% [markdown]
# ## 7 · La debilidad de FGSM: un solo paso es poco
#
# FGSM da **un** paso de tamaño completo $\epsilon$. O se pasa (la
# linealización ya no vale ahí) o se queda corto (la imagen no llega a la
# frontera). La solución — iterar con proyección — es el tema del notebook
# 03 (PGD). Déjalo comprobado con tus propios ojos: para el mismo $\epsilon$,
# un ataque iterativo engaña a más imágenes:

# %%
from adversarial_playground.attacks import BIM

bim = BIM(model, eps=eps, steps=10)
r_bim = bim(images, labels)
r_fgsm = FGSM(model, eps=eps)(images, labels)
print(f"FGSM (1 paso):      éxito {r_fgsm.success_rate:.2%}")
print(f"BIM (10 pasos):     éxito {r_bim.success_rate:.2%}")

# %% [markdown]
# ## Ejercicios propuestos
#
# 1. Cambia `sign(grad)` por `grad / ||grad||` (paso normalizado L2). ¿Por qué
#    rinde peor bajo la restricción $L_\infty$?
# 2. Implementa el ataque de **ruido aleatorio**: $x_{adv} = x + \epsilon\cdot u$
#    con $u \sim U(-1, 1)^d$. Compara la tasa de éxito con FGSM al mismo
#    $\epsilon$ (spoiler: el gradiente importa muchísimo).
# 3. Comprueba la **transferibilidad**: ataca `mnist_cnn.pt` y evalúa los
#    adversariales sobre `mnist_cnn_adv.pt`. ¿Qué tasa de éxito tiene? ¿Por
#    qué el modelo entrenado adversarialmente transfiere peor?
# 4. Repite la curva de confianza con `eps` negativo. ¿Qué ocurre y por qué?
