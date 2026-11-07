# %% [markdown]
# # 03 · PGD y los ataques iterativos: el estándar de facto
#
# FGSM (notebook 02) da **un paso** de tamaño completo $\epsilon$. PGD
# (*Projected Gradient Descent*, Madry et al. 2017) hace lo mismo pero
# **muchas veces**, proyectando tras cada paso para no salir de la bola:
#
# $$x^{(0)} = x + U(-\epsilon, \epsilon) \quad\text{(arranque aleatorio)}$$
# $$x^{(t+1)} = \Pi_{B_\epsilon(x)}\Big(\mathrm{clip}_{[0,1]}\big(x^{(t)} + \alpha\cdot \mathrm{sign}(\nabla_x L)\big)\Big)$$
#
# donde $\Pi_{B_\epsilon(x)}$ es la **proyección** sobre la bola $L_\infty$
# de radio $\epsilon$ centrada en la imagen original — que en $L_\infty$ es
# simplemente recortar la perturbación a $\pm\epsilon$.
#
# Madry lo llama el *"ataque universal de primer orden"*: si un modelo
# resiste PGD (con suficientes pasos y arranques), resiste prácticamente
# cualquier ataque de gradiente. Por eso la robustez se mide **contra PGD**.

# %%
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

import matplotlib.pyplot as plt
import torch

from adversarial_playground.attacks import BIM, FGSM, PGD
from adversarial_playground.attacks.pgd import project_l2, project_linf
from adversarial_playground.utils.data import get_loaders, load_checkpoint
from adversarial_playground.utils.seeding import set_seed

set_seed(42)
model, meta = load_checkpoint("checkpoints/mnist_cnn.pt")
model.eval()
_, test_loader = get_loaders("mnist", batch_size=128)
images, labels = next(iter(test_loader))

# %% [markdown]
# ## 1 · La proyección, concretada
#
# Proyectar sobre $B_\infty(x, \epsilon)$ = recortar cada píxel de la
# perturbación a $\pm\epsilon$. En L2 es reescalar si te pasas. Comprobemos
# que nuestras funciones lo hacen bien:

# %%
delta = torch.rand(4, 1, 28, 28) * 2 - 1  # perturbaciones arbitrarias en [-1, 1]

d_inf = project_linf(delta, eps=0.3)
print("L∞:  min =", float(d_inf.min()), " max =", float(d_inf.max()), "(debe estar en [-0.3, 0.3])")

d_l2 = project_l2(delta, eps=0.5)
print("L2:  normas =", [round(float(v), 3) for v in d_l2.flatten(1).norm(p=2, dim=1)],
      "(deben ser ≤ 0.5)")

# %% [markdown]
# ## 2 · PGD desde cero (el bucle completo, ~15 líneas)

# %%
def pgd_desde_cero(model, images, labels, eps=0.3, alpha=0.075, steps=40, random_start=True):
    """PGD L∞ con arranque aleatorio y proyección tras cada paso."""
    x_orig = images.clone().detach()
    if random_start:
        x = (x_orig + torch.empty_like(x_orig).uniform_(-eps, eps)).clamp(0, 1)
    else:
        x = x_orig.clone()

    for _ in range(steps):
        x.requires_grad_(True)
        loss = torch.nn.functional.cross_entropy(model(x), labels)
        (grad,) = torch.autograd.grad(loss, x)

        with torch.no_grad():
            x = x + alpha * grad.sign()
            delta = (x - x_orig).clamp(-eps, eps)      # proyección L∞
            x = (x_orig + delta).clamp(0, 1)           # dominio de píxeles
    return x.detach()


x_adv = pgd_desde_cero(model, images, labels)
with torch.no_grad():
    adv_preds = model(x_adv).argmax(dim=1)
print(f"PGD casero (eps=0.3, 40 pasos): éxito {float((adv_preds != labels).float().mean()):.2%}")

# %% [markdown]
# ## 3 · ¿Para qué sirve el arranque aleatorio?
#
# Sin arranque aleatorio, PGD parte siempre del mismo punto y sigue siempre
# el mismo camino — si el modelo "enmascara" gradientes cerca del origen,
# el ataque se queda atascado. Con arranque aleatorio, cada imagen se ataca
# desde puntos distintos de la bola. Veámoslo visualmente: adversariales de
# la **misma imagen** desde 6 arranques distintos — predicciones distintas,
# perturbaciones distintas, éxito el mismo.

# %%
image = images[0:1]
label = labels[0:1]

fig, axes = plt.subplots(1, 6, figsize=(12, 2.2))
for i, ax in enumerate(axes):
    r = PGD(model, eps=0.3, alpha=0.075, steps=40, random_start=True)(image, label)
    pred = int(r.adv_labels[0])
    ax.imshow(r.adv_images[0, 0], cmap="gray")
    ax.set_title(f"→ {pred}", color="#c62828" if pred != int(label) else "#1b5e20", fontsize=11)
    ax.axis("off")
fig.suptitle(f"Misma imagen (real: {int(label[0])}), 6 arranques aleatorios de PGD", y=1.06)
plt.show()

# %% [markdown]
# ## 4 · FGSM vs BIM vs PGD: la escalera de potencia
#
# - **FGSM**: 1 paso. Rápido, débil.
# - **BIM** (*basic iterative method*): FGSM iterado **sin** arranque aleatorio,
#   pasos finos ($\alpha = \epsilon/10$). Históricamente anterior a PGD.
# - **PGD**: BIM **con** arranque aleatorio. El estándar.
#
# Comparamos los tres sobre 1024 imágenes del test al mismo $\epsilon$:

# %%
big_images, big_labels = [], []
for i, (x, y) in enumerate(test_loader):
    big_images.append(x)
    big_labels.append(y)
    if sum(b.shape[0] for b in big_images) >= 1024:
        break
big_images = torch.cat(big_images)[:1024]
big_labels = torch.cat(big_labels)[:1024]

eps = 0.2
resultados = {
    "FGSM (1 paso)": FGSM(model, eps=eps)(big_images, big_labels),
    "BIM (10 pasos)": BIM(model, eps=eps, steps=10)(big_images, big_labels),
    "PGD (40 pasos)": PGD(model, eps=eps, alpha=eps / 4, steps=40)(big_images, big_labels),
}

print(f"{'ataque':<16} {'éxito':>8} {'‖δ‖₂ medio':>12}")
for name, r in resultados.items():
    print(f"{name:<16} {r.success_rate:>7.2%} {r.mean_l2:>12.3f}")

# %% [markdown]
# ## 5 · La bola importa: ataque L∞ vs ataque L2
#
# La restricción define el ataque. En la bola **L2** el presupuesto se
# reparte entre todos los píxeles (perturbaciones "globales", suaves); en
# **L∞** cada píxel tiene su propio presupuesto (perturbaciones "de grano
# fino"). El mismo $\epsilon$ numérico significa cosas muy distintas.

# %%
r_inf = PGD(model, eps=0.3, alpha=0.075, steps=40, norm="linf")(big_images[:256], big_labels[:256])
r_l2 = PGD(model, eps=3.0, alpha=0.75, steps=40, norm="l2")(big_images[:256], big_labels[:256])

fig, axes = plt.subplots(2, 4, figsize=(11, 5))
for j, (r, titulo) in enumerate([(r_inf, "PGD L∞ (eps=0.3)"), (r_l2, "PGD L2 (eps=3.0)")]):
    for i in range(4):
        ax = axes[j][i]
        d = (r.adv_images[i] - big_images[i])
        dv = (d - d.min()) / (d.max() - d.min() + 1e-12)
        ax.imshow(dv[0], cmap="viridis")
        ax.set_title(f"{titulo}\n‖δ‖₂={float(r.l2_norms[i]):.2f}", fontsize=9)
        ax.axis("off")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 6 · DeepFool: no gastar más de lo necesario
#
# FGSM/BIM/PGD agotan (casi) todo el presupuesto. **DeepFool** hace lo
# contrario: busca la perturbación **mínima** que cruce la frontera de
# decisión, iterando una aproximación lineal del clasificador. El resultado
# es 10-50× más pequeño en norma — y es la medida honesta de "cuánto margen
# le queda a cada imagen".

# %%
from adversarial_playground.attacks import DeepFool

sub_images, sub_labels = big_images[:64], big_labels[:64]
df = DeepFool(model)(sub_images, sub_labels)
pgd_small = PGD(model, eps=0.3, alpha=0.075, steps=40)(sub_images, sub_labels)

print(f"DeepFool:  éxito {df.success_rate:.2%} · ‖δ‖₂ medio {df.mean_l2:.3f} · "
      f"iteraciones medias {float(df.info['iterations'].mean()):.1f}")
print(f"PGD 0.3:   éxito {pgd_small.success_rate:.2%} · ‖δ‖₂ medio {pgd_small.mean_l2:.3f}")

# %% [markdown]
# ## 7 · El zoo completo
#
# | Ataque | Tipo | Norma | Coste | Fortaleza |
# |---|---|---|---|---|
# | FGSM | gradiente, 1 paso | L∞ | ★ | ★ |
# | BIM/I-FGSM | gradiente iterativo | L∞ | ★★ | ★★★ |
# | PGD | gradiente iterativo + arranque aleatorio | L∞/L2 | ★★ | ★★★★ |
# | DeepFool | linearización iterativa | L2 | ★★★ | mínima-norma |
# | C&W | optimización (Adam + búsqueda binaria) | L2 | ★★★★★ | ★★★★★ |
#
# C&W (Carlini & Wagner 2017) es el ataque de referencia en fuerza: optimiza
# directamente `min ||δ||² + c·f(x+δ)` con búsqueda binaria sobre `c`. En el
# repo está implementado (`adversarial_playground.attacks.CarliniWagnerL2`)
# — pruébalo sobre un subconjunto, es lento pero es el juez final.

# %% [markdown]
# ## Ejercicios
#
# 1. Cambia `alpha` de PGD a `eps` (paso = bola entera). ¿Qué pasa y por qué
#    se parece el resultado a FGSM?
# 2. Mide la robustez del modelo con `steps` ∈ {5, 10, 20, 40, 100} a
#    `eps=0.3` fijo. ¿Se estabiliza? Ese es el coste oculto de evaluar "contra PGD".
# 3. Implementa PGD-L2 desde cero (pista: normaliza el gradiente por muestra
#    y reescala el delta si supera `eps`).
# 4. ¿Cuántas imágenes del test tienen margen (DeepFool ‖δ‖₂) menor que 0.5?
#    ¿Qué relación ves entre esa distancia y la confianza limpia?
