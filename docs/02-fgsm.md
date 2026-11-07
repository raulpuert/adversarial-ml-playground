# 02 · La matemática de FGSM

El *Fast Gradient Sign Method* (Goodfellow, Shlens & Szegedy, 2014) es el
ataque más simple que funciona — y precisamente por eso es la mejor puerta
de entrada al campo. Todo lo demás (BIM, PGD, EOT...) son variaciones del
mismo tema.

## 1. El problema del atacante

Dado un clasificador $f$, una imagen $x$ con etiqueta verdadera $y$, el
atacante busca una perturbación $\delta$ que:

$$
\begin{aligned}
& \text{maximice} & & L(f(x+\delta), y) \\
& \text{sujeto a} & & \|\delta\|_\infty \le \epsilon \\
& & & x + \delta \in [0, 1]^d \quad \text{(píxeles válidos)}
\end{aligned}
$$

Esto es un problema de optimización no convexo (porque $f$ es una red
profunda). Resolverlo de forma exacta es inviable; la jugada de FGSM es
**linealizar y resolver exactamente el problema lineal**.

## 2. Linealización de primer orden

Aproximamos la pérdida alrededor de $x$ con Taylor de primer orden:

$$
L(f(x+\delta), y) \;\approx\; L(f(x), y) \;+\; \nabla_x L(f(x), y)^\top \delta
$$

El primer término es constante respecto a $\delta$, así que maximizar la
pérdida ≈ maximizar el producto lineal $g^\top \delta$ con $g = \nabla_x L$.

## 3. La solución de la bola L∞ es `sign`

Queremos $\max_{\|\delta\|_\infty \le \epsilon} g^\top \delta$. El producto
se expande por componentes:

$$
g^\top \delta = \sum_{i=1}^{d} g_i\, \delta_i, \qquad -\epsilon \le \delta_i \le \epsilon
$$

Cada término se maximiza **independientemente** (la restricción es por
componente): elige $\delta_i = \epsilon \cdot \mathrm{sign}(g_i)$. Es la
solución cerrada:

$$
\delta^\star = \epsilon \cdot \mathrm{sign}(\nabla_x L)
\qquad\Longrightarrow\qquad
\boxed{\; x_{adv} = \mathrm{clip}_{[0,1]}\!\big(x + \epsilon \cdot \mathrm{sign}(\nabla_x L(f(x), y))\big) \;}
$$

**Intuición del `sign`:** no importa *cuánto* quiere empujar el gradiente
cada píxel — bajo L∞, lo óptimo es que **cada píxel empuje lo máximo
permitido en la dirección correcta**. El gradiente aporta la dirección por
píxel; $\epsilon$, la magnitud uniforme.

**Coste:** una pasada forward + una backward. En GPU, milisegundos.

## 4. El papel de ε

ε es el **presupuesto del atacante**: la negociación entre
imperceptibilidad y poder de ataque.

| ε (sobre píxeles en [0,1]) | Perceptibilidad | Comentario |
|---|---|---|
| 0.01 | invisible | ataques quirúrgicos; a menudo basta |
| 0.05 – 0.15 | apenas perceptible | rango más usado en investigación |
| 0.3 (MNIST) | ruido visible | el estándar de robustez de Madry |

La regla de oro: **evalúa siempre un barrido de ε**, nunca un punto — la
curva *precisión robusta vs ε* es la firma de robustez de un modelo (ver
`scripts/evaluate.py`).

## 5. Versión dirigida

FGSM no dirigido solo exige "que falle". La versión **dirigida** hacia una
clase objetivo $t$ desciende la pérdida respecto a $t$:

$$
x_{adv} = \mathrm{clip}_{[0,1]}\big(x - \epsilon \cdot \mathrm{sign}(\nabla_x L(f(x), t))\big)
$$

Nótese el **signo menos**: subir la pérdida hacia $y$ ≡ bajarla hacia $t$
cuando $t \ne y$. Los ataques dirigidos son los que permiten ataques con
"mensaje" (por ejemplo, hacer que un clasificador de matrícula lea siempre
la misma).

## 6. Por qué FGSM se queda corto

Un paso de tamaño completo $\epsilon$ tiene dos modos de fallo:

1. **Sobrepasa**: la linealización solo vale cerca de $x$; al alejarse
   $\epsilon$ de golpe, la pérdida real puede bajar aunque la lineal suba.
2. **Se queda corto**: la frontera está a distancia $< \epsilon$ pero en
   una dirección que un solo paso no explota bien.

Los ataques iterativos (BIM, PGD) corrigen ambos con **pasos pequeños +
proyección**: ver [03 · PGD y ataques iterativos](03-pgd.md).

## 7. Implementación de referencia

La implementación completa del paquete (`attacks/fgsm.py`) equivale a esto:

```python
def fgsm(model, images, labels, eps):
    x = images.clone().detach().requires_grad_(True)
    loss = torch.nn.functional.cross_entropy(model(x), labels)
    (grad,) = torch.autograd.grad(loss, x)          # gradiente respecto a la ENTRADA
    x_adv = x.detach() + eps * grad.sign()          # paso en la dirección del signo
    return x_adv.clamp(0.0, 1.0)                    # dominio de píxeles válido
```

Los tres detalles que se suelen hacer mal y aquí están cuidados:

1. **`clone().detach()` antes de `requires_grad_`**: si no, acumulas
   gradientes en tensores compartidos con el dataset.
2. **`autograd.grad(loss, x)`**, no `loss.backward()`: el primero calcula y
   devuelve el gradiente **sin acumularlo en `.grad`** — lo que quieres
   cuando el mismo modelo también se entrena (adversarial training).
3. **La normalización dentro del modelo** (`models/layers.py`): así el
   ataque opera en píxeles [0,1] reales y ε tiene unidades interpretables.

## 8. Ejercicios

1. Demuestra que $\mathrm{sign}(\nabla_x L)$ es la solución del problema
   de maximización sobre la bola L∞ (pista: dualidad / desigualdad de
   Hölder).
2. ¿Qué ocurre si sustituyes $\mathrm{sign}$ por el gradiente normalizado
   L2 (`g / ||g||₂`)? ¿Qué norma estás atacando entonces?
3. Comprueba empíricamente la linealidad: mide $L(f(x + \epsilon\,\mathrm{sign}(g)), y)$
   real frente a la predicción lineal para ε ∈ [0, 0.5]. ¿Dónde se rompe
   la aproximación?
