# 04 · Defensas: el mapa honesto de lo que funciona

El campo de defensas adversariales tiene una historia de promesas rotas:
de las 9 defensas aceptadas en ICLR 2018, **7 fueron rotas** en el paper
siguiente (Athalye et al., 2018). Este documento resume las tres familias
con historial real, por qué fallan las demás, y cómo evaluar una defensa
sin autoengañarse.

## 1. Familia 1: entrenamiento adversarial (PGD-AT)

**Idea** (Madry et al., 2017): en vez de entrenar solo con imágenes
limpias, genera adversariales *al vuelo* en cada batch y entrena con ellos:

$$
\min_\theta \; \mathbb{E}_{(x,y)}\Big[\max_{\|\delta\|\le\epsilon} L\big(f_\theta(x+\delta), y\big)\Big]
$$

El máximo interior se aproxima con PGD (5-10 pasos bastan para
entrenamiento). Es un problema minimax: el modelo aprende a ser correcto
en **toda la bola**, no solo en los puntos del dataset.

**Propiedades medidas** (ver tabla de resultados del README):

- robustez bajo PGD pasa de ~0% a ~90%+ (MNIST, ε=0.3);
- precio: precisión limpia baja 1-3 puntos;
- coste: entrenamiento ×5-10 más lento (cada batch = PGD interno + paso
  de optimización);
- escala bien a ImageNet (con variantes: Fast-AT, Free-AT, TRADES).

**Variantes que conviene conocer**: TRADES (Zhang et al., 2019) equilibra
explícitamente exactitud y robustez con un término de suavidad local;
Free-AT (Shafahi et al., 2019) recicla backward passes para abaratar;
MADRY-eps-schedule entrena con ε creciente.

**Es la única defensa empírica que ha sobrevivido** a una década de
ataques adaptados. Si despliegas un modelo que puede ser atacado, es el
mínimo exigible.

## 2. Familia 2: transformaciones de entrada

Idea: "limpiar" la entrada antes del modelo — cuantizar píxeles
(`BitDepthReduction`), recomprimir JPEG, ruido gaussiano, minimización de
varianza total (implementados en `defenses/transforms.py`).

**Por qué parecen funcionar**: son no diferenciables o estocásticas; el
gradiente que ve el ataque deja de apuntar a donde el modelo realmente es
sensible. Bajo FGSM/PGD "de fábrica", la precisión robusta sube.

**Por qué fallan**: eso mismo — *gradient masking*. El atacante adapta:

- **Backward pass diferenciable**: sustituir la no-diferenciabilidad por
  un gradiente aproximado (p. ej., tratar la cuantización como identidad).
- **EOT** (*Expectation over Transformation*): optimizar
  $\mathbb{E}_{t\sim T}[L(f(t(x+\delta)), y)]$ para que la perturbación
  funcione *tras* cualquier transformación de la defensa.
- **Nes/attack-and-verify**: técnicas de gradiente con cero orden (SPSA)
  cuando el gradiente está roto del todo.

En `results/mnist_defenses.csv` del repo puedes ver el efecto medido:
mejoras marginales bajo PGD no adaptado, y la precisión limpia siempre
pagando.

**Veredicto**: útiles como capa barata de *dificultad añadida*, nunca
como defensa principal.

## 3. Familia 3: randomized smoothing — robustez certificable

**Idea** (Cohen, Rosenfeld & Kolter, 2019): no arregles el modelo base,
define un clasificador suavizado por voto mayoritario:

$$
g(x) = \mathrm{argmax}_{c \in \mathcal{Y}} \; P\big(f(x + \delta) = c\big),
\qquad \delta \sim \mathcal{N}(0, \sigma^2 I)
$$

**La garantía**: si con confianza $1-\alpha$ (cota de Clopper-Pearson
sobre las n muestras) la clase ganadora $A$ tiene $p_A > p_B$ para toda
otra clase $B$, entonces $g$ devuelve $A$ para **cualquier** ataque con
$\|\delta\|_2 \le r$, donde:

$$
r = \sigma \cdot \Phi^{-1}(\underline{p_A})
$$

con $\underline{p_A}$ la cota inferior binomial. Esto es un **certificado
matemático por imagen**: no una observación empírica sino una prueba.

**Costes**: ruido ⇒ precisión limpia baja; certificar es caro (n = 10⁴+
pasadas por imagen); la garantía es L2 (no L∞); funciona mejor en
modelos entrenados *con* ruido gaussiano (`GaussianNoise` en el training
augmentation).

En el repo: `defenses/smoothing.py` implementa predicción suavizada y el
radio certificado completo (`certify()`).

## 4. Otras familias (y por qué no las implementamos)

| Familia | Ejemplos | Estado |
|---|---|---|
| Detección de adversariales | statistical tests, feature squeezes | rota por ataques adaptados (que generan adversariales *indetectables*) |
| Autoencoders denoisers / PixelDefend | reconstruir antes de clasificar | costoso; roto con EOT |
| Gradiente enmascarado por diseño | distillation (Papernot), vortex | roto por C&W y backward-pass tricks |
| Verificación formal exacta | MIP, k-ReLU | solo redes diminutas; escalado abierto |
| Regularización de curvatura / Lipschitz | redes 1-Lipschitz, certificados parse-tree | activa pero con precio de precisión alto |

Ninguna de estas está "muerta" del todo, pero ninguna tiene el historial
de PGD-AT + smoothing. Prefirimos enseñar lo estable.

## 5. Cómo NO engañarte evaluando una defensa

Checklist de Athalye (adaptado):

1. **Ataque adaptado**: evalúa con EOT sobre la aleatoriedad de la defensa
   y backward-pass diferenciable sobre sus no-diferenciabilidades.
2. **Ataques sin gradiente**: C&W clásico, SPSA, o búsqueda de píxeles —
   si tu defensa solo resiste gradientes, no resiste nada.
3. **Barrido de ε y de presupuesto de ataque**: una defensa que solo
   aguanta PGD-5 pasos no es evidencia.
4. **Precisión limpia declarada**: una defensa que destruye la utilidad
   no es una defensa.
5. **Caso límite ε→0**: la defensa debe degradar a "sin defensa".

## 6. Lecturas

- Madry et al., *Towards Deep Learning Models Resistant to Adversarial
  Attacks* — [arXiv:1706.06083](https://arxiv.org/abs/1706.06083)
- Athalye, Carlini & Wagner, *Obfuscated Gradients Give a False Sense of
  Security* — [arXiv:1802.00420](https://arxiv.org/abs/1802.00420)
- Cohen, Rosenfeld & Kolter, *Certified Adversarial Robustness via
  Randomized Smoothing* — [arXiv:1902.02918](https://arxiv.org/abs/1902.02918)
- Zhang et al., *Theoretically Principled Trade-off (TRADES)* —
  [arXiv:1901.08573](https://arxiv.org/abs/1901.08573)
- Carlini & Wagner, *Adversarial Examples Are Not Easily Detected*
  (bypassing 10 detection methods) — [arXiv:1705.07263](https://arxiv.org/abs/1705.07263)
