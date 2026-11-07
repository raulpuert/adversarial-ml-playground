# 03 · PGD y los ataques iterativos

> *"PGD is the universal attack among first-order adversaries."*
> — Madry et al. (2017)

## 1. De un paso a muchos: BIM

FGSM da **un** paso de tamaño completo ε (ver [02-fgsm.md](02-fgsm.md)).
El *Basic Iterative Method* (Kurakin et al., 2016) reparte el mismo
presupuesto en pasos más pequeños:

$$
x^{(t+1)} = \mathrm{clip}_{[0,1]}\big(x^{(t)} + \alpha \cdot \mathrm{sign}(\nabla_x L)\big),
\qquad \alpha = \epsilon/10,\; t = 0..T-1
$$

Sin más control, la suma de pasos puede exceder ε respecto a la imagen
original. BIM clásico "clipa duro" al cubo $[x-\epsilon, x+\epsilon]$ en
cada paso — que es, dicho formalmente, **proyectar**.

## 2. PGD: BIM con proyección bien hecha y arranque aleatorio

*Projected Gradient Descent* (Madry et al., 2017) formaliza la idea:

$$
x^{(0)} = x + U(-\epsilon, \epsilon) \qquad \text{(arranque aleatorio uniforme)}
$$

$$
x^{(t+1)} = \Pi_{B_\epsilon(x)}\!\Big(\mathrm{clip}_{[0,1]}\big(x^{(t)} + \alpha \cdot \mathrm{sign}(\nabla_x L(f(x^{(t)}), y))\big)\Big)
$$

- $B_\epsilon(x)$: bola L∞ de radio ε centrada en la imagen original.
- $\Pi_{B_\epsilon(x)}$: **proyección** — el punto más cercano de la bola.
  En L∞ es `clamp(δ, -ε, ε)` por componente; en L2, reescalar si
  $\|\delta\|_2 > \epsilon$:

$$
\Pi^{L_2}(\delta) = \delta \cdot \min\!\Big(1, \frac{\epsilon}{\|\delta\|_2}\Big)
$$

- $\alpha$: tamaño de paso. Reglas prácticas: $\alpha = \epsilon/4$ (L∞,
  agresivo) o $\epsilon/10$ (fino). Debe cumplir $0 < \alpha \le \epsilon$.

Con `random_start=False` recuperas exactamente BIM — en el repo,
`BIM` es una subclase de `PGD` con esa opción fija.

## 3. ¿Por qué importa el arranque aleatorio?

Tres razones, de práctica a teoría:

1. **Cobertura**: PGD es ascenso local; distintos arranques encuentran
   distintos máximos de pérdida dentro de la bola. Repetir con arranques
   aleatorios (o elegir el peor de k arranques) estima mejor la robustez.
2. **Antifraude**: defensas que "aplanan" los gradientes cerca del punto
   original (el caso típico: cuantizaciones) engañan a BIM, que parte
   siempre del origen. PGD con arranque aleatorio ya les pega más fuerte.
3. **Teoría**: Madry demuestra que el ascenso por proyecciones con arranque
   aleatorio es una heurística razonable para el problema de maximización
   interior de la bola — el "adversario universal de primer orden".

**Consecuencia metodológica**: si tu modelo resiste PGD con T=40-100 pasos
y arranques aleatorios, es fuerte contra *cualquier* ataque de gradiente.
Si solo resiste FGSM o BIM, no ha resistido nada.

## 4. La variante L2

Misma estructura, dos cambios:

| Paso | L∞ | L2 |
|---|---|---|
| dirección | `sign(g)` | `g / ‖g‖₂` (normalizado por muestra) |
| proyección | `clamp(δ, ±ε)` | `δ · min(1, ε/‖δ‖₂)` |

El mismo ε significa presupuestos radicalmente distintos (ver notebook 03,
sección 5): L∞ reparte ε *por píxel*; L2 da un presupuesto *global*.

## 5. Cómo se usa PGD en evaluación (protocolo del repo)

1. **Barrido de ε** — precisión robusta para ε ∈ {0.05, ..., 0.45}
   (`robust_accuracy_sweep`).
2. **Pasos suficientes** — 40 pasos es el mínimo serio para ε grandes;
   verifica estabilidad con 100 (si la precisión robusta cambia al subir
   pasos, tu evaluación estaba incompleta).
3. **Varias semillas** en arranques aleatorios para ataques de referencia.
4. **Confirma con C&W** (optimización sin gradientes del mismo tipo) si el
   resultado va a defenderse en un paper o informe.

## 6. DeepFool y C&W: más allá del gradiente por signo

- **DeepFool** responde otra pregunta: no "cuál es el peor δ con
  presupuesto ε" sino "**cuál es el δ mínimo que cambia la decisión**".
  Itera la linearización del clasificador y avanza al hiperplano de la
  clase más cercana; la norma resultante estima la distancia de la imagen
  a la frontera. Útil para análisis (márgenes, distribución de distancias),
  no para maximizar daño a presupuesto fijo.
- **C&W** ataca el problema de optimización directamente con Adam sobre
  una reparametrización tanh, minimizando `‖δ‖₂² + c·f(x+δ)` con búsqueda
  binaria de `c`. Es el más fuerte del zoo clásico y el más caro
  (`binary_search_steps × max_iter` pasadas). Cuando C&W también te engaña,
  el margen era real.

### Notas de comportamiento medidas en este repo (CNN MNIST, 3 épocas)

Estos son datos reales de `results/mnist_attacks.csv` — y son más
interesantes que los números "ideales" de los papers:

- **DeepFool engaña a ~50% de las imágenes** con perturbaciones de
  ‖δ‖₂≈1.9. La mitad de las muestras agota las 50 iteraciones. ¿Por qué?
  Con logits muy confiados (gap 10-20), el *paso linealizado completo*
  de DeepFool sale disparado fuera del dominio [0,1]ⁿ; al recortar al
  cubo, el punto recortado no cruza la frontera y el ataque oscila.
  Verificamos que **torchattacks se comporta idéntico** (48.4% con la
  misma configuración): no es un bug de implementación sino una propiedad
  del algoritmo sobre modelos confiados bajo restricciones de caja.
- **C&W con presupuesto corto falla estrepitosamente** (búsqueda binaria
  ×3, 150 iteraciones ⇒ ~1.4% de éxito, ‖δ‖₂≈0.002 = no se mueve): no
  encuentra valores de `c` suficientemente grandes para vencer márgenes
  grandes. Con presupuesto serio (×5 rondas, 300 iteraciones, lr 0.02)
  sube al **100% de éxito con ‖δ‖₂≈2**. Moraleja doble: (1) C&W es el
  juez final, pero (2) un ataque débil *no* es evidencia de robustez —
  es el error de evaluación más común del campo, y aquí lo tienes medido.

## 7. Pseudocódigo del PGD del repo (`attacks/pgd.py`)

```python
x_orig = images.clone().detach()
delta = uniform(-eps, eps) if random_start else 0        # arranque
x = clamp(x_orig + delta, 0, 1)

for _ in range(steps):
    x.requires_grad_(True)
    loss = cross_entropy(model(x), labels)
    grad = autograd.grad(loss, x)[0]

    step = alpha * grad.sign()            # L∞ (o g/‖g‖₂ escalado en L2)
    delta = project((x - x_orig) + step)  # clamp ±ε  (o reescalar en L2)
    x = clamp(x_orig + delta, 0, 1)

# x es el adversarial; el éxito se mide con model(x).argmax != y
```

Detalles de implementación que importan:

- Trabajar sobre `delta` (relativo a `x_orig`) para que la proyección sea
  exacta — proyectar la imagen absoluta es un bug sutil.
- `autograd.grad` (no `backward`) para no contaminar `.grad` de los
  parámetros: el mismo ataque se usa dentro del entrenamiento adversarial.
- Clip final al rango de píxeles **después** de proyectar: proyección y
  clip no conmutan en general.

## 8. Ejercicios

1. Implementa PGD con proyección **sobre la imagen absoluta** (el bug
   mencionado) y comprueba que viola la bola ε. Entender el fallo vale
   más que la corrección.
2. Mide la robustez de un modelo con T ∈ {5, 10, 20, 40, 100} a ε fijo.
   ¿Converge la curva? ¿Cuál es el coste marginal de cada duplicación?
3. Implementa el ataque **de peor caso sobre k arranques**: para cada
   imagen, el adversarial que maximice la pérdida entre k arranques
   aleatorios. ¿Cuánto sube la tasa de éxito con k = 5?
4. Compara PGD-L∞ ε=0.3 con PGD-L2 ε=3.0 visualizando las perturbaciones.
   ¿Qué estructura tiene cada una y por qué?
