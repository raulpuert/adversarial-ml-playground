# 01 · Fundamentos: modelos de amenaza, normas y por qué ocurre

> *"Adversarial examples are not a bug of deep learning — they are a
> consequence of its linear structure in high-dimensional space."*
> — adaptado de Goodfellow et al. (2014)

## 1. El problema en una frase

Un **ejemplo adversarial** es una entrada manipulada de forma *deliberada y
casi imperceptible* para que un modelo la clasifique mal con alta confianza.
No es ruido aleatorio (eso el modelo lo ignora): es una señal construida
usando el propio modelo contra sí mismo.

```
   imagen + δ = imagen adversarial
   ‖δ‖ pequeño  →  imperceptible para un humano
   f(imagen + δ) ≠ f(imagen)  →  el modelo cambia de opinión
```

El descubrimiento data de **Szegedy et al. (2013)**: redes que tenían 99%+
de precisión en ImageNet fallaban catastróficamente con perturbaciones que
ni un humano detectaba. La ilustración icónica (Goodfellow et al., 2014):
panda → `+ ε·sign(∇L)` → gibón con 99.3% de confianza.

## 2. Modelos de amenaza

Toda afirmación sobre robustez debe especificar **qué puede hacer el
atacante**. Las dimensiones relevantes:

| Dimensión | Opciones |
|---|---|
| Conocimiento | **caja blanca** (conoce pesos y arquitectura) · **caja negra** (solo entradas/salidas) · **caja gris** (conoce arquitectura, no pesos) |
| Control | puede modificar la **entrada** (evasión) · solo puede mirar (extracción) · puede envenenar el **entrenamiento** (data poisoning) |
| Objetivo | **no dirigido** (que falle, da igual cómo) · **dirigido** (que prediga una clase concreta) |
| Presupuesto | norma y radio: L∞ con ε, L2, L0... |

Este playground implementa ataques de **evasión en caja blanca** sobre
imágenes. La evasión es el escenario más estudiado y el más comparable
entre papers.

## 3. Las normas: cómo se mide "imperceptible"

| Norma | Definición | Lectura intuitiva | Ataque típico |
|---|---|---|---|
| **L∞** | `max |δᵢ|` | *cada* píxel puede cambiar como mucho ε | FGSM, PGD |
| **L2** | `√(Σ δᵢ²)` | el presupuesto total se reparte entre píxeles | DeepFool, C&W-L2 |
| **L0** | `#{δᵢ ≠ 0}` | número de píxeles modificados | One-pixel attack |
| **L1** | `Σ |δᵢ|` | "tinta" total gastada | sparse attacks |

El mismo ε numérico da perturbaciones muy distintas según la bola: en L∞
obtienes "grano fino" distribuido por toda la imagen; en L2 una mancha
suave concentrada donde más importa. Por eso **los resultados no son
comparables entre normas** y toda tabla de robustez debe declarar la suya.

## 4. ¿Por qué ocurre? Las tres explicaciones

**a) La hipótesis lineal (Goodfellow).** Las redes modernas apilan
operaciones casi lineales (afines + ReLU). En la escala de un píxel, la
salida responde linealmente: `f(x + δ) ≈ f(x) + ∇f·δ`. En una imagen de
784-3072 dimensiones, 784 contribuciones individuales pequeñas **suman**
un cambio grande en el logit. Los modelos con mucho margen de decisión
pero con hiperplanos mal orientados son frágiles.

**b) Geometría del espacio.** Con miles de dimensiones, las direcciones
"que casi nadie ve" abundan: la imagen vive en una subvariedad de baja
dimensión y hay muchísimas direcciones ortogonales a lo que un humano
percibe donde los logits varían mucho. La frontera de decisión corta esa
región muy cerca de cada imagen natural.

**c) Sobreajuste a la métrica de entrenamiento.** El entrenamiento solo
optimiza precisión sobre puntos de la distribución; nada empuja a la
función a ser *consistente* en la bola alrededor de cada punto. La red
aprende atajos estadísticos que funcionan en la distribución y fallan
fuera de ella.

Las tres explicaciones son compatibles y hoy conviven en la literatura.

## 5. Transferibilidad: la pesadilla de las cajas negras

Un adversarial construido contra el modelo A engaña al modelo B con
probabilidad sorprendentemente alta — aunque A y B tengan arquitecturas,
datos y semillas distintas. Implicación brutal: **no necesitas la caja
negra para atacarla**; entrena tu propio modelo sustituto, atácalo en
caja blanca, y transfiere.

Es la razón por la que los sistemas reales (APIs comerciales, modelos en
producción) son atacables aunque no publiquen pesos: la transferibilidad
convierte caja negra en caja gris.

## 6. Taxonomía de ataques implementados en este repo

| Ataque | Papel | Norma | Coste | Fuerza |
|---|---|---|---|---|
| [FGSM](https://arxiv.org/abs/1412.6572) | el clásico de 1 paso; el "hola mundo" | L∞ | ★ | ★ |
| [BIM / I-FGSM](https://arxiv.org/abs/1607.02533) | FGSM iterado con pasos finos | L∞ | ★★ | ★★★ |
| [PGD](https://arxiv.org/abs/1706.06083) | BIM + arranque aleatorio; el estándar | L∞/L2 | ★★ | ★★★★ |
| [DeepFool](https://arxiv.org/abs/1511.04599) | perturbación *mínima*; mide márgenes | L2 | ★★★ | (no compite: mide distancia) |
| [C&W](https://arxiv.org/abs/1608.04644) | optimización completa; el juez final | L2 | ★★★★★ | ★★★★★ |

Regla de uso en evaluación: **barre ε con PGD** para curvas de robustez,
usa **C&W** para confirmar que los márgenes son reales, y **DeepFool**
para estudiar la geometría de la frontera.

## 7. ¿Importa en el mundo real?

Sí, y no solo como curiosidad académica:

- **Visión en automoción**: parches físicos que hacen que una señal de
  stop se lea como "límite 45" (Eykholt et al., 2017) — perturbaciones
  *impresas* en el mundo físico.
- **Reconocimiento facial**: gafas impresas que confunden sistemas de
  identificación (Sharif et al., 2016).
- **Moderación de contenido / antimalware**: evasión de clasificadores
  de texto e imágenes modificando ejemplos con sinónimos o recodificaciones.
- **LLMs**: prompt injection es el primo textual del ataque adversarial
  (y la seguridad de LLMs es hoy una de las áreas más activas).

## 8. Hoja de ruta de lectura

1. Goodfellow et al. (2014) — la explicación fundacional ([arXiv:1412.6572](https://arxiv.org/abs/1412.6572))
2. Madry et al. (2017) — PGD y el entrenamiento adversarial ([arXiv:1706.06083](https://arxiv.org/abs/1706.06083))
3. Carlini & Wagner (2017) — el ataque fuerte de referencia ([arXiv:1608.04644](https://arxiv.org/abs/1608.04644))
4. Athalye et al. (2018) — por qué fallan las defensas ([arXiv:1802.00420](https://arxiv.org/abs/1802.00420))
5. Cohen et al. (2019) — robustez certificable ([arXiv:1902.02918](https://arxiv.org/abs/1902.02918))

Continúa en [02 · La matemática de FGSM](02-fgsm.md).
