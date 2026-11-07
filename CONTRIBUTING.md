# Contribuir a adversarial-ml-playground

¡Gracias por el interés! Este es un proyecto educativo, así que las
contribuciones más valiosas son las que **enseñan algo mejor** que lo que
hay. Guía rápida:

## Cómo proponer cambios

1. Abre un *issue* describiendo la mejora (o el bug con pasos para
   reproducirlo).
2. Haz fork, crea una rama descriptiva: `git checkout -b ataque-eot`.
3. Asegúrate de que `pytest -m "not slow"` y `ruff check .` pasan.
4. Abre un PR explicando *por qué* (no solo *qué*).

## Ideas bienvenidas

- **Ataques**: JSMA, one-pixel, SPSA (caja negra), EOT completo,
  PGD con momentum (MI-FGSM), Square Attack.
- **Defensas**: TRADES, Free-AT, denoisers con comparativa honesta,
  certificados con difuminado semántico.
- **Datasets**: Fashion-MNIST, SVHN (basta añadirlos a `utils/data.py`
  y registrar stats/nombres de clase).
- **Evaluación**: curvas de *confianza calibrada* bajo ataque,
  comparativas de transferibilidad entre modelos.
- **Docs y notebooks**: cualquier explicación más clara que la actual.

## Convenciones

- Código en inglés (identificadores), comentarios y docstrings en
  **español** — es un proyecto educativo en español; la coherencia importa.
- Todo ataque nuevo debe: heredar de `BaseAttack`, devolver `AttackResult`,
  respetar la bola declarada (hay tests de invariantes que se pueden
  parametrizar) y traer test propio.
- Los modelos van en `MODEL_REGISTRY` con checkpoint reproducible
  (semilla, épocas y configuración en los metadatos).
- Los resultados numéricos del README **siempre** se regeneran con
  `scripts/make_assets.py` — no se editan a mano.

## Entorno

```bash
git clone https://github.com/raulp/adversarial-ml-playground
cd adversarial-ml-playground
python -m venv .venv && source .venv/bin/activate   # o .venv\Scripts\activate en Windows
pip install -e ".[dev,app]"
pytest -m "not slow"
```
