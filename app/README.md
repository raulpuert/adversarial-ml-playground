# Demo interactiva (Gradio)

Ataca dígitos MNIST en vivo: elige ataque, presupuesto ε y modelo objetivo,
y observa la rejilla *original | perturbación ×10 | adversarial*.

## Ejecutar en local

```bash
pip install ".[app]"
python app/app.py
# → http://127.0.0.1:7860
```

Requiere los checkpoints entrenados (`mnist_cnn.pt` y, opcionalmente,
`mnist_cnn_adv.pt` para la pestaña de comparación).

## Desplegar gratis en Hugging Face Spaces

1. Crea un Space nuevo con SDK **Gradio**.
2. Sube: `app/app.py` (renombrado a `app.py`), el paquete
   `adversarial_playground/` y los `checkpoints/`.
3. Añade un `requirements.txt` con:

```
torch
torchvision
gradio
numpy
matplotlib
pillow
```

4. En `app.py`, sustituye la carga local de checkpoints por una descarga
   desde un *release* de GitHub o el repo (HF Spaces tiene disco efímero,
   así que lo simple es subir los `.pt` al Space).
