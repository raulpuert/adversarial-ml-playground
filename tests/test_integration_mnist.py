"""Test de integración sobre MNIST real (marcado ``slow``).

Se salta si no hay datos descargados, y usa el checkpoint entrenado si
existe (si no, entrena 1 época rápida). Ejecutar con::

    pytest -v -m slow
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]


def _mnist_available() -> bool:
    try:
        from adversarial_playground.utils.data import get_datasets

        _, test = get_datasets("mnist", ROOT / "data")
        return len(test) > 0
    except Exception:  # noqa: BLE001 - cualquier fallo de red/disco ⇒ skip
        return False


@pytest.mark.slow
@pytest.mark.skipif(not _mnist_available(), reason="MNIST no descargado")
class TestRealMNIST:
    """Invariante clave sobre datos reales: el modelo limpio es preciso pero
    frágil, y los ataques respetan sus bolas."""

    @pytest.fixture()
    def model_and_loader(self):
        from adversarial_playground.utils.data import get_loaders, load_checkpoint
        from adversarial_playground.utils.seeding import set_seed

        set_seed(42)
        ckpt = ROOT / "checkpoints" / "mnist_cnn.pt"
        if ckpt.exists():
            model, _ = load_checkpoint(ckpt)
        else:  # modelo sin entrenar: solo vale para invariantes de forma
            from adversarial_playground.models import MnistCNN

            model = MnistCNN()
        model.eval()
        _, loader = get_loaders("mnist", batch_size=64, root=ROOT / "data")
        return model, loader

    def test_clean_accuracy_over_95(self, model_and_loader):
        model, loader = model_and_loader
        if not (ROOT / "checkpoints" / "mnist_cnn.pt").exists():
            pytest.skip("sin checkpoint entrenado")
        correct = total = 0
        with torch.no_grad():
            for x, y in loader:
                correct += int((model(x).argmax(1) == y).sum())
                total += len(y)
                if total >= 2000:
                    break
        assert correct / total > 0.95

    def test_attacks_respect_balls_on_real_images(self, model_and_loader):
        from adversarial_playground.attacks.factory import build_attack

        model, loader = model_and_loader
        images, labels = next(iter(loader))
        for name, eps in (("fgsm", 0.2), ("pgd", 0.2), ("bim", 0.2)):
            attack = build_attack(name, model, eps=eps)
            result = attack(images, labels)
            assert (result.linf_norms <= eps + 1e-6).all(), name
            assert result.adv_images.min() >= 0 and result.adv_images.max() <= 1

    def test_pgd_beats_fgsm_on_real_images(self, model_and_loader):
        from adversarial_playground.attacks.factory import build_attack

        model, loader = model_and_loader
        images, labels = next(iter(loader))
        fgsm_rate = build_attack("fgsm", model, eps=0.3)(images, labels).success_rate
        pgd_rate = build_attack("pgd", model, eps=0.3, steps=20)(
            images, labels
        ).success_rate
        assert pgd_rate >= fgsm_rate - 0.05
