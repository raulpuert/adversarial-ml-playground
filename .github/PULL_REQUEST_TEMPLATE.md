## Qué cambia

<!-- Describe el cambio y por qué. Si cierra un issue, añade "Closes #N". -->

## Checklist

- [ ] `pytest -m "not slow"` en verde
- [ ] `ruff check adversarial_playground scripts tests tools` sin avisos
- [ ] Si añade un ataque/defensa: hereda de `BaseAttack`/contrato correspondiente,
      trae tests de invariantes y docstring con la referencia arXiv
- [ ] Si cambia resultados numéricos: `scripts/make_assets.py` regenerado
      (los números del README nunca se editan a mano)
- [ ] Comentarios y docstrings en español (convención del repo)
