# Dados

O MVP não duplica o dataset. Passe a pasta de origem no argumento `--data-dir`.

Ele reconhece os dois formatos abaixo:

```text
# Dataset atual, plano
caso.tiff
caso_generate.tiff
caso_mask.tiff

# Formato organizado, para uso futuro
data/
  original/caso.tiff
  synthetic/caso_generate.tiff
  masks/caso_mask.tiff
```
