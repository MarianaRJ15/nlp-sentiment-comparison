# Dataset IMDB

## Origen
El dataset IMDB proviene del repositorio oficial de Stanford NLP y está disponible en Hugging Face bajo el identificador `stanfordnlp/imdb`.

## Estructura
- **text:** reseña de película en inglés.
- **label:** 0 = negativo, 1 = positivo.

## Obtención
El dataset se carga automáticamente mediante:

```python
from datasets import load_dataset
dataset = load_dataset("stanfordnlp/imdb")
