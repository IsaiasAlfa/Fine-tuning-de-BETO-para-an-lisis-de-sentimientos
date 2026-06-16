# Fine-tuning de BETO para Análisis de Sentimientos en Reseñas Latinoamericanas
 
Clasificador binario de sentimientos (positivo/negativo) en español latinoamericano mediante fine-tuning supervisado (SFT) del modelo BETO sobre el dataset MELISA.
 
## Descripción
 
Este proyecto aplica fine-tuning a [BETO](https://github.com/dccuchile/beto) (`dccuchile/bert-base-spanish-wwm-uncased`), un modelo BERT pre-entrenado sobre texto en español, para clasificar reseñas de productos de Mercado Libre como positivas o negativas. El entrenamiento utiliza el dataset [MELISA](https://huggingface.co/datasets/lpsc-fiuba/melisa), diseñado específicamente para capturar variantes del español latinoamericano.
 
El dataset MELISA fue seleccionado por sobre alternativas más populares (como `amazon_reviews_multi`) debido a su cobertura del español latinoamericano de múltiples países (Argentina, Colombia, Perú, Uruguay, Chile, Venezuela y México), lo que lo hace más representativo del dominio objetivo.
 
## Stack tecnológico
 
| Componente | Tecnología |
|---|---|
| Lenguaje | Python 3.12 |
| Framework de DL | PyTorch 2.12 + CUDA 13.0 |
| Modelos y tokenizadores | HuggingFace Transformers |
| Carga de datasets | HuggingFace Datasets |
| Métricas | scikit-learn |
| Hardware | NVIDIA GeForce RTX 5070 |
| Sistema operativo | Fedora Linux |
 
## Modelo
 
**Base:** `dccuchile/bert-base-spanish-wwm-uncased` (BETO)
 
BETO es un modelo BERT entrenado sobre el corpus Spanish Wikipedia con Whole Word Masking. Sobre él se agrega una capa de clasificación lineal (`Linear(768 → 2)`) que utiliza el embedding del token `[CLS]` para producir la predicción binaria.
 
```
Texto → Tokenizador BETO → BETO (12 capas Transformer) → Embedding [CLS] → Linear(768→2) → {Positivo, Negativo}
```
 
## Dataset
 
**MELISA** (`lpsc-fiuba/melisa`) — reseñas de Mercado Libre en español latinoamericano con etiquetas de 1 a 5 estrellas.
 
La conversión a clasificación binaria se realiza de la siguiente forma:
 
- ⭐ 1–2 estrellas → **Negativo (0)**
- ⭐ 3 estrellas → **Descartado** (ambiguo)
- ⭐ 4–5 estrellas → **Positivo (1)**
## Pipeline
 
1. Detección de GPU y configuración del entorno
2. Carga y división del dataset MELISA (train/validation)
3. Tokenización con el tokenizador de BETO (truncation + padding a 128 tokens)
4. Carga de BETO con capa de clasificación binaria
5. Definición de hiperparámetros (épocas, batch size, learning rate)
6. Bucle de entrenamiento en GPU con evaluación por época
7. Guardado del modelo fine-tuneado
## Métricas de evaluación
 
- Accuracy
- Precision
- Recall
- F1-Score (macro)
## Instalación
 
```bash
# Clonar el repositorio
git clone https://github.com/IsaiasAlfa/Fine-tuning-de-BETO-para-an-lisis-de-sentimientos.git
cd Fine-tuning-de-BETO-para-an-lisis-de-sentimientos
 
# Crear entorno virtual con Python 3.12
python3.12 -m venv .venv
source .venv/bin/activate
 
# Instalar dependencias
pip install -r requirements.txt
```
 
> **Nota:** Se requiere una GPU con soporte CUDA. El proyecto fue desarrollado con CUDA 13.0 y PyTorch 2.12.
 
## Uso
 
```bash
# Entrenamiento
python train.py
 
# Evaluación
python evaluate.py
```
 
## Resultados
 
*En progreso — se actualizarán al completar el entrenamiento.*
 
| Métrica | Valor |
|---|---|
| Accuracy | — |
| Precision | — |
| Recall | — |
| F1-Score | — |
 
## Estructura del proyecto
 
```
├── README.md
├── requirements.txt
├── .gitignore
├── train.py           # Bucle de entrenamiento
├── evaluate.py        # Evaluación del modelo
├── dataset.py         # Carga y preprocesamiento de MELISA
├── model.py           # Definición del modelo
└── model/             # Modelo guardado (no incluido en el repo)
```
 
## Referencias
 
- [BETO: Spanish BERT](https://github.com/dccuchile/beto) — Cañete et al., 2020
- [MELISA Dataset](https://huggingface.co/datasets/lpsc-fiuba/melisa) — LPSC, FIUBA
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
## Autor
 
**Isaías Alberto Alfaro Ugalde**  
[GitHub](https://github.com/IsaiasAlfa/Fine-tuning-de-BETO-para-an-lisis-de-sentimientos)
