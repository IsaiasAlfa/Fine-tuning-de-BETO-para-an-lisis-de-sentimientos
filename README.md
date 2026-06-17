# Fine-tuning de BETO para Detección de Toxicidad en Español

Clasificador binario de toxicidad (tóxico/no tóxico) en español mediante fine-tuning supervisado (SFT) del modelo BETO sobre el dataset ES Benchmark Binary.

## Descripción

Este proyecto aplica fine-tuning a [BETO](https://github.com/dccuchile/beto) (`dccuchile/bert-base-spanish-wwm-uncased`), un modelo BERT pre-entrenado sobre texto en español, para clasificar textos en español como tóxicos o no tóxicos. El entrenamiento utiliza el dataset [ES Benchmark Binary](https://huggingface.co/datasets/mrinaldi/es_benchmark_binary), un benchmark de clasificación binaria en español con 67,924 ejemplos derivados del IberLEF Toxicity Detection 2021.

La detección de toxicidad es una tarea de alto impacto en moderación de contenido, redes sociales y plataformas digitales, con implicaciones directas en debates sobre libertad de expresión y seguridad en línea.

## Stack tecnológico

| Componente | Tecnología |
|---|---|
| Lenguaje | Python 3.12 |
| Framework de DL | PyTorch 2.12 + CUDA 13.0 |
| Modelos y tokenizadores | HuggingFace Transformers 5.12.1 |
| Carga de datasets | HuggingFace Datasets 5.0.0 |
| Métricas | scikit-learn 1.9.0 |
| Hardware | NVIDIA GeForce RTX 5070 |
| Sistema operativo | Fedora Linux |

## Modelo

**Base:** `dccuchile/bert-base-spanish-wwm-uncased` (BETO)

BETO es un modelo BERT entrenado sobre el corpus Spanish Wikipedia con Whole Word Masking. Sobre él se agrega una capa de clasificación lineal (`Linear(768 → 2)`) que utiliza el embedding del token `[CLS]` para producir la predicción binaria.

```
Texto → Tokenizador BETO → BETO (12 capas Transformer) → Embedding [CLS] → Linear(768→2) → {Tóxico, No tóxico}
```

## Dataset

**ES Benchmark Binary** (`mrinaldi/es_benchmark_binary`) — benchmark de clasificación binaria en español con 67,924 ejemplos derivados del IberLEF Toxicity Detection 2021.

| Label | Clase | Ejemplos |
|---|---|---|
| 0 | No tóxico | 43,663 |
| 1 | Tóxico | 24,261 |

> **Nota:** El dataset presenta un desbalance de clases de aproximadamente 2:1 (no tóxico vs tóxico), lo que puede sesgar las predicciones hacia la clase mayoritaria.

## Hiperparámetros

| Parámetro | Valor |
|---|---|
| Épocas | 3 |
| Batch size | 32 |
| Learning rate | 2e-5 |
| Max tokens | 128 |
| Train/Val split | 90% / 10% |
| Optimizador | AdamW |
| Función de pérdida | CrossEntropyLoss |

## Pipeline

1. Detección de GPU y configuración del entorno
2. Carga del dataset ES Benchmark Binary (train/validation)
3. Tokenización con el tokenizador de BETO (truncation + padding a 128 tokens)
4. Carga de BETO con capa de clasificación binaria
5. Definición de hiperparámetros
6. Bucle de entrenamiento en GPU con evaluación por época
7. Guardado del modelo fine-tuneado

## Resultados

| Época | Loss | Accuracy | F1 |
|---|---|---|---|
| 1 | 0.3706 | 87.19% | 85.25% |
| 2 | 0.1998 | 90.76% | 89.89% |
| 3 | 0.0937 | 92.07% | 91.26% |

## Limitaciones

### Sesgo de dominio
El modelo fue entrenado exclusivamente con tweets políticos en español de España (IberLEF 2021). Como consecuencia, sus predicciones son confiables únicamente dentro de ese dominio específico. Al evaluar el modelo con textos de otros dominios se observa degradación severa:

- **Insultos directos** (`"Eres un idiota"`, `"Ojalá te mueras"`) → clasificados incorrectamente como **no tóxicos**
- **Elogios y textos neutros** (`"Me encanta este producto"`, `"Eres un genio"`) → clasificados incorrectamente como **tóxicos**

### Sarcasmo como señal dominante
El dataset contiene una alta proporción de sarcasmo político con estructura de elogio falso. El modelo aprendió a asociar ciertas estructuras gramaticales positivas con toxicidad, lo que produce falsos positivos en textos genuinamente positivos.


### Trabajo futuro
- Reentrenar con un dataset de toxicidad explícita y criterios de anotación más claros
- Aplicar técnicas de balanceo de clases (oversampling, class weights) para corregir el desbalance 2:1

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
python main.py

# Predicción
python predict.py
```

## Estructura del proyecto

```
.
├── README.md
├── requirements.txt
├── .gitignore
├── main.py
├── predict.py
└── src/
    ├── __init__.py
    ├── config.py
    ├── dataset.py
    ├── model.py
    ├── train.py
    └── evaluate.py
```

## Referencias

- [BETO: Spanish BERT](https://github.com/dccuchile/beto) — Cañete et al., 2020
- [ES Benchmark Binary](https://huggingface.co/datasets/mrinaldi/es_benchmark_binary) — mrinaldi, 2025
- [IberLEF Toxicity Detection 2021](https://competitions.codalab.org/competitions/28679)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)

## Autor

**Isaías Alberto Alfaro Ugalde**  
[GitHub](https://github.com/IsaiasAlfa/Fine-tuning-de-BETO-para-an-lisis-de-sentimientos)