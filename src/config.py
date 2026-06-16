import torch

import torch

# Modelo y dataset
MODEL_NAME = "dccuchile/bert-base-spanish-wwm-uncased"
DATASET_NAME = "lpsc-fiuba/melisa"
MAX_LEN = 128

# Hiperparámetros de entrenamiento
EPOCHS = 3
BATCH_SIZE = 32
LEARNING_RATE = 2e-5
TRAIN_SPLIT = 0.9
SEED = 42

# Sistema
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_SAVE_PATH = "model/"