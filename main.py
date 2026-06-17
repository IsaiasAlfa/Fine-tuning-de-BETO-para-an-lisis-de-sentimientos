import torch
from src.config import DEVICE, SEED, MODEL_SAVE_PATH
from src.model import cargar_modelo
from src.dataset import obtener_dataloaders
from src.train import entrenar

def main():
  torch.manual_seed(SEED)

  print(f"Usando dispositivo: {DEVICE}")

  print("Cargando modelo y tokenizador...")
  model, tokenizer = cargar_modelo()

  print("Cargando dataset Bench...")
  loader_train, loader_val = obtener_dataloaders(tokenizer)

  print("Iniciando entrenamiento...")
  entrenar(model, loader_train, loader_val)

  print("Guardando modelo...")
  model.save_pretrained(MODEL_SAVE_PATH)
  tokenizer.save_pretrained(MODEL_SAVE_PATH)
  print(f"Modelo guardado en {MODEL_SAVE_PATH}")


if __name__ == "__main__":
  main()