import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from src.config import MODEL_NAME, DATASET_NAME, MAX_LEN, BATCH_SIZE, TRAIN_SPLIT, SEED

def cargar_dataset():
  # Cargar el dataset de Hugging Face en español
  dataset = load_dataset(DATASET_NAME, "es", trust_remote_code=True)
  # Pasaar a pandas y filtrar las reseñas con calificación de 3
  datos = dataset["train"].to_pandas()
  datos = datos[datos["review_rate"] != 3]
  datos["label"] = (datos["review_rate"] >= 4).astype(int)
  return datos[["review_body", "label"]].reset_index(drop=True)

class MelisaDataset(Dataset):
  def __init__(self, textos, labels, tokenizer):
    self.textos = textos
    self.labels = labels
    self.tokenizer = tokenizer

  def __len__(self):
    return len(self.textos)

  def __getitem__(self, idx):
    encoding = self.tokenizer(
      self.textos[idx],
      max_length=MAX_LEN,
      padding="max_length",
      truncation=True,
      return_tensors="pt"
    )
    return {
      "input_ids": encoding["input_ids"].squeeze(),
      "attention_mask": encoding["attention_mask"].squeeze(),
      "label": torch.tensor(self.labels[idx], dtype=torch.long)
    }
  
  def obtener_dataloaders(tokenizer):
    datos = cargar_dataset()
    
    textos = datos["review_body"].tolist()
    labels = datos["label"].tolist()
    
    textos_train, textos_val, labels_train, labels_val = train_test_split(
      textos, labels,
      test_size=1 - TRAIN_SPLIT,
      random_state=SEED,
      stratify=labels
    )
    
    dataset_train = MelisaDataset(textos_train, labels_train, tokenizer)
    dataset_val = MelisaDataset(textos_val, labels_val, tokenizer)
    
    loader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)
    loader_val = DataLoader(dataset_val, batch_size=BATCH_SIZE, shuffle=False)
    
    return loader_train, loader_val