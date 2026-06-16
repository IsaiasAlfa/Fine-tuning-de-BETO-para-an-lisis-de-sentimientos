from transformers import AutoModelForSequenceClassification, AutoTokenizer
from src.config import MODEL_NAME, DEVICE

def cargar_modelo():
  tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
  model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2
  )
  model = model.to(DEVICE)
  return model, tokenizer

