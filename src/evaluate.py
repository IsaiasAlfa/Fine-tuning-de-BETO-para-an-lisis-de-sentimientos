import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from src.config import DEVICE


def evaluar(model, loader):
  model.eval()
  preds_todas = []
  labels_todas = []

  with torch.no_grad():
    for batch in loader:
      input_ids = batch["input_ids"].to(DEVICE)
      attention_mask = batch["attention_mask"].to(DEVICE)
      labels = batch["label"].to(DEVICE)

      outputs = model(input_ids=input_ids, attention_mask=attention_mask)
      preds = torch.argmax(outputs.logits, dim=1)

      preds_todas.extend(preds.cpu().numpy())
      labels_todas.extend(labels.cpu().numpy())

  precision, recall, f1, _ = precision_recall_fscore_support(
    labels_todas, preds_todas, average="macro"
  )
  accuracy = accuracy_score(labels_todas, preds_todas)

  return {
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1": f1,
  }
