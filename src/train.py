import torch
from torch.optim import AdamW
from src.config import EPOCHS, LEARNING_RATE, DEVICE
from src.evaluate import evaluar

def entrenar(model, loader_train, loader_val):
  optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
  criterion = torch.nn.CrossEntropyLoss()

  for epoca in range(EPOCHS):
    model.train()
    loss_total = 0

    for batch in loader_train:
      input_ids = batch["input_ids"].to(DEVICE)
      attention_mask = batch["attention_mask"].to(DEVICE)
      labels = batch["label"].to(DEVICE)

      optimizer.zero_grad()
      outputs = model(input_ids=input_ids, attention_mask=attention_mask)
      loss = criterion(outputs.logits, labels)
      loss.backward()
      optimizer.step()

      loss_total += loss.item()

    loss_promedio = loss_total / len(loader_train)
    metricas = evaluar(model, loader_val)

    print(f"Época {epoca + 1}/{EPOCHS} | Loss: {loss_promedio:.4f} | Accuracy: {metricas['accuracy']:.4f} | F1: {metricas['f1']:.4f}")