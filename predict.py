import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.config import MODEL_SAVE_PATH, MAX_LEN, DEVICE

def cargar_modelo():
  tokenizer = AutoTokenizer.from_pretrained(MODEL_SAVE_PATH)
  model = AutoModelForSequenceClassification.from_pretrained(MODEL_SAVE_PATH)
  model = model.to(DEVICE)
  model.eval()
  return model, tokenizer

def predecir(texto, model, tokenizer):
  encoding = tokenizer(
    texto,
    max_length=MAX_LEN,
    padding="max_length",
    truncation=True,
    return_tensors="pt"
  )
  input_ids = encoding["input_ids"].to(DEVICE)
  attention_mask = encoding["attention_mask"].to(DEVICE)

  with torch.no_grad():
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    pred = torch.argmax(outputs.logits, dim=1).item()

  etiquetas = {0: "No tóxico", 1: "Tóxico"}
  return etiquetas[pred]


if __name__ == "__main__":
  model, tokenizer = cargar_modelo()

  print("=== Textos que el modelo clasifica bien ===\n")
  textos_adecuados = [
    "así me gusta, que se maten entre ellos.",
    "los mejores, más preparados para cortar calabazas.",
    "lástima que no se volvió loco del todo.",
    "llévatelos a todos, con familia incluida.",
    "qué bien que se los lleven a todos de una vez.",
    "como les gusta el afeitado en seco a esta gente.",
    "los detuvieron en ronda málaga, un saludo.",
    "mucho texto, aguardo a la noticia en la sexta.",
    "no, en ronda no tenemos playa, somos del interior.",
    "el voluntario de Cruz Roja que lo atendió al llegar.",
  ]

  for texto in textos_adecuados:
    resultado = predecir(texto, model, tokenizer)
    print(f"[{resultado}] {texto}")

  print("\n=== Textos fuera del dominio del modelo ===\n")
  textos_inadecuados = [
    "Eres un idiota y no sabes nada.",
    "Ojalá te mueras maldito.",
    "Me encanta este producto, funciona perfecto.",
    "El servicio al cliente fue excelente.",
    "Esta película es una basura absoluta.",
    "Te voy a romper la cara.",
    "Eres lo peor que me ha pasado en la vida.",
    "Qué bonito día hace hoy.",
    "No me gustó nada la comida del restaurante.",
    "Eres un genio, gracias por tu ayuda.",
  ]

  for texto in textos_inadecuados:
    resultado = predecir(texto, model, tokenizer)
    print(f"[{resultado}] {texto}")