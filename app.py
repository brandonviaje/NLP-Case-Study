from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import DistilBertTokenizer
from model.distilbert_model import MyDistilBERT

model = MyDistilBERT()
state_dict = torch.load("model/distilbert_state_dict.pth", map_location="cpu")
model.model.load_state_dict(state_dict)
model.eval()

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

app = FastAPI()

class TextRequest(BaseModel):
    text: str

@app.post("/predict")
def predict(request: TextRequest):
    # Tokenize
    inputs = tokenizer(
        request.text, 
        return_tensors="pt", 
        truncation=True, 
        padding="max_length", 
        max_length=128
    )

    # Forward pass
    with torch.no_grad():
        logits = model(inputs["input_ids"], inputs["attention_mask"])

    pred = torch.argmax(logits, dim=1).item()
    label_map = {0: "negative", 1: "positive"}
    return {"prediction": label_map[pred]}