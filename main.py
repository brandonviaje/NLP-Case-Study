import torch
from transformers import DistilBertTokenizer
from model.distilbert_model import MyDistilBERT
import gradio as gr

model = MyDistilBERT()
state_dict = torch.load("model/distilbert_state_dict.pth", map_location="cpu")
model.model.load_state_dict(state_dict)
model.eval()

# Load tokenizer
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

def predict(text):
    # Tokenize
    inputs = tokenizer(
        text, 
        return_tensors="pt", 
        truncation=True, 
        padding="max_length", 
        max_length=128
    )

    # Forward pass
    with torch.no_grad():
        logits = model(**inputs).logits

    probs = torch.softmax(logits, dim=1)
    positive = float(probs[0][1])
    negative = float(probs[0][0])
    
    return {
        "Positive": positive,
        "Negative": negative
    }

iface = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(label="Enter text"),
    outputs=gr.Label(label="Sentiment"),
    title="Sentiment Analysis",
    description="Your custom fine-tuned DistilBERT sentiment model"
)

iface.launch()
