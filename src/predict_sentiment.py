import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class SentimentPredictor:
    def __init__(self, model_path="models/sentiment"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.eval()
        self.id2label = {0: "Positive", 1: "Neutral", 2: "Negative"}

    def predict(self, text: str):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1)
            confidence, label_id = torch.max(probs, dim=-1)
        
        return self.id2label[label_id.item()], confidence.item()

def predict_sentiment(text: str):
    predictor = SentimentPredictor()
    return predictor.predict(text)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        print(predict_sentiment(sys.argv[1]))
    else:
        print(predict_sentiment("สินค้านี้ดีมาก ชอบมากครับ"))
