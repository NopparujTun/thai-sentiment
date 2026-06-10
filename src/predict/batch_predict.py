import pandas as pd
import sys
import time
from src.predict.predict_sentiment import SentimentPredictor
from src.predict.predict_intent import IntentPredictor

def batch_predict(input_csv: str, output_csv: str):
    print(f"Loading data from {input_csv}...")
    df = pd.read_csv(input_csv)
    
    text_col = None
    for col in ['review_text', 'text', 'texts', 'review']:
        if col in df.columns:
            text_col = col
            break
            
    if not text_col:
        raise ValueError("Could not find a text column. Expected one of: 'review_text', 'text', 'texts'")

    print("Loading models...")
    sentiment_predictor = SentimentPredictor()
    intent_predictor = IntentPredictor()
    
    print("Running predictions...")
    start_time = time.time()
    
    results = []
    for text in df[text_col]:
        if pd.isna(text):
            results.append({
                "review_text": text,
                "sentiment": None,
                "sentiment_score": None,
                "intent": None,
                "intent_score": None
            })
            continue
            
        s_label, s_score = sentiment_predictor.predict(str(text))
        i_label, i_score = intent_predictor.predict(str(text))
        
        results.append({
            "review_text": text,
            "sentiment": s_label,
            "sentiment_score": s_score,
            "intent": i_label,
            "intent_score": i_score
        })
        
    out_df = pd.DataFrame(results)
    out_df.to_csv(output_csv, index=False)
    
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"Processed {len(df)} rows in {elapsed:.2f} seconds ({elapsed/len(df):.4f}s per row).")
    print(f"Predictions saved to {output_csv}")

if __name__ == "__main__":
    if len(sys.argv) == 3:
        batch_predict(sys.argv[1], sys.argv[2])
    else:
        print("Usage: python src/batch_predict.py <input.csv> <output.csv>")
