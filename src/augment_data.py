import os
import pandas as pd
import time
from openai import OpenAI

# Initialize DeepSeek client
client = OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

classes = ["Refund Request", "Delivery Issue", "Product Defect", "Product Question"]

def generate_samples(issue_class, count=50):
    prompt = f"""
    Generate exactly {count} distinct, highly realistic Thai customer reviews/complaints that fall under the category: "{issue_class}".
    
    CRITICAL Requirements:
    - Use authentic, everyday Thai language (spoken style, slang, typos, missing vowels, intense emotions).
    - Vary the sentence lengths drastically (some very short, some paragraph-long).
    - Vary the tone (furious, polite, confused, desperate).
    - DO NOT use repetitive templates. Every single sentence must be uniquely phrased.
    - OUTPUT FORMAT: Output strictly plain text with one review per line. No headers, no markdown blocks, no numbering (e.g., no "1. "), no quotes. Just the raw text.
    """
    
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are an expert data generation assistant specialized in Thai NLP and slang."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.9
        )
        content = response.choices[0].message.content.strip()
        
        # Clean up markdown if the model hallucinated it
        if content.startswith('```'):
            lines = content.split('\n')[1:-1]
        else:
            lines = content.split('\n')
            
        # Clean up empty lines, list numbers, and quotes
        cleaned = []
        for line in lines:
            line = line.strip('"- \t')
            # Remove potential numbering (e.g., "1. ", "2. ")
            if len(line) > 2 and line[0].isdigit() and line[1] in ['.', ')']:
                line = line[2:].strip()
            elif len(line) > 3 and line[:2].isdigit() and line[2] in ['.', ')']:
                line = line[3:].strip()
                
            if line and not line.lower().startswith('review'):
                cleaned.append(line)
                
        return cleaned
    except Exception as e:
        print(f"Error generating for {issue_class}: {e}")
        return []

def main():
    target_per_class = 300
    batch_size = 50
    data = []
    
    for c in classes:
        print(f"Generating diverse data for {c}...")
        collected = 0
        while collected < target_per_class:
            samples = generate_samples(c, count=batch_size)
            for s in samples:
                data.append({"text": s, "label": c})
            collected += len(samples)
            print(f"  Collected {collected}/{target_per_class} for {c}...")
            time.sleep(1) # slight pause for rate limit
            
    df = pd.DataFrame(data)
    
    # Shuffle the dataset
    df = df.sample(frac=1).reset_index(drop=True)
    
    # Save the augmented dataset, overwriting the old template-based one
    output_path = "data/processed/custom_issues.csv"
    df.to_csv(output_path, index=False)
    print(f"Success! Saved {len(df)} diverse samples to {output_path}!")

if __name__ == "__main__":
    main()
