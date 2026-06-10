import pandas as pd
import random
import os
from datasets import load_dataset
from src.pipeline.preprocessing import clean_data

def prepare_wisesight():
    print("Preparing Wisesight dataset...")
    # Load from huggingface datasets
    dataset = load_dataset("wisesight_sentiment", trust_remote_code=True)
    
    # We will combine train, valid, test for processing and later splits if needed
    dfs = []
    for split in ["train", "validation", "test"]:
        df = pd.DataFrame(dataset[split])
        dfs.append(df)
        
    full_df = pd.concat(dfs, ignore_index=True)
    
    # Map label integers to string for clarity or keep as integers
    # Wisesight labels: 0: pos, 1: neu, 2: neg, 3: q
    # As per PRD we only need Positive, Neutral, Negative, maybe we map 3 -> neutral or drop it?
    # PRD says (Positive, Neutral, Negative). Let's drop question (label 3).
    full_df = full_df[full_df['category'] != 3].copy()
    
    # Clean the texts
    # wisesight_sentiment dataset has 'texts' and 'category'
    cleaned_df = clean_data(full_df, 'texts')
    
    # Save to processed
    cleaned_df.to_csv("data/processed/wisesight_sentiment.csv", index=False)
    print(f"Wisesight dataset prepared: {len(cleaned_df)} rows.")

def generate_intent_dataset():
    print("Generating custom Thai intent dataset...")
    # Generate mock data for 4 classes: Delivery Issue, Product Defect, Product Question, Refund Request
    # 350 samples each
    classes = ["Delivery Issue", "Product Defect", "Product Question", "Refund Request"]
    templates = {
        "Delivery Issue": [
            "ของยังไม่ถึงเลย สั่งไปตั้งแต่วันที่ {}",
            "เช็คสถานะพัสดุให้หน่อยค่ะ รอมา {} วันแล้ว",
            "ทำไมส่งช้าจัง ตามของได้ที่ไหน",
            "ขนส่งแย่มาก โทรมาแล้วก็ไม่รอส่งของ",
            "ของตีกลับเพราะอะไรคะ ไม่เห็นมีใครโทรมา"
        ],
        "Product Defect": [
            "สินค้าเสียตั้งแต่แกะกล่อง เปิดไม่ติดเลย",
            "หน้าจอมีรอยแตกนะคะ ขอเคลมได้ไหม",
            "ใช้งานไป {} วันก็พังแล้ว ห่วยมาก",
            "สีลอก อุปกรณ์ไม่ครบตามที่แจ้งไว้",
            "เสียบไฟแล้วไฟไม่เข้า เครื่องร้อนมาก"
        ],
        "Product Question": [
            "มีสีดำหรือเปล่าคะ รุ่นนี้",
            "ขนาด กว้างxยาว เท่าไหร่คะ",
            "ใช้กับไฟ {} โวลต์ได้ไหม",
            "มีของพร้อมส่งไหม หรือต้องพรีออเดอร์",
            "วิธีใช้งานเบื้องต้นทำยังไงคะ มีคู่มือไหม"
        ],
        "Refund Request": [
            "ขอคืนเงินได้ไหมคะ สั่งผิดรุ่น",
            "ยกเลิกออเดอร์ไปแล้ว รอกี่วันถึงจะได้เงินคืน",
            "สินค้าไม่ตรงปก ต้องการคืนของและขอเงินคืนค่ะ",
            "หักเงินไปแล้วแต่สถานะยังไม่ขึ้นจ่ายเงิน รบกวนคืนเงินด้วย",
            "เปลี่ยนใจไม่รับแล้ว ขอเงินคืนเต็มจำนวนนะคะ"
        ]
    }
    
    data = []
    for c in classes:
        for i in range(350):
            template = random.choice(templates[c])
            order_no = f"ORD{random.randint(10000, 99999)}"
            if "{}" in template:
                text = template.format(random.randint(2, 100))
            else:
                text = template + (" ครับ" if random.choice([True, False]) else " ค่ะ")
            
            # Make it more unique
            text = f"{text} (ออเดอร์ {order_no})"
            data.append({"text": text, "label": c})
            
    df = pd.DataFrame(data)
    # Add some messy data to test cleaning
    df.loc[10, "text"] = df.loc[10, "text"] + " https://shopee.co.th/product/123 "
    df.loc[100, "text"] = df.loc[100, "text"] + "   "
    df.loc[200, "text"] = ""
    
    # Save raw
    df.to_csv("data/raw/custom_intents.csv", index=False)
    
    # Clean and save processed
    cleaned_df = clean_data(df, 'text')
    cleaned_df.to_csv("data/processed/custom_intents.csv", index=False)
    print(f"Custom intent dataset prepared: {len(cleaned_df)} rows.")

if __name__ == "__main__":
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    prepare_wisesight()
    generate_intent_dataset()
