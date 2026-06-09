import pandas as pd
from src.preprocessing import remove_urls, normalize_whitespace, clean_data

def test_remove_urls():
    text = "เช็คของได้ที่ https://example.com/track นะคะ"
    assert remove_urls(text).strip() == "เช็คของได้ที่  นะคะ"

def test_normalize_whitespace():
    text = "ของ   มา ถึง    แล้ว"
    assert normalize_whitespace(text) == "ของ มา ถึง แล้ว"

def test_clean_data():
    data = {
        "text": [
            "  ของสวยมากค่ะ  ",
            "ดูรายละเอียด https://shopee.co.th",
            "แย่มาก   ส่งช้า",
            "  ของสวยมากค่ะ  ", # Duplicate after cleaning
            "", # Empty
            None # Missing
        ],
        "label": [1, 2, 0, 1, 2, 0]
    }
    df = pd.DataFrame(data)
    cleaned_df = clean_data(df, "text")
    
    # Expected: "ของสวยมากค่ะ", "ดูรายละเอียด", "แย่มาก ส่งช้า"
    assert len(cleaned_df) == 3
    assert cleaned_df.iloc[0]["text"] == "ของสวยมากค่ะ"
    assert cleaned_df.iloc[1]["text"] == "ดูรายละเอียด"
    assert cleaned_df.iloc[2]["text"] == "แย่มาก ส่งช้า"
