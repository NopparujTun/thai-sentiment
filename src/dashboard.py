import streamlit as st
import pandas as pd
import plotly.express as px
import time
import sys
import os

# Add the project root to sys.path so we can import src modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.predict_sentiment import SentimentPredictor
from src.predict_intent import IntentPredictor

# Page config
st.set_page_config(page_title="Thai Customer Feedback Analyzer", layout="wide", page_icon="🇹🇭")

# Custom CSS for premium aesthetics
st.markdown("""
<style>
    /* Global styles */
    .stApp {
        background-color: #0E1117;
        font-family: 'Inter', 'Roboto', sans-serif;
    }
    
    /* Sleek typography */
    h1, h2, h3 {
        background: -webkit-linear-gradient(45deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }

    /* Glassmorphism Metric Cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 20px;
        padding: 25px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        text-align: center;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        transition: transform 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275), box-shadow 0.3s ease;
        margin-bottom: 20px;
    }
    .metric-card:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 15px 40px 0 rgba(78, 205, 196, 0.2);
    }
    .metric-value {
        font-size: 3rem;
        font-weight: 900;
        color: #ffffff;
        text-shadow: 0 0 20px rgba(255,255,255,0.1);
    }
    .metric-label {
        color: #A0AEC0;
        font-size: 1.1rem;
        font-weight: 600;
        margin-top: 10px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Enhance the file uploader */
    .stFileUploader > div > div {
        background-color: rgba(255, 255, 255, 0.02) !important;
        border: 1px dashed rgba(78, 205, 196, 0.5) !important;
        border-radius: 15px !important;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    return SentimentPredictor(), IntentPredictor()

st.title("✨ Thai Customer Feedback Analyzer")
st.markdown("**Automated NLP pipeline for classifying sentiment and intent categories from Thai text.**")

uploaded_file = st.file_uploader("Upload CSV containing customer reviews", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # Identify text column
    text_col = None
    for col in ['review_text', 'text', 'texts', 'review']:
        if col in df.columns:
            text_col = col
            break
            
    if not text_col:
        st.error("❌ Could not find a valid text column. Expected one of: 'review_text', 'text', 'texts', 'review'.")
    else:
        st.success(f"✅ Loaded dataset with **{len(df):,}** reviews using column `{text_col}`.")
        
        # Cap at 5000 rows as per PRD "handle up to 5000 reviews"
        if len(df) > 5000:
            st.warning("⚠️ Dataset exceeds 5,000 rows. Truncating to 5,000 for dashboard performance.")
            df = df.head(5000)
            
        st.dataframe(df.head())
        
        if st.button("🚀 Run Analysis", use_container_width=True):
            with st.spinner("Loading NLP Models (WangchanBERTa)..."):
                sentiment_predictor, intent_predictor = load_models()
                
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            results = []
            start_time = time.time()
            total_rows = len(df)
            
            for i, text in enumerate(df[text_col]):
                if pd.isna(text):
                    results.append({"review_text": text, "sentiment": "Unknown", "sentiment_score": 0.0, "intent": "Unknown", "intent_score": 0.0})
                else:
                    s_label, s_score = sentiment_predictor.predict(str(text))
                    i_label, i_score = intent_predictor.predict(str(text))
                    results.append({"review_text": text, "sentiment": s_label, "sentiment_score": s_score, "intent": i_label, "intent_score": i_score})
                
                # Update progress smoothly
                if i % 10 == 0 or i == total_rows - 1:
                    progress_bar.progress((i + 1) / total_rows)
                    status_text.markdown(f"**Processing row {i+1} of {total_rows}...**")
                    
            elapsed = time.time() - start_time
            out_df = pd.DataFrame(results)
            
            status_text.empty()
            progress_bar.empty()
            st.toast(f"Analysis complete! Processed {total_rows:,} reviews.", icon="🎉")
            
            st.markdown("---")
            st.subheader("📊 Analytics Dashboard")
            
            # Metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f'<div class="metric-card"><div class="metric-value">{total_rows:,}</div><div class="metric-label">Total Reviews</div></div>', unsafe_allow_html=True)
            with col2:
                neg_count = len(out_df[out_df["sentiment"] == "Negative"])
                st.markdown(f'<div class="metric-card"><div class="metric-value" style="color:#FF6B6B;">{neg_count:,}</div><div class="metric-label">Negative Reviews</div></div>', unsafe_allow_html=True)
            with col3:
                most_common_intent = out_df["intent"].mode()[0] if not out_df["intent"].empty else "N/A"
                st.markdown(f'<div class="metric-card"><div class="metric-value" style="font-size: 2rem; padding-top: 15px;">{most_common_intent}</div><div class="metric-label">Top Intent</div></div>', unsafe_allow_html=True)
                
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Charts
            chart_col1, chart_col2 = st.columns(2)
            
            with chart_col1:
                # Sentiment Distribution
                sentiment_counts = out_df["sentiment"].value_counts().reset_index()
                sentiment_counts.columns = ["Sentiment", "Count"]
                # Map colors consistently
                color_map = {"Positive": "#4ECDC4", "Negative": "#FF6B6B", "Neutral": "#FFE66D", "Unknown": "#A0AEC0"}
                fig_sent = px.pie(sentiment_counts, values="Count", names="Sentiment", title="Sentiment Distribution",
                                 color="Sentiment", color_discrete_map=color_map, hole=0.5)
                fig_sent.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#FFF"))
                fig_sent.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=16)
                st.plotly_chart(fig_sent, use_container_width=True)
                
            with chart_col2:
                # Intent Distribution
                intent_counts = out_df["intent"].value_counts().reset_index()
                intent_counts.columns = ["Intent", "Count"]
                fig_intent = px.bar(intent_counts, x="Count", y="Intent", title="Intent Categories Breakdown", orientation='h',
                                  color="Intent", color_discrete_sequence=px.colors.qualitative.Pastel)
                fig_intent.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#FFF"), showlegend=False)
                st.plotly_chart(fig_intent, use_container_width=True)
                
            st.subheader("📋 Predicted Data")
            st.dataframe(out_df, use_container_width=True)
            
            # Download
            csv = out_df.to_csv(index=False).encode('utf-8')
            st.download_button("💾 Download Predictions CSV", data=csv, file_name="predictions.csv", mime="text/csv", type="primary")

