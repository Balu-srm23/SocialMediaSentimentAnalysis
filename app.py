import streamlit as st
import pandas as pd
import re
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# Custom CSS for modern styling
st.markdown("""
    <style>
    .main {background-color: #f0f2f6;}
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .header {
        font-size: 24px;
        font-weight: bold;
        color: #333;
        margin-bottom: 10px;
    }
    .help-text {
        font-size: 14px;
        color: #555;
        margin-top: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# App title and description
st.title("Social Media Sentiment Analysis Dashboard")
st.markdown("Analyze sentiments of social media posts for brand monitoring using TextBlob. Explore dataset insights, adjust thresholds, and learn about the project.")

# Sidebar for inputs
st.sidebar.header("Input Settings")
input_option = st.sidebar.radio("Input Type", ["Single Post", "Batch Posts"])
platform_input = st.sidebar.selectbox("Select Platform", ["All", "X", "Instagram"])
pos_threshold = st.sidebar.slider("Positive Sentiment Threshold", 0.0, 1.0, 0.1, help="Set the polarity score above which a post is Positive.")
neg_threshold = st.sidebar.slider("Negative Sentiment Threshold", -1.0, 0.0, -0.1, help="Set the polarity score below which a post is Negative.")

# Threshold help section
with st.sidebar.expander("How to Adjust Thresholds"):
    st.markdown("""
    <div class="help-text">
    <b>Understanding Sentiment Thresholds</b><br>
    - <b>Polarity Score</b>: Ranges from -1 (very negative) to 1 (very positive), with 0 being neutral.<br>
    - <b>Positive Threshold</b>: Posts with polarity above this value are classified as Positive.<br>
    - <b>Negative Threshold</b>: Posts with polarity below this value are classified as Negative.<br>
    - <b>Neutral</b>: Posts with polarity between the thresholds are Neutral.<br>
    <br>
    <b>Example</b>:<br>
    - Post: "I love this brand!" → Polarity: ~0.5<br>
      - Thresholds (0.1, -0.1): Positive<br>
      - Thresholds (0.3, -0.3): Neutral<br>
    - Post: "Terrible service!" → Polarity: ~-0.5<br>
      - Thresholds (0.1, -0.1): Negative<br>
      - Thresholds (0.3, -0.3): Negative<br>
    <b>Tip</b>: Increase Positive threshold for stricter Positive classification; decrease Negative threshold for stricter Negative classification.
    </div>
    """, unsafe_allow_html=True)

# Load dataset for insights
@st.cache_data
def load_dataset():
    try:
        df = pd.read_csv('synthetic_social_media_dataset.csv')
        return df
    except FileNotFoundError:
        st.warning("Dataset not found. Some features may be disabled.")
        return None

dataset = load_dataset()

# Clean text function
def clean_text(text):
    text = re.sub(r'http\S+|www\S+', '', text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r'#\w+|\@\w+', '', text)  # Remove hashtags and mentions
    return text.lower().strip()

# Sentiment analysis function
def get_sentiment(text, pos_threshold, neg_threshold):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > pos_threshold:
        return 'Positive', polarity
    elif polarity < neg_threshold:
        return 'Negative', polarity
    else:
        return 'Neutral', polarity

# Main content with tabs
tab1, tab2, tab3, tab4 = st.tabs(["Sentiment Analysis", "Dataset Insights", "About", "Developed By"])

with tab1:
    st.markdown('<div class="header">Sentiment Analysis</div>', unsafe_allow_html=True)
    
    if input_option == "Single Post":
        text_input = st.text_area("Enter a social media post:", placeholder="Type your post here, e.g., 'Love this brand!'")
        if st.button("Analyze Sentiment"):
            if text_input.strip():
                cleaned_text = clean_text(text_input)
                sentiment, polarity = get_sentiment(cleaned_text, pos_threshold, neg_threshold)
                
                # Display results in a card
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.write(f"**Sentiment**: {sentiment}")
                st.write(f"**Polarity Score**: {polarity:.4f}")
                st.write(f"**Platform**: {platform_input}")
                st.write(f"**Post Length**: {len(cleaned_text)} characters")
                st.write(f"**Cleaned Text**: {cleaned_text}")
                
                # Bar chart
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.barplot(x=['Positive', 'Negative', 'Neutral'],
                           y=[1 if sentiment == 'Positive' else 0,
                              1 if sentiment == 'Negative' else 0,
                              1 if sentiment == 'Neutral' else 0],
                           palette=['#4CAF50', '#F44336', '#FFC107'])
                ax.set_title("Predicted Sentiment")
                ax.set_ylabel("Score")
                st.pyplot(fig)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # TF-IDF features
                try:
                    tfidf = joblib.load('tfidf_vectorizer.pkl')
                    tfidf_features = tfidf.transform([cleaned_text]).toarray()
                    st.write("**TF-IDF Features**: Extracted successfully.")
                except FileNotFoundError:
                    st.warning("TF-IDF vectorizer not found.")
            else:
                st.error("Please enter a post.")
    
    else:  # Batch Posts
        uploaded_file = st.file_uploader("Upload a CSV with posts (column: 'Post_Text')", type="csv")
        if uploaded_file and st.button("Analyze Batch"):
            batch_df = pd.read_csv(uploaded_file)
            if 'Post_Text' in batch_df.columns:
                batch_df['Cleaned_Text'] = batch_df['Post_Text'].apply(clean_text)
                batch_df['Sentiment'], batch_df['Polarity'] = zip(*batch_df['Cleaned_Text'].apply(
                    lambda x: get_sentiment(x, pos_threshold, neg_threshold)))
                batch_df['Platform'] = platform_input
                batch_df['Post_Length'] = batch_df['Cleaned_Text'].apply(len)
                
                # Display results
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.write("**Batch Analysis Results**")
                st.dataframe(batch_df[['Post_Text', 'Sentiment', 'Polarity', 'Platform', 'Post_Length']])
                
                # Sentiment distribution
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.countplot(x='Sentiment', data=batch_df, palette=['#4CAF50', '#F44336', '#FFC107'])
                ax.set_title("Sentiment Distribution of Batch Posts")
                st.pyplot(fig)
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.error("CSV must contain a 'Post_Text' column.")

with tab2:
    st.markdown('<div class="header">Dataset Insights</div>', unsafe_allow_html=True)
    if dataset is not None:
        # Filter by platform
        filtered_df = dataset if platform_input == "All" else dataset[dataset['Platform'] == platform_input]
        
        # Dataset summary
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.write("**Dataset Summary**")
        st.write(f"Total Posts: {len(filtered_df)}")
        st.write(f"Sentiment Counts: {filtered_df['Sentiment_Label'].value_counts().to_dict()}")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Display EDA plots
        plot_files = [
            'sentiment_by_platform.png',
            'engagement_by_sentiment.png',
            'post_length_by_sentiment.png',
            'sentiment_trends_over_time.png',
            'top_hashtags_by_sentiment.png'
        ]
        for plot in plot_files:
            if os.path.exists(plot):
                st.markdown(f'<div class="card"><b>{plot.replace(".png", "").replace("_", " ").title()}</b></div>', unsafe_allow_html=True)
                st.image(plot)
            else:
                st.warning(f"{plot} not found. Run the notebook to generate it.")
    else:
        st.error("Dataset not available for insights.")

with tab3:
    st.markdown('<div class="header">About</div>', unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.write("""
    **Project Overview**  
    This application performs real-time sentiment analysis on social media posts to monitor brand perception. It uses TextBlob for sentiment classification and supports both single and batch post analysis. The dataset includes 50,000 synthetic posts from platforms like X and Instagram, with features such as post text, engagement metrics, and hashtags.  

    **Key Features**  
    - Sentiment prediction (Positive, Negative, Neutral) based on adjustable thresholds.  
    - Batch analysis via CSV upload.  
    - Interactive dataset visualizations (e.g., sentiment trends, hashtag frequencies).  
    - Developed as part of an internship project at Edubot Technologies.  

    **Methodology**  
    - Text preprocessing: Remove URLs, hashtags, and mentions.  
    - Sentiment analysis: TextBlob polarity scores (-1 to 1).  
    - Visualizations: Generated using Pandas, Matplotlib, and Seaborn in the accompanying notebook.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

with tab4:
    st.markdown('<div class="header">Developed By</div>', unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.write("""
    **Developer**: Balaiah Dongala  
    **Institution**: SRM University, AP.  
    **Contact**: Email: balaiah_dongala@srmap.edu.in
             Linkedin: https://in.linkedin.com/in/balaiah-dongala-624b82290  

    This project was developed as part of an internship at Edubot Technologies to build a tool for real-time social media sentiment analysis. Special thanks to the open-source community for tools like Streamlit and TextBlob.
    """)
    st.markdown('</div>', unsafe_allow_html=True)