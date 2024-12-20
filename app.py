import streamlit as st
import pandas as pd
import torch
from sentiment_analyzer import RobertaSentimentAnalyzer  # Import from your existing code

def load_model():
    """Load the saved model"""
    try:
        return RobertaSentimentAnalyzer.load_model('./saved_model')
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def analyze_sentiment(analyzer, text):
    """Analyze sentiment of given text"""
    try:
        prediction = analyzer.predict([text])[0]
        sentiment_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
        return sentiment_map[prediction]
    except Exception as e:
        st.error(f"Error analyzing sentiment: {str(e)}")
        return None

def get_confidence(analyzer, text):
    """Get prediction confidence"""
    try:
        analyzer.model.eval()
        encoding = analyzer.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=128,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        with torch.no_grad():
            input_ids = encoding['input_ids'].to(analyzer.device)
            attention_mask = encoding['attention_mask'].to(analyzer.device)
            outputs = analyzer.model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
            return float(torch.max(probs).item() * 100)
    except Exception as e:
        st.error(f"Error calculating confidence: {str(e)}")
        return None

def main():
    st.set_page_config(
        page_title="Brand Review Sentiment Analyzer",
        page_icon="👟",
        layout="wide"
    )
    
    st.title("👟 Brand Review Sentiment Analyzer")
    st.write("Analyze sentiment of shoe brand reviews using RoBERTa")
    
    # Initialize session state
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = load_model()
    
    if st.session_state.analyzer is None:
        st.error("Failed to load model. Please ensure the model is saved in './saved_model' directory")
        return
    
    # Sidebar
    st.sidebar.title("About")
    st.sidebar.info(
        "This app uses a fine-tuned RoBERTa model to analyze "
        "sentiment in shoe brand reviews. The model classifies "
        "reviews as Positive, Neutral, or Negative."
    )
    
    # Main content
    tab1, tab2 = st.tabs(["Single Review Analysis", "Batch Analysis"])
    
    with tab1:
        st.header("Single Review Analysis")
        
        # Brand selection
        brand = st.selectbox(
            "Select Brand",
            ["Nike", "Adidas", "Puma"],
            key="single_brand"
        )
        
        # Review input
        review_text = st.text_area(
            "Enter your review",
            height=100,
            placeholder="Type your review here..."
        )
        
        if st.button("Analyze Sentiment"):
            if review_text.strip():
                with st.spinner("Analyzing sentiment..."):
                    sentiment = analyze_sentiment(st.session_state.analyzer, review_text)
                    confidence = get_confidence(st.session_state.analyzer, review_text)
                    
                    if sentiment and confidence:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric("Sentiment", sentiment)
                        
                        with col2:
                            st.metric("Confidence", f"{confidence:.1f}%")
                        
                        # Display color-coded box based on sentiment
                        sentiment_colors = {
                            "Positive": "success",
                            "Neutral": "warning",
                            "Negative": "error"
                        }
                        st.markdown(
                            f'''
                            <div style="padding: 20px; border-radius: 5px; background-color: {'#00ff00' if sentiment == 'Positive' else '#ff0000' if sentiment == 'Negative' else '#ffff00'}; opacity: 0.3;">
                                <h3 style="color: black;">{sentiment} Review</h3>
                                <p style="color: black;">{review_text}</p>
                            </div>
                            ''',
                            unsafe_allow_html=True
                        )
            else:
                st.warning("Please enter a review text")
    
    with tab2:
        st.header("Batch Analysis")
        
        uploaded_file = st.file_uploader(
            "Upload CSV file with reviews",
            type=['csv'],
            help="CSV file should have 'brand' and 'content' columns"
        )
        
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                if 'brand' not in df.columns or 'content' not in df.columns:
                    st.error("CSV must contain 'brand' and 'content' columns")
                    return
                
                if st.button("Analyze Batch"):
                    with st.spinner("Analyzing reviews..."):
                        # Analyze sentiments
                        predictions = st.session_state.analyzer.predict(df['content'].tolist())
                        df['sentiment'] = [
                            {0: 'Negative', 1: 'Neutral', 2: 'Positive'}[p]
                            for p in predictions
                        ]
                        
                        # Display results
                        st.subheader("Results Summary")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("Overall Sentiment Distribution")
                            st.bar_chart(df['sentiment'].value_counts())
                        
                        with col2:
                            st.write("Sentiment by Brand")
                            st.table(pd.crosstab(df['brand'], df['sentiment']))
                        
                        # Display detailed results
                        st.subheader("Detailed Results")
                        st.dataframe(df)
                        
                        # Download results
                        csv = df.to_csv(index=False)
                        st.download_button(
                            "Download Results",
                            csv,
                            "sentiment_analysis_results.csv",
                            "text/csv",
                            key='download-csv'
                        )
                        
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")

if __name__ == "__main__":
    main()