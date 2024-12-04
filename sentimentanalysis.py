import streamlit as st
import requests
import json
import pandas as pd
import numpy as np
from transformers import pipeline
import plotly.graph_objects as go
from datetime import datetime, timedelta
import dateutil.parser
import yfinance as yf
from fuzzywuzzy import fuzz
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# Set page config for a more professional look
st.set_page_config(page_title="Financial News Sentiment Analysis", layout="wide")

# Custom CSS for black and white theme
st.markdown("""
<style>
    body {
        color: white;
        background-color: black;
    }
    .stApp {
        background-color: black;
    }
    .stButton>button {
        color: black;
        background-color: white;
        border: 1px solid white;
    }
    .stTextInput>div>div>input {
        color: white;
        background-color: black;
        border: 1px solid white;
    }
    .stSelectbox>div>div>select {
        color: white;
        background-color: black;
        border: 1px solid white;
    }
</style>
""", unsafe_allow_html=True)

EXCLUDED_TERMS = ['BSE', 'NSE', 'SENSEX', 'NIFTY']

@st.cache_data(ttl=3600)
def fetch_news_from_newsapi(days=7):
    api_key = "a797da8e4c264cf3baeafc36afbef763"
    base_url = "https://newsapi.org/v2/everything"
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    params = {
        'q': '(stock OR market OR finance) AND (India OR BSE OR NSE)',
        'language': 'en',
        'from': start_date.isoformat(),
        'to': end_date.isoformat(),
        'sortBy': 'publishedAt',
        'apiKey': api_key,
        'pageSize': 100  # Fetch only one page of results
    }
    
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        return response.json()['articles']
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 426:
            st.warning("NewsAPI free tier limit reached. Consider upgrading for more results.")
        else:
            st.error(f"Error fetching news from NewsAPI: {e}")
        return []
    except Exception as e:
        st.error(f"Unexpected error fetching news from NewsAPI: {e}")
        return []

@st.cache_data(ttl=3600)
def fetch_news_from_finnhub(days=7):
    api_key = "cr807lpr01qotnb4a490cr807lpr01qotnb4a49g"
    base_url = "https://finnhub.io/api/v1/news"
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    params = {
        'category': 'general',
        'token': api_key,
        'from': start_date.strftime('%Y-%m-%d'),
        'to': end_date.strftime('%Y-%m-%d')
    }
    
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        articles = response.json()
        return articles[:min(len(articles), 1000)]  # Increase to up to 1000 articles if available
    except Exception as e:
        st.error(f"Error fetching news from Finnhub: {e}")
        return []

def fetch_news(days=7):
    newsapi_articles = fetch_news_from_newsapi(days)
    finnhub_articles = fetch_news_from_finnhub(days)
    
    combined_news = []
    
    for article in newsapi_articles:
        combined_news.append({
            'Title': article['title'],
            'Source': article['source']['name'],
            'URL': article['url'],
            'Date': article['publishedAt']
        })
    
    for article in finnhub_articles:
        combined_news.append({
            'Title': article['headline'],
            'Source': article['source'],
            'URL': article['url'],
            'Date': article['datetime']
        })
    
    return combined_news

@st.cache_resource
def load_sentiment_model(use_gpu):
    device = 0 if use_gpu and torch.cuda.is_available() else -1
    return pipeline("sentiment-analysis", model="ProsusAI/finbert", device=device)

def analyze_sentiment(text, model):
    result = model(text)[0]
    return result['label'], result['score']

def parse_date(date_str):
    try:
        return dateutil.parser.parse(date_str).date()
    except (ValueError, TypeError):
        return datetime.now().date()

def categorize_sentiment(score, mean, std):
    if std == 0:  # Avoid division by zero
        return "Hold"
    z_score = (score - mean) / std
    if z_score > 0.67:  # Top 25%
        return "Buy"
    elif z_score < -0.67:  # Bottom 25%
        return "Sell"
    else:  # Middle 50%
        return "Hold"

@st.cache_data(ttl=24*3600)
def fetch_indian_stocks():
    # Fetch Nifty 500 (includes large and mid-cap)
    nifty500 = pd.read_csv("https://archives.nseindia.com/content/indices/ind_nifty500list.csv")
    symbols = nifty500['Symbol'].tolist()
    
    stock_dict = {}
    with st.spinner("Fetching stock information..."):
        for symbol in symbols:
            try:
                stock = yf.Ticker(f"{symbol}.NS")
                info = stock.info
                stock_dict[symbol] = {
                    'name': info.get('longName', ''),
                    'sector': info.get('sector', ''),
                    'industry': info.get('industry', '')
                }
            except Exception as e:
                st.warning(f"Error fetching info for {symbol}: {e}")
    
    return stock_dict

def match_stock(text, stocks):
    words = text.split()
    matches = []
    
    for symbol, info in stocks.items():
        # Check for exact symbol match
        if symbol in words:
            # Additional check for BSE and NSE
            if symbol in EXCLUDED_TERMS:
                # Check if it's not just "BSE" or "NSE" alone
                index = words.index(symbol)
                if index > 0 and words[index-1].lower() in ['the', 'on', 'in', 'at']:
                    continue  # Skip this match as it's likely referring to the exchange
            matches.append(symbol)
        
        # Check for company name match
        company_name = info['name'].lower()
        text_lower = text.lower()
        if company_name in text_lower:
            matches.append(symbol)
        
        # Check for partial matches (for company names with multiple words)
        name_parts = company_name.split()
        if len(name_parts) > 1:
            for part in name_parts:
                if len(part) > 3 and part in text_lower:  # Avoid matching very short words
                    matches.append(symbol)
                    break
    
    if not matches:
        print(f"No matches found for text: {text}")
    else:
        print(f"Matches found for text: {text}")
        print(f"Matched stocks: {matches}")
    
    return list(set(matches))  # Remove duplicates

def main():
    st.title("Financial News Sentiment Analysis")

    # Sidebar for filters
    st.sidebar.header("Filters")
    end_date = st.sidebar.date_input("End date", datetime.now().date())
    days_range = st.sidebar.slider("Number of days to include", 1, 30, 7)
    start_date = end_date - timedelta(days=days_range-1)
    sources = st.sidebar.multiselect("Filter by source", ["All Sources"])
    use_gpu = st.sidebar.checkbox("Use GPU (if available)", value=True)

    # Fetch Indian stocks
    with st.spinner("Fetching Indian stocks..."):
        indian_stocks = fetch_indian_stocks()
    
    if not indian_stocks:
        st.error("Failed to fetch Indian stocks. Please try again later.")
        return

    st.write("Sample of fetched stocks:")
    st.write(dict(list(indian_stocks.items())[:10]))  # Display first 10 stocks

    # Fetch news
    with st.spinner("Fetching news data..."):
        news_data = fetch_news(days_range)
    
    if news_data:
        st.info(f"Fetched {len(news_data)} news articles.")
        with st.spinner("Analyzing sentiment..."):
            model = load_sentiment_model(use_gpu)

            # Process news and calculate sentiment scores
            processed_news = []
            for item in news_data:
                title = item.get('Title', 'No title')
                source = item.get('Source', 'Unknown')
                url = item.get('URL', '#')
                date = parse_date(item.get('Date', ''))
                
                if start_date <= date <= end_date and (not sources or "All Sources" in sources or source in sources):
                    matched_stocks = match_stock(title, indian_stocks)
                    for stock in matched_stocks:
                        sentiment, score = analyze_sentiment(title, model)
                        processed_news.append({
                            'Title': title,
                            'Source': source,
                            'URL': url,
                            'Date': date,
                            'Sentiment': sentiment,
                            'Score': score,
                            'Stock': stock
                        })

            st.info(f"Processed {len(processed_news)} news articles related to specific stocks.")

            if processed_news:
                df = pd.DataFrame(processed_news)
                
                # Display unique stocks found
                unique_stocks = df['Stock'].nunique()
                st.info(f"Found news for {unique_stocks} unique stocks.")

                # Calculate mean and standard deviation of scores
                mean_score = df['Score'].mean()
                std_score = df['Score'].std()

                # Categorize stocks based on Gaussian distribution
                df['Category'] = df['Score'].apply(lambda x: categorize_sentiment(x, mean_score, std_score))

                # Display news in a table
                st.dataframe(df)

                # Sentiment distribution chart
                st.subheader("Sentiment Distribution")
                fig = go.Figure(data=[go.Pie(labels=df['Sentiment'].value_counts().index, 
                                             values=df['Sentiment'].value_counts().values,
                                             hole=.3)])
                fig.update_layout(paper_bgcolor="black", plot_bgcolor="black", font_color="white")
                st.plotly_chart(fig)

                # Stock categorization table
                st.header("Top 10 Stocks by Sentiment Category")
                
                # Group by stock and calculate average sentiment score
                stock_sentiment = df.groupby('Stock').agg({
                    'Score': 'mean',
                    'Category': lambda x: x.mode().iloc[0] if not x.empty else 'Unknown'  # Most common category
                }).reset_index()

                # Recategorize based on average scores
                stock_sentiment['Category'] = stock_sentiment['Score'].apply(
                    lambda x: categorize_sentiment(x, mean_score, std_score)
                )

                # Get top 10 for each category
                buy_stocks = stock_sentiment[stock_sentiment['Category'] == 'Buy'].sort_values('Score', ascending=False).head(10)
                hold_stocks = stock_sentiment[stock_sentiment['Category'] == 'Hold'].sort_values('Score', ascending=False).head(10)
                sell_stocks = stock_sentiment[stock_sentiment['Category'] == 'Sell'].sort_values('Score', ascending=True).head(10)

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.subheader("Buy")
                    st.table(buy_stocks[['Stock', 'Score']])
                with col2:
                    st.subheader("Hold")
                    st.table(hold_stocks[['Stock', 'Score']])
                with col3:
                    st.subheader("Sell")
                    st.table(sell_stocks[['Stock', 'Score']])

                # Stock-specific sentiment analysis
                st.header("Stock-specific Sentiment")
                stock_name = st.text_input("Enter stock name:")
                if stock_name:
                    relevant_news = df[df['Stock'].str.contains(stock_name, case=False)]
                    if not relevant_news.empty:
                        st.dataframe(relevant_news)
                        avg_sentiment = relevant_news['Score'].mean()
                        st.metric("Average Sentiment Score", f"{avg_sentiment:.2f}")
                    else:
                        st.write(f"No news found for {stock_name}")

                # Export data
                if st.button("Export Data"):
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name="financial_news_sentiment.csv",
                        mime="text/csv",
                    )
            else:
                st.warning("No news articles found for the selected date range and sources.")
    else:
        st.error("Failed to fetch news data. Please check the error messages above and try again later.")

if __name__ == "__main__":
    main()