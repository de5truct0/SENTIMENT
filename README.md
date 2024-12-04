# Financial News Sentiment Analysis App

## Overview
This Streamlit app provides financial sentiment analysis for Indian stocks based on news articles fetched from **NewsAPI** and **Finnhub**. The app uses the **FinBERT model** (via Hugging Face's `transformers` library) to determine the sentiment of news articles and categorizes stocks into "Buy," "Hold," or "Sell" recommendations based on a Gaussian distribution of sentiment scores.

---

## Features

1. **Financial News Fetching**:
   - Fetches the latest financial news using NewsAPI and Finnhub.
   - Filters articles based on date range and specific sources.

2. **Sentiment Analysis**:
   - Leverages the FinBERT model to perform sentiment analysis on news headlines.
   - Supports GPU acceleration for faster processing.

3. **Stock Matching**:
   - Matches news articles to Indian stocks using company names and symbols from the **NIFTY 500** list.

4. **Stock Categorization**:
   - Categorizes stocks into "Buy," "Hold," or "Sell" based on sentiment scores.
   - Visualizes sentiment distribution using interactive charts.

5. **Stock-specific Insights**:
   - Allows users to search for specific stocks and view associated sentiment and news.

6. **Data Export**:
   - Provides an option to export processed data as a CSV file.

---

## Requirements

### Python Libraries
Ensure the following Python libraries are installed:
- `streamlit`
- `requests`
- `pandas`
- `numpy`
- `transformers`
- `torch`
- `dateutil`
- `yfinance`
- `fuzzywuzzy`
- `plotly`

### API Keys
- **NewsAPI**: Obtain an API key from [NewsAPI](https://newsapi.org/).
- **Finnhub**: Obtain an API key from [Finnhub](https://finnhub.io/).

Add your API keys directly in the `fetch_news_from_newsapi` and `fetch_news_from_finnhub` functions.

---

## How to Run the App

1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Install Dependencies**:
   Install required Python libraries using `pip`:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the App**:
   Execute the Streamlit app:
   ```bash
   streamlit run app.py
   ```

4. **Access the App**:
   Open the link displayed in your terminal (default: `http://localhost:8501`).

---

## File Structure

- `app.py`: Main application file containing the Streamlit code.
- `requirements.txt`: List of required Python libraries.
- `README.md`: Documentation for the app.

---

## Customization

1. **Adjust Date Ranges**:
   - Modify the default date range in the sidebar slider.

2. **Filter by Sources**:
   - Update the default sources in the `sources` dropdown.

3. **Stock List**:
   - Replace the **NIFTY 500** stock list with your own list by updating the `fetch_indian_stocks` function.

4. **Styling**:
   - Customize the app's appearance by modifying the custom CSS in the `st.markdown` block.

---

## Limitations

- **API Limitations**:
  - NewsAPI and Finnhub have rate limits on free-tier accounts. Upgrade your plans for more extensive usage.
  
- **Stock Matching**:
  - Matches are based on text similarity and might not always perfectly align with stocks.

- **Sentiment Model**:
  - The FinBERT model is tuned for general financial sentiment and may not capture all nuances.

---

## Future Enhancements

- Integrate real-time stock price data and sentiment correlation.
- Add support for additional news APIs (e.g., Bloomberg, Reuters).
- Improve stock-matching logic with enhanced fuzzy matching.
- Provide more robust visualizations and dashboards.

---

## Contact
For questions or issues, please raise a ticket on the repository or contact the developer.
