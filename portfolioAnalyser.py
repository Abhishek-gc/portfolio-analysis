import warnings
warnings.filterwarnings("ignore")
import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import google.generativeai as genai
import json, os
from dotenv import load_dotenv
import plotly.express as px

# Configure Gemini API
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def create_sparkline(hist_data, days=90):
    """Create a sparkline chart for 3-month price movement"""
    try:
        prices = hist_data['Close'].tail(days)

        if prices.empty:
            st.error("Insufficient data to create sparkline.")
            return None

        # Convert to DataFrame so Plotly can parse it safely
        df = prices.reset_index()
        df.columns = ["Date", "Close"]

        fig = px.line(
            df,
            x="Date",
            y="Close",
            height=50,
            width=200
        )

        # Minimalist design
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0, r=0, t=0, b=0),
            showlegend=False,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False)
        )

        # Trend color
        color = 'green' if df["Close"].iloc[-1] > df["Close"].iloc[0] else 'red'
        fig.update_traces(line_color=color)

        return fig

    except Exception as e:
        st.error(f"Error creating sparkline: {e}")
        return None


def fetch_stock_data(ticker):
    """Fetch current stock data and recent performance"""
    try:
        # Get historical data for 1 year
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        
        # Add .NS if not already present
        if not ticker.endswith('.NS'):
            ticker = f"{ticker}.NS"
            
        hist_data = yf.download(ticker, start=start_date, end=end_date)
        
        if hist_data.empty:
            st.error(f"No price data found for {ticker}")
            return None, None
            
        # Calculate percentage changes
        current_price = float(hist_data['Close'].iloc[-1])
        
        # Get prices for different periods
        week_ago_price = float(hist_data['Close'].iloc[-6] if len(hist_data) > 5 else hist_data['Close'].iloc[0])
        month_ago_price = float(hist_data['Close'].iloc[-22] if len(hist_data) > 21 else hist_data['Close'].iloc[0])
        three_month_ago_price = float(hist_data['Close'].iloc[-63] if len(hist_data) > 62 else hist_data['Close'].iloc[0])
        year_ago_price = float(hist_data['Close'].iloc[0])
        
        # Calculate changes
        weekly_change = ((current_price - week_ago_price) / week_ago_price) * 100
        monthly_change = ((current_price - month_ago_price) / month_ago_price) * 100
        three_month_change = ((current_price - three_month_ago_price) / three_month_ago_price) * 100
        yearly_change = ((current_price - year_ago_price) / year_ago_price) * 100
        
        stock_data = {
            'symbol': ticker,
            'shortName': ticker.replace('.NS', ''),
            'currentPrice': current_price,
            'fiftyTwoWeekHigh': float(hist_data['High'].max()),
            'fiftyTwoWeekLow': float(hist_data['Low'].min()),
            'volume': float(hist_data['Volume'].iloc[-1]),
            'weekly_change': weekly_change,
            'monthly_change': monthly_change,
            'three_month_change': three_month_change,
            'yearly_change': yearly_change
        }
        
        return stock_data, hist_data
        
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {str(e)}")
        return None, None

def get_chatgpt_recommendation(stock_data, hist_data, buy_price, quantity):
    """Get investment recommendation from Gemini"""
    
    # Calculate key metrics
    current_price = stock_data.get('currentPrice', 0)
    price_change = ((current_price - buy_price) / buy_price) * 100
    
    # Prepare prompt for Gemini
    prompt = f"""
    You are a professional stock market analyst. Analyze this stock investment and provide a clear HOLD or EXIT recommendation.
    
    Investment Details:
    - Stock: {stock_data.get('symbol')}
    - Purchase Price: ₹{buy_price:.2f}
    - Current Price: ₹{current_price:.2f}
    - Price Change: {price_change:.2f}%
    - Quantity: {quantity}
    
    Technical Indicators:
    - 52-Week High: ₹{stock_data.get('fiftyTwoWeekHigh', 0):.2f}
    - 52-Week Low: ₹{stock_data.get('fiftyTwoWeekLow', 0):.2f}
    - Weekly Change: {stock_data.get('weekly_change', 0):.2f}%
    - Monthly Change: {stock_data.get('monthly_change', 0):.2f}%
    - 3-Month Change: {stock_data.get('three_month_change', 0):.2f}%
    - Yearly Change: {stock_data.get('yearly_change', 0):.2f}%

    Provide your analysis in this exact JSON format:
    {{
        "recommendation": "HOLD or EXIT",
        "reasons": [
            "First key reason with specific numbers and percentages",
            "Second reason focusing on technical analysis",
            "Third reason about risk/reward"
        ],
        "risks": [
            "Primary risk factor",
            "Secondary risk factor",
            "Additional consideration"
        ]
    }}

    Focus on:
    1. Price trend analysis
    2. Recent performance metrics
    3. Risk vs reward at current levels
    4. Technical support/resistance levels
    
    Use actual numbers and percentages in your analysis. Be specific and quantitative.
    """
    
    try:
        # Initialize Gemini model
        model = genai.GenerativeModel('gemini-2.5-flash-lite')
        
        # Generate response
        response = model.generate_content(prompt)
        
        # Try to extract JSON from the response
        try:
            # First attempt: direct parsing
            analysis = json.loads(response.text)
        except json.JSONDecodeError:
            # Second attempt: try to find JSON-like structure
            import re
            json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
            if json_match:
                analysis = json.loads(json_match.group())
            else:
                raise Exception("Could not parse JSON from response")
        
        return analysis
    
    except Exception as e:
        st.error(f"Error getting recommendation: {e}")
        # Return a default analysis structure
        return {
            "recommendation": "HOLD",
            "reasons": [
                "Error in analysis",
                "Using default recommendation",
                "Please try again"
            ],
            "risks": [
                "Analysis failed",
                "Data may be incomplete",
                "Manual review recommended"
            ]
        }

def display_stock_analysis(stock_data, hist_data, analysis):
    """Display stock analysis and recommendation"""
    
    # Create two columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Stock Information")
        st.write(f"**Company:** {stock_data.get('shortName')}")
        st.write(f"**Sector:** {stock_data.get('sector')}")
        st.write(f"**Current Price:** ${stock_data.get('currentPrice', 0):.2f}")
        st.write(f"**Market Cap:** ${stock_data.get('marketCap', 0)/1e9:.2f}B")
        
    with col2:
        st.subheader("Technical Indicators")
        st.write(f"**52-Week High:** ${stock_data.get('fiftyTwoWeekHigh', 0):.2f}")
        st.write(f"**52-Week Low:** ${stock_data.get('fiftyTwoWeekLow', 0):.2f}")
        st.write(f"**P/E Ratio:** {stock_data.get('forwardPE', 'N/A')}")
        st.write(f"**Volume:** {stock_data.get('volume', 0):,}")
    
    # Display recommendation
    st.subheader("AI Analysis")
    recommendation_color = "green" if analysis['recommendation'] == "HOLD" else "red"
    st.markdown(f"**Recommendation:** :{recommendation_color}[{analysis['recommendation']}]")
    
    st.write("**Key Reasons:**")
    for reason in analysis['reasons']:
        st.write(f"- {reason}")
    
    st.write("**Risks to Consider:**")
    for risk in analysis['risks']:
        st.write(f"- {risk}")


def analyze_portfolio(pnl_df):
    """Analyze entire portfolio and get recommendations for each stock"""
    portfolio_analysis = []
    
    for _, row in pnl_df.iterrows():
        ticker = row['ticker']  # Using lowercase column name from our DataFrame
        stock_data, hist_data = fetch_stock_data(ticker)
        
        if stock_data and hist_data is not None:
            # Calculate cost per share
            cost_per_share = row['buy_price']  # Using lowercase column name
            quantity = row['quantity']  # Using lowercase column name
            current_price = stock_data['currentPrice']
            current_value = stock_data['currentPrice'] * quantity
            profit_loss = current_value - row['total_cost']
            
            analysis = get_chatgpt_recommendation(stock_data, hist_data, cost_per_share, quantity)
            
            if analysis:
                portfolio_analysis.append({
                    'ticker': ticker,
                    'investment': quantity,
                    'cost': row['total_cost'],
                    'current_value': current_value,
                    'current_price': current_price,
                    'profit_loss': profit_loss,
                    'recommendation': analysis['recommendation'],
                    'reasons': analysis['reasons'],
                    'risks': analysis['risks'],
                    'weekly_change': stock_data.get('weekly_change', 0),
                    'monthly_change': stock_data.get('monthly_change', 0),
                    'three_month_change': stock_data.get('three_month_change', 0),
                    'yearly_change': stock_data.get('yearly_change', 0),
                    'hist_data': hist_data
                })
    
    return portfolio_analysis

def display_portfolio_analysis(portfolio_analysis):
    """Display analysis for entire portfolio"""
    st.header("Portfolio Analysis")
    
    # Portfolio Summary
    total_investment = sum(stock['cost'] for stock in portfolio_analysis)
    total_current_value = sum(stock['current_value'] for stock in portfolio_analysis)
    total_pnl = sum(stock['profit_loss'] for stock in portfolio_analysis)
    total_return_pct = (total_pnl / total_investment) * 100
    
    # Summary metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Investment", f"₹{total_investment:,.2f}")
    with col2:
        st.metric("Current Value", f"₹{total_current_value:,.2f}")
    with col3:
        st.metric("Overall P&L", f"₹{total_pnl:,.2f}", delta=f"{total_return_pct:.1f}%")
    
    st.write("")  # Add spacing
    
    # Individual Stock Analysis
    for stock in portfolio_analysis:
       
        with st.expander(f"{stock['ticker']} (P&L: ₹{stock['profit_loss']:,.2f})", expanded=True):
            col1, col2 = st.columns([1, 1])
            
            # Left Column - Investment Details
            with col1:
                st.subheader("Investment Details:")
                st.write(f"**Quantity:** {stock['investment']}")
                st.write(f"**Purchase price:** ₹{stock['cost']/stock['investment']:.2f}")
                st.write(f"**Current price:** ₹{stock['current_price']:,.2f}")
                st.write(f"**Current value:** ₹{stock['current_value']:,.2f}")
                
                st.write("")  # Add spacing
                
                st.subheader("Performance:")
                metrics = [
                    ("Weekly", stock['weekly_change']),
                    ("Monthly", stock['monthly_change']),
                    ("3 Months", stock['three_month_change']),
                    ("1 Year", stock.get('yearly_change', 0))
                ]
                
                for label, value in metrics:
                    color = "green" if value > 0 else "red"
                    st.write(f"{label}: :{color}[{value:.2f}%]")
                
                # Add Sparkline for the last 3 months
                st.subheader("3-Month Price Movement")
                sparkline_fig = create_sparkline(stock['hist_data'], days=63)  # 63 trading days approx. for 3 months
                if sparkline_fig:
                    st.plotly_chart(sparkline_fig, width=500, height=300)  # Set height for small sparkline
                
            # Right Column - AI Analysis
            with col2:
                st.subheader("AI Analysis:")
                recommendation_color = "green" if stock['recommendation'] == "HOLD" else "red"
                st.write(f"**Recommendation:** :{recommendation_color}[{stock['recommendation']}]")
                
                st.write("")  # Add spacing
                
                st.write("**Key Reasons:**")
                for reason in stock['reasons']:
                    st.write(f"○ {reason}")
                
                st.write("")  # Add spacing
                
                st.write("**Risks:**")
                for risk in stock['risks']:
                    st.write(f"○ {risk}")

@st.cache_data
def load_nifty500_tickers():
    """Load Nifty 500 tickers from CSV file"""
    try:
        # Load the CSV file directly
        df = pd.read_csv('nifty500_Tickers.csv')
        
        # Create a dictionary with format: {Display Name: Ticker}
        return {f"{row['Symbol']} - {row['Ticker']}": row['Ticker'] for _, row in df.iterrows()}
    
    except FileNotFoundError:
        st.error("Could not find nifty500_Tickers.csv. Please ensure the file is in the correct directory.")
        return {}
    except Exception as e:
        st.error(f"Error loading Nifty 500 tickers: {str(e)}")
        return {}

def main():
    st.set_page_config(layout="wide")  # Set wide layout at the start
    st.title("🤖 Portfolio Advisor")
    
    # Initialize session state for portfolio and form inputs
    if 'portfolio' not in st.session_state:
        st.session_state.portfolio = []
    
    # Portfolio Input Section
    st.subheader("Add Stock to Portfolio")
    
    # Load Nifty 500 tickers for dropdown
    tickers_dict = load_nifty500_tickers()
    
    if tickers_dict:
        # Adjust column widths for better spacing
        col1, col2, col3 = st.columns([3, 1, 1])
        
        with col1:
            selected_stock = st.selectbox(
                "Select Stock",
                options=[""] + list(tickers_dict.keys()),
                format_func=lambda x: "Choose a stock..." if x == "" else x
            )
        
        with col2:
            buy_price = st.number_input(
                "Purchase Price (₹)", 
                min_value=1,
            
            )
            
        with col3:
            quantity = st.number_input(
                "Quantity", 
                min_value=1, 
                step=1
            )
        
        # Center the Add button
        if st.button("Add to Portfolio", use_container_width=True):
            if selected_stock and selected_stock != "":
                new_stock = {
                    'ticker': tickers_dict[selected_stock],
                    'symbol': selected_stock,
                    'quantity': quantity,
                    'buy_price': buy_price,
                    'total_cost': quantity * buy_price
                }
                st.session_state.portfolio.append(new_stock)
                st.success(f"Added {selected_stock} to portfolio")

    # Display Current Portfolio
    if st.session_state.portfolio:
        st.divider()  # Add visual separation
        st.subheader("Your Portfolio")
        
        # Convert portfolio to DataFrame for display
        df = pd.DataFrame(st.session_state.portfolio)
        
        # Display the portfolio with better formatting
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )
        
        # Center the Analyze button
        if st.button("Analyze Portfolio", use_container_width=True):
            with st.spinner("Analyzing your portfolio..."):
                portfolio_analysis = analyze_portfolio(df)
                display_portfolio_analysis(portfolio_analysis)
                
    else:
        st.info("Add stocks to your portfolio to get started")

if __name__ == "__main__":
    main() 