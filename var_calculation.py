"""
Value at Risk (VaR) Calculation using Historical Simulation Method
==================================================================

This script calculates the 95% Value at Risk for a portfolio of three stocks:
- Apple (AAPL)
- Microsoft (MSFT) 
- Google (GOOGL)

The portfolio uses equal weights (1/3 each) with a total value of $100,000.
VaR is calculated using the historical simulation method (5th percentile).

Required libraries: yfinance, pandas, numpy, matplotlib
Install with: pip install yfinance pandas numpy matplotlib
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def download_stock_data(tickers, start_date, end_date):
    """
    Download historical stock data from Yahoo Finance.
    
    Parameters:
    tickers (list): List of stock ticker symbols
    start_date (str): Start date in 'YYYY-MM-DD' format
    end_date (str): End date in 'YYYY-MM-DD' format
    
    Returns:
    pandas.DataFrame: DataFrame with adjusted closing prices
    """
    print("Downloading historical stock data...")
    
    try:
        # Download data for all tickers
        data = yf.download(tickers, start=start_date, end=end_date, progress=False, auto_adjust=True)
        
        # Extract closing prices (auto_adjust=True means we get adjusted prices)
        if len(tickers) == 1:
            adj_close = data['Close']
        else:
            adj_close = data['Close']
        
        # Check if data was successfully downloaded
        if adj_close.empty:
            raise ValueError("No data was downloaded. Please check ticker symbols and dates.")
        
        print(f"Successfully downloaded data for {len(tickers)} stocks")
        print(f"Data range: {adj_close.index[0].strftime('%Y-%m-%d')} to {adj_close.index[-1].strftime('%Y-%m-%d')}")
        print(f"Number of trading days: {len(adj_close)}")
        
        return adj_close
        
    except Exception as e:
        print(f"Error downloading data: {e}")
        print("Please check your internet connection and ticker symbols.")
        return None

def calculate_returns(prices):
    """
    Calculate daily returns from price data.
    
    Parameters:
    prices (pandas.DataFrame): DataFrame with stock prices
    
    Returns:
    pandas.DataFrame: DataFrame with daily returns
    """
    print("Calculating daily returns...")
    
    # Calculate daily returns: (P_t - P_{t-1}) / P_{t-1}
    returns = prices.pct_change()
    
    # Drop the first row (NaN due to pct_change)
    returns = returns.dropna()
    
    print(f"Returns calculated for {len(returns)} trading days")
    return returns

def create_portfolio_returns(stock_returns, weights):
    """
    Create portfolio returns using specified weights.
    
    Parameters:
    stock_returns (pandas.DataFrame): DataFrame with individual stock returns
    weights (list): List of portfolio weights
    
    Returns:
    pandas.Series: Portfolio daily returns
    """
    print("Creating portfolio returns...")
    
    # Calculate weighted portfolio returns
    portfolio_returns = (stock_returns * weights).sum(axis=1)
    
    print(f"Portfolio returns calculated with weights: {weights}")
    return portfolio_returns

def calculate_var(returns, confidence_level=0.95, portfolio_value=100000):
    """
    Calculate Value at Risk using historical simulation method.
    
    Parameters:
    returns (pandas.Series): Portfolio daily returns
    confidence_level (float): VaR confidence level (default: 0.95 for 95%)
    portfolio_value (float): Total portfolio value in dollars
    
    Returns:
    tuple: (VaR percentage, VaR dollar amount)
    """
    print(f"Calculating {confidence_level*100}% VaR...")
    
    # Calculate the percentile for VaR
    # For 95% VaR, we want the 5th percentile (worst 5% of returns)
    percentile = (1 - confidence_level) * 100
    
    # Calculate VaR as percentage
    var_percentage = np.percentile(returns, percentile)
    
    # Convert to dollar amount
    var_dollar = abs(var_percentage * portfolio_value)
    
    print(f"{confidence_level*100}% VaR: {var_percentage:.4f} ({var_percentage*100:.2f}%)")
    print(f"VaR in dollars: ${var_dollar:,.2f}")
    
    return var_percentage, var_dollar

def create_var_plot(returns, var_percentage, var_dollar, confidence_level=0.95):
    """
    Create histogram of portfolio returns with VaR indicator.
    
    Parameters:
    returns (pandas.Series): Portfolio daily returns
    var_percentage (float): VaR as percentage
    var_dollar (float): VaR in dollars
    confidence_level (float): VaR confidence level
    """
    print("Creating VaR visualization...")
    
    # Create figure and axis
    plt.figure(figsize=(12, 8))
    
    # Create histogram
    plt.hist(returns, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    
    # Add VaR line
    plt.axvline(x=var_percentage, color='red', linestyle='--', linewidth=2, 
                label=f'{confidence_level*100}% VaR: {var_percentage:.4f} (${var_dollar:,.0f})')
    
    # Customize plot
    plt.xlabel('Daily Returns', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Portfolio Daily Returns Distribution with 95% VaR', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Add statistics text
    stats_text = f'Mean: {returns.mean():.4f}\nStd Dev: {returns.std():.4f}\nMin: {returns.min():.4f}\nMax: {returns.max():.4f}'
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # Save plot
    plt.savefig('var_histogram.png', dpi=300, bbox_inches='tight')
    print("Plot saved as 'var_histogram.png'")
    
    # Show plot
    plt.show()

def main():
    """
    Main function to execute the VaR calculation workflow.
    """
    print("=" * 60)
    print("VALUE AT RISK (VaR) CALCULATION")
    print("Historical Simulation Method")
    print("=" * 60)
    
    # Configuration
    tickers = ['AAPL', 'MSFT', 'GOOGL']
    start_date = '2023-01-01'
    end_date = '2024-12-31'  # Use current date range
    weights = [1/3, 1/3, 1/3]  # Equal weights
    portfolio_value = 100000
    confidence_level = 0.95
    
    print(f"Portfolio: {', '.join(tickers)}")
    print(f"Weights: {[f'{w:.1%}' for w in weights]}")
    print(f"Portfolio Value: ${portfolio_value:,.0f}")
    print(f"Confidence Level: {confidence_level*100}%")
    print()
    
    # Step 1: Download data
    prices = download_stock_data(tickers, start_date, end_date)
    if prices is None:
        return
    
    # Step 2: Calculate returns
    returns = calculate_returns(prices)
    
    # Step 3: Create portfolio returns
    portfolio_returns = create_portfolio_returns(returns, weights)
    
    # Step 4: Calculate VaR
    var_percentage, var_dollar = calculate_var(portfolio_returns, confidence_level, portfolio_value)
    
    # Step 5: Create visualization
    create_var_plot(portfolio_returns, var_percentage, var_dollar, confidence_level)
    
    # Step 6: Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"95% VaR: {var_percentage:.4f} ({var_percentage*100:.2f}%)")
    print(f"VaR in Dollars: ${var_dollar:,.2f}")
    print(f"Portfolio Value: ${portfolio_value:,.0f}")
    print(f"Maximum Daily Loss (95% confidence): ${var_dollar:,.2f}")
    print("=" * 60)

if __name__ == "__main__":
    main() 