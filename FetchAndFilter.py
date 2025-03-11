# Language: python
from FetchIndex import IndexFetcher  # [FetchIndex.py](FetchIndex.py)
from StockAnalyzer import StockAnalyzer  # [StockAnalyzer.py](StockAnalyzer.py)
import time
import random
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import yfinance as yf

def passes_filters(analysis, cagr):
    # Fundamental Filters
    # Use reported ROE if available else calculated ROE
    roe = analysis['roe_reported'] if analysis['roe_reported'] is not None else analysis['roe_calculated']
    if roe is None or roe <= 10:
        print(f"Failed Fundamental Filter: ROE value {roe} is None or <= expected minimum of 10%")
        return False
        
    # ROCE check - skip if None
    if analysis['roce'] is not None and analysis['roce'] <= 10:
        print(f"Failed Fundamental Filter: ROCE value {analysis['roce']} is <= expected minimum of 10%")
        return False

    # Debt/Equity check - skip if None
    if analysis['debt_to_equity'] is not None and analysis['debt_to_equity'] >= 100:
        print(f"Failed Fundamental Filter: Debt/Equity value {analysis['debt_to_equity']} is >= expected maximum of 1")
        return False

    if cagr['eps_cagr'] is None or cagr['eps_cagr'] <= 10:
        print(f"Failed Fundamental Filter: EPS Growth (3-Year CAGR) value {cagr['eps_cagr']} is None or <= expected minimum of 10%")
        return False
    if cagr['revenue_cagr'] is None or cagr['revenue_cagr'] <= 10:
        print(f"Failed Fundamental Filter: Revenue Growth (3-Year CAGR) value {cagr['revenue_cagr']} is None or <= expected minimum of 10%")
        return False
    if analysis['actual_one_year_return'] is None or analysis['actual_one_year_return'] <= 10:
        print(f"Failed Fundamental Filter: 1 Year Return value {analysis['actual_one_year_return']} is None or <= expected minimum of 10%")
        return False
    if analysis['net_profit_margin'] is None or analysis['net_profit_margin'] <= 0.07:
        print(f"Failed Fundamental Filter: Net Profit Margin value {analysis['net_profit_margin']} is None or <= expected minimum of 7%")
        return False

    # # Separate P/E ratio checks
    # if analysis['pe_ratio'] is None:
    #     print("Failed Fundamental Filter: P/E is None")
    #     return False
    # if analysis['sector_pe'] is None:
    #     print("Failed Fundamental Filter: Sector P/E is None")
    #     return False
    # if analysis['pe_ratio'] >= analysis['sector_pe']:
    #     print(f"Failed Fundamental Filter: P/E mismatch condition. Received P/E: {analysis['pe_ratio']}, expected less than Sector P/E: {analysis['sector_pe']}")
    #     return False

    # Momentum Filters
    if analysis['RSI'] is None or analysis['RSI'] <= 50:
        print(f"Failed Momentum Filter: RSI value {analysis['RSI']} is None or <= expected minimum of 50")
        return False
    # Golden Cross: Price > 50 EMA and 50 EMA > 200 EMA
    if analysis['Close'] is None or analysis['50_EMA'] is None or analysis['200_EMA'] is None:
        print(f"Failed Momentum Filter: Missing Price/EMA values. Received Close: {analysis.get('Close')}, 50_EMA: {analysis.get('50_EMA')}, 200_EMA: {analysis.get('200_EMA')}")
        return False
    if not (analysis['Close'] > analysis['50_EMA'] and analysis['50_EMA'] > analysis['200_EMA']):
        print(f"Failed Momentum Filter: Golden Cross condition failed. Received Close: {analysis['Close']}, 50_EMA: {analysis['50_EMA']}, 200_EMA: {analysis['200_EMA']} but expected Close > 50_EMA > 200_EMA")
        return False
    # SMA Checks: 50, 100 and 200 Day SMAs
    if analysis['50_SMA'] is None or analysis['Close'] <= analysis['50_SMA']:
        print(f"Failed Momentum Filter: 50 Day SMA condition failed. Received Close: {analysis['Close']}, 50_SMA: {analysis['50_SMA']} but expected Close > 50_SMA")
        return False
    if analysis['100_SMA'] is None or analysis['Close'] <= analysis['100_SMA']:
        print(f"Failed Momentum Filter: 100 Day SMA condition failed. Received Close: {analysis['Close']}, 100_SMA: {analysis['100_SMA']} but expected Close > 100_SMA")
        return False
    if analysis['200_SMA'] is None or analysis['Close'] <= analysis['200_SMA']:
        print(f"Failed Momentum Filter: 200 Day SMA condition failed. Received Close: {analysis['Close']}, 200_SMA: {analysis['200_SMA']} but expected Close > 200_SMA")
        return False
    # MACD > Signal Line
    if analysis['MACD'] is None or analysis['Signal_Line'] is None or analysis['MACD'] <= analysis['Signal_Line']:
        print(f"Failed Momentum Filter: MACD condition failed. Received MACD: {analysis['MACD']}, Signal_Line: {analysis['Signal_Line']} but expected MACD > Signal_Line")
        return False

    return True

def main():
    # Ask the user which index to fetch
    index_name = input("Enter the index name to fetch (e.g., NIFTY50, NIFTY100, NIFTYBANK): ")
    
    # Fetch stocks from the specified index
    fetcher = IndexFetcher(index_name)  # [FetchIndex.py](FetchIndex.py)
    tickers = fetcher.get_tickers()
    # Append the ".NS" suffix to each ticker
    tickers = [ticker + ".NS" for ticker in tickers]

    passed_stocks = []
    failed_stocks = []

    for ticker in tickers:
        try:
            print(f"Analyzing {ticker}...")
            analyzer = StockAnalyzer(ticker)  # [StockAnalyzer.py](StockAnalyzer.py)
            analysis_df = analyzer.get_complete_analysis()
            
            if analysis_df.empty:
                print(f"No data available for {ticker}")
                failed_stocks.append(ticker)
                continue

            # Get the analysis as a Series (since get_complete_analysis returns a DataFrame)
            analysis = analysis_df.iloc[0]

            # Recalculate for 3-year CAGR fundamentals
            cagr = analyzer.calculate_cagr(3)
            
            if passes_filters(analysis, cagr):
                print(f"{ticker} passes all filters")
                passed_stocks.append(ticker)
            else:
                failed_stocks.append(ticker)
                
            # Add a small random delay between requests
            time.sleep(random.uniform(1.0, 3.0))
            
        except Exception as e:
            print(f"Error analyzing {ticker}: {e}")
            failed_stocks.append(ticker)
            # Add a longer delay after an error
            time.sleep(random.uniform(5.0, 10.0))
            continue

    print("\n----- RESULTS -----")
    print(f"Total stocks analyzed: {len(tickers)}")
    print(f"Stocks passing all filters: {len(passed_stocks)}")
    print(f"Stocks failing filters or with errors: {len(failed_stocks)}")
    print("\nStocks passing all filters:")
    for stock in passed_stocks:
        print(stock)

if __name__ == "__main__":
    main()

