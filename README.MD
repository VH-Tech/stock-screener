# Indian Stock Screener

An automated stock screening tool that analyzes Indian stocks based on various technical and fundamental criteria to identify high-quality investment opportunities.

## Overview

This stock screener analyzes stocks from Indian market indices (NIFTY50, NIFTY100, NIFTYBANK, etc.) using a combination of technical and fundamental filters to identify strong companies with good growth potential and momentum.

## Features

- **Index-based screening:** Screen all stocks from major Indian indices.
- **Fundamental analysis:**
  - Return on Equity (ROE) > 10%
  - Return on Capital Employed (ROCE) > 10%
  - Debt-to-Equity < 1.0 (100%)
  - 3-year EPS growth (CAGR) > 10%
  - 3-year Revenue growth (CAGR) > 10%
  - 1-year return > 10%
  - Net profit margin > 7%
- **Technical analysis:**
  - RSI > 50
  - Golden Cross (Price > 50 EMA and 50 EMA > 200 EMA)
  - Price above 50, 100, and 200-day SMAs
  - MACD above Signal Line
- **Rate limit handling:** Built-in retry mechanism for Yahoo Finance API rate limiting.
- **Detailed output:** Provides comprehensive analysis and reasons for inclusion/exclusion.

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/indian-stock-screener.git
   cd indian-stock-screener
   ```
2. Install the required packages:
   ```bash
   pip install pandas numpy yfinance tenacity
   ```

## Usage

Run the main script:
```bash
python FetchAndFilter.py
```

When prompted, enter the index you want to analyze (e.g., NIFTY50, NIFTY100, NIFTYBANK).

### Workflow:
1. Fetch all stocks in the selected index.
2. Download historical data for each stock.
3. Apply fundamental and technical filters.
4. Display which stocks passed all criteria.
5. Show detailed reasons why stocks failed specific filters.

## Project Structure

- **`FetchAndFilter.py`** - Main script that orchestrates the screening process.
- **`FetchIndex.py`** - Retrieves the list of stocks for a given index.
- **`StockAnalyzer.py`** - Performs detailed technical and fundamental analysis on individual stocks.

## Example Output

```
Enter the index name to fetch (e.g., NIFTY50, NIFTY100, NIFTYBANK): NIFTY50
Analyzing ADANIPORTS.NS...
Failed Momentum Filter: RSI value 48.2 is None or <= expected minimum of 50
Analyzing ASIANPAINT.NS...
Analyzing AXISBANK.NS...
Failed Fundamental Filter: Revenue Growth (3-Year CAGR) value 8.7 is None or <= expected minimum of 10%
...

----- RESULTS -----
Total stocks analyzed: 50
Stocks passing all filters: 12
Stocks failing filters or with errors: 38

Stocks passing all filters:
HDFCBANK.NS
INFY.NS
RELIANCE.NS
TCS.NS
...
```

## Notes

- The screener uses Yahoo Finance data, which may occasionally have rate-limiting issues.
- The retry mechanism with exponential backoff helps handle rate limiting.
- For best results, run the script during non-peak hours.
- You can adjust the filter thresholds in the `passes_filters` function in `FetchAndFilter.py`.

