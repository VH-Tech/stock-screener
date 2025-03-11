import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time

class StockAnalyzer:
    def __init__(self, ticker):
        """
        Initialize StockAnalyzer with a ticker symbol.
        
        Parameters:
        ticker (str): Stock ticker symbol
        """
        self.ticker = ticker
        self.stock = yf.Ticker(ticker)
        # Cache market data at init
        try:
            self.market = yf.download('^NSEI', period='1mo')
            time.sleep(1)  # Add delay between requests
        except Exception as e:
            print(f"Failed to fetch market data: {e}")
            self.market = None

        
    def calculate_rsi(self, data, window=14):
        """Calculate Relative Strength Index."""
        delta = data['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=window, min_periods=1).mean()
        avg_loss = loss.rolling(window=window, min_periods=1).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def calculate_cagr(self, years=5):
        """Calculate CAGR for EPS and Revenue."""
        income_statement = self.stock.income_stmt
        
        try:
            net_income = income_statement.loc["Net Income"]
            revenue = income_statement.loc["Total Revenue"]
            shares_outstanding = self.stock.financials.loc["Diluted Average Shares"]
            
            # Drop NaN values
            net_income = net_income.dropna()
            revenue = revenue.dropna()
            shares_outstanding = shares_outstanding.dropna()
            
            if len(net_income) == 0 or len(shares_outstanding) == 0 or len(revenue) == 0:
                return pd.Series({'eps_cagr': None, 'revenue_cagr': None})
            
            # Convert to numpy arrays and calculate EPS
            net_income = np.array(net_income)
            revenue = np.array(revenue)
            shares_outstanding = np.array(shares_outstanding)
            eps_values = net_income / shares_outstanding
            
            # Sort values in ascending order
            eps_values = eps_values[::-1]
            revenue = revenue[::-1]
            
            # Calculate CAGR
            eps_start, eps_end = float(eps_values[0]), float(eps_values[years])
            revenue_start, revenue_end = float(revenue[0]), float(revenue[years])
            
            eps_cagr = ((eps_end / eps_start) ** (1 / years) - 1) * 100 if eps_start > 0 and eps_end > 0 else None
            revenue_cagr = ((revenue_end / revenue_start) ** (1 / years) - 1) * 100 if revenue_start > 0 and revenue_end > 0 else None
            
            return pd.Series({'eps_cagr': eps_cagr, 'revenue_cagr': revenue_cagr})
            
        except KeyError:
            return pd.Series({'eps_cagr': None, 'revenue_cagr': None})
    
    def calculate_technical_indicators(self):
        """Calculate technical indicators for the stock."""
        try:
            # Get historical data
            df = self.stock.history(period='1y')
            if df.empty:
                return None
                
            # Calculate SMAs
            df['50_SMA'] = df['Close'].rolling(window=50).mean()
            df['100_SMA'] = df['Close'].rolling(window=100).mean()
            df['200_SMA'] = df['Close'].rolling(window=200).mean()
            
            # Calculate EMAs
            df['12_EMA'] = df['Close'].ewm(span=12, adjust=False).mean()
            df['26_EMA'] = df['Close'].ewm(span=26, adjust=False).mean()
            df['50_EMA'] = df['Close'].ewm(span=50, adjust=False).mean()
            df['200_EMA'] = df['Close'].ewm(span=200, adjust=False).mean()
            
            # Calculate MACD
            df["MACD"] = df["12_EMA"] - df["26_EMA"]
            df["Signal_Line"] = df["MACD"].ewm(span=9, adjust=False).mean()
            
            # Calculate RSI
            df["RSI"] = self.calculate_rsi(df)
            
            # Calculate 1-week Beta
            df['Market_Return'] = self.market['Close'].pct_change()
            # print(self.market['Close'])
            df['Stock_Return'] = df['Close'].pct_change()
            # print(df['Close'])
            
            # Calculate 1-week (5 trading days) beta
            cov = df['Stock_Return'].tail(5).cov(df['Market_Return'].tail(5))
            var = df['Market_Return'].tail(5).var()
            df['Beta_1W'] = cov/var if var != 0 else None
            
            time.sleep(1)  # Add delay between requests
            return df[["Close", "50_SMA", "100_SMA", "200_SMA", "50_EMA", 
                      "200_EMA", "MACD", "Signal_Line", "RSI", "Beta_1W"]].iloc[-1]
                      
        except Exception as e:
            print(f"Error calculating technical indicators: {e}")
            return None
    
    def calculate_roce(self):
        """Calculate Return on Capital Employed (ROCE)."""
        try:
            # Get financial statements using existing stock instance
            income_stmt = self.stock.financials
            balance_sheet = self.stock.balance_sheet
            
            if income_stmt.empty or balance_sheet.empty:
                return None
                
            # Extract EBIT
            ebit_keys = ["EBIT", "Operating Income", "Operating Profit"]
            ebit = None
            for key in ebit_keys:
                if key in income_stmt.index:
                    ebit = income_stmt.loc[key].iloc[0]
                    break
            if ebit is None:
                return None

            # Extract Total Assets 
            total_assets_keys = ["Total Assets"]
            total_assets = None
            for key in total_assets_keys:
                if key in balance_sheet.index:
                    total_assets = balance_sheet.loc[key].iloc[0]
                    break
            if total_assets is None:
                return None
            
            # Extract Current Liabilities
            current_liabilities_keys = ["Current Liabilities", "Total Current Liabilities"]
            current_liabilities = None
            for key in current_liabilities_keys:
                if key in balance_sheet.index:
                    current_liabilities = balance_sheet.loc[key].iloc[0]
                    break
            if current_liabilities is None:
                return None
            
            # Calculate ROCE
            capital_employed = total_assets - current_liabilities
            roce = (ebit / capital_employed) * 100

            return roce
                
        except Exception:
            return None
    
    def calculate_roce_from_statements(self):
        """Calculate ROCE from financial statements."""
        try:
            income_stmt = self.stock.financials
            balance_sheet = self.stock.balance_sheet
            
            if income_stmt.empty or balance_sheet.empty:
                return None

            # Try to get EBIT directly first
            try:
                ebit = income_stmt.loc["EBIT"].iloc[0]
            except KeyError:
                # If EBIT not found, calculate from components
                print("EBIT not found, calculating from components...")
                try:
                    pretax_income = income_stmt.loc["Pretax Income"].iloc[0]
                    interest_expense = income_stmt.loc["Interest Expense"].iloc[0]
                    ebit = pretax_income + interest_expense
                except KeyError:
                    print("Could not calculate EBIT, missing components.")
                    return None

            # print(balance_sheet.index)
            # Get capital employed components
            try:
                total_assets = balance_sheet.loc["Total Assets"].iloc[0]
                # print("found total assets")
                current_liabilities = balance_sheet.loc["Total Current Liabilities"].iloc[0]
            except KeyError:
            
                return None

            # Calculate ROCE
            capital_employed = total_assets - current_liabilities
            roce = (ebit / capital_employed) * 100

            return roce

        except Exception:
            return None
    
    def calculate_returns(self):
        """Calculate 1-year returns."""
        hist = self.stock.history(period="1y")
        if not hist.empty:
            start_price = hist['Close'].iloc[0]
            end_price = hist['Close'].iloc[-1]
            return ((end_price - start_price) / start_price) * 100
        return None
    
    def calculate_roe(self, period='annual'):
        """
        Calculate Return on Equity (ROE) from financial statements.
        
        Parameters:
        period (str): 'annual' or 'quarterly' to specify which statements to use
        
        Returns:
        pd.Series: ROE values for available periods
        """
        try:
            # Get income statement and balance sheet
            income_stmt = self.stock.income_stmt if period == 'annual' else self.stock.quarterly_income_stmt
            balance_sheet = self.stock.balance_sheet if period == 'annual' else self.stock.quarterly_balance_sheet
            
            if income_stmt.empty or balance_sheet.empty:
                return pd.Series()
            
            # Get net income
            net_income = income_stmt.loc['Net Income']
            
            # Get shareholders' equity components
            try:
                total_equity = balance_sheet.loc['Stockholders Equity']
            except KeyError:
                # Some datasets use different keys for stockholders' equity
                try:
                    total_equity = balance_sheet.loc['Total Stockholder Equity']
                except KeyError:
                    # If direct equity value not available, calculate it
                    total_assets = balance_sheet.loc['Total Assets']
                    total_liabilities = balance_sheet.loc['Total Liabilities']
                    total_equity = total_assets - total_liabilities
            
            # Calculate ROE for each period
            # We use the average equity between start and end of period
            roe_values = {}
            
            for date in net_income.index:
                try:
                    # Get equity values for start and end of period
                    equity_end = total_equity[total_equity.index <= date].iloc[0]
                    equity_start = total_equity[total_equity.index <= date].iloc[1]
                    
                    # Calculate average equity
                    avg_equity = (equity_start + equity_end) / 2
                    
                    # Calculate ROE
                    roe = (net_income[date] / avg_equity) * 100
                    roe_values[date] = roe
                except (IndexError, KeyError):
                    # If we can't get both start and end equity, use end equity only
                    try:
                        equity_end = total_equity[total_equity.index <= date].iloc[0]
                        roe = (net_income[date] / equity_end) * 100
                        roe_values[date] = roe
                    except (IndexError, KeyError):
                        continue
            
            return pd.Series(roe_values)
        
        except Exception as e:
            print(f"Error calculating ROE: {str(e)}")
            return pd.Series()
    
    def get_complete_analysis(self):
        """Get complete stock analysis."""
        try:
            # Get stock info
            stock_info = self.stock.info
            time.sleep(1)  # Add delay between requests
            
            # Calculate ROE from financial statements
            roe_series = self.calculate_roe()
            calculated_roe = roe_series.iloc[0] if not roe_series.empty else None
            
            # Create analysis dictionary
            analysis = {
                'ticker': self.ticker,
                'company_name': stock_info.get('longName'),
                'current_price': stock_info.get('currentPrice'),
                'roe_calculated': calculated_roe,  # ROE calculated from statements
                'roe_reported': stock_info.get('returnOnEquity') * 100 if stock_info.get('returnOnEquity') is not None else None,  # ROE from Yahoo Finance
                'debt_to_equity': stock_info.get('debtToEquity'),
                'one_year_return': stock_info.get('oneYearReturn'),
                'net_profit_margin': stock_info.get('profitMargins'),
                'pe_ratio': stock_info.get('trailingPE'),
                'sector_pe': stock_info.get('sectorPE'),
                'roce': self.calculate_roce_from_statements(),
                'actual_one_year_return': self.calculate_returns()
            }
            
            # Add CAGR calculations
            cagr = self.calculate_cagr(3)
            analysis['eps_growth_3y'] = cagr['eps_cagr']
            analysis['revenue_growth_3y'] = cagr['revenue_cagr']
            
            # Add technical indicators
            technical_indicators = self.calculate_technical_indicators()
            for indicator_name, value in technical_indicators.items():
                analysis[indicator_name] = value
            
            # Convert to DataFrame
            return pd.DataFrame([analysis])
        
        except yf.exceptions.YFRateLimitError:
            print("Rate limit hit, retrying with backoff...")
            raise  # Trigger retry
        except Exception as e:
            print(f"Error in analysis: {e}")
            return None
    
# Example usage
if __name__ == "__main__":
    analyzer = StockAnalyzer("TCS.NS")
    analysis_df = analyzer.get_complete_analysis()
    # print(analysis_df['roce_calculated'])
    print(analysis_df)
    # print each column seperately
    for column in analysis_df.columns:
        print(f"{column}: {analysis_df[column].iloc[0]}")