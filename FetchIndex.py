from nsetools import Nse
import pandas as pd
import requests

class IndexFetcher:
    def __init__(self, index_name='NIFTY50'):
        self.index_name = index_name
        self.tickers = self.download_index()

    def get_tickers(self):
        return self.tickers

    def download_index(self,verbose=True):
        url = "https://archives.nseindia.com/content/indices/ind_"+self.index_name.lower()+"list.csv"
        print(url)
        headers =	{
						'Accept': 'application/json, text/javascript, */*; q=0.01',
						'Accept-Language': 'en-US,en;q=0.5',
						'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.113 Safari/537.36'
					}
        try:
            # Read CSV file directly from the URL
            df = pd.read_csv(url)
            # Get list of stock symbols
            stock_list = df['Symbol'].tolist()
            return stock_list
        except Exception as e:
            print(f"Error fetching "+self.index_name+" stocks: {e}")
            return []

                  
