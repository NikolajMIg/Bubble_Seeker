# Sector_Data.py (Enhanced with better error handling)
"""
Sector Data Engine - Handles data collection for all sectors with improved error handling
"""
#=> inform you must run this script with python 3 or above
#!/usr/bin/python3
#=> inform we are usige an UTF-8 code ( otherwhise some chars such as é, è, .. could be wrongly displayed) 
# -*- coding: utf-8 -*-

import  yfinance    as yf
import  General_ML
import  numpy       as np
from    datetime    import datetime
import  asyncio
from    Tools       import Display_info

class SectorDataEngine:
    def __init__(self, sectors_config):
        self.sectors = sectors_config
        self.start_date = "2020-01-01"
        self.end_date = datetime.now().strftime("%Y-%m-%d")
        self.failed_tickers = {}  # Track failed tickers by sector
        
    async def collect_all_sector_data(self, app):
        """Collect comprehensive data for all sectors with error handling"""
        sector_data = {}
        self.failed_tickers = {}
        
        for sector_name, tickers in self.sectors.items(): 
            app.Tag_OK_Last_Line_N()
            Display_info(f"Collecting {sector_name} sector data...", app)
            self.failed_tickers[sector_name] = []
            sector_data[sector_name] = await self._collect_sector_data(sector_name, tickers, app)
            
            # Log failed tickers for this sector
            if self.failed_tickers[sector_name]:
                Display_info(f"Warning: Failed to collect data for {len(self.failed_tickers[sector_name])} tickers in {sector_name}: {self.failed_tickers[sector_name]}", app)
        
        return sector_data
    
    async def _collect_sector_data(self, sector_name, tickers, app):
        """Collect detailed data for a specific sector with retry logic"""
        sector_info = {
            'metadata': {
                'sector_name': sector_name,
                'tickers': tickers,
                'analysis_date': datetime.now(),
                'successful_tickers': [],
                'failed_tickers': []
            },
            'stocks': {},
            'valuation_metrics': {},
            'fundamental_data': {}
        }
        
        # Download individual stock data with rate limiting
        tasks = []
        for ticker in tickers:
            H = self._download_stock_data_with_retry(ticker, sector_name, app)
            tasks.append(H)
            await asyncio.sleep(0.1)  # Rate limiting to avoid overwhelming the API
        #['data']['historical']['Close']
        stock_data = await asyncio.gather(*tasks, return_exceptions=True)
        
        for i, data in enumerate(stock_data):
            ticker = tickers[i]
            if data and not isinstance(data, Exception):
                if data.get('success', False):
                    sector_info['stocks'][ticker] = data['data']
                    sector_info['metadata']['successful_tickers'].append(ticker)
                else:
                    sector_info['metadata']['failed_tickers'].append(ticker)
                    self.failed_tickers[sector_name].append(ticker)
            else:
                sector_info['metadata']['failed_tickers'].append(ticker)
                self.failed_tickers[sector_name].append(ticker)
                Display_info(f"Failed to download data for {ticker}: {data if isinstance(data, Exception) else 'Unknown error'}", app)                       
        
        # Calculate sector aggregates only if we have successful data
        if sector_info['stocks']:
            sector_info = self._calculate_sector_aggregates(sector_info, app)
        else:
            Display_info(f"Warning: No successful data collected for {sector_name} sector", app)
                 
        return sector_info
    
    async def _download_stock_data_with_retry(self, ticker, sector,app, max_retries=2):
        """Download stock data with retry logic and error handling"""
        for attempt in range(max_retries):
            try:
                result = await self._download_stock_data(ticker, sector,app)
                
                import  pandas as pd
                import   os
                import  Tools
                from    Tools import Jours_from_1st_January_M_d
                h= result['data']['historical']['Close']
                h2= result['data']['historical']['Volume']
                #Excel_Csv_Access    =       pd.DataFrame()
                Excel_Csv_Access    =       pd.DataFrame(h)
                
                #Excel_Csv_Access2   =       pd.DataFrame()
                Excel_Csv_Access2   =       pd.DataFrame(h2)
               
                '''
                print(CSV_output_fl)
                AAPL-Technology.csv
                MSFT-Technology.csv
                GOOGL-Technology.csv
                META-Technology.csv
                NVDA-Technology.csv
                '''
                ISEC=-1
                while True:
                    ISEC +=1
                    if ISEC >=len(Tools.Sektor):
                        print('ERROR SOFT SECTOR=', sector, ' ', Tools.Sektor)
                        ISEC=-1
                        break
                    if sector == Tools.Sektor[ISEC]: 
                        JSEC=-1
                        while True:
                            JSEC+=1
                            if JSEC >= len(Tools.Sect_Ticket[ISEC]):
                                print('ERROR SOFT SECTOR=', sector, '   TICKER=',ticker)
                                JSEC=-1
                                break
                            if ticker==Tools.Sect_Ticket[ISEC][JSEC]:
                                break
                            # end whike true
                        break
                        # end if sector==Tools.Sektor[ISEC]
                    #end while true
                #print(ISEC, JSEC)
                
                if (ISEC>=0) and (JSEC>=0):
                    _00_ = (ISEC==0) and (JSEC==0)
                    if _00_:
                        Tools.DDT_ALL.clear()
                        Tools.DDT_ALL_TXT.clear()
                        Tools.DDT_Y.clear()
                        Tools.DDT_M.clear()
                        Tools.DDT_D.clear()
                        for i in range(0, len(Tools.Sektor)):  
                            for j in range(0, len(Tools.Sect_Ticket[i])):                                
                                Tools.Value_at_Close[i][j].clear() # Value_at_Close[sector][ticket][n] : data
                                Tools.Volume_at_dt[i][j].clear()
          
                    CSV_output_fl       =   str(ISEC)+'-'+str(JSEC)+'.csv'
                    Excel_Csv_Access.to_csv( # Excel_Csv_Access.to_excel(output_file) devrait marcher ....  
                        CSV_output_fl,     # pour faire un xlsx mais ......???
                        sep = ';'
                        )
                    Excel_Csv_Access2.to_csv( # Excel_Csv_Access.to_excel(output_file) devrait marcher ....  
                        '_' + CSV_output_fl,     # pour faire un xlsx mais ......???
                        sep = ';'
                        )
                    #input('ANY KEY')
                    '''
                    Tools.Value_at_Close[i][j].clear() # Value_at_Close[sector][ticket][n] : data
                                Tools.Volume_at_dt[i][j]
                    '''
                    if os.path.isfile(CSV_output_fl):
                        if os.path.isfile('_' + CSV_output_fl):
                            fp          =   open('_' + CSV_output_fl, 'r') 
                            line        =   fp.readline()       # remove header line
                            while True:
                                line    =   fp.readline()
                                if not line:
                                    break
                                line    =   line[:-1]     # remove \n  char 16.02.2024
                                #2022-01-07 00:00:00-05:00;32720000
                                a=line.find(';')+1
                                line=line[a:] 
                                Tools.Volume_at_dt[ISEC][JSEC].append(float(line))
                                
                            if fp != None:
                                fp.close()
                                fp=None 
                            General_ML.Klean('_' + CSV_output_fl)
                        #end if if os.path.isfile('_' + CSV_output_fl)
                        fp          =   open(CSV_output_fl, 'r') 
                        line        =   fp.readline()       # remove header line
                        while True:
                            line    =   fp.readline()
                            if not line:
                                break
                            line    =   line[:-1]     # remove \n  char 16.02.2024
                            if _00_:
                                Y, M, D = Tools.get_YMD(line[:10])  
                                Tools.DDT_Y.append(Y)
                                Tools.DDT_M.append(M)
                                Tools.DDT_D.append(D)
                                dy = Jours_from_1st_January_M_d(Y, M, D)
                                #print(line[:10], Y, M, D, Y + (dy-1)/366)
                                Tools.DDT_ALL_TXT.append(line[:10])                          
                                Tools.DDT_ALL.append(Y + (dy-1)/366)
                            a=line.find(';')+1# 2022-01-04 00:00:00-05:00;140.56649780273438
                            #print(line)   2025-11-17 00:00:00-05:00;506.54193115234375
                            line=line[a:]    
                            #print(line)506.54193115234375
                            Tools.Value_at_Close[ISEC][JSEC].append(float(line))
                            #end if indicator is not None:
                        if fp != None:
                            fp.close()
                            fp=None 
                        General_ML.Klean(CSV_output_fl)
                    # fin de if ISEC >=0 & JSEC >=0

                     
                #jjjpppmmm
                '''
                print(ticker)
                print(sector)
                
                print(result['data']['historical']['Close']) 
                x=input('XXXXXXXXXXXXXXXXXX')
                AAPL
                Technology
                Date
                2022-01-03 00:00:00-05:00    178.270325
                2022-01-04 00:00:00-05:00    176.007782
                                ...
                2025-11-20 00:00:00-05:00    266.250000
                '''
                if result and result.get('success', False):
                    return result
                else:
                    Display_info(f"Attempt {attempt + 1} failed for {ticker}, retrying...", app)
                    await asyncio.sleep(1)  # Wait before retry
            except Exception as e:
                Display_info(f"Attempt {attempt + 1} error for {ticker}: {e}", app)
                if attempt < max_retries - 1:
                    await asyncio.sleep(1)
        
        # All retries failed
        return {'success': False, 'ticker': ticker, 'error': 'All retries failed'}
    
    async def _download_stock_data(self, ticker, sector, app):
        """Download comprehensive stock data with robust error handling"""
        try:
            # Clean ticker symbol
            ticker = str(ticker).strip().upper()
            
            stock = yf.Ticker(ticker)
            
            # Get info first to check if ticker is valid
            info = stock.info
            if not info:
                return {'success': False, 'ticker': ticker, 'error': 'No info available'}
            
            # Check if we have basic required info
            if 'symbol' not in info or not info.get('symbol'):
                return {'success': False, 'ticker': ticker, 'error': 'Invalid symbol'}
            
            # Historical price data with error handling
            try:
                hist = stock.history(start="2022-01-01", end=self.end_date, auto_adjust=True)
                if hist.empty:
                    Display_info(f"Warning: No historical data for {ticker}", app)
                    # Try with a longer period
                    hist = stock.history(period="1y", auto_adjust=True)
                    if hist.empty:
                        return {'success': False, 'ticker': ticker, 'error': 'No historical data'}
            except Exception as e:
                Display_info(f"Error getting history for {ticker}: {e}", app)
                return {'success': False, 'ticker': ticker, 'error': f'History error: {str(e)}'}
            
            # Get additional info with error handling
            try:
                recommendations = stock.recommendations
                if recommendations is not None and not recommendations.empty:
                    latest_rec = recommendations.iloc[-1] if len(recommendations) > 0 else None
                    if latest_rec is not None:
                        info['latest_recommendation'] = latest_rec.to_dict()
            except Exception:
                info['latest_recommendation'] = None  # Ignore if recommendations fail
            
            # Extract valuation metrics
            valuation_data = self._extract_valuation_metrics(info, hist, app)
            
            stock_data = {
                'sector': sector,
                'historical': hist,
                'info': info,
                'valuation': valuation_data
            }
            
            return {
                'success': True,
                'ticker': ticker,
                'data': stock_data
            }
            
        except Exception as e:
            Display_info(f"Error downloading {ticker}: {str(e)}", app)
            return {'success': False, 'ticker': ticker, 'error': str(e)}
    
    def _extract_valuation_metrics(self, info, hist, app):
        """Extract comprehensive valuation metrics with safe defaults"""
        try:
            # Get current price safely
            current_price = 0
            if not hist.empty and 'Close' in hist.columns:
                current_price = hist['Close'].iloc[-1] if len(hist) > 0 else 0
            
            # Get market cap with fallbacks
            market_cap = info.get('marketCap', 0)
            if not market_cap or market_cap == 0:
                # Try to calculate from shares and price
                shares = info.get('sharesOutstanding', 0)
                if shares and current_price:
                    market_cap = shares * current_price
            
            # Safely extract all metrics with defaults
            return {
                'pe_ratio': info.get('trailingPE', info.get('forwardPE', 0)) or 0,
                'forward_pe': info.get('forwardPE', 0) or 0,
                'price_to_sales': info.get('priceToSalesTrailing12Months', 0) or 0,
                'price_to_book': info.get('priceToBook', 0) or 0,
                'ev_to_ebitda': info.get('enterpriseToEbitda', 0) or 0,
                'market_cap': market_cap or 0,
                'earnings_growth': info.get('earningsGrowth', 0) or 0,
                'revenue_growth': info.get('revenueGrowth', 0) or 0,
                'profit_margins': info.get('profitMargins', 0) or 0,
                'return_on_equity': info.get('returnOnEquity', 0) or 0,
                'current_price': current_price
            }
        except Exception as e:
            Display_info(f"Error extracting valuation metrics: {e}", app)
            # Return default metrics
            return {
                'pe_ratio': 0,
                'forward_pe': 0,
                'price_to_sales': 0,
                'price_to_book': 0,
                'ev_to_ebitda': 0,
                'market_cap': 0,
                'earnings_growth': 0,
                'revenue_growth': 0,
                'profit_margins': 0,
                'return_on_equity': 0,
                'current_price': 0
            }
    
    def _calculate_sector_aggregates(self, sector_info, app):
        """Calculate sector-level aggregate metrics with error handling"""
        stocks = sector_info['stocks']
        if not stocks:
            return sector_info
            
        try:
            # Calculate median valuation metrics
            valuation_metrics = ['pe_ratio', 'forward_pe', 'price_to_sales', 'price_to_book', 'ev_to_ebitda']
            aggregates = {}
            
            for metric in valuation_metrics:
                values = []
                for stock_data in stocks.values():
                    value = stock_data['valuation'].get(metric, 0)
                    # Only include valid, non-zero values
                    if value and value != 0 and not np.isnan(value):
                        values.append(value)
                
                if values:
                    aggregates[metric] = np.median(values)
                else:
                    aggregates[metric] = 0
                    
            sector_info['valuation_metrics'] = aggregates
            return sector_info
            
        except Exception as e:
            Display_info(f"Error calculating sector aggregates: {e}", app)
            # Return empty aggregates on error
            sector_info['valuation_metrics'] = {metric: 0 for metric in valuation_metrics}
            return sector_info