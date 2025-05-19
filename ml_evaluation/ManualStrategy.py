import datetime as dt  		  	   		 	 	 			  		 			     			  	 
import random  		  	   		 	 	 			  		 			     			  	 
import indicators	  		 			     			  	 
import pandas as pd  		  	   		 	 	 			  		 			     			  	 
import util as ut
from marketsimcode import compute_portvals
import matplotlib.pyplot as plt

  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
class ManualStrategy(object):

    def __init__(self, verbose=False, impact=0.0, commission=0.0):  		  	   		 	 	 			  		 			     			  	 
        """  		  	   		 	 	 			  		 			     			  	 
        Constructor method  		  	   		 	 	 			  		 			     			  	 
        """  		  	   		 	 	 			  		 			     			  	 
        self.verbose = verbose  		  	   		 	 	 			  		 			     			  	 
        self.impact = impact  		  	   		 	 	 			  		 			     			  	 
        self.commission = commission
        self.signals = None

    def add_evidence(  		  	   		 	 	 			  		 			     			  	 
        self,  		  	   		 	 	 			  		 			     			  	 
        symbol="IBM",  		  	   		 	 	 			  		 			     			  	 
        sd=dt.datetime(2008, 1, 1),  		  	   		 	 	 			  		 			     			  	 
        ed=dt.datetime(2009, 1, 1),  		  	   		 	 	 			  		 			     			  	 
        sv=10000,  		  	   		 	 	 			  		 			     			  	 
    ):  		  	   		 	 	 			  		 			     			  	 
        """  		  	   		 	 	 			  		 			     			  	 
        Trains your strategy learner over a given time frame.  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
        :param symbol: The stock symbol to train on  		  	   		 	 	 			  		 			     			  	 
        :type symbol: str  		  	   		 	 	 			  		 			     			  	 
        :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		 	 	 			  		 			     			  	 
        :type sd: datetime  		  	   		 	 	 			  		 			     			  	 
        :param ed: A datetime object that represents the end date, defaults to 1/1/2009  		  	   		 	 	 			  		 			     			  	 
        :type ed: datetime  		  	   		 	 	 			  		 			     			  	 
        :param sv: The starting value of the portfolio  		  	   		 	 	 			  		 			     			  	 
        :type sv: int  		  	   		 	 	 			  		 			     			  	 
        """  		
        pass
    
    def testPolicy(  		  	   		 	 	 			  		 			     			  	 
        self,  		  	   		 	 	 			  		 			     			  	 
        symbol="JPM",  		  	   	
        sd=dt.datetime(2008, 1, 1),  		  	   		 	 	 			  		 			     			  	 
        ed=dt.datetime(2009, 12, 31),  		  	   		 	 	 			  		 			     			  	 
        sv=100000,  		  	   		 	 	 			  		 			     			  	 
    ):
        """
        Tests your learner using data outside of the training data

        Parameters
            symbol (str) – The stock symbol that you trained on on
            sd (datetime) – A datetime object that represents the start date, defaults to 1/1/2008
            ed (datetime) – A datetime object that represents the end date, defaults to 1/1/2009
            sv (int) – The starting value of the portfolio
        Returns
            A DataFrame with values representing trades for each day. Legal values are +1000.0 indicating
            a BUY of 1000 shares, -1000.0 indicating a SELL of 1000 shares, and 0.0 indicating NOTHING.
            Values of +2000 and -2000 for trades are also legal when switching from long to short or short to
            long so long as net holdings are constrained to -1000, 0, and 1000.

        Return type
            pandas.DataFrame
        """

        # Get all stock data, including extra data for indicators
        df = ut.get_data([symbol], pd.date_range(sd - dt.timedelta(days=100), ed)).drop(columns='SPY')
        high_df = ut.get_data([symbol],  pd.date_range(sd - dt.timedelta(days=100), ed), colname="High").drop(columns="SPY")
        low_df = ut.get_data([symbol], pd.date_range(sd - dt.timedelta(days=100), ed), colname="Low").drop(columns="SPY")
        close_df = ut.get_data([symbol], pd.date_range(sd - dt.timedelta(days=100), ed), colname="Close").drop(columns="SPY")
        sd_extra = df.index[df.index < sd][-20]

        stock_data = df.loc[sd_extra:].copy()
        high_df = high_df.loc[sd_extra:].copy()
        low_df = low_df.loc[sd_extra:].copy()
        close_df = close_df.loc[sd_extra:].copy()

        #### Bollinger Bands ####
        bb = indicators.calc_bollinger(stock_data, symbol, window=10, num_std=2)
        bb = bb.reindex(stock_data.index)
        stock_data['%B'] = bb

        #### MACD ####
        new_data = indicators.calc_macd(stock_data[[symbol]], short_per=12, long_per=26, signal_per=9)
        macd = new_data.iloc[:new_data.shape[0] // 2].reindex(stock_data.index)
        macd_sig = new_data.iloc[new_data.shape[0] // 2:].reindex(stock_data.index)
        stock_data['MACD'] = macd
        stock_data['MAC_Signal'] = macd_sig
        stock_data['MACD_diff'] = stock_data['MACD'] - stock_data['MAC_Signal']
        stock_data['MACD_delta'] = stock_data['MACD_diff'].diff()

        #### Momentum ####
        mom = indicators.calc_momentum(stock_data[[symbol]], symbol, window=20)
        mom = mom.reindex(stock_data.index)
        stock_data['Mom'] = mom
        stock_data['Mom_prev'] = stock_data['Mom'].shift(1).fillna(0)
        
        #### SMA ####
        sma = indicators.calc_sma(stock_data[[symbol]], symbol, window=50)
        sma = sma.reindex(stock_data.index)
        stock_data['SMA'] = sma

        #### CCI ####
        cci = indicators.calc_cci(close_df, high_df, low_df, symbol, window=50)
        cci = cci.reindex(stock_data.index)
        stock_data['CCI'] = cci


        # Set conditions for buy/sell signals
        condition_df = pd.DataFrame({
            'cond1': stock_data['%B'] < 0.1,
            'cond2': stock_data['MACD'] > stock_data['MAC_Signal'],
            'cond3': stock_data[symbol] > stock_data['SMA'],
            'cond4': stock_data['CCI'] > 100
        })

        sell_condition_df = pd.DataFrame({
            'cond1': stock_data['%B'] > 1,
            'cond2': stock_data['MACD'] < (stock_data['MAC_Signal'] - 0.2), 
            'cond3': stock_data[symbol] < stock_data['SMA'],
            'cond4': stock_data['CCI'] < - 100
        })

        # Convert boolean to int (True=1, False=0), sum across conditions
        buy_score = condition_df.astype(int).sum(axis=1)
        sell_score = sell_condition_df.astype(int).sum(axis=1)

        # Set signal based on net score * momentum
        net_score = (buy_score - sell_score) * stock_data['Mom']
        stock_data['Signal'] = 0
        stock_data.loc[(net_score > 0.15), 'Signal'] = 1
        stock_data.loc[(net_score < -0.25), 'Signal'] = -1

        # Filter to desired date range
        stock_data = stock_data.loc[sd:]

        # Store signals for use in Experiment 1
        self.signals = stock_data['Signal']

        # 
        trades_df = self.get_trade_orders(stock_data['Signal'])
        stock_data.index = pd.to_datetime(stock_data.index)

        return trades_df

    
    def get_trade_orders(self, signal_df):
        trades_df = pd.DataFrame(0, index=signal_df.index, columns=["Shares"])
    
        # Get previous day trades as df
        signal_shifted = signal_df.shift(1).fillna(0)

        # Entry signals (from 0 to non-zero)
        trades_df["Shares"] = ((signal_shifted == 0) & (signal_df != 0)) * (signal_df * 1000)

        # Exit signals (from non-zero to zero)
        trades_df["Shares"] += ((signal_shifted != 0) & (signal_df == 0)) * (-signal_shifted * 1000)

        # Position reversal signals (switching from +1 to -1 or vice versa)
        trades_df["Shares"] += ((signal_shifted != 0) & (signal_df != 0) & (signal_df != signal_shifted)) * (-2 * signal_shifted * 1000)

        # Make trades on the next day
        trades_df['Shares'] = trades_df['Shares'].shift(1, fill_value=0)
  
        return trades_df

    
    def author(self):  		  	   		 	 	 			  		 			     			  	 
        """  		  	   		 	 	 			  		 			     			  	 
        :return: The GT username of the student  		  	   		 	 	 			  		 			     			  	 
        :rtype: str  		  	   		 	 	 			  		 			     			  	 
        """  		  	   		 	 	 			  		 			     			  	 
        return "afitton3"  # replace tb34 with your Georgia Tech username.  		'

    def study_group(self):
        """
        :return: A comma separated string of GT_Name of each member of your study group
        :rtype: str
        """
        return "afitton3" 
    
if __name__ == "__main__":
    strat = ManualStrategy()
    strat.testPolicy()