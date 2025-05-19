""""""  		  	   		 	 	 			  		 			     			  	 
"""  		  	   		 	 	 			  		 			     			  	 
Template for implementing StrategyLearner  (c) 2016 Tucker Balch  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		 	 	 			  		 			     			  	 
Atlanta, Georgia 30332  		  	   		 	 	 			  		 			     			  	 
All Rights Reserved  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
Template code for CS 4646/7646  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		 	 	 			  		 			     			  	 
works, including solutions to the projects assigned in this course. Students  		  	   		 	 	 			  		 			     			  	 
and other users of this template code are advised not to share it with others  		  	   		 	 	 			  		 			     			  	 
or to make it available on publicly viewable websites including repositories  		  	   		 	 	 			  		 			     			  	 
such as github and gitlab.  This copyright statement should not be removed  		  	   		 	 	 			  		 			     			  	 
or edited.  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
We do grant permission to share solutions privately with non-students such  		  	   		 	 	 			  		 			     			  	 
as potential employers. However, sharing with other current or future  		  	   		 	 	 			  		 			     			  	 
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		 	 	 			  		 			     			  	 
GT honor code violation.  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
-----do not edit anything above this line---  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
Student Name: Andrew Fitton		  	   		 	 	 			  		 			     			  	 
GT User ID: afitton3		  	   		 	 	 			  		 			     			  	 
GT ID: 903560281		  	   		 	 	 			  		 			     			  	 
"""  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
import datetime as dt  		  	   		 	 	 			  		 			     			  	 
import random  		  	   		 	 	 			  		 			     			  	 
import indicators		  		 			     			  	 
import pandas as pd  		  	   		 	 	 			  		 			     			  	 
# import util as ut  		  	 
import numpy as np  		 
import matplotlib.pyplot as plt
from BagLearner import BagLearner	 
from RTLearner import RTLearner	
# import warnings
# warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)		  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
class StrategyLearner(object):  		  	   		 	 	 			  		 			     			  	 
    """  		  	   		 	 	 			  		 			     			  	 
    A strategy learner that can learn a trading policy using the same indicators used in ManualStrategy.  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
    :param verbose: If “verbose” is True, your code can print out information for debugging.  		  	   		 	 	 			  		 			     			  	 
        If verbose = False your code should not generate ANY output.  		  	   		 	 	 			  		 			     			  	 
    :type verbose: bool  		  	   		 	 	 			  		 			     			  	 
    :param impact: The market impact of each transaction, defaults to 0.0  		  	   		 	 	 			  		 			     			  	 
    :type impact: float  		  	   		 	 	 			  		 			     			  	 
    :param commission: The commission amount charged, defaults to 0.0  		  	   		 	 	 			  		 			     			  	 
    :type commission: float  		  	   		 	 	 			  		 			     			  	 
    """  		  	   		 	 	 			  		 			     			  	 
    # constructor  		  	   		 	 	 			  		 			     			  	 
    def __init__(self, verbose=False, impact=0.0, commission=0.0):  		  	   		 	 	 			  		 			     			  	 
        """  		  	   		 	 	 			  		 			     			  	 
        Constructor method  		  	   		 	 	 			  		 			     			  	 
        """  		  	   		 	 	 			  		 			     			  	 
        self.verbose = verbose  		  	   		 	 	 			  		 			     			  	 
        self.impact = impact  		  	   		 	 	 			  		 			     			  	 
        self.commission = commission  
        self.learner = BagLearner(RTLearner, kwargs={'leaf_size':5}, bags=20, boost=False, verbose=False) 	 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
    # this method should create a QLearner, and train it for trading  		  	   		 	 	 			  		 			     			  	 
    def add_evidence(  		  	   		 	 	 			  		 			     			  	 
        self,  		  	   		 	 	 			  		 			     			  	 
        df=None,	  	   		 	 	 			  		 			     			  	 
        sd=dt.datetime(2020, 1, 1),  		  	   		 	 	 			  		 			     			  	 
        ed=dt.datetime(2025, 1, 1),  	   		 	 	 			  		 			     			  	 
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
        # Rolling Window
        N = 5		

        # Get stock return data along with indicators for each date
        # stock_data = df#.loc[sd:ed].copy()
        df = df.copy()
        df.set_index('date', inplace=True)
        # print(df)
        # k1, k2 = 1.0, 0.5

        # Calculate slopes for MACD and MACD Signal
        # df['macd_slope'] = df['MACD'].diff()
        # df['macd_signal_slope'] = df['MACD_Signal'].diff()

        # Calculate distance between MACD and MACD Signal
        df['macd_score'] = df['MACD'] - df['MACD_Signal']
        df['macd_slope'] = df['macd_score'].diff()


        df['rsi_score'] = (df['RSI'] - df['RSI'].min()) / (df['RSI'].max() - df['RSI'].min())
        df['bb_pct'] = (df['adj_close'] - df['lower_bollinger']) / (df['upper_bollinger'] - df['lower_bollinger'])

        # print(df[['macd_score', 'macd_slope']].iloc[-100:-50])

        # Debugging: Print relevant columns for inspection
        feature_cols = ['macd_score', 'macd_slope', 'rsi_score', 'bb_pct']

        # Get stock returns, with date barrier to avoid NA values later
        returns_df = df[['adj_close']].copy()
        returns_df = returns_df[returns_df['adj_close'] > 0]
        
        # Calculate % returns for N days ahead of current date
        df['N_day_ret'] = ((df['adj_close'].shift(-N) / df['adj_close'])) - 1

        df = df.dropna()
        # returns_df = returns_df.dropna()
        X = df[feature_cols].to_numpy()



        # Create Y dataframe to set to buy, sell, hold signals when Rolling Returns is high or low
        df['Action'] = np.where(df['N_day_ret'] > 0.02, 1, np.where(df['N_day_ret'] < -.02, -1, 0))
        print(df['Action'])
        Y = df['Action'].to_numpy()
        # print(Y)

        # Train learner on training data
        self.learner.add_evidence(X, Y)
  		  	   		 	 	 			  		 			     			  	  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
    # this method should use the existing policy and test it against new data  		  	   		 	 	 			  		 			     			  	 
    def testPolicy(  		  	   		 	 	 			  		 			     			  	 
        self,  		  	   		 	 	 			  		 			     			  	 
        df=None,	  	   		 	 	 			  		 			     			  	 
        sd=dt.datetime(2008, 1, 1),  		  	   		 	 	 			  		 			     			  	 
        ed=dt.datetime(2009, 1, 1),  	   		 	 	 			  		 			     			  	 
        sv=10000,   		 	 	 			  		 			     			  	 
    ):  		  	   		 	 	 			  		 			     			  	 
        """  		  	   		 	 	 			  		 			     			  	 
        Tests your learner using data outside of the training data  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
        :param symbol: The stock symbol that you trained on on  		  	   		 	 	 			  		 			     			  	 
        :type symbol: str  		  	   		 	 	 			  		 			     			  	 
        :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		 	 	 			  		 			     			  	 
        :type sd: datetime  		  	   		 	 	 			  		 			     			  	 
        :param ed: A datetime object that represents the end date, defaults to 1/1/2009  		  	   		 	 	 			  		 			     			  	 
        :type ed: datetime  		  	   		 	 	 			  		 			     			  	 
        :param sv: The starting value of the portfolio  		  	   		 	 	 			  		 			     			  	 
        :type sv: int  		  	   		 	 	 			  		 			     			  	 
        :return: A DataFrame with values representing trades for each day. Legal values are +1000.0 indicating  		  	   		 	 	 			  		 			     			  	 
            a BUY of 1000 shares, -1000.0 indicating a SELL of 1000 shares, and 0.0 indicating NOTHING.  		  	   		 	 	 			  		 			     			  	 
            Values of +2000 and -2000 for trades are also legal when switching from long to short or short to  		  	   		 	 	 			  		 			     			  	 
            long so long as net holdings are constrained to -1000, 0, and 1000.  		  	   		 	 	 			  		 			     			  	 
        :rtype: pandas.DataFrame  		  	   		 	 	 			  		 			     			  	 
        """  		  	   		 	 	 			  		 			     			  	 
        # Get stock data along with indicator features for StrategyLearner input
        df = df.copy()
        df.set_index('date', inplace=True)
        # print(df)
        # k1, k2 = 1.0, 0.5

        # Calculate slopes for MACD and MACD Signal
        # df['macd_slope'] = df['MACD'].diff()
        # df['macd_signal_slope'] = df['MACD_Signal'].diff()

        # Calculate distance between MACD and MACD Signal
        df['macd_score'] = df['MACD'] - df['MACD_Signal']
        df['macd_slope'] = df['macd_score'].diff()


        df['rsi_score'] = (df['RSI'] - df['RSI'].min()) / (df['RSI'].max() - df['RSI'].min())
        df['bb_pct'] = df['adj_close'] - df['lower_bollinger'] / (df['upper_bollinger'] - df['lower_bollinger'])

        # print(df[['macd_score', 'macd_slope']].iloc[-100:-50])

        # Debugging: Print relevant columns for inspection
        feature_cols = ['macd_score', 'macd_slope', 'rsi_score', 'bb_pct']
        X = df[feature_cols].to_numpy()

        # Get action choice from learner
        actions = self.learner.query(X)
        
        # Convert numpy array of outputs to a dataframe
        preds_df = pd.DataFrame(actions, index=df.index, columns=['Action'])

        # Set signals_df to buy and sell signals based upon Learner predictions
        preds_df['Shares'] = np.where(preds_df['Action'] > 0.2, 1,
                                    np.where(preds_df['Action'] < -0.2, -1, 0))
        signal_df = preds_df['Shares']

        # Get and return trade orders based on learner output
        trades_df = self.get_trade_orders(signal_df)


        return trades_df

    
    # def get_trade_orders(self, signal_df):
    #     trades_df = pd.DataFrame(0, index=signal_df.index, columns=["Shares"])
    
    #     # Get previous day trades as df
    #     signal_shifted = signal_df.shift(1).fillna(0)

    #     # Entry signals (from 0 to non-zero)
    #     trades_df["Shares"] = ((signal_shifted == 0) & (signal_df != 0)) * (signal_df * 1000)

    #     # Exit signals (from non-zero to zero)
    #     trades_df["Shares"] += ((signal_shifted != 0) & (signal_df == 0)) * (-signal_shifted * 1000)

    #     # Position reversal signals (switching from +1 to -1 or vice versa)
    #     trades_df["Shares"] += ((signal_shifted != 0) & (signal_df != 0) & (signal_df != signal_shifted)) * (-2 * signal_shifted * 1000)

    #     # Buy or sell at the next days price
    #     trades_df['Shares'] = trades_df['Shares']

    #     return trades_df

    def get_trade_orders(self, signal_df):
        trades_df = pd.DataFrame(0, index=signal_df.index, columns=["Shares"])
        
        current_position = 0  # Start with no position
        
        for i in range(len(signal_df)):
            signal = signal_df.iloc[i]
            
            if signal == 1:
                if current_position == 0:
                    trades_df.iloc[i] = 100  # Buy
                    current_position = 100
                else:
                    trades_df.iloc[i] = 0  # Hold
            elif signal == 0:
                # if current_position == 1000:
                #     trades_df.iloc[i] = -1000  # Sell to 0
                #     current_position = 0
                # else:
                trades_df.iloc[i] = 0  # Already at 0
            elif signal == -1:
                if current_position == 100:
                    trades_df.iloc[i] = -100  # Sell to 0
                    current_position = 0
                else:
                    trades_df.iloc[i] = 0  # No position, nothing to do
            else:
                trades_df.iloc[i] = 0  # Unknown signal or hold

        return trades_df


    # def author(self):  		  	   		 	 	 			  		 			     			  	 
    #     """  		  	   		 	 	 			  		 			     			  	 
    #     :return: The GT username of the student  		  	   		 	 	 			  		 			     			  	 
    #     :rtype: str  		  	   		 	 	 			  		 			     			  	 
    #     """  		  	   		 	 	 			  		 			     			  	 
    #     return "afitton3"  # replace tb34 with your Georgia Tech username.  		'

    # def study_group(self):
    #     """
    #     :return: A comma separated string of GT_Name of each member of your study group
    #     :rtype: str
    #     """
    #     return "afitton3"  	 	  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
if __name__ == "__main__":  		  	   		 	 	 			  		 			     			  	 
    print("One does not simply think up a strategy")  		  	   		 	 	 			  		 			     			  	 
