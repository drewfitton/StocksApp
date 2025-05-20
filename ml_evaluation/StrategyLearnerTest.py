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
    ):  		  	   		 	 	 			  		 			     			  	 
        """  		  	   		 	 	 			  		 			     			  	 
        Trains your strategy learner over a given time frame.  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
        :param df: Dataframe of stock data	  	   		 	 	 			  		 			     			  	 		  	   		 	 	 			  		 			     			  	 
        :rtype: None	  	   		 	 	 			  		 			     			  	 
        """  		  	   	
        # Rolling Window
        N = 10

        # Set index
        df = df.copy()
        df.set_index('date', inplace=True)


        # Get MACD Difference from signal and crossover points
        df['macd_score'] = df['MACD'] - df['MACD_Signal']
        df['macd_crossover'] = 0
        df.loc[
            (df['MACD'].shift(1) < df['MACD_Signal'].shift(1)) & 
            (df['MACD'] > df['MACD_Signal']), 
            'macd_crossover'
        ] = 1

        df.loc[
            (df['MACD'].shift(1) > df['MACD_Signal'].shift(1)) & 
            (df['MACD'] < df['MACD_Signal']), 
            'macd_crossover'
        ] = -1

        # Normalize RSI and get RSI trend
        df['rsi_score'] = df['RSI'] / 100
        df['rsi_trend'] = df['RSI'].diff()

        # Normalize Bollinger Bands
        df['bb_pct'] = (df['adj_close'] - df['lower_bollinger']) / (df['upper_bollinger'] - df['lower_bollinger'])

        # Normalize Volume
        df['norm_volume'] = (df['volume'] - df['volume'].min()) / (df['volume'].max() - df['volume'].min())

        # Set feature columns
        feature_cols = ['macd_score', 'MACD', 'macd_crossover', 'rsi_score', 'rsi_trend', 'bb_pct', 'norm_volume']

        
        # Calculate % returns for N days ahead of current date
        df['N_day_ret'] = ((df['adj_close'].shift(-N) / df['adj_close'])) - 1

        df = df.dropna()
        

        # Create Y dataframe to set to buy, sell, hold signals when Rolling Returns is high or low
        buy_thresh = df['N_day_ret'].quantile(0.80)   # Top 20% → Buy
        sell_thresh = df['N_day_ret'].quantile(0.20)  # Bottom 20% → Sell
        df['Action'] = np.where(df['N_day_ret'] > buy_thresh, 1,
                                np.where(df['N_day_ret'] < sell_thresh, -1, 0))
        
        # Set X, Y and train the learner
        X = df[feature_cols].to_numpy()
        Y = df['Action'].to_numpy()
        self.learner.add_evidence(X, Y)
  		  	   		 	 	 			  		 			     			  	  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
    # this method should use the existing policy and test it against new data  		  	   		 	 	 			  		 			     			  	 
    def testPolicy(  		  	   		 	 	 			  		 			     			  	 
        self,  		  	   		 	 	 			  		 			     			  	 
        df=None,	  	   		 	 	 			  		 			     			  	  		 	 	 			  		 			     			  	 
    ):  		  	   		 	 	 			  		 			     			  	 
        """  		  	   		 	 	 			  		 			     			  	 
        Tests your learner using data outside of the training data  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
        :param df: Dataframe of stock data	  	   		 	 	 			  		 			     			  	 		  	   		 	 	 			  		 			     			  	 
        :rtype: pandas.DataFrame  		  	   		 	 	 			  		 			     			  	 
        """  		  	   		 	 	 			  		 			     			  	 
        # Set index
        df = df.copy()
        df.set_index('date', inplace=True)


        # Get MACD Difference from signal and crossover points
        df['macd_score'] = df['MACD'] - df['MACD_Signal']
        df['macd_crossover'] = 0
        df.loc[
            (df['MACD'].shift(1) < df['MACD_Signal'].shift(1)) & 
            (df['MACD'] > df['MACD_Signal']), 
            'macd_crossover'
        ] = 1

        df.loc[
            (df['MACD'].shift(1) > df['MACD_Signal'].shift(1)) & 
            (df['MACD'] < df['MACD_Signal']), 
            'macd_crossover'
        ] = -1

        # Normalize RSI and get RSI trend
        df['rsi_score'] = df['RSI'] / 100
        df['rsi_trend'] = df['RSI'].diff()

        # Normalize Bollinger Bands
        df['bb_pct'] = (df['adj_close'] - df['lower_bollinger']) / (df['upper_bollinger'] - df['lower_bollinger'])

        # Normalize Volume
        df['norm_volume'] = (df['volume'] - df['volume'].min()) / (df['volume'].max() - df['volume'].min())

        # Set feature columns
        feature_cols = ['macd_score', 'MACD', 'macd_crossover', 'rsi_score', 'rsi_trend', 'bb_pct', 'norm_volume']

        # Set X and get actions from learner
        X = df[feature_cols].to_numpy()
        actions = self.learner.query(X)
        
        # Convert numpy array of outputs to a dataframe
        preds_df = pd.DataFrame(actions, index=df.index, columns=['Action'])

        preds_df['RollingAction'] = preds_df['Action'].rolling(window=3).mean()

        # Define conditions on the 3-day rolling average instead of raw 'Action'
        conditions = [
            preds_df['RollingAction'] > 0.2,
            preds_df['RollingAction'] > 0.1,
            preds_df['RollingAction'] < -0.2,
            preds_df['RollingAction'] < -0.1
        ]

        choices = [2, 1, -2, -1]

        # Apply decisions based on rolling average
        preds_df['Shares'] = np.select(conditions, choices, default=0)

        # Final signals
        signal_df = preds_df['Shares']

        # Get and return trade orders based on learner output
        trades_df = self.get_trade_orders(signal_df)


        return trades_df


    def get_trade_orders(self, signal_df, base_amount=1000):
        trades_df = pd.DataFrame(0, index=signal_df.index, columns=["Shares"])
        
        current_position = 0
        last_trade_day = None
        
        for i in range(len(signal_df)):
            signal = signal_df.iloc[i]
            current_date = signal_df.index[i]

            can_trade = (
                last_trade_day is None or 
                (current_date - last_trade_day).days >= 5
            )

            if not can_trade:
                continue

            trade_amount = 0

            if signal == 2:        # Strong buy
                trade_amount = base_amount
            elif signal == 1:      # Weak buy
                trade_amount = base_amount / 2
            elif signal == -1:     # Weak sell
                trade_amount = -base_amount / 2
            elif signal == -2:     # Strong sell
                trade_amount = -base_amount

            # Prevent overbuying/selling more than we have
            new_position = current_position + trade_amount
            if new_position < 0:
                trade_amount = -current_position  # Can't go below 0
            if new_position > base_amount:
                trade_amount = base_amount - current_position  # Cap at full position

            if trade_amount != 0:
                trades_df.iloc[i] = trade_amount
                current_position += trade_amount
                last_trade_day = current_date

        return trades_df
  		 			     			  	 
if __name__ == "__main__":  		  	   		 	 	 			  		 			     			  	 
    print("One does not simply think up a strategy")  		  	   		 	 	 			  		 			     			  	 
