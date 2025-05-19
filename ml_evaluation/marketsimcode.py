""""""  		  	   		 	 	 			  		 			     			  	 
"""MC2-P1: Market simulator.  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
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
import os  		  	  
from config import DB_DETAILS 		 	 	 			  		 			     			  	   
import psycopg2			  		 			     			  	 
import numpy as np  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
import pandas as pd  		  	   		 	 	 			  		 			     			  	 

def author():  		  	   		 	 	 			  		 			     			  	 
    """  		  	   		 	 	 			  		 			     			  	 
    :return: The GT username of the student  		  	   		 	 	 			  		 			     			  	 
    :rtype: str  		  	   		 	 	 			  		 			     			  	 
    """  		  	   		 	 	 			  		 			     			  	 
    return "afitton3"	  	   		 	 	 			  		 			     			  	 

def study_group():
    """
    : return: a comma separated string of GT_Name of each member of your study group
    : rtype: str
    """
    return "afitton3"   
  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
def compute_portvals(  		  	   		 	 	 			  		 			     			  	 
    orders,
    stock_id,		  	   		 	 	 			  		 			     			  	 
    start_val=1000000,  		  	   		 	 	 			  		 			     			  	 
    commission=0.0,  		  	   		 	 	 			  		 			     			  	 
    impact=0.0,  		  	   		 	 	 			  		 			     			  	 
):  		  	   		 	 	 			  		 			     			  	 
    """  		  	   		 	 	 			  		 			     			  	 
    Computes the portfolio values.  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
    :param orders_file: Path of the order file or the file object  		  	   		 	 	 			  		 			     			  	 
    :type orders_file: str or file object  		  	   		 	 	 			  		 			     			  	 
    :param start_val: The starting value of the portfolio  		  	   		 	 	 			  		 			     			  	 
    :type start_val: int  		  	   		 	 	 			  		 			     			  	 
    :param commission: The fixed amount in dollars charged for each transaction (both entry and exit)  		  	   		 	 	 			  		 			     			  	 
    :type commission: float  		  	   		 	 	 			  		 			     			  	 
    :param impact: The amount the price moves against the trader compared to the historical data at each transaction  		  	   		 	 	 			  		 			     			  	 
    :type impact: float  		  	   		 	 	 			  		 			     			  	 
    :return: the result (portvals) as a single-column dataframe, containing the value of the portfolio for each trading day in the first column from start_date to end_date, inclusive.  		  	   		 	 	 			  		 			     			  	 
    :rtype: pandas.DataFrame  		  	   		 	 	 			  		 			     			  	 
    """  		  	   		 	 	 			  		 			     			  	 
    # this is the function the autograder will call to test your code  		  	   		 	 	 			  		 			     			  	 
    # NOTE: orders_file may be a string, or it may be a file object. Your  		  	   		 	 	 			  		 			     			  	 
    # code should work correctly with either input  		  	   		 	 	 			  		 			     			  	 
    # TODO: Your code here  		  	   		 	 	 			  		 			     			  	 	  	   		 	 	 			  		 			     			  	 

    # Read orders and convert the dates to datetime format to sort
    orders['Date'] = pd.to_datetime(orders.index)
    orders = orders.sort_values(by='Date')

    # Get unique stock names from orders and first and last date of orders
    # stocks = set(orders['Symbol'])
    # start_date, end_date = orders['Date'].iloc[0], orders['Date'].iloc[-1]


    # Get prices data for the stocks I am ordering
    # prices = get_data(list(stocks), pd.date_range(start_date, end_date))
    conn = psycopg2.connect(**DB_DETAILS)
    cur = conn.cursor()

    cur.execute("""
        SELECT * from stock_price
        WHERE stock_id = %s
        ORDER BY date ASC 
    """, (str(stock_id)))
    prices = pd.DataFrame(cur.fetchall(), columns=[desc[0] for desc in cur.description])[-len(orders):]
    # print(prices)
    prices['date'] = pd.to_datetime(prices['date'])
    prices["adj_close"] = prices["adj_close"].astype(float)
    prices["close"] = prices["close"].astype(float)
    prices["open"] = prices["open"].astype(float)
    prices["high"] = prices["high"].astype(float)
    prices["low"] = prices["low"].astype(float)
    prices["volume"] = prices["volume"].astype(float)
    prices = prices.set_index('date')
    cur.close()
    conn.close()
    # print(prices)

    # Create holdings dataframe from prices, and add cash column with starting cash first row
    # This dataframe will track our stock quantity holdings at each date, along with cash available
    holdings = pd.DataFrame(0, index=prices.index, columns=['Shares', 'Cash'])
    holdings['Cash'] = 0.0
    holdings.at[orders['Date'].iloc[0], 'Cash'] = start_val  # Set initial cash
    holdings['stock_id'] = stock_id

    # print(holdings)
    # print(orders[50:100])

    # Iterate over my orders
    for _, row in orders.iterrows():
        
        # Extract order details for current order
        # print(row)
        date, symbol, order, shares = row['Date'], row['Symbol'], row['Order'], row['Shares']
        # print(order)
        
        if order == 'BUY':
            # print(1)
            # Adjust trade price for market impact, add ordered shares to our holdings, subtract cash from our purchase
            trade_price = prices.loc[date, 'adj_close'] * (1 + impact)
            cost = trade_price * shares
            holdings.loc[date, 'Shares'] += shares
            # print(cost)
            holdings.loc[date, 'Cash'] -= (cost + commission)
        elif order == 'SELL':
            # print(2)
            # print(cost)
            # Adjust trade price for market impact, remove sold shares from our holdings, add cash from the sale
            trade_price = prices.loc[date, 'adj_close'] * (1 - impact)
            cost = trade_price * shares
            holdings.loc[date, 'Shares'] -= shares 
            holdings.loc[date, 'Cash'] += (cost - commission)

    # print(holdings.head(50))
    # Get rolling sum of holdings over date range
    holdings = holdings.cumsum()
    # print(holdings)

    # Add const cash column to prices prices
    prices['Cash'] = 1
    # print(holdings)
    # print(prices)

    # Multiply holdings and prices to get holding values
    portvals = (holdings['Shares'] * prices['adj_close']) + holdings['Cash']
    # print(portvals)

    # Add holdings values with cash holdings
    # portvals = portvals.sum(axis=1)

    # print(portvals)	  		 			     			  	  		  	   		 	 	 			  		 			     			  	 
    return portvals  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
def test_code():  		  	   		 	 	 			  		 			     			  	 
    """  		  	   		 	 	 			  		 			     			  	 
    Helper function to test code  		  	   		 	 	 			  		 			     			  	 
    """  		  	   		 	 	 			  		 			     			  	 
    # this is a helper function you can use to test your code  		  	   		 	 	 			  		 			     			  	 
    # note that during autograding his function will not be called.  		  	   		 	 	 			  		 			     			  	 
    # Define input parameters  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
    of = "./orders/orders2.csv"  		  	   		 	 	 			  		 			     			  	 
    sv = 1000000  		   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
    # Process orders  		  	   		 	 	 			  		 			     			  	 
    portvals = compute_portvals(orders_file=of, start_val=sv)		  	   		 	 	 			  		 			     			  	 
    if isinstance(portvals, pd.DataFrame):  		  	   		 	 	 			  		 			     			  	 
        portvals = portvals[portvals.columns[0]]  # just get the first column  		  	   		 	 	 			  		 			     			  	 
    else:  		  	   		 	 	 			  		 			     			  	 
        "warning, code did not return a DataFrame"  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
    # Get portfolio stats  		  	   		 	 	 			  		 			     			  	 
    # Here we just fake the data. you should use your code from previous assignments.  		  	   		 	 	 			  		 			     			  	 
    start_date, end_date = portvals.index[0], portvals.index[-1]
    cum_ret = (portvals.iloc[-1] / portvals.iloc[0]) - 1

    # Avg daily returns and std dev daily returns
    daily_returns = portvals.pct_change()
    avg_daily_ret = daily_returns.mean()
    std_daily_ret = daily_returns.std()

    #Sharpe Ratio
    sharpe_ratio = (avg_daily_ret / std_daily_ret) * np.sqrt(252) 	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
    #Compare portfolio against $SPX  		  	   		 	 	 			  		 			     			  	 
    # print(f"Date Range: {start_date} to {end_date}")  		  	   		 	 	 			  		 			     			  	 
    # print()  		  	   		 	 	 			  		 			     			  	 
    # print(f"Sharpe Ratio of Fund: {sharpe_ratio}")  		  	   		 	 	 			  		 			     			  	 
    # # print(f"Sharpe Ratio of SPY : {sharpe_ratio_SPY}")  		  	   		 	 	 			  		 			     			  	 
    # print()  		  	   		 	 	 			  		 			     			  	 
    # print(f"Cumulative Return of Fund: {cum_ret}")  		  	   		 	 	 			  		 			     			  	 
    # # print(f"Cumulative Return of SPY : {cum_ret_SPY}")  		  	   		 	 	 			  		 			     			  	 
    # print()  		  	   		 	 	 			  		 			     			  	 
    # print(f"Standard Deviation of Fund: {std_daily_ret}")  		  	   		 	 	 			  		 			     			  	 
    # # print(f"Standard Deviation of SPY : {std_daily_ret_SPY}")  		  	   		 	 	 			  		 			     			  	 
    # print()  		  	   		 	 	 			  		 			     			  	 
    # print(f"Average Daily Return of Fund: {avg_daily_ret}")  		  	   		 	 	 			  		 			     			  	 
    # # print(f"Average Daily Return of SPY : {avg_daily_ret_SPY}")  		  	   		 	 	 			  		 			     			  	 
    # print()  		  	   		 	 	 			  		 			     			  	 
    # print(f"Final Portfolio Value: {portvals[-1]}")  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
if __name__ == "__main__":  		  	   		 	 	 			  		 			     			  	 
    test_code()  		  	   		 	 	 			  		 			     			  	 
