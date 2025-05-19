# import ManualStrategy as ms
import csv
from StrategyLearnerTest import StrategyLearner as slt
from StrategyLearner import StrategyLearner as sl
import matplotlib.pyplot as plt
import datetime as dt
import pandas as pd
import psycopg2
from marketsimcode import compute_portvals
from config import DB_DETAILS
import indicators
import numpy as np

def populdate_ml_ind():
    conn = psycopg2.connect(**DB_DETAILS)
    cur = conn.cursor()

    cur.execute("""
        SELECT * from stock_price
    """)

    df = pd.DataFrame(cur.fetchall(), columns=[desc[0] for desc in cur.description])
    df["date"] = pd.to_datetime(df["date"])
    df["adj_close"] = df["adj_close"].astype(float)
    df["close"] = df["close"].astype(float)
    df["open"] = df["open"].astype(float)
    df["high"] = df["high"].astype(float)
    df["low"] = df["low"].astype(float)
    df["volume"] = df["volume"].astype(float)

    df_sorted = df.sort_values(by=["stock_id", "date"])

    indications = {}
    i = 0
    returns = {}
    learner = sl()
    for stock_id, group_df in df_sorted.groupby("stock_id"):

        for ind in ['Bollinger', 'RSI', 'MACD']:
            ind_df = indicators.__dict__[f"calc_{ind}"](group_df[['adj_close']], 'adj_close')
            # print(ind_df)
            group_df = pd.concat([ind_df, group_df], axis=1)
        
        group_df = group_df.bfill().ffill()

        learner.add_evidence(df=group_df[:-5])

    for stock_id, group_df in df_sorted.groupby("stock_id"):

        for ind in ['Bollinger', 'RSI', 'MACD']:
            ind_df = indicators.__dict__[f"calc_{ind}"](group_df[['adj_close']], 'adj_close')
            # print(ind_df)
            group_df = pd.concat([ind_df, group_df], axis=1)
        
        group_df = group_df.bfill().ffill()


        mlInds = learner.testPolicy(df=group_df[-5:])

        mlInds = np.array(mlInds).flatten()


        indications[stock_id] = float(mlInds[-1])

    
    for id in indications.keys():
        cur.execute("""
            UPDATE stock
            SET ml_ind = %s
            WHERE id = %s
        """, (indications[id], id))

    conn.commit()
    cur.close()
    conn.close()

def test_strategy_learner():
    conn = psycopg2.connect(**DB_DETAILS)
    cur = conn.cursor()

    cur.execute("""
        SELECT * from stock_price
    """)

    df = pd.DataFrame(cur.fetchall(), columns=[desc[0] for desc in cur.description])
    df["date"] = pd.to_datetime(df["date"])
    df["adj_close"] = df["adj_close"].astype(float)
    df["close"] = df["close"].astype(float)
    df["open"] = df["open"].astype(float)
    df["high"] = df["high"].astype(float)
    df["low"] = df["low"].astype(float)
    df["volume"] = df["volume"].astype(float)

    df_sorted = df.sort_values(by=["stock_id", "date"])

    indications = {}
    i = 0
    returns = {}
    learner = slt()
    for stock_id, group_df in df_sorted.groupby("stock_id"):

        for ind in ['Bollinger', 'RSI', 'MACD']:
            ind_df = indicators.__dict__[f"calc_{ind}"](group_df[['adj_close']], 'adj_close')
            # print(ind_df)
            group_df = pd.concat([ind_df, group_df], axis=1)
        
        group_df = group_df.bfill().ffill()

        learner.add_evidence(df=group_df[:-251])

    with open('results.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['StockId', 'portval']) 

        for stock_id, group_df in df_sorted.groupby("stock_id"):

            for ind in ['Bollinger', 'RSI', 'MACD']:
                ind_df = indicators.__dict__[f"calc_{ind}"](group_df[['adj_close']], 'adj_close')
                # print(ind_df)
                group_df = pd.concat([ind_df, group_df], axis=1)
            
            group_df = group_df.bfill().ffill()

            # learner.add_evidence(df=group_df[:-251])

            trades = learner.testPolicy(df=group_df[-251:])

            # print(trades.head(50))
            orders_df = create_orders(clean_trades=trades, symbol=stock_id)

            port_vals = compute_portvals(orders_df, stock_id, start_val=100000, commission=0, impact=0)

            port_val = port_vals.iloc[-1] / port_vals.iloc[0]

            writer.writerow([stock_id, port_val])

            break
        

        # print(port_vals.tail(50))
    

        # mlInds = np.array(mlInds).flatten()

        # # Calculate actual returns
        # N = 5
        # test_df = group_df[-50:].copy()
        # test_df['N_day_ret'] = test_df['adj_close'].shift(-N) / test_df['adj_close']
        # test_df = test_df[:-N]  # Drop rows that don't have valid future return

        # dates = test_df.index
        # actual_returns = test_df['adj_close'].values / max(test_df['adj_close']) - 1
        # mlInds = mlInds[:len(actual_returns)] # Match lengths

        # Plot
        # plt.figure(figsize=(10, 5))
        # plt.plot(dates, actual_returns, label="Actual 5-Day Returns", color='blue')
        # plt.plot(dates, mlInds, label="ML Predictions", color='orange')
        # plt.xlabel("Date")
        # plt.ylabel("Return / Prediction")
        # plt.title(f"Stock {stock_id} - ML Predictions vs Actual Returns")
        # plt.legend()
        # plt.grid(True)
        # plt.xticks(rotation=45)
        # plt.tight_layout()
        # plt.show()
        # print(mlInd_df)


        # indications[stock_id] = float(mlInds[-1])

    conn.commit()
    cur.close()
    conn.close()


# def create_orders(clean_trades, symbol=1):

#     # Create psuedo orders file with trade signals
#     orders = pd.DataFrame(index=clean_trades.index.values, columns=["Symbol", "Order", "Shares"])

#     # Clean
#     orders["Symbol"] = symbol
#     # Use np.select to classify the trades
#     conditions = [clean_trades > 0, clean_trades < 0, clean_trades == 0]
#     choices = ["BUY", "SELL", "HOLD"]
#     orders["Order"] = np.select(conditions, choices)
#     # orders = orders.drop(orders[orders["Shares"] == 0].index)
#         # Use np.select to classify the trades
#     orders["Shares"] = abs(clean_trades).astype(int)

#     return orders

def create_orders(clean_trades, symbol=1):

    # Create psuedo orders file with trade signals
    orders = pd.DataFrame(index=clean_trades.index.values, columns=["Symbol", "Order", "Shares"])

    # Clean
    orders["Symbol"] = symbol
    orders["Order"] = clean_trades.where(clean_trades < 0, "BUY").where(clean_trades > 0, "SELL").where(((clean_trades > 0) | (clean_trades < 0)), "HOLD")
    orders["Shares"] = abs(clean_trades)
    # orders = orders.drop(orders[orders["Shares"] == 0].index)

    return orders


# def run_testproject(symbol):

#     sv = 100000
#     # ----------------------------------------------- #
#     # -------------- In Sample - MANUAL ------------- #
#     # ----------------------------------------------- #
#     manual = ms.ManualStrategy()

#     sd = dt.datetime(2008, 1, 1)
#     ed = dt.datetime(2009, 12, 31)

#     stock_vals = get_data([symbol], pd.date_range(sd, ed)).drop(columns="SPY")

#     benchmark = get_data([symbol], pd.date_range(sd, ed)).drop(columns="SPY") * 1000
#     benchmark["Benchmark"] = benchmark / benchmark.iloc[0]

#     trades_df = manual.testPolicy(symbol=symbol, sd=sd, ed=ed, sv=sv)
#     orders_df = create_orders(clean_trades=trades_df, symbol=symbol)
#     port_vals = compute_portvals(orders_df, start_val=sv, commission=9.95, impact=0.005)

#     # 1. Calculate daily returns for both portfolios
#     in_returns = port_vals.pct_change().dropna()
#     in_stock_returns = stock_vals.pct_change().dropna()

#     # 2. Cumulative Return (from the first value to the last value)
#     in_cum_return = (port_vals.iloc[-1] / port_vals.iloc[0]) - 1
#     in_stock_cum_return = (stock_vals.iloc[-1] / stock_vals.iloc[0]) - 1

#     # 3. Standard Deviation of daily returns
#     in_stdev = in_returns.std()
#     in_stock_stdev = in_stock_returns.std()

#     # 4. Mean of daily returns
#     in_mean_return = in_returns.mean()
#     in_stock_mean_return = in_stock_returns.mean()

#     port_vals = port_vals / port_vals.iloc[0]

#     fig = plt.figure(figsize=(9, 6))
#     plt.plot(benchmark['Benchmark'], color='purple')
#     plt.plot(port_vals, color='r')
    
#     long_signals = manual.signals[(manual.signals.shift(1) != 1) & (manual.signals > 0)].index
#     short_signals = manual.signals[(manual.signals.shift(1) != -1) & (manual.signals < 0)].index
#     long_prices = port_vals.loc[long_signals]
#     short_prices = port_vals.loc[short_signals]
#     ylim = plt.ylim()
#     plt.vlines(x=long_signals, ymin=long_prices, ymax=ylim[1], color='blue', linestyle='-', linewidth=1.5, label="Long Entry")
#     plt.vlines(x=short_signals, ymin=ylim[0], ymax=short_prices, color='black', linestyle='-', linewidth=1.5, label="Short Entry")  

#     plt.legend(["Benchmark", "Manual Portfolio"])
#     plt.xlabel("Date")
#     plt.ylabel("Normalized Portfolio Value")
#     plt.title("Manual Strategy vs Benchmark Portfolio - In Sample")
#     plt.tick_params(axis='x', rotation=45)
#     plt.savefig("./images/ManualStrategy_InSample.png")
#     plt.close()

#     # ----------------------------------------------- #
#     # ------------ Out of Sample - MANUAL ----------- #
#     # ----------------------------------------------- #
#     manual = ms.ManualStrategy()

#     sd = dt.datetime(2010, 1, 1)
#     ed = dt.datetime(2011, 12, 31)


#     stock_vals = get_data([symbol], pd.date_range(sd, ed)).drop(columns="SPY")

#     benchmark = get_data([symbol], pd.date_range(sd, ed)).drop(columns="SPY") * 1000
#     benchmark["Benchmark"] = benchmark / benchmark.iloc[0]

#     trades_df = manual.testPolicy(symbol=symbol, sd=sd, ed=ed, sv=sv)
#     orders_df = create_orders(clean_trades=trades_df, symbol=symbol)
#     port_vals = compute_portvals(orders_df, start_val=sv, commission=9.95, impact=0.005)

#     # 1. Calculate daily returns for both portfolios
#     out_returns = port_vals.pct_change().dropna()
#     out_stock_returns = stock_vals.pct_change().dropna()

#     # 2. Cumulative Return (from the first value to the last value)
#     out_cum_return = (port_vals.iloc[-1] / port_vals.iloc[0]) - 1
#     out_stock_cum_return = (stock_vals.iloc[-1] / stock_vals.iloc[0]) - 1

#     # 3. Standard Deviation of daily returns
#     out_stdev = out_returns.std()
#     out_stock_stdev = out_stock_returns.std()

#     # 4. Mean of daily returns
#     out_mean_return = out_returns.mean()
#     out_stock_mean_return = out_stock_returns.mean()

#     port_vals = port_vals / port_vals.iloc[0]

#     fig = plt.figure(figsize=(9, 6))
#     plt.plot(benchmark['Benchmark'], color='purple')
#     plt.plot(port_vals, color='r')
#     long_signals = manual.signals[(manual.signals.shift(1) != 1) & (manual.signals > 0)].index
#     short_signals = manual.signals[(manual.signals.shift(1) != -1) & (manual.signals < 0)].index
#     long_prices = port_vals.loc[long_signals]
#     short_prices = port_vals.loc[short_signals]
#     ylim = plt.ylim()
#     plt.vlines(x=long_signals, ymin=long_prices, ymax=ylim[1], color='blue', linestyle='-', linewidth=1.5, label="Long Entry")
#     plt.vlines(x=short_signals, ymin=ylim[0], ymax=short_prices, color='black', linestyle='-', linewidth=1.5, label="Short Entry")  

#     plt.legend(["Benchmark", "Manual Portfolio"])
#     plt.xlabel("Date")
#     plt.ylabel("Normalized Portfolio Value")
#     plt.title("Manual Strategy vs Benchmark Portfolio - Out of Sample")
#     plt.tick_params(axis='x', rotation=45)
#     plt.savefig("./images/ManualStrategy_OutSample.png")
#     plt.close()

#     # ----------------------------------------------- #
#     # -------------- PERFORMANCE CHART -------------- #
#     # ----------------------------------------------- #
#     data = {
#         'Metric': [
#             'Cumulative Return', 'Standard Deviation (STDEV)', 'Mean Return'
#         ],
#         'In Sample Manual': [
#             (f"{in_cum_return * 100:.4f} %"), (f"{in_stdev * 100:.4f} %"), (f"{in_mean_return * 100:.4f} %")
#         ],
#         'In Sample Stock': [
#             (f"{in_stock_cum_return.values[0] * 100:.4f} %"), (f"{in_stock_stdev.values[0] * 100:.4f} %"), (f"{in_stock_mean_return.values[0] * 100:.4f} %")
#         ],
#         'Out of Sample Manual': [
#             (f"{out_cum_return * 100:.4f} %"), (f"{out_stdev * 100:.4f} %"), (f"{out_mean_return * 100:.4f} %")
#         ],
#         'Out of Sample Stock': [
#             (f"{out_stock_cum_return.values[0] * 100:.4f} %"), (f"{out_stock_stdev.values[0] * 100:.4f} %"), (f"{out_stock_mean_return.values[0] * 100:.4f} %")
#         ]
#     }

#     # Create a pandas DataFrame
#     df = pd.DataFrame(data)

#     # Set the 'Metric' column as the index
#     df.set_index('Metric', inplace=True)

#     # Write to a CSV file
#     df.to_csv('./images/Manual_Strategy_Metrics.csv')

    
# def author():  		  	   		 	 	 			  		 			     			  	 
#     """  		  	   		 	 	 			  		 			     			  	 
#     :return: The GT username of the student  		  	   		 	 	 			  		 			     			  	 
#     :rtype: str  		  	   		 	 	 			  		 			     			  	 
#     """  		  	   		 	 	 			  		 			     			  	 
#     return "afitton3"  # replace tb34 with your Georgia Tech username.  		'

# def study_group():
#     """
#     :return: A comma separated string of GT_Name of each member of your study group
#     :rtype: str
#     """
#     return "afitton3" 

if __name__ == "__main__":
    symbol = 'JPM'
    # run_testproject(symbol)
    # experiment1.run_exp1_insample(symbol)
    # experiment1.run_exp1_outsample(symbol)
    # experiment2.run_exp2(symbol)
    test_strategy_learner()