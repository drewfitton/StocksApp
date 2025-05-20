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
        
        group_df = group_df.bfill().ffill()[20:]

        learner.add_evidence(df=group_df[:-50])

    for stock_id, group_df in df_sorted.groupby("stock_id"):

        for ind in ['Bollinger', 'RSI', 'MACD']:
            ind_df = indicators.__dict__[f"calc_{ind}"](group_df[['adj_close']], 'adj_close')
            # print(ind_df)
            group_df = pd.concat([ind_df, group_df], axis=1)
        
        group_df = group_df.bfill().ffill()


        mlInds = learner.testPolicy(df=group_df[-50:])

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
    port_returns = []
    learner = slt()

    with open('results.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['StockId', 'portval']) 
        for i in range(0, 10):
    
            for stock_id, group_df in df_sorted.groupby("stock_id"):

                for ind in ['Bollinger', 'RSI', 'MACD']:
                    ind_df = indicators.__dict__[f"calc_{ind}"](group_df[['adj_close']], 'adj_close')
                    # print(ind_df)
                    group_df = pd.concat([ind_df, group_df], axis=1)
                
                group_df = group_df.bfill().ffill()[20:]

                learner.add_evidence(df=group_df[:-251])


            for stock_id, group_df in df_sorted.groupby("stock_id"):

                for ind in ['Bollinger', 'RSI', 'MACD']:
                    ind_df = indicators.__dict__[f"calc_{ind}"](group_df[['adj_close']], 'adj_close')
                    # print(ind_df)
                    group_df = pd.concat([ind_df, group_df], axis=1)
                
                group_df = group_df.bfill().ffill()[20:]

                # learner = slt()

                # learner.add_evidence(df=group_df[:-251])

                trades = learner.testPolicy(df=group_df[-251:])

                # print(trades.head(50))
                orders_df = create_orders(clean_trades=trades, symbol=stock_id)

                port_vals = compute_portvals(orders_df, stock_id, start_val=100000, commission=0, impact=0)

                port_val = port_vals.iloc[-1] / port_vals.iloc[0]
                port_returns.append(port_val)
                N = 5
                test_df = group_df[-251:].copy()
                dates = test_df.index
                benchmark_base = test_df['adj_close'] * 1000
                benchmark = benchmark_base / benchmark_base.iloc[0]
                port_vals_df = port_vals / port_vals.iloc[0]
                trades_df = trades / 1000

                # Plot
                # plt.figure(figsize=(10, 5))
                # plt.plot(dates, benchmark, label="Benchmark", color='blue')
                # plt.plot(dates, trades_df, label="Portfolio Value", color='orange')
                # plt.xlabel("Date")
                # plt.ylabel("Return / Prediction")
                # plt.title(f"Stock {stock_id} - ML Predictions vs Actual Returns")
                # plt.legend()
                # plt.grid(True)
                # plt.xticks(rotation=45)
                # plt.tight_layout()
                # plt.show()

                # writer.writxerow([stock_id, port_val])

                # if i >= 5:
                #     break
            writer.writerow(['Average', np.mean(port_returns)])
            # print(port_vals.tail(50))
    

        # mlInds = np.array(mlInds).flatten()

        # # Calculate actual returns



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


if __name__ == "__main__":
    symbol = 'JPM'
    # run_testproject(symbol)
    # experiment1.run_exp1_insample(symbol)
    # experiment1.run_exp1_outsample(symbol)
    # experiment2.run_exp2(symbol)
    populdate_ml_ind()
    # test_strategy_learner()