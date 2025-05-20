import yfinance as yf
import psycopg2
from config import DB_DETAILS, TICKER_DICT, TICKER_TO_DOMAIN
from StrategyLearner import StrategyLearner as sl
from sqlalchemy import create_engine
import pandas as pd
from psycopg2.extras import execute_values
from datetime import timedelta
import indicators
import numpy as np


def get_prices(date):
    df = yf.download(list(TICKER_DICT.keys()), interval='1d', start=date, auto_adjust=False, group_by='ticker')

    df = df.stack(level=0, future_stack=True).reset_index()
    # Convert 'date' to datetime format
    df["Date"] = pd.to_datetime(df["Date"])

    # Rename columns to match SQL table schema (case-sensitive!)
    df = df.rename(columns={
        "Date": "date",
        "Ticker": "ticker",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Volume": "volume"
    })

    return df


def insert_price_data(df):
    # Connect to the database
    conn = psycopg2.connect(**DB_DETAILS)
    cur = conn.cursor()

    # 1. Insert all stocks first
    stock_data = [(symbol, company) for symbol, company in TICKER_DICT.items()]
    execute_values(cur,
        "INSERT INTO stock (symbol, company) VALUES %s ON CONFLICT(symbol) DO NOTHING",
        stock_data
    )

    # 2. Fetch all stock ids at once
    cur.execute("SELECT id, symbol FROM stock")
    stock_rows = cur.fetchall()
    symbol_to_id = {symbol: stock_id for stock_id, symbol in stock_rows}

    # 3. Prepare all stock_price inserts
    stock_price_data = []
    missing_symbols = set()

    for _, row in df.iterrows():
        stock_id = symbol_to_id.get(row["ticker"])
        if stock_id:
            stock_price_data.append((
                stock_id,
                row["date"],
                row["open"],
                row["high"],
                row["low"],
                row["close"],
                row["adj_close"],
                row["volume"]
            ))
        else:
            missing_symbols.add(row["ticker"])

    if missing_symbols:
        print(f"Warning: Missing stock IDs for tickers: {missing_symbols}")

    # 4. Bulk insert stock_price rows
    if stock_price_data:
        execute_values(cur, """
            INSERT INTO stock_price (stock_id, date, open, high, low, close, adj_close, volume)
            VALUES %s
            ON CONFLICT (stock_id, date) DO NOTHING
        """, stock_price_data)

    # Commit and close
    conn.commit()
    cur.close()
    conn.close()


def update_prices():
    conn = psycopg2.connect(**DB_DETAILS)
    cur = conn.cursor()

    # Step 1: Get the latest date for each stock
    cur.execute("""
        SELECT s.symbol, MAX(sp.date)
        FROM stock s
        LEFT JOIN stock_price sp ON s.id = sp.stock_id
        GROUP BY s.symbol
    """)
    latest_dates = cur.fetchall()

    # Build a dict: {symbol: latest_date}
    symbol_to_latest_date = {symbol: latest_date for symbol, latest_date in latest_dates}

    conn.commit()
    cur.close()
    conn.close()

    # Step 2: Determine the earliest date to fetch
    # If a stock has no data, start from a default
    start_date = min([
        (date + pd.Timedelta(days=1)) if date else pd.Timestamp('2015-01-01')
        for date in symbol_to_latest_date.values()
    ]).strftime('%Y-%m-%d')

    # Step 3: Call get_prices with start_date
    df = get_prices(start_date)

    # Step 4: Insert new data
    insert_price_data(df)

def insert_stock_metadata(cur, ticker_data):
    """
    ticker_data: list of tuples like (symbol, company, ret_1w, ret_1m, ret_6m, ret_1y, ret_ytd, ret_5y)
    """
    stock_rows = [
        (
            symbol, company, ret_1w, ret_1m, ret_6m, ret_1y, ret_ytd, ret_5y, 0  # ML_indicator = 0
        )
        for symbol, company, ret_1w, ret_1m, ret_6m, ret_1y, ret_ytd, ret_5y in ticker_data
    ]

    insert_sql = """
        INSERT INTO stock (symbol, company, one_week, one_month, six_month, one_year, ytd, five_year, ML_ind)
        VALUES %s
        ON CONFLICT(symbol)
        DO UPDATE SET
            one_week = EXCLUDED.one_week,
            one_month = EXCLUDED.one_month,
            six_month = EXCLUDED.six_month,
            one_year = EXCLUDED.one_year,
            ytd = EXCLUDED.ytd,
            five_year = EXCLUDED.five_year,
            ML_ind = EXCLUDED.ML_ind
    """

    execute_values(cur, insert_sql, stock_rows)


def calculate_returns(df):
    df['date'] = pd.to_datetime(df['date'])

    results = []

    for ticker, group in df.groupby('ticker'):
        group = group.sort_values('date')
        group.set_index('date', inplace=True)

        latest_date = group.index.max()
        latest_price = group.loc[latest_date]['adj_close']

        def get_return(days_ago=None, months_ago=None, years_ago=None, use_ytd=False):
            if use_ytd:
                target_date = pd.Timestamp(year=latest_date.year, month=1, day=1)
            else:
                target_date = pd.Timestamp.today()
                if days_ago:
                    target_date -= timedelta(days=days_ago)
                if months_ago:
                    target_date -= pd.DateOffset(months=months_ago)
                if years_ago:
                    target_date -= pd.DateOffset(years=years_ago)
            # Get nearest price on or before target_date
            try:
                nearest_date = group.loc[target_date:].index.min()
                past_price = group.loc[nearest_date]['adj_close']
                return round((latest_price - past_price) / past_price * 100, 2)
            except:
                return None

        ret_1w = float(get_return(days_ago=7))
        ret_1m = float(get_return(months_ago=1))
        ret_6m = float(get_return(months_ago=6))
        ret_1y = float(get_return(years_ago=1))
        ret_ytd = float(get_return(use_ytd=True))
        ret_5y = float(get_return(years_ago=5))

        company = TICKER_DICT.get(ticker, "Unknown")

        results.append((ticker, company, ret_1w, ret_1m, ret_6m, ret_1y, ret_ytd, ret_5y))

    return results

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


if __name__ == "__main__":
    update_prices()

    df = get_prices('2020-01-01')
    returns_data = calculate_returns(df)

    conn = psycopg2.connect(**DB_DETAILS)
    cur = conn.cursor()

    insert_stock_metadata(cur, returns_data)
    conn.commit()
    cur.close()
    conn.close()

    populdate_ml_ind()