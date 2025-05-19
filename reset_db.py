import yfinance as yf
import psycopg2
from config import DB_DETAILS, TICKER_DICT, TICKER_TO_DOMAIN, RETURNS_TIMES
from sqlalchemy import create_engine
import pandas as pd
from psycopg2.extras import execute_values
from datetime import timedelta

def reset_db():
    conn = psycopg2.connect(**DB_DETAILS)
    cur = conn.cursor()

    cur.execute("DROP TABLE IF EXISTS stock CASCADE;")
    cur.execute("DROP TABLE IF EXISTS stock_price CASCADE;")

    cur.execute("""
        CREATE TABLE IF NOT EXISTS stock (
            id SERIAL PRIMARY KEY,
            symbol TEXT NOT NULL UNIQUE,
            company TEXT NOT NULL,
            one_week NUMERIC,
            one_month NUMERIC,
            six_month NUMERIC,
            one_year NUMERIC,
            ytd NUMERIC,
            five_year NUMERIC,
            ML_ind NUMERIC
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS stock_price (
            id SERIAL PRIMARY KEY,
            stock_id INTEGER,
            date DATE NOT NULL,
            open NUMERIC NOT NULL,
            high NUMERIC NOT NULL,
            low NUMERIC NOT NULL,
            close NUMERIC NOT NULL,
            adj_close NUMERIC NOT NULL,
            volume NUMERIC NOT NULL,
            FOREIGN KEY (stock_id) REFERENCES stock (id) ON DELETE CASCADE,
            UNIQUE (stock_id, date)
        )
    """)


    cur.execute("""
        ALTER TABLE stock_price
        ADD CONSTRAINT unique_stock_date UNIQUE (stock_id, date)
    """)

    conn.commit()
    cur.close()
    conn.close()

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
        ON CONFLICT(symbol) DO NOTHING
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


if __name__ == "__main__":
    df = get_prices('2020-01-01')
    reset_db()

    returns_data = calculate_returns(df)

    conn = psycopg2.connect(**DB_DETAILS)
    cur = conn.cursor()

    insert_stock_metadata(cur, returns_data)
    conn.commit()
    cur.close()
    conn.close()

    insert_price_data(df)  # This still handles price rows

