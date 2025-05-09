import yfinance as yf
import psycopg2
from config import DB_DETAILS, TICKER_DICT, TICKER_TO_DOMAIN
from sqlalchemy import create_engine
import pandas as pd
# from tickers import tickers

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


def reset_db():
    conn = psycopg2.connect(**DB_DETAILS)
    cur = conn.cursor()

    cur.execute("DROP TABLE IF EXISTS stock CASCADE;")
    cur.execute("DROP TABLE IF EXISTS stock_price CASCADE;")

    cur.execute("""
        CREATE TABLE IF NOT EXISTS stock (
            id SERIAL PRIMARY KEY,
            symbol TEXT NOT NULL UNIQUE,
            company TEXT NOT NULL
        )
    """)

    # cur.execute("""
    #     CREATE TABLE IF NOT EXISTS stock_price (
    #         id SERIAL PRIMARY KEY,
    #         stock_id INTEGER,
    #         date DATE NOT NULL,
    #         open NUMERIC NOT NULL,
    #         high NUMERIC NOT NULL,
    #         low NUMERIC NOT NULL,
    #         close NUMERIC NOT NULL,
    #         adj_close NUMERIC NOT NULL,
    #         volume NUMERIC NOT NULL,
    #         FOREIGN KEY (stock_id) REFERENCES stock (id) ON DELETE CASCADE
    #     )
    # """)
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

# def insert_price_data(df):
#     # Connect to the database
#     conn = psycopg2.connect(**DB_DETAILS)
#     cur = conn.cursor()

#     for symbol, company in TICKER_DICT.items():
#         cur.execute("INSERT INTO stock (symbol, company) VALUES (%s, %s) ON CONFLICT(symbol) DO NOTHING", (symbol, company))
        
#     for i, row in df.iterrows():
#         print(f"Inserting {i}: {row['ticker']} on {row['date']}")
#         cur.execute("SELECT id FROM stock WHERE symbol = %s", (row["ticker"],))
#         stock_id = cur.fetchone()
#         print(f"Fetched stock_id: {stock_id}")

#         if stock_id:
#             stock_id = stock_id[0]
#             cur.execute("""
#                 INSERT INTO stock_price (stock_id, date, open, high, low, close, adj_close, volume)
#                 VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
#             """, (stock_id, row["date"], row["open"], row["high"], row["low"], row["close"], row["adj_close"], row["volume"]))
#             print(f"Inserted price for {row['ticker']} on {row['date']}")
#         else:
#             print(f"Stock {row['ticker']} not found in stock table.")

#     # Commit changes and close connection
#     # print(1)
#     conn.commit()
#     cur.close()
#     conn.close()

from psycopg2.extras import execute_values

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


def get_stock_data(
    category: str,
    period: str,  # e.g., '1y', '6m', '2020-01-01'
    offset: int,
    limit: int,
    sort: str
):
    conn = psycopg2.connect(**DB_DETAILS)
    cur = conn.cursor()

    tickers = list(TICKER_DICT.keys())  # TODO: filter by category if needed
    cur.execute("SELECT id, symbol FROM stock WHERE symbol = ANY(%s)", (tickers,))
    stock_rows = cur.fetchall()
    symbol_to_id = {symbol: stock_id for stock_id, symbol in stock_rows}
    ids = list(symbol_to_id.values())

    if not ids:
        cur.close()
        conn.close()
        return {"results": [], "total": 0}

    cur.execute(f"""
        SELECT stock_id, date, open, high, low, close, adj_close, volume
        FROM stock_price
        WHERE stock_id = ANY(%s) AND date >= %s
        ORDER BY stock_id, date ASC
    """, (ids, period))  # assumes `period` is a start date string like '2020-01-01'

    df = pd.DataFrame(cur.fetchall(), columns=[desc[0] for desc in cur.description])
    df["date"] = pd.to_datetime(df["date"])
    df_sorted = df.sort_values(by=["stock_id", "date"])

    # Compute returns
    returns_df = df_sorted.groupby("stock_id")["adj_close"].agg(["first", "last"])
    returns_df["returns"] = (returns_df["last"] - returns_df["first"]) / returns_df["first"] * 100
    returns_df["returns"] = returns_df["returns"].round(2)

    if sort == "returns_desc":
        sorted_ids = returns_df.sort_values(by="returns", ascending=False).index.tolist()
    else:
        sorted_ids = returns_df.sort_values(by="returns", ascending=True).index.tolist()

    total_count = len(sorted_ids)
    page_ids = sorted_ids[offset: offset + limit]

    df_page = df_sorted[df_sorted["stock_id"].isin(page_ids)]

    # Ensure returns available
    returns_lookup = returns_df["returns"].to_dict()

    stock_entries = []
    for stock_id, group_df in df_page.groupby("stock_id"):
        ticker = next((sym for sym, sid in symbol_to_id.items() if sid == stock_id), None)
        if not ticker:
            continue

        stock_entries.append({
            "id": stock_id,
            "ticker": ticker,
            "company": TICKER_DICT.get(ticker, "Unknown Company"),
            "img": f"https://logo.clearbit.com/{TICKER_TO_DOMAIN.get(ticker, 'Unknown Domain')}",
            "returns": returns_lookup.get(stock_id, 0),
            "date": group_df["date"].dt.strftime("%Y-%m-%d").tolist(),
            "open": group_df["open"].tolist(),
            "high": group_df["high"].tolist(),
            "low": group_df["low"].tolist(),
            "close": group_df["close"].tolist(),
            "adj_close": group_df["adj_close"].tolist(),
            "volume": group_df["volume"].tolist()
        })

    cur.close()
    conn.close()

    return {"results": stock_entries, "total": total_count}

if __name__ == "__main__":
    # get_stock_data('AAPL', '2023-10-01')
    # df = get_prices('2015-01-01')
    # print(df)
    # reset_db()
    # # print(1)
    # insert_price_data(df)
    # get_stock_data()
    update_prices()
    # get_stock_returns('AAPL', '2023-01-01')