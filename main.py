from fastapi import FastAPI
from fastapi.responses import JSONResponse
import yfinance as yf
import pandas as pd
from config import DB_DETAILS, TICKER_DICT, TICKER_TO_DOMAIN, STOCK_CATEGORIES
from fastapi.middleware.cors import CORSMiddleware
import psycopg2

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],  # Angular dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/stock/data/{ticker}+{date}")
def get_stock_data(ticker: str, date: str):
    conn = psycopg2.connect(**DB_DETAILS)
    cur = conn.cursor()

    if ticker not in TICKER_DICT:
        return {"error": f"Ticker '{ticker}' not supported."}

    # Get stock id for the ticker
    cur.execute(
        "SELECT id FROM stock WHERE symbol = %s",
        (ticker,)
    )
    stock_row = cur.fetchone()

    if not stock_row:
        cur.close()
        conn.close()
        return {"error": f"No stock found for ticker '{ticker}'."}

    stock_id = stock_row[0]

    # Get stock price data for the specified stock_id and date
    cur.execute("""
        SELECT stock_id, date, open, high, low, close, adj_close, volume
        FROM stock_price
        WHERE stock_id = %s AND date >= %s
        ORDER BY date ASC
    """, (stock_id, date))

    price_rows = cur.fetchall()
    colnames = [desc[0] for desc in cur.description]

    df = pd.DataFrame(price_rows, columns=colnames)
    df["date"] = pd.to_datetime(df["date"])

    company_name = TICKER_DICT.get(ticker, "Unknown Company")
    domain = f"https://logo.clearbit.com/{TICKER_TO_DOMAIN.get(ticker, 'Unknown Domain')}"
    # Get total returns over the period
    returns = round(((df["adj_close"].iloc[-1] - df["adj_close"].iloc[0]) / df["adj_close"].iloc[0]) * 100, 2)

    stock_entry = {
        "id": stock_id,
        "ticker": ticker,
        "company": company_name,
        "returns": returns,
        "img": domain,
        "date": df["date"].dt.strftime("%Y-%m-%d").tolist(),
        "open": df["open"].tolist(),
        "high": df["high"].tolist(),
        "low": df["low"].tolist(),
        "close": df["close"].tolist(),
        "adj_close": df["adj_close"].tolist(),
        "volume": df["volume"].tolist()
    }

    cur.close()
    conn.close()

    return stock_entry
# @app.get("/stock/data")
# def get_stock_data():
#     conn = psycopg2.connect(**DB_DETAILS)
#     cur = conn.cursor()
#     date = "2016-01-01"

#     tickers = list(TICKER_DICT.keys())

#     # 1. Get stock ids in bulk
#     cur.execute(
#         "SELECT id, symbol FROM stock WHERE symbol = ANY(%s)",
#         (tickers,)
#     )
#     stock_rows = cur.fetchall()  # [(id, symbol), ...]

#     # Build mappings
#     symbol_to_id = {symbol: stock_id for stock_id, symbol in stock_rows}
#     ids = list(symbol_to_id.values())

#     if not ids:
#         cur.close()
#         conn.close()
#         return []  # No valid tickers found

#     # 2. Get all stock prices in bulk
#     cur.execute(f"""
#         SELECT stock_id, date, open, high, low, close, adj_close, volume
#         FROM stock_price
#         WHERE stock_id = ANY(%s) AND date >= %s
#         ORDER BY stock_id, date ASC
#     """, (ids, date))

#     price_rows = cur.fetchall()
#     # print(price_rows)

#     # Optional: get column names
#     colnames = [desc[0] for desc in cur.description]

#     # 3. Create one big DataFrame
#     df = pd.DataFrame(price_rows, columns=colnames)
#     df["date"] = pd.to_datetime(df["date"])


#     # numeric_cols = ["open", "high", "low", "close", "adj_close", "volume"]
#     # df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
#     # print(len(df))

#     # 4. Group by stock_id
#     stock_entries = []

#     for stock_id, group_df in df.groupby("stock_id"):
#         # Find the ticker
#         ticker = next((symbol for symbol, id_ in symbol_to_id.items() if id_ == stock_id), None)
#         # print(ticker)
#         if not ticker:
#             continue

#         company_name = TICKER_DICT.get(ticker, "Unknown Company")
#         domain = f"https://logo.clearbit.com/{TICKER_TO_DOMAIN.get(ticker, 'Unknown Domain')}"

#         stock_entry = {
#             "id": stock_id,
#             "ticker": ticker,
#             "company": company_name,
#             "img": domain,
#             "date": group_df["date"].dt.strftime("%Y-%m-%d").tolist(),
#             "open": group_df["open"].tolist(),
#             "high": group_df["high"].tolist(),
#             "low": group_df["low"].tolist(),
#             "close": group_df["close"].tolist(),
#             "adj_close": group_df["adj_close"].tolist(),
#             "volume": group_df["volume"].tolist()
#         }

#         # print(stock_entry["ticker"])

#         stock_entries.append(stock_entry)
#     # print(stock_entries)
#     cur.close()
#     conn.close()

#     return stock_entries

@app.get("/stock/returns/{ticker}+{date}")
def get_stock_returns(ticker, date):
    conn = psycopg2.connect(**DB_DETAILS)
    cur = conn.cursor()

    # Get the stock ID first
    cur.execute("SELECT id FROM stock WHERE symbol = %s", (ticker,))
    stock = cur.fetchone()

    if not stock:
        cur.close()
        conn.close()
        return None

    stock_id = stock[0]

    # Get ALL stock prices after the given date
    cur.execute("""
        SELECT date, open, high, low, close, adj_close, volume
        FROM stock_price
        WHERE stock_id = %s AND date >= %s
        ORDER BY date ASC
    """, (stock_id, date))

    rows = cur.fetchall()

    # Optional: get column names
    colnames = [desc[0] for desc in cur.description]

    # Create DataFrame
    df = pd.DataFrame(rows, columns=colnames)
    df["date"] = pd.to_datetime(df["date"])

    cur.close()
    conn.close()

    # Calculate returns since date
    # print(df)
    df["returns"] = df["adj_close"].pct_change() * 100

    return df

@app.get("/stock/category/{category}")
def get_stock_by_category(category):
    if category == "All":
        return JSONResponse(content=TICKER_DICT)
    
    stocks = STOCK_CATEGORIES.get(category, [])
    stocks_dict = {ticker: TICKER_DICT[ticker] for ticker in stocks if ticker in TICKER_DICT}
    return JSONResponse(content=stocks_dict)

@app.get("/stock/stock_names")
def get_stock_names():
    # Create a dictionary to map ticker symbols to company names
    return JSONResponse(content=TICKER_DICT)


if __name__ == "__main__":
   get_stock_data('AAPL', '2023-01-01')