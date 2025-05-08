from datetime import datetime, timedelta
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
import yfinance as yf
import pandas as pd
from config import DB_DETAILS, TICKER_DICT, TICKER_TO_DOMAIN, STOCK_CATEGORIES
from fastapi.middleware.cors import CORSMiddleware
import psycopg2
from typing import Optional, List
import indicators

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],  # Angular dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/stock/data/")
def get_stock_data(
    category: str = Query(...),
    period: str = Query(...),
    offset: int = Query(0),
    limit: int = Query(20),
    sort: str = Query("returns_desc"),
    inds: List[str] = Query(default=[])
):
    conn = psycopg2.connect(**DB_DETAILS)
    cur = conn.cursor()
    # print(inds)

    tickers = STOCK_CATEGORIES[category] if category != "All" else list(TICKER_DICT.keys()) 
    cur.execute("SELECT id, symbol FROM stock WHERE symbol = ANY(%s)", (tickers,))
    stock_rows = cur.fetchall()
    symbol_to_id = {symbol: stock_id for stock_id, symbol in stock_rows}
    ids = list(symbol_to_id.values())

    if not ids:
        cur.close()
        conn.close()
        return {"results": [], "total": 0}

    adjusted_period = (datetime.strptime(period, '%Y-%m-%d') - timedelta(days=50)).strftime('%Y-%m-%d')

    # Now pass `adjusted_period` into your query
    cur.execute(f"""
        SELECT stock_id, date, open, high, low, close, adj_close, volume
        FROM stock_price
        WHERE stock_id = ANY(%s) AND date >= %s
        ORDER BY stock_id, date ASC
    """, (ids, adjusted_period))

    df = pd.DataFrame(cur.fetchall(), columns=[desc[0] for desc in cur.description])
    # print(df)
    df["date"] = pd.to_datetime(df["date"])
    df_sorted = df.sort_values(by=["stock_id", "date"])

    original_start_date = pd.to_datetime(period)
    ### Compute returns ###
    returns_filtered = df_sorted[df_sorted["date"] >= original_start_date]
    returns_df = returns_filtered.groupby("stock_id")["adj_close"].agg(["first", "last"])
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

        # print(group_df)
        for ind in inds:
            # print(group_df)
            ind_df = indicators.__dict__[f"calc_{ind}"](group_df[['adj_close']], 'adj_close')
            # print(ind_df)
            group_df = pd.concat([ind_df, group_df], axis=1)
            # print(group_df[ind])
            # print(group_df)

        # Filter after indicator computation
        group_df = group_df[group_df["date"] >= original_start_date]
        # print(group_df)
        stock_entries.append({
            "id": stock_id,
            "ticker": ticker,
            "company": TICKER_DICT.get(ticker, "Unknown Company"),
            "img": f"https://logo.clearbit.com/{TICKER_TO_DOMAIN.get(ticker, 'Unknown Domain')}",
            "returns": round(returns_lookup.get(stock_id, 0),2),
            "date": group_df["date"].dt.strftime("%Y-%m-%d").tolist(),
            "open": group_df["open"].tolist(),
            "high": group_df["high"].tolist(),
            "low": group_df["low"].tolist(),
            "close": group_df["close"].tolist(),
            "adj_close": group_df["adj_close"].tolist(),
            "lower_bollinger": group_df["lower_bollinger"].tolist() if "lower_bollinger" in group_df else None,
            "upper_bollinger": group_df["upper_bollinger"].tolist() if "upper_bollinger" in group_df else None,
            "rsi": group_df["RSI"].tolist() if "RSI" in group_df else None,
            "macd": group_df["MACD"].tolist() if "MACD" in group_df else None,
            "macd_signal": group_df["MACD_Signal"].tolist() if "MACD_Signal" in group_df else None,
            "volume": group_df["volume"].tolist()
        })

    cur.close()
    conn.close()

    return {"results": stock_entries, "total": total_count}



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