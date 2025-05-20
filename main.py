from datetime import datetime, timedelta
from time import time
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
import yfinance as yf
import pandas as pd
from config import DB_DETAILS, TICKER_DICT, TICKER_TO_DOMAIN, STOCK_CATEGORIES, RETURNS_TIMES
from fastapi.middleware.cors import CORSMiddleware
import psycopg2
from typing import Optional, List
import indicators
# from ml_evaluation.StrategyLearner import StrategyLearner as sl
import matplotlib.pyplot as plt
import matplotlib

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
    returns_period: str = Query(...),
    offset: int = Query(0),
    limit: int = Query(20),
    sort: str = Query("returns_desc"),
    inds: List[str] = Query(default=[])
):
    conn = psycopg2.connect(**DB_DETAILS)
    cur = conn.cursor()
    # print(inds)
    inds = ['Bollinger', 'MACD', 'RSI']

    tickers = STOCK_CATEGORIES[category] if category != "All" else list(TICKER_DICT.keys()) 
    cur.execute("SELECT id, symbol FROM stock WHERE symbol = ANY(%s)", (tickers,))
    stock_rows = cur.fetchall()
    symbol_to_id = {symbol: stock_id for stock_id, symbol in stock_rows}
    ids = list(symbol_to_id.values())

    total_count = len(ids)

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

    ret_column = RETURNS_TIMES.get(returns_period, "ytd")

    if sort.startswith("returns_"):
        sort_order = "DESC" if sort == "returns_desc" else "ASC"
        cur.execute(
            f"""
            SELECT id, symbol, {ret_column}, ml_ind
            FROM stock
            WHERE id = ANY(%s)
            ORDER BY {ret_column} {sort_order}
            LIMIT %s OFFSET %s
            """,
            (ids, limit, offset)
        )
    elif sort.startswith("ml_ind_"):
        sort_order = "DESC" if sort == "ml_ind_desc" else "ASC"
        cur.execute(
            f"""
            SELECT id, symbol, {ret_column}, ml_ind
            FROM stock
            WHERE id = ANY(%s)
            ORDER BY ml_ind {sort_order}
            LIMIT %s OFFSET %s
            """,
            (ids, limit, offset)
        )

    sorted_rows = cur.fetchall()
    page_ids = [row[0] for row in sorted_rows]
    returns_lookup = {row[0]: float(row[2]) if row[2] is not None else 0 for row in sorted_rows}
    ml_ind_lookup = {row[0]: row[3] if row[3] is not None else 0 for row in sorted_rows}


    df_page = df_sorted[df_sorted["stock_id"].isin(page_ids)]


    stock_entries = []
    for stock_id, group_df in df_page.groupby("stock_id"):
        ticker = next((sym for sym, sid in symbol_to_id.items() if sid == stock_id), None)
        if not ticker:
            continue

        for ind in inds:
            ind_df = indicators.__dict__[f"calc_{ind}"](group_df[['adj_close']], 'adj_close')
            group_df = pd.concat([ind_df, group_df], axis=1)

        # Filter after indicator computation
        group_df = group_df[group_df["date"] >= original_start_date]
        group_df = group_df.bfill().ffill()

        
        stock_entries.append({
            "id": stock_id,
            "ticker": ticker,
            "company": TICKER_DICT.get(ticker, "Unknown Company"),
            "img": f"https://logo.clearbit.com/{TICKER_TO_DOMAIN.get(ticker, 'Unknown Domain')}",
            "returns": round(returns_lookup.get(stock_id, 0),2),
            "ml_ind": round(ml_ind_lookup.get(stock_id, 0),5),
            "date": group_df["date"].dt.strftime("%Y-%m-%d").tolist(),
            "open": group_df["open"].astype(float).tolist(),
            "high": group_df["high"].astype(float).tolist(),
            "low": group_df["low"].astype(float).tolist(),
            "close": group_df["close"].astype(float).tolist(),
            "adj_close": group_df["adj_close"].astype(float).tolist(),
            "lower_bollinger": group_df["lower_bollinger"].astype(float).tolist() if "lower_bollinger" in group_df else None,
            "upper_bollinger": group_df["upper_bollinger"].astype(float).tolist() if "upper_bollinger" in group_df else None,
            "rsi": group_df["RSI"].astype(float).tolist() if "RSI" in group_df else None,
            "macd": group_df["MACD"].astype(float).tolist() if "MACD" in group_df else None,
            "macd_signal": group_df["MACD_Signal"].astype(float).tolist() if "MACD_Signal" in group_df else None,
            "volume": group_df["volume"].astype(float).tolist(),
        })
        
    cur.close() 
    conn.close()

    return {"results": stock_entries, "total": total_count}

@app.get("/stock/all_returns/")
def get_all_stock_returns(
    category: str = Query(...),
    period: str = Query(...),
):
    conn = psycopg2.connect(**DB_DETAILS)
    cur = conn.cursor()

    tickers = STOCK_CATEGORIES[category] if category != "All" else list(TICKER_DICT.keys()) 

    # Join to get ticker and company info
    cur.execute("""
        SELECT sp.stock_id, s.symbol, s.company, sp.date, sp.adj_close
        FROM stock_price sp
        JOIN stock s ON sp.stock_id = s.id
        WHERE s.symbol = ANY(%s) AND sp.date >= %s
        ORDER BY sp.stock_id, sp.date ASC
    """, (tickers, period))

    rows = cur.fetchall()
    colnames = [desc[0] for desc in cur.description]
    cur.close()
    conn.close()

    df = pd.DataFrame(rows, columns=colnames)
    df["date"] = pd.to_datetime(df["date"])

    # Group by stock and calculate total return for each
    result = []
    for stock_id, group in df.groupby("stock_id"):
        group = group.sort_values("date")
        if len(group) > 1:
            initial = group.iloc[0]["adj_close"]
            final = group.iloc[-1]["adj_close"]
            total_return = ((final - initial) / initial) * 100
        else:
            total_return = 0.0

        result.append({
            "id": stock_id,
            "ticker": group.iloc[0]["symbol"],
            "company": group.iloc[0]["company"],
            "returns": round(total_return, 2)
        })
    total_count = len(result)
    return {'results': result, 'total': total_count}

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