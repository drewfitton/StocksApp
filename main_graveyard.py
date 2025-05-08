# @app.get("/stock/data")
# def get_stock_data():
#     conn = psycopg2.connect(**DB_DETAILS)
#     cur = conn.cursor()
#     date = "2020-01-01"

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