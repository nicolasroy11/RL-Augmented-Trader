import duckdb
import finplot as fplt
import pandas as pd
import os
from db.data_store import db_path

# --- Connect to your ticks.duckdb file ---
con = duckdb.connect(db_path, read_only=True)

# Fetch the data
df = con.execute("SELECT * FROM ticks ORDER BY timestamp ASC").fetchdf()
df['timestamp'] = pd.to_datetime(df['timestamp'])
df.set_index('timestamp', inplace=True)

# Create a new window
price_ax, rsi_ax, macd_ax = fplt.create_plot('BTCUSDC Indicators', rows=3)

# Plot price
fplt.plot(df['price'], ax=price_ax, legend='Price')

# Plot EMAs
fplt.plot(df['ema_short'], ax=price_ax, color='#5f5', legend='EMA Short')
fplt.plot(df['ema_mid'], ax=price_ax, color='#55f', legend='EMA Mid')
fplt.plot(df['ema_long'], ax=price_ax, color='#f55', legend='EMA Long')
fplt.plot(df['ema_xlong'], ax=price_ax, color='#aaa', legend='EMA XLong')

# Plot RSI indicators
fplt.plot(df['rsi_5'], ax=rsi_ax, legend='RSI 5')
fplt.plot(df['rsi_7'], ax=rsi_ax, legend='RSI 7')
fplt.plot(df['rsi_12'], ax=rsi_ax, legend='RSI 12')
# fplt.add_line((df.index[0], 70), (df.index[-1], 70), ax=rsi_ax, color='#f55', style='--')  # Overbought
# fplt.add_line((df.index[0], 30), (df.index[-1], 30), ax=rsi_ax, color='#5f5', style='--')  # Oversold

# Plot MACD
fplt.plot(df['macd_line'], ax=macd_ax, color='cyan', legend='MACD Line')
fplt.plot(df['macd_signal'], ax=macd_ax, color='orange', legend='MACD Signal')
colors = df['macd_hist'].apply(lambda x: '#0f0' if x >= 0 else '#f00').tolist()
fplt.plot(df['macd_hist'], ax=macd_ax, colors=colors, legend='MACD Histogram')


# Auto-align and run the UI
fplt.show()