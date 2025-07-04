import streamlit as st
import pandas as pd
from sqlalchemy import create_engine

# --- Setup ---
st.set_page_config(page_title="Market Summary", layout="wide")

st.title("Nifty 50 - Market Summary Dashboard")

# --- Database connection ---
engine = create_engine("mysql+pymysql://root:root@localhost:3306/stock_data")

@st.cache_data
def load_data():
    df = pd.read_sql("SELECT * FROM stocks", engine)
    df['date'] = pd.to_datetime(df['date'])
    return df

df = load_data()

# --- Yearly return calculation ---
returns = []
for ticker in df['Ticker'].unique():
    temp = df[df['Ticker'] == ticker].sort_values("date")
    if len(temp) > 1:
        first_close = temp.iloc[0]['close']
        last_close = temp.iloc[-1]['close']
        yearly_return = ((last_close - first_close) / first_close) * 100
        returns.append({
            "Ticker": ticker,
            "COMPANY": temp.iloc[0]['COMPANY'],
            "sector": temp.iloc[0]['sector'],
            "Return (%)": round(yearly_return, 2)
        })

returns_df = pd.DataFrame(returns).sort_values("Return (%)", ascending=False)

# --- Top 10 Green / Red Stocks ---
top_10 = returns_df.head(10)
bottom_10 = returns_df.tail(10)

# --- Green vs Red Stocks ---
green_count = (returns_df['Return (%)'] > 0).sum()
red_count = (returns_df['Return (%)'] <= 0).sum()

# --- Average price & volume ---
avg_price = round(df['close'].mean(), 2)
avg_volume = int(df['volume'].mean())

# --- Volatility (std dev of daily returns per stock) ---
volatility = []
for ticker in df['Ticker'].unique():
    temp = df[df['Ticker'] == ticker].sort_values("date")
    temp['daily_return'] = temp['close'].pct_change()
    std_dev = temp['daily_return'].std()
    if pd.notna(std_dev):
        volatility.append({"Ticker": ticker, "Volatility": std_dev})

vol_df = pd.DataFrame(volatility).sort_values("Volatility", ascending=False).head(10)

# --- Streamlit layout ---

st.subheader( "Market Summary")
col1, col2, col3 = st.columns(3)
col1.metric("Green Stocks", green_count)
col2.metric("Red Stocks", red_count)
col3.metric("Avg. Close Price", f"₹{avg_price}")

st.metric("Avg. Volume", f"{avg_volume:,}")

st.subheader("Top 10 Gainers")
st.dataframe(top_10, use_container_width=True)

st.subheader(" Top 10 Losers")
st.dataframe(bottom_10, use_container_width=True)

st.subheader("Top 10 Most Volatile Stocks")
st.dataframe(vol_df, use_container_width=True)

import matplotlib.pyplot as plt

st.subheader("Volatility Bar Chart (Top 10 Stocks)")

fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(vol_df["Ticker"], vol_df["Volatility"], color="orange")
ax.set_ylabel("Volatility (Std. Dev of Daily Returns)")
ax.set_xlabel("Ticker")
ax.set_title("Top 10 Most Volatile Stocks")
st.pyplot(fig)

import matplotlib.pyplot as plt

st.subheader("Cumulative Return Over Time")

# Calculate cumulative return for each stock
cum_returns = pd.DataFrame()

for ticker in df['Ticker'].unique():
    temp = df[df['Ticker'] == ticker].sort_values("date")
    temp['daily_return'] = temp['close'].pct_change()
    temp['cumulative_return'] = (1 + temp['daily_return']).cumprod()
    temp['cumulative_return'] = temp['cumulative_return'].fillna(1.0)
    temp['Ticker'] = ticker
    cum_returns = pd.concat([cum_returns, temp[['date', 'Ticker', 'cumulative_return']]])

# Get final return per stock for ranking
final_returns = cum_returns.groupby("Ticker").tail(1).sort_values("cumulative_return", ascending=False)
top_5_tickers = final_returns['Ticker'].head(5).tolist()

# Filter data for those top 5
top_5_data = cum_returns[cum_returns['Ticker'].isin(top_5_tickers)]

# Plot the line chart
fig, ax = plt.subplots(figsize=(10, 5))
for ticker in top_5_tickers:
    stock_data = top_5_data[top_5_data['Ticker'] == ticker]
    ax.plot(stock_data['date'], stock_data['cumulative_return'], label=ticker)

ax.set_title("Cumulative Return of Top 5 Performing Stocks")
ax.set_ylabel("Cumulative Return")
ax.set_xlabel("Date")
ax.legend()
ax.grid(True)
st.pyplot(fig)


st.subheader("Sector-wise Performance")

# Group by sector and calculate average yearly return
sector_avg = returns_df.groupby("sector")["Return (%)"].mean().sort_values(ascending=False).reset_index()

# Display as table
st.dataframe(sector_avg, use_container_width=True)

# Plot bar chart
fig, ax = plt.subplots(figsize=(10, 5))
ax.barh(sector_avg["sector"], sector_avg["Return (%)"], color="skyblue")
ax.set_xlabel("Average Yearly Return (%)")
ax.set_title("Average Yearly Return by Sector")
ax.invert_yaxis()
ax.grid(axis="x", linestyle="--", alpha=0.7)
st.pyplot(fig)


import seaborn as sns
import matplotlib.pyplot as plt

st.subheader("Stock Price Correlation Heatmap")

# Fix: average 'close' per date-ticker to remove duplicates
pivot_df = df.groupby(['date', 'Ticker'])['close'].mean().reset_index()

# Pivot safely
price_matrix = pivot_df.pivot(index='date', columns='Ticker', values='close')

# Compute % returns and correlation
correlation = price_matrix.pct_change().corr()

# Plot heatmap
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(correlation, cmap="coolwarm", annot=False, ax=ax)
ax.set_title("Correlation Between Stocks' Daily Returns")
st.pyplot(fig)



st.subheader("Top 5 Monthly Gainers and Losers")

monthly_returns = df.copy()
monthly_returns['month'] = pd.to_datetime(monthly_returns['date']).dt.to_period('M')
monthly_returns.sort_values(['Ticker', 'date'], inplace=True)

# Calculate monthly return per stock
monthly_returns['prev_close'] = monthly_returns.groupby(['Ticker', 'month'])['close'].shift(1)
monthly_returns['monthly_return'] = (monthly_returns['close'] - monthly_returns['prev_close']) / monthly_returns['prev_close'] * 100

# Drop rows with no prev_close
monthly_returns = monthly_returns.dropna()

# Group month-wise and plot
months = monthly_returns['month'].unique()

for m in months:
    st.markdown(f"Month: {m}")
    month_df = monthly_returns[monthly_returns['month'] == m]
    avg_returns = month_df.groupby('Ticker')['monthly_return'].mean().reset_index()

    # Top 5 gainers and losers
    top5 = avg_returns.sort_values('monthly_return', ascending=False).head(5)
    bottom5 = avg_returns.sort_values('monthly_return').head(5)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("Top 5 Gainers")
        fig1, ax1 = plt.subplots()
        ax1.barh(top5['Ticker'], top5['monthly_return'], color='green')
        ax1.invert_yaxis()
        st.pyplot(fig1)

    with col2:
        st.markdown("Top 5 Losers")
        fig2, ax2 = plt.subplots()
        ax2.barh(bottom5['Ticker'], bottom5['monthly_return'], color='red')
        ax2.invert_yaxis()
        st.pyplot(fig2)
