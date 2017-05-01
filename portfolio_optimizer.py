import os
import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np
import scipy.optimize as spo

# Calculate CUMULATIVE RETURNS
def compute_cumulative_returns(df):
    cumulative_returns = df.copy()
    cumulative_returns = (df[0:] / df[0].values) - 1
    print(df[0].values)
    return cumulative_returns


# Calculate DAILY RETURNS
def compute_daily_returns(df):
    daily_returns = df.copy()
    daily_returns[1:] = (df[1:] / df[:-1].values) - 1
    daily_returns.ix[0, :] = 0
    return daily_returns


# function to get data of the stocks in the 'symbol' for the date range givewn in 'dates'
def get_data(symbols, dates):
    """Read stock data (adjusted close) for given symbols from CSV files."""
    df = pd.DataFrame(index=dates)
    if 'SPY' not in symbols:  # add SPY for reference, if absent
        symbols.insert(0, 'SPY')

    for symbol in symbols:
        df_temp = pd.read_csv(symbol_to_path(symbol), index_col='Date',
                              parse_dates=True, usecols=['Date', 'Adj Close'], na_values=['nan'])
        df_temp = df_temp.rename(columns={'Adj Close': symbol})
        df = df.join(df_temp)
        if symbol == 'SPY':  # drop dates SPY did not trade
            df = df.dropna(subset=["SPY"])

    return df


# Calculate ROLLING MEAN
def get_rolling_mean(values, window):
    """Return rolling mean of given values, using specified window size."""
    return pd.rolling_mean(values, window=window)


# Calculate ROLLING STANDARD DEVIATION
def get_rolling_std(values, window):
    """Return rolling standard deviation of given values, using specified window size."""
    return pd.rolling_std(values, window)


# function to normalize data
def normalize_data(df):
    return df/df.ix[0, :]


# function to PLOT data
def plot_data(df, title="Stock Market"):
    """Plot stock prices with a custom title and meaningful axis labels."""
    ax = df.plot(title=title, fontsize=12)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend(loc="best")
    # plt.axhline(0, color='w', linestyle='dashed', linewidth=2)     #line y = 0
    plt.show()


def symbol_to_path(symbol, base_dir="data"):
    """Return CSV file path given ticker symbol."""
    return os.path.join(base_dir, "{}.csv".format(str(symbol)))


# Calculate SHARPE RATIO for allocation = 'alloc'
def sharpe_ratio_portfolio(alloc):
    dates = pd.date_range('2009-01-01', '2011-12-31')
    symbols = ['SPY', 'AAPL', 'IBM', 'GOOG', 'GLD']
    # fetching data of the stocks mentioned in the symbols
    df = get_data(symbols, dates)

    df = normalize_data(df)

    risk_free_rate = .000110        # 4% annually

    normed = df[['AAPL','IBM','GOOG','GLD']]

    pos_val = normed * alloc

    # pos_val = alloc * start_val

    port_val = pos_val.sum(axis=1)
    daily_returns = port_val.copy()
    daily_returns[1:] = (daily_returns[1:] / daily_returns[:-1].values) - 1

    daily_returns = daily_returns[1:]

    cumulative_return = (port_val[-1] / port_val[0]) - 1
    # print("cumulative return of portfolio is = {}".format(cumulative_return))

    avg_daily_return = daily_returns.mean()
    # print("average daily returns = {}".format(avg_daily_return))

    std_daily_return = daily_returns.std()  # risk
    # print("standard deviation of daily returns = {}".format(std_daily_return))

    sharp_num = daily_returns - risk_free_rate
    sharp_ratio = (252 ** .5) * sharp_num.mean() / std_daily_return

    return sharp_ratio * (-1)


# CONSTRAINTS defined for minimizer
def constraints(alloc):
    return alloc[0] + alloc[1] + alloc[2] + alloc[3] -1


def constraints2(alloc):
    c = 0
    for a in alloc:
        if(a==0):
            b = 0
        else:
            b = math.ceil(a-1)
        c += b
    return c




def test_run():
    # Read data
    dates = pd.date_range('2009-01-01', '2011-12-31')
    symbols = ['SPY', 'AAPL', 'IBM', 'GOOG', 'GLD']

    # fetching data of the stocks mentioned inthe symbols

    df = get_data(symbols, dates)
    df = normalize_data(df)

    alloc = [.25, .25, .25, .25]                            # initial guess
    print("initial sharpe ratio : {}".format(sharpe_ratio_portfolio(alloc) * -1))

    cons = {'type': 'eq', 'fun': constraints}               # constraints
    bounds = ((0,1),(0,1),(0,1),(0,1))                      # bounds
    min_result = spo.minimize(sharpe_ratio_portfolio, alloc, method='SLSQP',constraints=cons, bounds=bounds, options={'disp': True})
    print("Minimum found at : ")
    alloc = min_result.x

    sharpe_ratio = min_result.fun * -1
    print(alloc)
    print(sharpe_ratio)

    port = df[['AAPL','IBM','GOOG','GLD']]
    portfolio = port * alloc
    portfolio = portfolio.sum(axis=1)

    ax = portfolio.plot(title="Comparison", label='portfolio')
    ax.set_xlabel("Date")
    ax.set_ylabel("Return")
    df['SPY'].plot()

    plt.legend()
    plt.show()

if __name__ == "__main__":
    test_run()
