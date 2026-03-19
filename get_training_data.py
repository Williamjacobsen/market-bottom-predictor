import yfinance as yf
import pandas as pd
import datetime
import numpy as np


def generate_training_data(
    tickers, years=20, window_size=100, buffer=10, output_file="training_data.csv"
):
    all_rows = []
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=years * 365)

    for ticker in tickers:
        print(f"Processing {ticker}...")
        try:
            # Download data
            data = yf.download(ticker, start=start_date, end=end_date)

            if data.empty:
                print(f"No data found for {ticker}. Skipping.")
                continue

            # Extract Low prices correctly for both single and multi-ticker downloads
            if isinstance(data.columns, pd.MultiIndex):
                low_prices = data["Low"][ticker]
            else:
                low_prices = data["Low"]

            # Filter out any NaN values
            low_prices = low_prices.dropna()

            # Sliding window logic
            # Use step size 5 to get a good number of samples while keeping the file manageable
            step = 5

            ticker_samples = 0
            for i in range(0, len(low_prices) - window_size + 1, step):
                window = low_prices.iloc[i : i + window_size]

                # Find the index of the minimum value in the window
                min_date = window.idxmin()
                min_idx_in_window = window.index.get_loc(min_date)

                # Check reliability: not in first 10 or last 10 days
                if buffer <= min_idx_in_window < (window_size - buffer):
                    # For training a NN, it's often best to normalize prices relative to the window
                    # so the network learns shapes rather than absolute dollar values.
                    # We'll keep them raw as requested, but we'll add the ticker for context if needed.
                    prices = window.values.flatten().tolist()
                    row = prices + [min_idx_in_window]
                    all_rows.append(row)
                    ticker_samples += 1

            print(f"Added {ticker_samples} samples from {ticker}.")

        except Exception as e:
            print(f"Error processing {ticker}: {e}")

    # Create column names
    columns = [f"price_{j}" for j in range(window_size)] + ["target_min_idx"]

    training_df = pd.DataFrame(all_rows, columns=columns)
    training_df.to_csv(output_file, index=False)
    print(
        f"\nSuccessfully saved total {len(training_df)} training samples to {output_file}"
    )
    return training_df


if __name__ == "__main__":
    # Diverse set of tickers
    diverse_tickers = [
        "AAPL",  # Tech/Growth
        "XOM",  # Energy/Value
        "WMT",  # Retail/Stable
        "TSLA",  # High Volatility
        "GLD",  # Commodities/Gold
        "JPM",  # Financials
        "TLT",  # Bonds/Interest Rate sensitive
        "AMZN",  # E-commerce/Cloud
        "KO",  # Consumer Staples
        "BTC-USD",  # Crypto (Very different behavior)
    ]

    generate_training_data(diverse_tickers)