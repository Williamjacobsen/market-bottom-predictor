import yfinance as yf
import pandas as pd
import datetime
import numpy as np


def generate_granular_training_data(
    tickers, years=20, window_size=100, buffer=10, output_file="training_data.csv"
):
    all_data_points = []
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

            # Extract Low prices correctly
            if isinstance(data.columns, pd.MultiIndex):
                low_prices = data["Low"][ticker]
            else:
                low_prices = data["Low"]

            # Filter out any NaN values
            low_prices = low_prices.dropna()

            # Sliding window logic
            # Step size 10 to provide enough training data while keeping file size reasonable
            step = 10

            for i in range(0, len(low_prices) - window_size + 1, step):
                window = low_prices.iloc[i : i + window_size]

                # Find the index of the minimum value in the window
                min_date = window.idxmin()
                min_idx_in_window = window.index.get_loc(min_date)

                # RELIABILITY RULE: Skip if minimum is within the first 10 days
                if min_idx_in_window < buffer:
                    continue

                # For each point in the window, create a row
                for idx, price in enumerate(window.values.flatten()):
                    is_minima = 1 if idx == min_idx_in_window else 0
                    all_data_points.append(
                        {
                            "ticker": ticker,
                            "index": idx,
                            "price": float(price),
                            "is_minima": is_minima,
                        }
                    )

        except Exception as e:
            print(f"Error processing {ticker}: {e}")

    # Create the final DataFrame
    training_df = pd.DataFrame(all_data_points)
    training_df.to_csv(output_file, index=False)
    print(f"\nSuccessfully saved {len(training_df)} data points to {output_file}")
    return training_df


if __name__ == "__main__":
    # Diverse set of tickers
    diverse_tickers = [
        "AAPL",
        "XOM",
        "WMT",
        "TSLA",
        "GLD",
        "JPM",
        "TLT",
        "AMZN",
        "KO",
        "BTC-USD",
    ]

    generate_granular_training_data(diverse_tickers)

