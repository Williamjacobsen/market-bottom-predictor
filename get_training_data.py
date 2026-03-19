import yfinance as yf
import pandas as pd
import datetime
import numpy as np
import random


def generate_balanced_training_data(
    tickers, years=20, window_size=100, buffer=10, output_file="training_data.csv"
):
    class_0_rows = []
    class_1_rows = []
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
            # Step size 1 to maximize potential samples
            step = 1

            for i in range(0, len(low_prices) - window_size + 1, step):
                window = low_prices.iloc[i : i + window_size]

                # Find the index of the minimum value in the window
                min_date = window.idxmin()
                min_idx_in_window = window.index.get_loc(min_date)

                # RELIABILITY RULE: Skip if minimum is within the first 10 days
                if min_idx_in_window < buffer:
                    continue

                # Collect the 100 prices
                prices = window.values.flatten().tolist()

                # BINARY CLASSIFICATION:
                # 1 if the minimum is at the last index (price_99), 0 otherwise
                if min_idx_in_window == (window_size - 1):
                    class_1_rows.append(prices + [1])
                else:
                    class_0_rows.append(prices + [0])

            print(
                f"Current count: Class 1: {len(class_1_rows)}, Class 0: {len(class_0_rows)}"
            )

        except Exception as e:
            print(f"Error processing {ticker}: {e}")

    # BALANCING LOGIC:
    # We want a 50/50 split. We'll take all of Class 1 and a random sample of Class 0.
    num_class_1 = len(class_1_rows)
    print(
        f"\nFinal count before balancing: Class 1: {num_class_1}, Class 0: {len(class_0_rows)}"
    )

    if len(class_0_rows) > num_class_1:
        sampled_class_0 = random.sample(class_0_rows, num_class_1)
    else:
        sampled_class_0 = class_0_rows
        print("Warning: Fewer Class 0 samples than Class 1. Dataset will be smaller.")

    all_rows = class_1_rows + sampled_class_0
    random.shuffle(all_rows)  # Shuffle to mix classes

    # Create column names
    columns = [f"price_{j}" for j in range(window_size)] + ["is_min_at_last"]

    training_df = pd.DataFrame(all_rows, columns=columns)
    training_df.to_csv(output_file, index=False)
    print(
        f"Successfully saved total {len(training_df)} balanced training samples to {output_file}"
    )
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

    generate_balanced_training_data(diverse_tickers)

