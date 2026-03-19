import yfinance as yf
import pandas as pd
import datetime
import numpy as np


def export_continuous_minima(
    tickers, years=20, window_size=100, buffer=10, output_file="training_data.csv"
):
    all_data = []
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=years * 365)

    for ticker in tickers:
        print(f"Processing {ticker}...")
        try:
            data = yf.download(ticker, start=start_date, end=end_date)

            if data.empty:
                print(f"No data found for {ticker}. Skipping.")
                continue

            if isinstance(data.columns, pd.MultiIndex):
                low_prices = data["Low"][ticker]
            else:
                low_prices = data["Low"]

            low_prices = low_prices.dropna()

            ticker_df = pd.DataFrame({"ticker": ticker, "price": low_prices.values})
            ticker_df["is_minima"] = 0
            ticker_df["index"] = range(len(ticker_df))

            for i in range(0, len(low_prices) - window_size + 1):
                window = low_prices.iloc[i : i + window_size]

                min_date = window.idxmin()
                min_idx_in_window = window.index.get_loc(min_date)

                if buffer <= min_idx_in_window < (window_size - buffer):
                    global_idx = i + min_idx_in_window
                    ticker_df.at[global_idx, "is_minima"] = 1

            ticker_df = ticker_df[["ticker", "index", "price", "is_minima"]]
            all_data.append(ticker_df)
            print(
                f"Collected {len(ticker_df)} days for {ticker}, with {ticker_df['is_minima'].sum()} minima."
            )

        except Exception as e:
            print(f"Error processing {ticker}: {e}")

    if not all_data:
        print("No data collected.")
        return None

    final_df = pd.concat(all_data, ignore_index=True)

    final_df.to_csv(output_file, index=False)
    print(f"\nSuccessfully saved {len(final_df)} total rows to {output_file}")
    return final_df


if __name__ == "__main__":
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

    export_continuous_minima(diverse_tickers)
