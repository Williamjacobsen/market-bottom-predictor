import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("predictions.csv")

tickers = df["ticker"].unique()

for ticker in tickers:
    ticker_df = df[df["ticker"] == ticker].sort_values("index")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    ax1.plot(ticker_df["index"], ticker_df["price"], label="Price", color="blue")
    minima = ticker_df[ticker_df["label"] == 1]
    ax1.scatter(
        minima["index"],
        minima["price"],
        color="green",
        s=100,
        label="Actual Minima",
        zorder=5,
    )
    ax1.set_ylabel("Price")
    ax1.set_title(f"{ticker} - Price and Minima Detection")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(
        ticker_df["index"],
        ticker_df["probability"],
        label="Probability",
        color="orange",
    )
    ax2.axhline(y=0.5, color="red", linestyle="--", alpha=0.5, label="Threshold (0.5)")
    ax2.scatter(minima["index"], minima["probability"], color="green", s=100, zorder=5)
    ax2.set_xlabel("Index")
    ax2.set_ylabel("Probability")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(f"plot_{ticker}.png")
    plt.close()

print("Plots saved as plot_<ticker>.png")
