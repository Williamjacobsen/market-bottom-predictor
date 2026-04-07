import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons
import numpy as np

df = pd.read_csv("predictions.csv")
tickers = sorted(df["ticker"].unique().tolist())

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
plt.subplots_adjust(left=0.25, bottom=0.15)


def update(label):
    ax1.clear()
    ax2.clear()

    ticker_df = df[df["ticker"] == label].sort_values(by=["index"])

    ax1.plot(ticker_df["index"], ticker_df["price"], color="blue", label="Price")
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
    ax1.set_title(f"{label} - Price and Minima Detection")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(
        ticker_df["index"],
        ticker_df["probability"],
        color="orange",
        label="Probability",
    )
    ax2.axhline(y=0.5, color="red", linestyle="--", alpha=0.5, label="Threshold (0.5)")
    ax2.scatter(minima["index"], minima["probability"], color="green", s=100, zorder=5)
    ax2.set_xlabel("Index")
    ax2.set_ylabel("Probability")
    ax2.legend()
    ax2.grid(True)

    fig.canvas.draw_idle()


ax_radio = plt.axes([0.02, 0.3, 0.15, 0.4])
radio = RadioButtons(ax_radio, tickers)
radio.on_clicked(update)

update(tickers[0])

plt.show()
