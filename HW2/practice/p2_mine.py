import numpy as np
import pandas as pd
from src.data_loader import load_prices
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- 1. Build samples: one continuous price series (2003-12-31, 2004, 2005) ----
extra = pd.read_csv("data/extra_2003-12-31.csv", parse_dates=["Date"], index_col="Date")
prices = pd.concat([extra, load_prices(2004), load_prices(2005)])
pct = 100 * prices.pct_change()
data = pd.DataFrame({
    "x1": pct["GSPC"],                # today's S&P 500 % change
    "x2": pct["EURUSD"],              # today's exchange-rate % change
    "y":  pct["FIGRX"].shift(-1),     # TOMORROW's FIGRX % change
}).dropna()

year = pd.DatetimeIndex(data.index).year   # tell Pylance the index holds dates
train = data[year == 2004]
test = data[year == 2005]

# ---- 2. Linear least-squares model ----
def design(df):
    return np.column_stack([np.ones(len(df)), df["x1"], df["x2"]])

w, *_ = np.linalg.lstsq(design(train), train["y"].to_numpy(), rcond=None)

def accuracy(df, w):
    pred = np.sign(design(df) @ w)
    keep = df["y"] != 0                       # skip zero-change days
    return np.mean(pred[keep] == np.sign(df["y"][keep]))

# ---- 3. Quadratic decision boundary ----
def quad_design(df):
    x1, x2 = df["x1"].to_numpy(), df["x2"].to_numpy()
    return np.column_stack([np.ones(len(df)), x1, x2, x1**2, x2**2, x1 * x2])

train_nz = train[train["y"] != 0]                 # drop zero-change days
labels = np.sign(train_nz["y"].to_numpy())        # +1 = UP, -1 = DOWN
v, *_ = np.linalg.lstsq(quad_design(train_nz), labels, rcond=None)

def accuracy_with(df, w, features):
    keep = df["y"] != 0
    pred = np.sign(features(df[keep]) @ w)
    return np.mean(pred == np.sign(df["y"][keep]))

# ---- 4. Decision boundary plots ----
UP_COLOR, DOWN_COLOR, INK = "#2a78d6", "#eb6834", "#0b0b0b"

def plot_boundary(ax, df, wt, features, title):
    # evaluate the model on a fine grid over the required (-2%, +2%) range
    g1, g2 = np.meshgrid(np.linspace(-2, 2, 400), np.linspace(-2, 2, 400))
    grid = pd.DataFrame({"x1": g1.ravel(), "x2": g2.ravel()})
    score = (features(grid) @ wt).reshape(g1.shape)
    # shaded regions + the boundary line g(x) = 0
    ax.contourf(g1, g2, score, levels=[-np.inf, 0, np.inf],
                colors=[DOWN_COLOR, UP_COLOR], alpha=0.12)
    ax.contour(g1, g2, score, levels=[0], colors=INK, linewidths=2)
    # actual outcomes
    up, down = df[df["y"] > 0], df[df["y"] < 0]
    ax.scatter(up["x1"], up["x2"], marker="^", s=28, color=UP_COLOR,
               edgecolor="white", linewidth=0.6, label="next day UP")
    ax.scatter(down["x1"], down["x2"], marker="v", s=28, color=DOWN_COLOR,
               edgecolor="white", linewidth=0.6, label="next day DOWN")
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.axhline(0, color="#d0d0cc", lw=0.8, zorder=0)
    ax.axvline(0, color="#d0d0cc", lw=0.8, zorder=0)
    ax.set_title(f"{title}\naccuracy {accuracy_with(df, wt, features):.1%}", fontsize=10)

def plot_all():
    fig, axes = plt.subplots(2, 2, figsize=(9, 9), sharex=True, sharey=True)
    models = [("Linear", w, design), ("Quadratic", v, quad_design)]
    for row, (name, wt, feat) in enumerate(models):
        plot_boundary(axes[row, 0], train, wt, feat, f"{name} boundary, 2004 training data")
        plot_boundary(axes[row, 1], test, wt, feat, f"{name} boundary, 2005 test data")
    for ax in axes[1]:
        ax.set_xlabel("x1: S&P 500 daily change (%)")
    for ax in axes[:, 0]:
        ax.set_ylabel("x2: EUR/USD daily change (%)")
    axes[0, 0].legend(loc="lower right", fontsize=8, framealpha=0.9)
    fig.suptitle("FIGRX trading classifiers: shaded = predicted UP (blue) / DOWN (orange)")
    fig.tight_layout()
    out = Path(__file__).parent / "p2_boundaries.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("saved", out)

# ---- 5. Trading simulation on 2005 ----
def simulate(df, invested):
    """invested: True = hold FIGRX tomorrow, False = hold cash."""
    daily = df["y"].to_numpy() / 100           # % -> fraction
    account = np.cumprod(1 + daily * invested)  # account value, start = 1
    gain = (account[-1] - 1) * 100
    exposure = np.mean(invested) * 100
    return gain, exposure, account

def trading_table():
    strategies = {
        "linear":     design(test) @ w > 0,
        "quadratic":  quad_design(test) @ v > 0,
        "buy & hold": np.ones(len(test), dtype=bool),
    }
    print(f"{'strategy':12}{'GAIN %':>9}{'EXPOSURE %':>12}")
    for name, invested in strategies.items():
        gain, exposure, _ = simulate(test, invested)
        print(f"{name:12}{gain:9.2f}{exposure:12.1f}")

def luck_check(invested, reps=10_000, seed=0):
    """Random strategies holding FIGRX on the same number of days."""
    rng = np.random.default_rng(seed)
    daily = test["y"].to_numpy() / 100
    n, k = len(daily), int(invested.sum())
    gains = []
    for _ in range(reps):
        pick = np.zeros(n, dtype=bool)
        pick[rng.choice(n, k, replace=False)] = True
        gains.append((np.prod(1 + daily * pick) - 1) * 100)
    gains = np.array(gains)
    target, _, _ = simulate(test, invested)
    print(f"random median {np.median(gains):.1f}%, share >= {target:.2f}%: {np.mean(gains >= target):.1%}")
    print("position switches:", int(np.sum(invested[1:] != invested[:-1])))



if __name__ == "__main__":
    print("quadratic v =", np.round(v, 4))
    print(f"{'':10}{'train':>8}{'test':>8}")
    for name, wt, feat in [("linear", w, design), ("quadratic", v, quad_design)]:
        print(f"{name:10}{accuracy_with(train, wt, feat):8.3f}{accuracy_with(test, wt, feat):8.3f}")
    keep = test["y"] != 0
    print(f"{'always-UP':10}{'':8}{np.mean(test['y'][keep] > 0):8.3f}")
    plot_all()
    trading_table()
    luck_check(design(test) @ w > 0)

