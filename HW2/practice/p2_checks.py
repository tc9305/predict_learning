"""Reproduce the Problem 2 numbers cited in the report that p2_mine.py does not print.

Run from the HW2 folder:  python -m practice.p2_checks
The output is also written to practice/p2_checks_output.txt.
"""
from pathlib import Path
import numpy as np
from src.data_loader import _ubc, _yahoo
from practice.p2_mine import design, quad_design, test, train, v, w

OUT = Path(__file__).parent / "p2_checks_output.txt"
lines = []


def log(text=""):
    print(text)
    lines.append(text)


def main():
    # ---- Data alignment ----
    log("Data alignment, 2004")
    gspc, fx = _yahoo("GSPC", 2004, "Close"), _ubc(2004, invert=False)
    common = set(gspc.index) & set(_yahoo("FIGRX", 2004, "Close").index) & set(fx.index)
    log(f"  common trading days: {len(common)}")
    log(f"  stock days without a UBC quote: {len(set(gspc.index) - common)}")
    log(f"  UBC days without stock prices:  {len(set(fx.index) - common)}")
    log(f"  training samples: {len(train)}, test samples: {len(test)}")

    # ---- Linear classifier ----
    log("\nLinear classifier")
    log(f"  DOWN threshold on x1 when x2 = 0: {-w[0] / w[1]:.2f}%")
    log(f"  share of training days predicted UP: {np.mean(design(train) @ w > 0):.1%}")

    # ---- Quadratic classifier: shape of the boundary ----
    log("\nQuadratic classifier")
    w3, w4, w5 = v[3], v[4], v[5]
    log(f"  w3 = {w3:.4f} (< 0), w3*w4 - w5^2/4 = {w3 * w4 - w5 ** 2 / 4:.4f} (> 0 means negative definite)")
    for name, df in (("2004", train), ("2005", test)):
        df = df[df["y"] != 0]
        inside = quad_design(df) @ v > 0
        log(f"  {name}: next-day UP rate inside the ellipse {np.mean(df['y'][inside] > 0):.1%} "
            f"({inside.sum()} days), outside {np.mean(df['y'][~inside] > 0):.1%} ({(~inside).sum()} days)")

    # ---- Trading details ----
    log("\nTrading, 2005")
    invested = design(test) @ w > 0
    daily = test["y"].to_numpy() / 100
    log(f"  linear: cash days {(~invested).sum()}, mean FIGRX change on cash days "
        f"{test['y'][~invested].mean():+.2f}%, on invested days {test['y'][invested].mean():+.2f}%")
    switches = np.r_[False, invested[1:] != invested[:-1]]
    gross = np.prod(1 + daily * invested) - 1
    net = np.prod((1 + daily * invested) * np.where(switches, 0.999, 1.0)) - 1
    log(f"  linear: {switches.sum()} switches, gain {gross:.2%} without costs, {net:.2%} with 0.1% per switch")
    log(f"  buy-and-hold gain: {np.prod(1 + daily) - 1:.2%}")

    # ---- Sampling error and the VC confidence term ----
    log("\nSampling error and VC confidence term")
    n_test = int((test["y"] != 0).sum())
    log(f"  one day in {n_test} test days = {100 / n_test:.2f} accuracy points; "
        f"standard error at 50% accuracy = {100 * np.sqrt(0.25 / n_test):.1f} points")
    log(f"  standard error for the 44 days outside the ellipse = {100 * np.sqrt(0.25 / 44):.1f} points")
    n = int((train["y"] != 0).sum())
    log(f"  confidence level 1 - eta with n = {n}: {1 - 4 / np.sqrt(n):.2%}")
    for big_n in (1_000, 5_000, 10_000):
        eta = min(4 / np.sqrt(big_n), 1)
        phi = np.sqrt((3 * (np.log(2 * big_n / 3) + 1) - np.log(eta / 4)) / big_n)
        log(f"  Phi for h = 3 at n = {big_n}: {phi:.3f}")

    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()
