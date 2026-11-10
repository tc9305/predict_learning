"""Reproducible numerical checks for Problem 1 (cited in the report).

Run from the HW2 folder:  python -m practice.p1_checks
The output is also written to practice/p1_checks_output.txt.

Check 1  Numerical rank of the design matrix (n = 30, 200 random samples, seed 1).
Check 2  Leverage shortcut vs. explicit leave-one-out refits on one sample (seed 0).
Check 3  cv with explicit refits for EVERY candidate DoF vs. the default rule
         (shortcut when full rank, refits when rank deficient): compares the DoF
         selected in each of the 300 realizations, for all four settings (seed 0).
Check 4  The 15 worst fpe realizations at n = 30 (seed 0), refitted with three
         LAPACK least-squares drivers through SciPy: gelsd and gelss (SVD-based)
         and gelsy (QR with column pivoting), all with SciPy's default cutoff.
Check 5  Other numbers quoted in the report text (training MSE example, penalty
         factors, DoF chosen in the worst realizations).

Rank threshold everywhere: numpy.linalg.matrix_rank default,
tol = max singular value * max(rows, cols) * machine epsilon.
"""
import sys
from pathlib import Path
import numpy as np
import scipy
import scipy.linalg as sl
from practice import p1_mine as P

OUT = Path(__file__).parent / "p1_checks_output.txt"
lines = []


def log(text=""):
    print(text)
    lines.append(text)


def selections(target, n, max_dof, loo_function):
    """Selected DoF per realization for every method, same seed and data as p1_mine.run."""
    original = P.loo_mse
    P.loo_mse = loo_function
    try:
        rng = np.random.default_rng(0)
        rng.uniform(0, 1, size=2000)                     # x_test draw, keeps the random stream aligned
        chosen = []
        for _ in range(300):
            x, y = P.make_data(target, n, rng)
            chosen.append(P.select_dof(x, y, max_dof))
        return chosen
    finally:
        P.loo_mse = original


def main():
    log(f"numpy {np.__version__}, scipy {scipy.__version__}, python {sys.version.split()[0]}")

    # ---- Check 1: numerical rank ----
    log("\nCheck 1: samples (of 200, n = 30, seed 1) whose design matrix is rank deficient")
    rng = np.random.default_rng(1)
    counts = {d: 0 for d in range(1, 20, 2)}
    for _ in range(200):
        x = rng.uniform(0, 1, 30)
        for d in counts:
            if np.linalg.matrix_rank(P.design(x, d)) < d:
                counts[d] += 1
    log("  " + ", ".join(f"DoF {d}: {c}" for d, c in counts.items()))

    # ---- Check 2: leverage shortcut vs explicit refits ----
    log("\nCheck 2: one sine-squared sample (n = 30, seed 0), shortcut formula vs 30 explicit refits")
    rng = np.random.default_rng(0)
    x, y = P.make_data("sine2", 30, rng)
    for d in (9, 13, 15, 17, 19):
        X = P.design(x, d)
        resid = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
        h = np.diag(X @ np.linalg.pinv(X))
        shortcut = np.mean((resid / (1 - h)) ** 2)
        log(f"  DoF {d:2d}  rank {np.linalg.matrix_rank(X):2d}  shortcut {shortcut:12.6f}"
            f"  refits {P.loo_mse_brute(x, y, d):10.6f}")

    # ---- Check 3: cv selections with refits for every DoF ----
    log("\nCheck 3: cv-selected DoF, default rule vs explicit refits for every candidate DoF")
    log("  (compares the selected DoF in each of the 300 realizations, seed 0)")
    for target in ("noise", "sine2"):
        for n, max_dof in ((30, 20), (100, 25)):
            default = selections(target, n, max_dof, P.loo_mse)
            refit = selections(target, n, max_dof, P.loo_mse_brute)
            differ = sum(a["cv"] != b["cv"] for a, b in zip(default, refit))
            log(f"  {target:5s} n = {n:3d}: realizations with a different cv choice: {differ} of 300")

    # ---- Check 4: solver comparison on the worst fpe fits ----
    log("\nCheck 4: 15 worst fpe realizations at n = 30 (seed 0), risk under three LAPACK drivers")
    for target in ("noise", "sine2"):
        rng = np.random.default_rng(0)
        x_test = rng.uniform(0, 1, size=2000)
        truth = P.true_f(target, x_test)
        rows = []
        for _ in range(300):
            x, y = P.make_data(target, 30, rng)
            d = P.select_dof(x, y, 20)["fpe"]
            X, Xt = P.design(x, d), P.design(x_test, d)
            risk = {drv: np.mean((Xt @ sl.lstsq(X, y, lapack_driver=drv)[0] - truth) ** 2)
                    for drv in ("gelsd", "gelss", "gelsy")}
            rows.append((risk["gelsd"], d, np.linalg.cond(X), np.linalg.matrix_rank(X), risk))
        rows.sort(key=lambda r: r[0])
        log(f"  {target}:  DoF  rank   cond      gelsd       gelss       gelsy   max rel. diff")
        for _, d, cond, rank, risk in rows[-15:]:
            vals = np.array(list(risk.values()))
            rel = (vals.max() - vals.min()) / vals.min()
            log(f"          {d:3d}  {rank:4d}  {cond:8.1e}  {risk['gelsd']:10.4g}  {risk['gelss']:10.4g}"
                f"  {risk['gelsy']:10.4g}   {rel:6.1%}")
        deficient = sum(r[3] < r[1] for r in rows)
        log(f"  {target}: fpe selections with a rank-deficient matrix: {deficient} of 300")

    # ---- Check 5: other numbers quoted in the text ----
    log("\nCheck 5: other numbers quoted in the report")
    rng = np.random.default_rng(0)
    x, y = P.make_data("sine2", 30, rng)
    log("  training MSE, one sine-squared sample (n = 30, seed 0): "
        + ", ".join(f"DoF {d}: {P.train_mse(x, y, d):.4f}" for d in (1, 7, 19)))
    for n, p in ((30, 0.1), (30, 0.5), (100, 0.1)):
        log(f"  penalty factors at n = {n}, p = {p}: "
            + ", ".join(f"{name} {r(p, n):.2f}" for name, r in P.PENALTIES.items()))
    rng = np.random.default_rng(0)
    x_test = rng.uniform(0, 1, size=2000)
    picks, risks = [], []
    for _ in range(300):
        x, y = P.make_data("noise", 30, rng)
        pk = P.select_dof(x, y, 20)
        picks.append(pk)
        risks.append({m: P.test_risk(x, y, d, "noise", x_test) for m, d in pk.items()})
    for m in ("fpe", "vc"):
        worst = sorted(range(300), key=lambda i: risks[i][m])[-15:]
        log(f"  pure noise n = 30, DoF chosen by {m} in its 15 highest-risk realizations: "
            f"{sorted(picks[i][m] for i in worst)}")

    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()
