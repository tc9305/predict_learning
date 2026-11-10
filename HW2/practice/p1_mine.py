import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def make_data(target, n, rng):
    """Return x ~ U[0,1] and y for target in {'noise', 'sine2'}."""
    x = rng.uniform(0, 1, size=n)
    if target == "noise":
        y = rng.normal(0, 1, size=n)
    else:
        y = np.sin(2 * np.pi * x) ** 2 + rng.normal(0, 0.2, size=n)
    return x, y

def design(x, dof):
    """Columns: 1, cos(2*pi*1*x), ..., cos(2*pi*(dof-1)*x)."""
    k = np.arange(dof)                      # [0, 1, ..., dof-1]
    return np.cos(2 * np.pi * np.outer(x, k))


def train_mse(x, y, dof):
    X = design(x, dof)
    w, *_ = np.linalg.lstsq(X, y, rcond=None)
    residual = y - X @ w
    return np.mean(residual ** 2)

def r_fpe(p, n):
    return (1 + p) / (1 - p)

def r_sc(p, n):
    return 1 + np.log(n) * p / (1 - p)

def r_gcv(p, n):
    return 1 / (1 - p) ** 2

def r_vc(p, n):
    inner = p - p * np.log(p) + np.log(n) / (2 * n)
    if inner >= 1:
        return np.inf
    return 1 / (1 - np.sqrt(inner))

PENALTIES = {"fpe": r_fpe, "sc": r_sc, "gcv": r_gcv, "vc": r_vc}

def loo_mse(x, y, dof):
    """Exact leave-one-out MSE via the hat-matrix shortcut."""
    X = design(x, dof)
    w, *_ = np.linalg.lstsq(X, y, rcond=None)
    residual = y - X @ w
    H = X @ np.linalg.pinv(X)     # hat matrix, n x n
    h = np.diag(H)                # leverages h_ii
    return np.mean((residual / (1 - h)) ** 2)

def loo_mse_brute(x, y, dof):
    """Leave-one-out MSE by actually refitting n times (slow, for checking)."""
    n = len(x)
    errors = []
    for i in range(n):
        keep = np.arange(n) != i            # True everywhere except position i
        X_train = design(x[keep], dof)
        w, *_ = np.linalg.lstsq(X_train, y[keep], rcond=None)
        y_pred = design(x[i:i+1], dof) @ w
        errors.append((y[i] - y_pred[0]) ** 2)
    return np.mean(errors)



def select_dof(x, y, max_dof):
    """Return {method: chosen DoF} using penalized training MSE."""
    n = len(x)
    dofs = np.arange(1, max_dof + 1)
    emp = np.array([train_mse(x, y, d) for d in dofs])
    picks = {}
    for name, r in PENALTIES.items():
        score = np.array([e * r(d / n, n) for e, d in zip(emp, dofs)])
        picks[name] = int(dofs[np.argmin(score)])
    loo = np.array([loo_mse(x, y, d) for d in dofs])
    picks["cv"] = int(dofs[np.argmin(loo)])
    return picks


def true_f(target, x):
    """Noise-free target function."""
    if target == "noise":
        return np.zeros_like(x)
    return np.sin(2 * np.pi * x) ** 2

def test_risk(x, y, dof, target, x_test):
    """Fit on (x, y), then MSE against the noise-free truth on x_test."""
    X = design(x, dof)
    w, *_ = np.linalg.lstsq(X, y, rcond=None)
    y_pred = design(x_test, dof) @ w
    return np.mean((y_pred - true_f(target, x_test)) ** 2)

def run(target, n, max_dof, reps=300, seed=0):
    rng = np.random.default_rng(seed)
    true_dof = 1 if target == "noise" else 3
    x_test = rng.uniform(0, 1, size=2000)
    methods = ["fpe", "sc", "gcv", "vc", "cv"]
    chosen = {m: [] for m in methods}
    risks = {m: [] for m in methods}
    for _ in range(reps):
        x, y = make_data(target, n, rng)
        picks = select_dof(x, y, max_dof)
        for m, d in picks.items():
            chosen[m].append(d)
            risks[m].append(test_risk(x, y, d, target, x_test))
    print(f"--- {target}, n = {n} ---")
    for m in methods:
        correct = np.mean(np.array(chosen[m]) == true_dof)
        p50, p95 = np.percentile(risks[m], [50, 95])
        print(f"{m:>4}: correct {correct:6.1%}   risk median {p50:.4f}   p95 {p95:.4f}")
    return chosen, risks

def plot_setting(n, max_dof, reps=300):
    targets = {"noise": ("Pure Gaussian noise, $\\sigma=1$", 1),
               "sine2": ("Sine-squared target, $\\sigma=0.2$", 3)}
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5), sharex=True)
    for col, (target, (title, true_dof)) in enumerate(targets.items()):
        chosen, risks = run(target, n, max_dof, reps)
        methods = list(risks)
        box = dict(whis=(5, 95), showfliers=False, tick_labels=methods)
        axes[0, col].boxplot([risks[m] for m in methods], **box)
        axes[0, col].set_yscale("log")
        axes[0, col].set_title(title)
        axes[1, col].boxplot([chosen[m] for m in methods], **box)
        axes[1, col].axhline(true_dof, ls="--", color="gray", lw=1)
        axes[1, col].set_ylim(0, max_dof + 1)
    axes[0, 0].set_ylabel("Risk (MSE)")
    axes[1, 0].set_ylabel("Degrees of freedom")
    fig.suptitle(f"Model selection, trigonometric polynomials, n = {n} ({reps} realizations)")
    fig.tight_layout()
    out = Path(__file__).parent / f"p1_n{n}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("saved", out)

if __name__ == "__main__":
    plot_setting(30, 20)
    plot_setting(100, 25)




