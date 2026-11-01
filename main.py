import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from pathlib import Path

TRUE_ATE = 2.0


def simulate_sample(
        n: int,
        rng: np.random.Generator
) -> pd.DataFrame:

    # 1. Generate covariates X1, X2 ~ N(0, 1)
    x1 = rng.standard_normal(size=n)
    x2 = rng.standard_normal(size=n)

    # 2. Compute the propensity score
    # Linear predictor: z = -0.3 + 0.8*X1 - 0.6*X2
    z = -0.3 + 0.8 * x1 - 0.6 * x2

    # Use scipy.special.expit for a numerically stable logistic transform (1 / (1 + exp(-z)))
    propensity = expit(z)

    # 3. Draw treatment assignment D ~ Bernoulli(p=propensity)
    treatment = rng.binomial(n=1, p=propensity)

    # 4. Generate the error term epsilon ~ N(0, 1) and compute the observed outcome Y
    # Y = 1 + 2D + 1.2*X1 - 0.8*X2 + 0.5*X1^2 + epsilon
    epsilon = rng.standard_normal(size=n)
    outcome = (
            1.0
            + TRUE_ATE * treatment
            + 1.2 * x1
            - 0.8 * x2
            + 0.5 * (x1 ** 2)
            + epsilon
    )

    # 5. Return everything as a DataFrame
    return pd.DataFrame({
        "x1": x1,
        "x2": x2,
        "treatment": treatment,
        "outcome": outcome,
        "propensity": propensity
    })


def estimate_naive(df: pd.DataFrame) -> float:
    # Compute the mean outcome within each treatment group
    group_means = df.groupby("treatment")["outcome"].mean()

    # Subtract the D=0 mean from the D=1 mean and return a scalar float
    return float(group_means.loc[1] - group_means.loc[0])


def estimate_ols(df: pd.DataFrame) -> float:
    # 1. Build a separate design matrix X
    X = pd.DataFrame({
        "treatment": df["treatment"],
        "x1": df["x1"],
        "x2": df["x2"],
        "x1_sq": df["x1"] ** 2
    })

    # 2. Extract the dependent variable y
    y = df["outcome"]

    # 3. Initialize and fit an ordinary OLS model
    model = LinearRegression()
    model.fit(X, y)

    # 4. Look up the treatment column position and extract its coefficient
    treatment_idx = X.columns.get_loc("treatment")
    return float(model.coef_[treatment_idx])


def estimate_ipw(df: pd.DataFrame) -> float:
    # 1. Extract covariates X and the treatment indicator d
    X = df[["x1", "x2"]]
    d = df["treatment"].to_numpy()
    y = df["outcome"].to_numpy()

    # 2. Fit a logistic regression to estimate the propensity score
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, d)

    # 3. Use predict_proba to get the predicted probability of D=1 (e_hat)
    # predict_proba returns an (n, 2) array; column 1 holds the probability of the treated group (D=1)
    e_hat = clf.predict_proba(X)[:, 1]

    # 4. Compute the Hájek weights
    w1 = d / e_hat
    w0 = (1.0 - d) / (1.0 - e_hat)

    # 5. Compute the self-normalized weighted means with np.average
    mu1 = np.average(y, weights=w1)
    mu0 = np.average(y, weights=w0)

    # 6. Return the difference as the treatment effect estimate
    return float(mu1 - mu0)

rng = np.random.default_rng(42)
df = simulate_sample(10_000, rng)

print("True ATE:", TRUE_ATE)
print("Naive:", estimate_naive(df))
print("OLS:", estimate_ols(df))
print("IPW:", estimate_ipw(df))


def run_monte_carlo(
        repetitions: int = 200,
        n: int = 1000,
        seed: int = 42
) -> pd.DataFrame:
    # 1. Initialize a dedicated random number generator
    rng = np.random.default_rng(seed)

    records = []

    # 2. Run the repeated simulations
    for rep in range(1, repetitions + 1):
        # Draw a fresh sample for each repetition
        df_sample = simulate_sample(n=n, rng=rng)

        # Compute all three estimates on the same sample
        naive_est = estimate_naive(df_sample)
        ols_est = estimate_ols(df_sample)
        ipw_est = estimate_ipw(df_sample)

        records.append({
            "repetition": rep,
            "naive": naive_est,
            "ols": ols_est,
            "ipw": ipw_est
        })

    return pd.DataFrame(records)


def summarize_monte_carlo(
    df_results: pd.DataFrame,
    true_ate: float = TRUE_ATE
) -> pd.DataFrame:
    estimators = [("naive", "Naive"), ("ols", "OLS"), ("ipw", "IPW")]
    summary_data = []

    for col_name, display_name in estimators:
        estimates = df_results[col_name].to_numpy()
        R = len(estimates)  # Number of repetitions R

        # 1. Mean estimate
        mean_est = float(np.mean(estimates))

        # 2. Formula: \widehat{Bias} = 1/R * \sum (\hat\tau_r - \tau)
        bias = float(np.sum(estimates - true_ate) / R)

        # 3. Empirical Monte Carlo standard deviation (ddof=1)
        sd = float(np.std(estimates, ddof=1))

        # 4. Formula: \widehat{RMSE} = \sqrt{ 1/R * \sum (\hat\tau_r - \tau)^2 }
        rmse = float(np.sqrt(np.sum((estimates - true_ate) ** 2) / R))

        summary_data.append({
            "Estimator": display_name,
            "Mean Estimate": mean_est,
            "Bias": bias,
            "SD": sd,
            "RMSE": rmse
        })

    return pd.DataFrame(summary_data)

results = run_monte_carlo(
    repetitions=200,
    n=1000,
    seed=42
)

summary = summarize_monte_carlo(results, true_ate=TRUE_ATE)

print(summary.to_string(index=False, float_format="%.4f"))

output_dir = Path("figures")
output_dir.mkdir(exist_ok=True)

fig, ax = plt.subplots(figsize=(8, 5))

# 1. Draw a boxplot of results[["naive", "ols", "ipw"]]
ax.boxplot(
    results[["naive", "ols", "ipw"]],
    tick_labels=["Naive", "OLS", "IPW"],
    patch_artist=True,
    boxprops=dict(facecolor="#dbeafe", color="#2563eb", alpha=0.8),
    medianprops=dict(color="#1d4ed8", linewidth=1.8),
    whiskerprops=dict(color="#2563eb"),
    capprops=dict(color="#2563eb"),
)

# 2. Mark the true ATE as a reference line with ax.axhline
ax.axhline(
    TRUE_ATE,
    color="#dc2626",
    linestyle="--",
    linewidth=1.5,
    label=f"True ATE ({TRUE_ATE})"
)

ax.set_ylabel("Estimated Treatment Effect")
ax.set_title("Monte Carlo Comparison of ATE Estimators")
ax.legend(loc="upper right")
ax.grid(axis="y", linestyle=":", alpha=0.5)

fig.tight_layout()
fig.savefig(
    output_dir / "estimator_comparison.png",
    dpi=200
)
plt.close(fig)