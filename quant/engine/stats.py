"""Multiple-testing correction. PROTECTED (operator-owned).

Deflated Sharpe Ratio, Bailey & Lopez de Prado (2014), implemented in the
units the paper uses. The guide's version has two bugs that make its
PASS/REJECT close to arbitrary:
  1. it plugs an ANNUALISED Sharpe into a formula that multiplies by
     sqrt(n_obs) and squares the Sharpe in the non-normality term - both
     need the PER-BAR Sharpe;
  2. it omits the cross-trial standard deviation of Sharpe ratios that
     scales the expected-maximum-of-noise benchmark.
"""
import numpy as np
from scipy.stats import norm

EULER = 0.5772156649015329


def expected_max_sharpe(n_trials, sr_std):
    """Expected maximum per-bar Sharpe of n_trials zero-skill strategies."""
    if n_trials <= 1:
        return 0.0
    return float(sr_std * ((1 - EULER) * norm.ppf(1 - 1 / n_trials)
                           + EULER * norm.ppf(1 - 1 / (n_trials * np.e))))


def probabilistic_sharpe(sr, sr_benchmark, n_obs, skew, kurtosis):
    """P(true per-bar Sharpe > sr_benchmark). kurtosis is Pearson (normal = 3)."""
    var = 1 - skew * sr + (kurtosis - 1) / 4 * sr ** 2
    if n_obs < 2 or var <= 0:
        return 0.0
    return float(norm.cdf((sr - sr_benchmark) * np.sqrt(n_obs - 1) / np.sqrt(var)))


def deflated_sharpe(sr, n_obs, skew, kurtosis, n_trials, trial_sharpes):
    """sr, trial_sharpes: PER-BAR Sharpe ratios. n_trials: every variation
    ever tried (from the trial registry, not self-reported).

    The cross-trial spread is floored at the sampling noise of a Sharpe
    estimate, 1/sqrt(n_obs): with few logged trials the sample variance is
    unreliable, and the floor keeps the benchmark from collapsing to zero.
    """
    spread = float(np.std(trial_sharpes, ddof=1)) if len(trial_sharpes) > 1 else 0.0
    sr_std = max(spread, 1 / np.sqrt(max(n_obs, 2)))
    sr0 = expected_max_sharpe(n_trials, sr_std)
    return {
        "deflated_sharpe": round(probabilistic_sharpe(sr, sr0, n_obs, skew, kurtosis), 4),
        "n_trials": int(n_trials),
        "trial_sharpe_std": round(sr_std, 6),
        "expected_max_sharpe_bar_from_noise": round(sr0, 6),
    }
