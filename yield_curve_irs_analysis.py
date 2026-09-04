"""Zero-curve bootstrap and vanilla interest-rate swap risk engine."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class InterestRateSwap:
    notional: float = 10_000_000.0
    maturity_years: int = 5
    fixed_rate: float = 0.0275
    pay_fixed: bool = True

    def __post_init__(self) -> None:
        if self.notional <= 0 or self.maturity_years < 1:
            raise ValueError("Notional and maturity must be positive.")
        if self.fixed_rate < 0:
            raise ValueError("Fixed rate cannot be negative in this illustrative model.")


def bootstrap_curve(par_rates: pd.Series) -> pd.DataFrame:
    """Bootstrap annual discount factors from annual-pay par swap rates."""
    rates = pd.Series(par_rates, dtype=float).sort_index()
    expected = np.arange(1, len(rates) + 1)
    if not np.array_equal(rates.index.to_numpy(dtype=int), expected):
        raise ValueError("Maturities must be consecutive integers starting at one year.")

    discount_factors: list[float] = []
    for maturity, swap_rate in rates.items():
        previous = sum(discount_factors)
        discount_factor = (1.0 - swap_rate * previous) / (1.0 + swap_rate)
        if discount_factor <= 0:
            raise ValueError("Inputs imply a non-positive discount factor.")
        discount_factors.append(discount_factor)

    curve = pd.DataFrame(
        {
            "maturity_years": rates.index.astype(int),
            "par_rate": rates.to_numpy(),
            "discount_factor": discount_factors,
        }
    )
    curve["zero_rate_cc"] = -np.log(curve["discount_factor"]) / curve["maturity_years"]
    previous_df = curve["discount_factor"].shift(1, fill_value=1.0)
    curve["one_year_forward"] = previous_df / curve["discount_factor"] - 1.0
    return curve


def par_swap_rate(curve: pd.DataFrame, maturity_years: int) -> float:
    dfs = curve.set_index("maturity_years")["discount_factor"]
    if maturity_years not in dfs.index:
        raise ValueError("Swap maturity is not available on the curve.")
    annuity = float(dfs.loc[1:maturity_years].sum())
    return float((1.0 - dfs.loc[maturity_years]) / annuity)


def swap_value(swap: InterestRateSwap, curve: pd.DataFrame) -> dict[str, float]:
    dfs = curve.set_index("maturity_years")["discount_factor"]
    if swap.maturity_years not in dfs.index:
        raise ValueError("Swap maturity exceeds the curve.")
    annuity = float(dfs.loc[1 : swap.maturity_years].sum())
    fixed_leg = swap.notional * swap.fixed_rate * annuity
    floating_leg = swap.notional * (1.0 - float(dfs.loc[swap.maturity_years]))
    payer_npv = floating_leg - fixed_leg
    return {
        "fixed_leg_pv_eur": fixed_leg,
        "floating_leg_pv_eur": floating_leg,
        "npv_eur": payer_npv if swap.pay_fixed else -payer_npv,
        "fair_fixed_rate": par_swap_rate(curve, swap.maturity_years),
        "annuity": annuity,
    }


def curve_from_zero_rates(base_curve: pd.DataFrame, shocked_zero_rates: np.ndarray) -> pd.DataFrame:
    result = base_curve.copy()
    maturities = result["maturity_years"].to_numpy(dtype=float)
    result["zero_rate_cc"] = shocked_zero_rates
    result["discount_factor"] = np.exp(-shocked_zero_rates * maturities)
    previous_df = result["discount_factor"].shift(1, fill_value=1.0)
    result["one_year_forward"] = previous_df / result["discount_factor"] - 1.0
    return result


def parallel_shift(curve: pd.DataFrame, basis_points: float) -> pd.DataFrame:
    shocked = curve["zero_rate_cc"].to_numpy() + basis_points / 10_000
    return curve_from_zero_rates(curve, shocked)


def swap_dv01(swap: InterestRateSwap, curve: pd.DataFrame) -> float:
    up = swap_value(swap, parallel_shift(curve, 1.0))["npv_eur"]
    down = swap_value(swap, parallel_shift(curve, -1.0))["npv_eur"]
    return float((up - down) / 2.0)


def stress_tests(swap: InterestRateSwap, curve: pd.DataFrame) -> pd.DataFrame:
    maturities = curve["maturity_years"].to_numpy(dtype=float)
    base_zero = curve["zero_rate_cc"].to_numpy()
    max_maturity = maturities.max()
    scenarios = {
        "Parallel +100 bp": np.full(len(curve), 0.0100),
        "Parallel -100 bp": np.full(len(curve), -0.0100),
        "Bear steepener": 0.0025 + 0.0075 * maturities / max_maturity,
        "Bull flattener": -0.0100 + 0.0075 * maturities / max_maturity,
    }
    base_npv = swap_value(swap, curve)["npv_eur"]
    rows = []
    for name, shock in scenarios.items():
        shocked_curve = curve_from_zero_rates(curve, base_zero + shock)
        shocked_npv = swap_value(swap, shocked_curve)["npv_eur"]
        rows.append(
            {
                "scenario": name,
                "shocked_npv_eur": shocked_npv,
                "pnl_vs_base_eur": shocked_npv - base_npv,
            }
        )
    return pd.DataFrame(rows)


def save_outputs(output_dir: Path = Path("outputs")) -> None:
    market_rates = pd.Series(
        [0.0210, 0.0225, 0.0240, 0.0255, 0.0270, 0.0282, 0.0292],
        index=range(1, 8),
        name="par_rate",
    )
    curve = bootstrap_curve(market_rates)
    swap = InterestRateSwap()
    valuation = swap_value(swap, curve)
    valuation["dv01_eur_per_bp"] = swap_dv01(swap, curve)
    stresses = stress_tests(swap, curve)

    output_dir.mkdir(parents=True, exist_ok=True)
    curve.to_csv(output_dir / "bootstrapped_curve.csv", index=False)
    pd.DataFrame([valuation]).to_csv(output_dir / "swap_valuation.csv", index=False)
    stresses.to_csv(output_dir / "swap_stress_tests.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(curve["maturity_years"], 100 * curve["par_rate"], marker="o", label="Par swap rate")
    ax.plot(curve["maturity_years"], 100 * curve["zero_rate_cc"], marker="s", label="Zero rate (cc)")
    ax.plot(curve["maturity_years"], 100 * curve["one_year_forward"], marker="^", label="1Y forward")
    ax.set_xlabel("Maturity (years)")
    ax.set_ylabel("Rate (%)")
    ax.set_title("Illustrative EUR interest-rate curve")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "yield_curve.png", dpi=180)
    plt.close(fig)

    print(curve.round(6).to_string(index=False))
    print("\nSwap valuation")
    for key, value in valuation.items():
        print(f"{key}: {value:,.6f}")
    print("\nStress tests")
    print(stresses.round(2).to_string(index=False))


if __name__ == "__main__":
    save_outputs()

