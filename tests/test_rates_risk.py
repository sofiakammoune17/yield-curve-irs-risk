import numpy as np
import pandas as pd
import pytest

from src.rates_risk import (
    InterestRateSwap,
    bootstrap_curve,
    par_swap_rate,
    parallel_shift,
    stress_tests,
    swap_dv01,
    swap_value,
)


@pytest.fixture
def curve():
    rates = pd.Series([0.02, 0.022, 0.024, 0.026, 0.028], index=range(1, 6))
    return bootstrap_curve(rates)


def test_discount_factors_are_positive_and_decreasing(curve):
    assert (curve["discount_factor"] > 0).all()
    assert curve["discount_factor"].is_monotonic_decreasing


def test_bootstrap_reprices_par_rates(curve):
    for maturity, expected in enumerate([0.02, 0.022, 0.024, 0.026, 0.028], start=1):
        assert par_swap_rate(curve, maturity) == pytest.approx(expected)


def test_swap_at_fair_rate_has_zero_npv(curve):
    fair_rate = par_swap_rate(curve, 5)
    swap = InterestRateSwap(maturity_years=5, fixed_rate=fair_rate)
    assert swap_value(swap, curve)["npv_eur"] == pytest.approx(0.0, abs=1e-8)


def test_payer_swap_gains_when_rates_rise(curve):
    swap = InterestRateSwap(maturity_years=5, fixed_rate=0.028)
    base = swap_value(swap, curve)["npv_eur"]
    up = swap_value(swap, parallel_shift(curve, 100))["npv_eur"]
    assert up > base


def test_payer_swap_dv01_is_positive(curve):
    swap = InterestRateSwap(maturity_years=5, fixed_rate=0.028)
    assert swap_dv01(swap, curve) > 0


def test_parallel_up_stress_has_positive_pnl_for_payer(curve):
    swap = InterestRateSwap(maturity_years=5, fixed_rate=0.028)
    results = stress_tests(swap, curve).set_index("scenario")
    assert results.loc["Parallel +100 bp", "pnl_vs_base_eur"] > 0

