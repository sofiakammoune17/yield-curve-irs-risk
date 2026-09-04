# Yield Curve, Interest Rate Swap Pricing & Risk

Python case study covering zero-coupon curve construction, forward-rate extraction, vanilla interest-rate swap valuation and interest-rate risk analysis.

## Business objective

The project reproduces a simplified Rates / Fixed Income workflow:

- bootstrap discount factors and zero rates from par swap rates;
- derive one-year forward rates;
- determine the fair fixed rate of a vanilla payer swap;
- value the fixed and floating legs;
- calculate NPV and DV01;
- apply parallel, steepening and flattening curve shocks;
- export auditable tables and charts.

All instruments and market inputs are illustrative. The implementation is educational and does not replace a production curve framework.

## Illustrative market inputs

Annual par swap rates are supplied for maturities from one to seven years. For each maturity `n`, the discount factor is bootstrapped from:

`DF(n) = [1 - S(n) × Σ DF(i)] / [1 + S(n)]`

The continuously compounded zero rate is:

`z(n) = -ln(DF(n)) / n`

The fair fixed rate is obtained by dividing the floating-leg present value by the swap annuity. The project values a **EUR 10 million, five-year annual-pay payer swap**.

## Repository structure

```text
src/rates_risk.py          Curve, IRS pricing and risk engine
tests/test_rates_risk.py   Financial-logic tests
outputs/                   Generated reports and chart
requirements.txt           Dependencies
```

## Run

```bash
python -m venv .venv
python -m pip install -r requirements.txt
python -m src.rates_risk
pytest -q
```

## Valuation and risk insights

A payer swap pays fixed and receives floating. Its value generally increases when market rates rise because the contractual fixed rate becomes relatively cheaper. DV01 approximates the change in value for a one-basis-point parallel shift. Curve-shape scenarios complement DV01 because a single parallel sensitivity does not capture steepening or flattening risk.

## Skills demonstrated

Fixed Income • Yield Curve • Bootstrap • Discount Factors • Forward Rates • Interest Rate Swaps • NPV • Par Rate • DV01 • Stress Testing • Python

## Author

Sofia Kammoune — MBA Trading & Finance de Marché, ESLSCA Business School Paris.

