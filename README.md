# Quantitative Research
Quantitative analysis of the Oops pattern by Larry Williams
Author: Simone Ciceri

---

# Oops Pattern – Quantitative Strategy Research

This repository contains the code used in a quantitative research project
focused on the Oops pattern originally introduced by Larry Williams.

The objective of the project is to systematically evaluate the statistical edge
of a discretionary trading pattern and progressively transform it into a fully
specified, rule-based trading strategy.

The research focuses on signal validation, return distribution analysis,
MAE/MFE dynamics, exit design, and basic robustness considerations.

---

## Research Overview

The project follows a structured, research-driven approach:

1. Formal definition of the Oops pattern using mechanical rules
2. Validation of the raw edge through distributional analysis
3. Regime analysis (micro & macro)
4. Study of adverse and favorable excursions (MAE / MFE)
5. Design and comparison of different exit strategies
6. Transition from a pure edge to a complete trading strategy

The goal is not to optimize performance, but to understand **why and when**
the pattern exhibits a statistical advantage.

---

## Data & Environment

- Research conducted using the **QuantConnect Research Environment**
- Python-based analysis
- Daily price data
- Assumptions:
  - No transaction costs
  - No slippage

All results should be interpreted as *research outputs*, not as live trading performance.

---

## Repository Structure

oops-larry-williams/
│
├── README.md
├── paper/
│   └── Oops_Pattern_Quantitative_Study.pdf
│
├── src/ (fare io qui)
│   ├── oops_signal.py
│   ├── mae_mfe_analysis.py
│   ├── exit_analysis.py
│   └── utils.py
│
├── notebooks/
│   └── oops_research.ipynb
│
└── requirements.txt


- `paper/` contains the full research paper describing methodology and results
- `src/` contains reusable Python modules used in the analysis
?? - `notebooks/` contains exploratory research notebooks

---

## Key Metrics Analyzed

- Win rate and expectancy
- Return distribution and skewness
- Maximum Adverse Excursion (MAE)
- Maximum Favorable Excursion (MFE)
- Time to MFE (TMFE)
- Percentage of MFE captured by different exit rules

These metrics are used to guide exit design and risk management decisions.

---

## Disclaimer

This project is intended for **educational and research purposes only**.
It does not constitute investment advice or a recommendation to trade
any financial instrument.

Past performance, whether simulated or historical, is not indicative
of future results.

---

For a full explanation of the methodology and results, see the attached research paper.

