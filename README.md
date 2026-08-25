# Differential Privacy Calculator (DP-SGD)

A pure-Python implementation of differential privacy mechanisms, composition theorems, and DP-SGD (Differentially Private Stochastic Gradient Descent). Uses only the Python standard library.

## Features

- **Gaussian Mechanism**: `noise = N(0, σ²)` where `σ = Δf × √(2ln(1.25/δ)) / ε`
- **Laplace Mechanism**: `noise = Lap(Δf / ε)`
- **Privacy Budget Composition**:
  - Basic composition: `k` mechanisms → `ε_total = k × ε_i`
  - Advanced composition (Dwork et al. 2010)
  - Rényi Differential Privacy (RDP) composition
- **DP-SGD**:
  - Per-sample gradient clipping (clip to norm C)
  - Gaussian noise injection: `σ = C × √(2ln(1.25/δ)) / ε`
  - Moments accountant for tracking cumulative privacy loss
- **Sensitivity Calculation**: Count, sum, mean, histogram queries
- **Privacy Budget Calculator**: Track and manage ε-δ budget across queries

## CLI Usage

```bash
# Gaussian mechanism
python cli.py gaussian --value 100 --sensitivity 1.0 --epsilon 1.0 --delta 1e-5

# Laplace mechanism
python cli.py laplace --value 50 --sensitivity 1.0 --epsilon 0.5

# Compute noise scale
python cli.py noise-scale --mechanism gaussian --sensitivity 1.0 --epsilon 1.0 --delta 1e-5

# Privacy composition
python cli.py composition --type basic --epsilon 0.1 --k 10
python cli.py composition --type advanced --epsilon 0.1 --delta 1e-6 --k 10 --delta-prime 1e-5
python cli.py composition --type rdp --epsilon 1.0 --delta 1e-5 --k 100 --sigma 1.0

# Sensitivity calculation
python cli.py sensitivity --query count
python cli.py sensitivity --query sum --lower 0 --upper 100
python cli.py sensitivity --query mean --lower 0 --upper 10 --n 1000

# DP-SGD step
python cli.py dp-sgd --clip-norm 1.0 --epsilon 1.0 --delta 1e-5 --batch-size 32

# Moments accountant
python cli.py accountant --sigma 1.0 --steps 1000 --delta 1e-5

# Privacy budget management
python cli.py budget --total-epsilon 10.0 --total-delta 1e-5
python cli.py budget --total-epsilon 10.0 --spend-epsilon 3.0 --description "count query"
```

## Python API

```python
from diff_privacy_engine.engine import (
    gaussian_mechanism, laplace_mechanism,
    basic_composition, advanced_composition,
    PrivacyBudgetCalculator, MomentsAccountant,
    dp_sgd_step, clip_gradient,
)

# Apply Gaussian mechanism
noisy_value = gaussian_mechanism(value=100.0, sensitivity=1.0, epsilon=1.0, delta=1e-5)

# Track privacy budget
budget = PrivacyBudgetCalculator(total_epsilon=10.0, total_delta=1e-5)
budget.spend(epsilon=1.0, delta=1e-6, description="count query")
print(budget.remaining_epsilon)  # 9.0

# DP-SGD
noisy_grad = dp_sgd_step(gradients, clip_norm=1.0, noise_multiplier=1.1)
```

## Running Tests

```bash
python -m pytest tests/ -v
```

## License

MIT License.
