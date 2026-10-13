"""
Differential Privacy Calculator Engine
Implements: Epsilon-Delta privacy budget, Gaussian/Laplace mechanisms,
privacy budget composition (basic, advanced, RDP), DP-SGD, and sensitivity calculation.
Uses only Python stdlib.
"""
import math
from random import SystemRandom
from typing import List, Dict, Any, Optional, Tuple

# OS-backed entropy; seeding the process-global pseudo-random generator must not
# make released noise reproducible. Floating-point DP remains best-effort.
_rng = SystemRandom()


def _finite_positive(value: float, name: str) -> None:
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


def _finite_nonnegative(value: float, name: str) -> None:
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and non-negative")


# ---------------------------------------------------------------------------
# Sensitivity Calculation
# ---------------------------------------------------------------------------

def count_sensitivity() -> float:
    """L2 sensitivity of a count query is 1 (adding/removing one record changes count by 1)."""
    return 1.0


def sum_sensitivity(lower: float, upper: float) -> float:
    """L2 sensitivity of a sum query with bounded data in [lower, upper]."""
    if not math.isfinite(lower) or not math.isfinite(upper) or lower > upper:
        raise ValueError("bounds must be finite and lower <= upper")
    return max(abs(lower), abs(upper))


def mean_sensitivity(lower: float, upper: float, n: int) -> float:
    """L2 sensitivity of a mean query over n records with values in [lower, upper]."""
    if n <= 0:
        raise ValueError("n must be positive")
    if not math.isfinite(lower) or not math.isfinite(upper) or lower > upper:
        raise ValueError("bounds must be finite and lower <= upper")
    # Fixed-size datasets with substitution adjacency.
    return (upper - lower) / n


def histogram_sensitivity() -> float:
    """L2 sensitivity of a histogram query is 1 (one record changes one bin by 1)."""
    return 1.0


# ---------------------------------------------------------------------------
# Gaussian Mechanism
# ---------------------------------------------------------------------------

def gaussian_noise_scale(sensitivity: float, epsilon: float, delta: float) -> float:
    """
    Compute sigma for the Gaussian mechanism.
    sigma = sensitivity * sqrt(2 * ln(1.25 / delta)) / epsilon
    """
    _finite_nonnegative(sensitivity, "sensitivity")
    _finite_positive(epsilon, "epsilon")
    if not math.isfinite(delta) or delta <= 0 or delta >= 1:
        raise ValueError("delta must be finite and in (0, 1)")
    # The classical 1.25/delta calibration is proved for 0 < epsilon <= 1.
    # For larger epsilon, use the conservative Gaussian RDP conversion bound.
    if epsilon <= 1:
        return sensitivity * math.sqrt(2.0 * math.log(1.25 / delta)) / epsilon
    log_inverse_delta = math.log(1.0 / delta)
    return sensitivity * (math.sqrt(2 * log_inverse_delta + 2 * epsilon)
                          + math.sqrt(2 * log_inverse_delta)) / (2 * epsilon)


def gaussian_mechanism(value: float, sensitivity: float, epsilon: float, delta: float) -> float:
    """Apply Gaussian mechanism: return value + N(0, sigma^2)."""
    sigma = gaussian_noise_scale(sensitivity, epsilon, delta)
    # Box-Muller transform for generating normal random variable
    u1 = _rng.random()
    u2 = _rng.random()
    while u1 == 0:
        u1 = _rng.random()
    z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
    return value + sigma * z


def gaussian_mechanism_vector(values: List[float], sensitivity: float,
                               epsilon: float, delta: float) -> List[float]:
    """Apply Gaussian mechanism to a vector of values."""
    sigma = gaussian_noise_scale(sensitivity, epsilon, delta)
    result = []
    for v in values:
        u1 = _rng.random()
        u2 = _rng.random()
        while u1 == 0:
            u1 = _rng.random()
        z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
        result.append(v + sigma * z)
    return result


# ---------------------------------------------------------------------------
# Laplace Mechanism
# ---------------------------------------------------------------------------

def laplace_noise_scale(sensitivity: float, epsilon: float) -> float:
    """
    Compute b (scale) for the Laplace mechanism.
    b = sensitivity / epsilon
    """
    _finite_nonnegative(sensitivity, "sensitivity")
    _finite_positive(epsilon, "epsilon")
    return sensitivity / epsilon


def _sample_laplace(b: float) -> float:
    """Sample from Laplace(0, b) using inverse CDF."""
    u = _rng.random() - 0.5
    # Avoid log(0)
    if u == 0:
        return 0.0
    return -b * math.copysign(1.0, u) * math.log(1.0 - 2.0 * abs(u))


def laplace_mechanism(value: float, sensitivity: float, epsilon: float) -> float:
    """Apply Laplace mechanism: return value + Lap(0, b)."""
    b = laplace_noise_scale(sensitivity, epsilon)
    return value + _sample_laplace(b)


def laplace_mechanism_vector(values: List[float], sensitivity: float,
                              epsilon: float) -> List[float]:
    """Apply Laplace mechanism to a vector of values."""
    b = laplace_noise_scale(sensitivity, epsilon)
    return [v + _sample_laplace(b) for v in values]


# ---------------------------------------------------------------------------
# Privacy Budget Composition
# ---------------------------------------------------------------------------

def basic_composition(epsilon_per_mechanism: float, k: int) -> float:
    """
    Basic composition theorem:
    k mechanisms each with (epsilon_i, delta_i) ->
    total epsilon = sum(epsilon_i), total delta = sum(delta_i)
    """
    _finite_nonnegative(epsilon_per_mechanism, "epsilon")
    if k < 0:
        raise ValueError("k must be non-negative")
    return k * epsilon_per_mechanism


def advanced_composition(epsilon_per_mechanism: float, delta_per_mechanism: float,
                          k: int, delta_prime: float) -> Tuple[float, float]:
    """
    Advanced composition theorem (Dwork, Rothblum, Vadhan 2010):
    k mechanisms each with (epsilon_i, delta_i) ->
    epsilon_total = sqrt(2 * k * ln(1/delta')) * epsilon_i + k * epsilon_i * (e^epsilon_i - 1)
    delta_total = k * delta_i + delta'
    """
    _finite_nonnegative(epsilon_per_mechanism, "epsilon")
    _finite_nonnegative(delta_per_mechanism, "delta")
    if k < 0:
        raise ValueError("k must be non-negative")
    if delta_prime <= 0 or delta_prime >= 1:
        raise ValueError("delta_prime must be in (0, 1)")

    eps_i = epsilon_per_mechanism
    delta_i = delta_per_mechanism

    epsilon_total = (math.sqrt(2.0 * k * math.log(1.0 / delta_prime)) * eps_i
                     + k * eps_i * (math.exp(eps_i) - 1.0))
    delta_total = k * delta_i + delta_prime

    return epsilon_total, delta_total


# ---------------------------------------------------------------------------
# Rényi Differential Privacy (RDP) Composition
# ---------------------------------------------------------------------------

def rdp_epsilon(alpha: float, sigma: float, sensitivity: float = 1.0) -> float:
    """
    Compute RDP epsilon for the Gaussian mechanism at order alpha.
    RDP epsilon = sensitivity^2 / (2 * sigma^2) * alpha
    """
    if not math.isfinite(alpha) or alpha <= 1.0:
        raise ValueError("alpha must be finite and > 1 for RDP")
    _finite_positive(sigma, "sigma")
    _finite_nonnegative(sensitivity, "sensitivity")
    return 0.5 * (sensitivity / sigma) ** 2 * alpha


def rdp_to_dp(rdp_epsilons: Dict[float, float], delta: float) -> float:
    """
    Convert RDP guarantees to (epsilon, delta)-DP.
    For each alpha: epsilon = rdp_epsilon(alpha) + ln(1/delta)/(alpha-1)
    Return the minimum epsilon over all tested alpha values.
    """
    if delta <= 0 or delta >= 1:
        raise ValueError("delta must be in (0, 1)")

    best_epsilon = float('inf')
    for alpha, rdp_eps in rdp_epsilons.items():
        if not math.isfinite(alpha) or alpha <= 1.0:
            raise ValueError("RDP orders must be finite and > 1")
        _finite_nonnegative(rdp_eps, "RDP epsilon")
        eps = rdp_eps + math.log(1.0 / delta) / (alpha - 1.0)
        best_epsilon = min(best_epsilon, eps)
    if not rdp_epsilons:
        raise ValueError("at least one valid RDP order is required")
    return best_epsilon


def rdp_composition_gaussian(sigma: float, k: int, alphas: List[float],
                              sensitivity: float = 1.0) -> Dict[float, float]:
    """
    RDP composition for k applications of Gaussian mechanism with scale sigma.
    RDP composes linearly: total_rdp_epsilon = k * per_mechanism_rdp_epsilon
    """
    _finite_positive(sigma, "sigma")
    _finite_nonnegative(sensitivity, "sensitivity")
    if k < 0:
        raise ValueError("k must be non-negative")

    result = {}
    for alpha in alphas:
        if alpha <= 1.0:
            continue
        per_eps = rdp_epsilon(alpha, sigma, sensitivity)
        result[alpha] = k * per_eps
    return result


def find_sigma_for_rdp(target_epsilon: float, delta: float, k: int,
                        alphas: Optional[List[float]] = None,
                        sensitivity: float = 1.0) -> float:
    """
    Find the minimum sigma such that k compositions of Gaussian mechanism
    with that sigma satisfy (epsilon, delta)-DP.
    Uses binary search over sigma.
    """
    _finite_positive(target_epsilon, "target_epsilon")
    if not math.isfinite(delta) or not 0 < delta < 1:
        raise ValueError("delta must be in (0, 1)")
    if k < 0:
        raise ValueError("k must be non-negative")
    if alphas is None:
        alphas = [1.0 + x / 10.0 for x in range(1, 100)] + list(range(11, 257))
    if not alphas or any(not math.isfinite(a) or a <= 1 for a in alphas):
        raise ValueError("alphas must be finite and > 1")
    # Conversion carries an irreducible ln(1/delta)/(alpha-1) penalty.
    # Reject unattainable requests instead of overflowing while doubling sigma.
    if min(math.log(1 / delta) / (alpha - 1) for alpha in alphas) >= target_epsilon:
        raise ValueError("RDP orders too small for the requested epsilon and delta")
    lo, hi = 0.0, 1.0
    while rdp_to_dp(rdp_composition_gaussian(hi, k, alphas, sensitivity), delta) > target_epsilon:
        hi *= 2
        if not math.isfinite(hi):
            raise ValueError("cannot calibrate finite sigma for these parameters")
    for _ in range(100):  # binary search iterations
        mid = (lo + hi) / 2.0
        rdp_eps = rdp_composition_gaussian(mid, k, alphas, sensitivity)
        eps = rdp_to_dp(rdp_eps, delta)
        if eps <= target_epsilon:
            hi = mid
        else:
            lo = mid
    return hi


# ---------------------------------------------------------------------------
# DP-SGD (Differentially Private Stochastic Gradient Descent)
# ---------------------------------------------------------------------------

def clip_gradient(gradient: List[float], clip_norm: float) -> List[float]:
    """
    Per-sample gradient clipping: clip gradient to have L2 norm <= clip_norm.
    If ||g||_2 > C, scale to C * g / ||g||_2.
    """
    _finite_positive(clip_norm, "clip_norm")
    if not gradient or any(not math.isfinite(g) for g in gradient):
        raise ValueError("gradient must be non-empty and finite")
    norm = math.hypot(*gradient)
    if norm <= clip_norm:
        return list(gradient)
    scale = clip_norm / norm
    return [g * scale for g in gradient]


def clip_gradients(gradients: List[List[float]], clip_norm: float) -> List[List[float]]:
    """Clip a batch of per-sample gradients."""
    return [clip_gradient(g, clip_norm) for g in gradients]


def aggregate_gradients(gradients: List[List[float]]) -> List[float]:
    """Aggregate clipped gradients by averaging."""
    if not gradients:
        raise ValueError("Empty gradient list")
    n = len(gradients)
    dim = len(gradients[0])
    if not dim or any(len(g) != dim or any(not math.isfinite(x) for x in g) for g in gradients):
        raise ValueError("gradients must be non-empty, finite, and equally sized")
    result = [0.0] * dim
    for g in gradients:
        for i in range(dim):
            result[i] += g[i]
    return [r / n for r in result]


def dp_sgd_noise(clip_norm: float, epsilon: float, delta: float, batch_size: int = 1) -> float:
    """
    Compute noise multiplier for DP-SGD.
    sigma = clip_norm * sqrt(2 * ln(1.25 / delta)) / epsilon
    For mini-batch, noise is divided by batch_size.
    """
    _finite_positive(clip_norm, "clip_norm")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    # Returns a dimensionless multiplier, not an absolute standard deviation.
    # The step adds noise with stddev = 2 * multiplier * clip_norm / batch_size
    # under replacement adjacency for a fixed-size batch.
    return gaussian_noise_scale(1.0, epsilon, delta)


def dp_sgd_step(gradients: List[List[float]], clip_norm: float,
                 noise_multiplier: float) -> List[float]:
    """
    One step of DP-SGD:
    1. Clip each gradient to norm C
    2. Aggregate (average)
    3. Add calibrated Gaussian noise
    """
    _finite_positive(noise_multiplier, "noise_multiplier")
    clipped = clip_gradients(gradients, clip_norm)
    aggregated = aggregate_gradients(clipped)

    # Add noise
    n = len(gradients)
    # Fixed-size datasets under replacement adjacency have sensitivity 2*C/n.
    # The multiplier is relative to that sensitivity (no subsampling credit).
    sigma = 2.0 * noise_multiplier * clip_norm / n
    noisy = []
    for g in aggregated:
        u1 = _rng.random()
        u2 = _rng.random()
        while u1 == 0:
            u1 = _rng.random()
        z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
        noisy.append(g + sigma * z)
    return noisy


# ---------------------------------------------------------------------------
# Moments Accountant (Privacy Accountant for DP-SGD)
# ---------------------------------------------------------------------------

class MomentsAccountant:
    """Gaussian RDP accountant for full-batch steps (no subsampling amplification).

    The supplied sigma is the dimensionless noise multiplier (noise stddev
    divided by clipped gradient L2 norm), not an absolute gradient stddev.
    """

    def __init__(self, target_delta: float):
        if not math.isfinite(target_delta) or not 0 < target_delta < 1:
            raise ValueError("target_delta must be in (0, 1)")
        self.target_delta = target_delta
        self.log_moments: List[float] = []  # Stored RDP epsilon at each order.
        self._orders: Optional[Tuple[float, ...]] = None
        self.steps = 0

    def accumulate(self, sigma: float, num_steps: int = 1,
                   alphas: Optional[List[float]] = None):
        """Add Gaussian steps using the non-subsampled RDP composition bound."""
        _finite_positive(sigma, "sigma")
        if not isinstance(num_steps, int) or num_steps < 0:
            raise ValueError("num_steps must be a non-negative integer")
        orders = tuple(alphas if alphas is not None else
                       [1.0 + x / 10.0 for x in range(1, 100)])
        if not orders or any(not math.isfinite(a) or a <= 1 for a in orders):
            raise ValueError("alphas must be finite and > 1")
        if self._orders is not None and orders != self._orders:
            raise ValueError("RDP orders cannot change after accumulation")
        self._orders = orders
        if not self.log_moments:
            self.log_moments = [0.0] * len(orders)
        for i, alpha in enumerate(orders):
            self.log_moments[i] += num_steps * alpha / (2 * sigma * sigma)
        self.steps += num_steps

    def get_privacy_spent(self, alphas: Optional[List[float]] = None) -> Tuple[float, float]:
        if self.steps == 0:
            return 0.0, self.target_delta
        if alphas is not None and tuple(alphas) != self._orders:
            raise ValueError("RDP orders must match those used in accumulate")
        return rdp_to_dp(dict(zip(self._orders, self.log_moments)),
                         self.target_delta), self.target_delta


# ---------------------------------------------------------------------------
# Utility: Epsilon-Delta Privacy Budget Calculator
# ---------------------------------------------------------------------------

class PrivacyBudgetCalculator:
    """Calculator for managing and tracking epsilon-delta privacy budgets."""

    def __init__(self, total_epsilon: float, total_delta: float):
        if total_epsilon <= 0:
            raise ValueError("total_epsilon must be positive")
        if total_delta < 0 or total_delta >= 1:
            raise ValueError("total_delta must be in [0, 1)")
        self.total_epsilon = total_epsilon
        self.total_delta = total_delta
        self.spent_epsilon = 0.0
        self.spent_delta = 0.0
        self.history: List[Dict[str, Any]] = []

    @property
    def remaining_epsilon(self) -> float:
        return max(0.0, self.total_epsilon - self.spent_epsilon)

    @property
    def remaining_delta(self) -> float:
        return max(0.0, self.total_delta - self.spent_delta)

    @property
    def is_exhausted(self) -> bool:
        return self.spent_epsilon >= self.total_epsilon

    def spend(self, epsilon: float, delta: float = 0.0, description: str = "") -> bool:
        """Spend privacy budget; reject non-finite or negative spends."""
        _finite_nonnegative(epsilon, "epsilon")
        _finite_nonnegative(delta, "delta")
        if self.spent_epsilon + epsilon > self.total_epsilon:
            return False
        if self.spent_delta + delta > self.total_delta:
            return False

        self.spent_epsilon += epsilon
        self.spent_delta += delta
        self.history.append({
            "epsilon": epsilon,
            "delta": delta,
            "cumulative_epsilon": self.spent_epsilon,
            "cumulative_delta": self.spent_delta,
            "description": description,
        })
        return True

    def budget_summary(self) -> Dict[str, Any]:
        return {
            "total_epsilon": self.total_epsilon,
            "total_delta": self.total_delta,
            "spent_epsilon": self.spent_epsilon,
            "spent_delta": self.spent_delta,
            "remaining_epsilon": self.remaining_epsilon,
            "remaining_delta": self.remaining_delta,
            "is_exhausted": self.is_exhausted,
            "num_queries": len(self.history),
        }


# ---------------------------------------------------------------------------
# Convenience: run a full DP query
# ---------------------------------------------------------------------------

def run_gaussian_query(value: float, sensitivity: float, epsilon: float,
                        delta: float) -> Dict[str, Any]:
    """Run a single Gaussian mechanism query and return results."""
    sigma = gaussian_noise_scale(sensitivity, epsilon, delta)
    noisy = gaussian_mechanism(value, sensitivity, epsilon, delta)
    return {
        "mechanism": "gaussian",
        "noisy_value": noisy,
        "sigma": sigma,
        "epsilon": epsilon,
        "delta": delta,
        "sensitivity": sensitivity,
    }


def run_laplace_query(value: float, sensitivity: float, epsilon: float) -> Dict[str, Any]:
    """Run a single Laplace mechanism query and return results."""
    b = laplace_noise_scale(sensitivity, epsilon)
    noisy = laplace_mechanism(value, sensitivity, epsilon)
    return {
        "mechanism": "laplace",
        "noisy_value": noisy,
        "scale_b": b,
        "epsilon": epsilon,
        "sensitivity": sensitivity,
    }


# ---------------------------------------------------------------------------
# Frontier Domain Engine (Threshold-based evaluation for agent auditing)
# ---------------------------------------------------------------------------

class FrontierDomainEngine:
    """
    Threshold-based evaluation engine used by DP agent sub-auditors.
    Each static method returns a dict with summary/details/remediation when
    a threshold is breached, or None when the value is within nominal bounds.
    """

    PRIMARY_THRESHOLD = 25.0
    SECONDARY_THRESHOLD = 12.0
    DISCORDANT_KEYWORDS = {"DISCORDANT", "ANOMALY", "MUTANT", "VIOLATION", "FAIL", "REJECT"}

    @staticmethod
    def evaluate_primary_parameter(primary_metric: float) -> Optional[Dict[str, str]]:
        """
        Evaluate primary parameter against reference threshold.
        Returns alert dict if threshold exceeded, None otherwise.
        """
        if primary_metric > FrontierDomainEngine.PRIMARY_THRESHOLD:
            return {
                "summary": "Primary Parameter Threshold Exceeded",
                "details": (
                    f"Primary metric ({primary_metric:.2f}) exceeds upper reference "
                    f"limit ({FrontierDomainEngine.PRIMARY_THRESHOLD:.2f})."
                ),
                "remediation": (
                    "Initiate recalibration workflow and review secondary parameters "
                    "under Differential Privacy (Mironov Renyi DP) specifications."
                ),
            }
        return None

    @staticmethod
    def evaluate_secondary_kinetics(secondary_metric: float, is_critical_flag: bool) -> Optional[Dict[str, str]]:
        """
        Evaluate secondary kinetics against safety boundary.
        Returns alert dict if threshold exceeded or critical flag set, None otherwise.
        """
        if is_critical_flag or secondary_metric > FrontierDomainEngine.SECONDARY_THRESHOLD:
            return {
                "summary": "Critical Safety Interlock Triggered",
                "details": (
                    f"CriticalFlag={is_critical_flag} with secondary index "
                    f"{secondary_metric:.2f} exceeding safety boundary "
                    f"({FrontierDomainEngine.SECONDARY_THRESHOLD:.2f})."
                ),
                "remediation": (
                    "Execute immediate closed-loop escalation and notify attending supervisor."
                ),
            }
        return None

    @staticmethod
    def audit_specification_conformance(status_descriptor: str, attributes: Dict[str, Any]) -> Optional[Dict[str, str]]:
        """
        Audit status descriptor for protocol conformance.
        Returns alert dict if discordant keywords detected, None otherwise.
        """
        desc_upper = str(status_descriptor).upper()
        if any(kw in desc_upper for kw in FrontierDomainEngine.DISCORDANT_KEYWORDS):
            return {
                "summary": "Protocol Conformance Discordance Detected",
                "details": (
                    f"Descriptor '{status_descriptor}' indicates discordance with "
                    f"Differential Privacy (Mironov Renyi DP) specifications."
                ),
                "remediation": (
                    "Re-evaluate input parameters or rerun secondary confirmation analysis."
                ),
            }
        return None
