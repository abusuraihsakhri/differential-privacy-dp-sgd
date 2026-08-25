"""
CLI for Differential Privacy Calculator
Supports: Gaussian/Laplace mechanisms, composition theorems, DP-SGD, sensitivity, budget tracking.
"""
import argparse
import json
import sys
import random

from diff_privacy_engine.engine import (
    gaussian_noise_scale, gaussian_mechanism,
    laplace_noise_scale, laplace_mechanism,
    basic_composition, advanced_composition,
    rdp_epsilon, rdp_to_dp, rdp_composition_gaussian, find_sigma_for_rdp,
    count_sensitivity, sum_sensitivity, mean_sensitivity, histogram_sensitivity,
    clip_gradient, dp_sgd_noise, dp_sgd_step, MomentsAccountant,
    PrivacyBudgetCalculator, run_gaussian_query, run_laplace_query,
)


def cmd_gaussian(args):
    """Run Gaussian mechanism."""
    result = run_gaussian_query(args.value, args.sensitivity, args.epsilon, args.delta)
    print(json.dumps(result, indent=2))


def cmd_laplace(args):
    """Run Laplace mechanism."""
    result = run_laplace_query(args.value, args.sensitivity, args.epsilon)
    print(json.dumps(result, indent=2))


def cmd_noise_scale(args):
    """Compute noise scale for a mechanism."""
    if args.mechanism == "gaussian":
        sigma = gaussian_noise_scale(args.sensitivity, args.epsilon, args.delta)
        print(json.dumps({"mechanism": "gaussian", "sigma": sigma,
                          "sensitivity": args.sensitivity,
                          "epsilon": args.epsilon, "delta": args.delta}, indent=2))
    else:
        b = laplace_noise_scale(args.sensitivity, args.epsilon)
        print(json.dumps({"mechanism": "laplace", "scale_b": b,
                          "sensitivity": args.sensitivity,
                          "epsilon": args.epsilon}, indent=2))


def cmd_composition(args):
    """Compute privacy budget composition."""
    if args.type == "basic":
        total_eps = basic_composition(args.epsilon, args.k)
        print(json.dumps({"type": "basic", "epsilon_per_mechanism": args.epsilon,
                          "k": args.k, "total_epsilon": total_eps}, indent=2))
    elif args.type == "advanced":
        total_eps, total_delta = advanced_composition(
            args.epsilon, args.delta, args.k, args.delta_prime)
        print(json.dumps({"type": "advanced", "epsilon_per_mechanism": args.epsilon,
                          "delta_per_mechanism": args.delta, "k": args.k,
                          "delta_prime": args.delta_prime,
                          "total_epsilon": total_eps, "total_delta": total_delta}, indent=2))
    elif args.type == "rdp":
        alphas = [float(a) for a in args.alphas.split(",")] if args.alphas else \
                 [1.0 + x / 10.0 for x in range(1, 100)]
        sigma = args.sigma if args.sigma else gaussian_noise_scale(
            args.sensitivity, args.epsilon, args.delta)
        rdp_eps = rdp_composition_gaussian(sigma, args.k, alphas, args.sensitivity)
        dp_eps = rdp_to_dp(rdp_eps, args.delta)
        # Convert dict keys to strings for JSON
        rdp_str = {str(k): v for k, v in rdp_eps.items()}
        print(json.dumps({"type": "rdp", "sigma": sigma, "k": args.k,
                          "delta": args.delta, "rdp_epsilons": rdp_str,
                          "dp_epsilon": dp_eps}, indent=2))


def cmd_sensitivity(args):
    """Compute sensitivity for common queries."""
    if args.query == "count":
        s = count_sensitivity()
    elif args.query == "sum":
        s = sum_sensitivity(args.lower, args.upper)
    elif args.query == "mean":
        s = mean_sensitivity(args.lower, args.upper, args.n)
    elif args.query == "histogram":
        s = histogram_sensitivity()
    else:
        print(f"Unknown query type: {args.query}", file=sys.stderr)
        return 1
    print(json.dumps({"query": args.query, "sensitivity": s}, indent=2))


def cmd_dp_sgd(args):
    """Run DP-SGD step."""
    # Generate sample gradients for demo
    if args.gradients:
        gradients = json.loads(args.gradients)
    else:
        gradients = [[random.gauss(0, 1) for _ in range(args.dim)]
                     for _ in range(args.batch_size)]

    noise_mult = args.noise_multiplier
    if noise_mult is None:
        noise_mult = dp_sgd_noise(args.clip_norm, args.epsilon, args.delta, args.batch_size)

    result_grad = dp_sgd_step(gradients, args.clip_norm, noise_mult)
    print(json.dumps({
        "clip_norm": args.clip_norm,
        "noise_multiplier": noise_mult,
        "batch_size": len(gradients),
        "gradient_dim": len(result_grad),
        "result_gradient": result_grad[:10],  # show first 10 dims
        "result_gradient_norm": sum(g**2 for g in result_grad) ** 0.5,
    }, indent=2))


def cmd_accountant(args):
    """Run moments accountant."""
    acct = MomentsAccountant(args.delta)
    alphas = [float(a) for a in args.alphas.split(",")] if args.alphas else None
    acct.accumulate(args.sigma, args.steps, alphas)
    eps, delta = acct.get_privacy_spent(alphas)
    print(json.dumps({
        "sigma": args.sigma,
        "steps": args.steps,
        "delta": delta,
        "epsilon": eps,
    }, indent=2))


def cmd_budget(args):
    """Manage privacy budget."""
    calc = PrivacyBudgetCalculator(args.total_epsilon, args.total_delta)
    if args.spend_epsilon:
        ok = calc.spend(args.spend_epsilon, args.spend_delta or 0.0,
                        args.description or "CLI query")
        print(json.dumps({
            "spend_accepted": ok,
            **calc.budget_summary(),
        }, indent=2))
    else:
        print(json.dumps(calc.budget_summary(), indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="differential-privacy-dp-sgd",
        description="Differential Privacy Calculator: mechanisms, composition, DP-SGD")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Gaussian mechanism
    p_gauss = subparsers.add_parser("gaussian", help="Apply Gaussian mechanism")
    p_gauss.add_argument("--value", type=float, required=True, help="True value")
    p_gauss.add_argument("--sensitivity", type=float, required=True, help="L2 sensitivity")
    p_gauss.add_argument("--epsilon", type=float, required=True, help="Privacy parameter epsilon")
    p_gauss.add_argument("--delta", type=float, required=True, help="Privacy parameter delta")

    # Laplace mechanism
    p_lap = subparsers.add_parser("laplace", help="Apply Laplace mechanism")
    p_lap.add_argument("--value", type=float, required=True, help="True value")
    p_lap.add_argument("--sensitivity", type=float, required=True, help="L1 sensitivity")
    p_lap.add_argument("--epsilon", type=float, required=True, help="Privacy parameter epsilon")

    # Noise scale
    p_noise = subparsers.add_parser("noise-scale", help="Compute noise scale")
    p_noise.add_argument("--mechanism", choices=["gaussian", "laplace"], required=True)
    p_noise.add_argument("--sensitivity", type=float, required=True)
    p_noise.add_argument("--epsilon", type=float, required=True)
    p_noise.add_argument("--delta", type=float, default=1e-5)

    # Composition
    p_comp = subparsers.add_parser("composition", help="Privacy budget composition")
    p_comp.add_argument("--type", choices=["basic", "advanced", "rdp"], required=True)
    p_comp.add_argument("--epsilon", type=float, required=True, help="Per-mechanism epsilon")
    p_comp.add_argument("--delta", type=float, default=1e-5, help="Per-mechanism delta")
    p_comp.add_argument("--k", type=int, required=True, help="Number of mechanisms")
    p_comp.add_argument("--delta-prime", type=float, default=1e-5, help="Delta' for advanced")
    p_comp.add_argument("--alphas", type=str, default=None, help="Comma-separated alpha values for RDP")
    p_comp.add_argument("--sigma", type=float, default=None, help="Noise sigma for RDP")
    p_comp.add_argument("--sensitivity", type=float, default=1.0)

    # Sensitivity
    p_sens = subparsers.add_parser("sensitivity", help="Compute query sensitivity")
    p_sens.add_argument("--query", choices=["count", "sum", "mean", "histogram"], required=True)
    p_sens.add_argument("--lower", type=float, default=0.0, help="Lower bound for sum/mean")
    p_sens.add_argument("--upper", type=float, default=1.0, help="Upper bound for sum/mean")
    p_sens.add_argument("--n", type=int, default=100, help="Number of records for mean")

    # DP-SGD
    p_sgd = subparsers.add_parser("dp-sgd", help="Run DP-SGD step")
    p_sgd.add_argument("--clip-norm", type=float, required=True, help="Gradient clip norm")
    p_sgd.add_argument("--epsilon", type=float, required=True)
    p_sgd.add_argument("--delta", type=float, required=True)
    p_sgd.add_argument("--batch-size", type=int, default=32)
    p_sgd.add_argument("--dim", type=int, default=10, help="Gradient dimension")
    p_sgd.add_argument("--noise-multiplier", type=float, default=None)
    p_sgd.add_argument("--gradients", type=str, default=None, help="JSON gradients array")

    # Accountant
    p_acct = subparsers.add_parser("accountant", help="Moments accountant")
    p_acct.add_argument("--sigma", type=float, required=True, help="Noise sigma")
    p_acct.add_argument("--steps", type=int, required=True, help="Number of steps")
    p_acct.add_argument("--delta", type=float, required=True)
    p_acct.add_argument("--alphas", type=str, default=None)

    # Budget
    p_budget = subparsers.add_parser("budget", help="Privacy budget calculator")
    p_budget.add_argument("--total-epsilon", type=float, required=True)
    p_budget.add_argument("--total-delta", type=float, default=1e-5)
    p_budget.add_argument("--spend-epsilon", type=float, default=None)
    p_budget.add_argument("--spend-delta", type=float, default=None)
    p_budget.add_argument("--description", type=str, default="")

    args = parser.parse_args(argv)

    cmd_map = {
        "gaussian": cmd_gaussian,
        "laplace": cmd_laplace,
        "noise-scale": cmd_noise_scale,
        "composition": cmd_composition,
        "sensitivity": cmd_sensitivity,
        "dp-sgd": cmd_dp_sgd,
        "accountant": cmd_accountant,
        "budget": cmd_budget,
    }
    return cmd_map[args.command](args)


if __name__ == "__main__":
    sys.exit(main() or 0)
