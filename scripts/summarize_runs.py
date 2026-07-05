from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml


def last_metric(row: pd.Series, name: str) -> float:
    return float(row[name]) if name in row and pd.notna(row[name]) else float("nan")


def best_metric(df: pd.DataFrame, name: str, direction: str) -> float:
    if name not in df:
        return float("nan")
    if direction == "max":
        return float(df[name].max())
    if direction == "min":
        return float(df[name].min())
    raise ValueError(f"Unsupported direction: {direction}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", default="runs/fal_longterm_fair_query")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--latest-per-strategy", action="store_true")
    parser.add_argument("--aggregate-seeds", action="store_true")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs_root = Path(args.runs_root)
    rows = []

    for run_dir in sorted(runs_root.glob("*")):
        if not run_dir.is_dir():
            continue
        if args.dataset and f"_{args.dataset}_" not in run_dir.name:
            continue
        metrics_path = run_dir / "round_metrics.csv"
        config_path = run_dir / "resolved_config.yaml"
        if not metrics_path.exists():
            continue
        config = {}
        if config_path.exists():
            with config_path.open("r", encoding="utf-8") as config_file:
                config = yaml.safe_load(config_file) or {}
        dataset_cfg = config.get("dataset", {})
        fed_cfg = config.get("federated", {})
        al_cfg = config.get("active_learning", {})
        df = pd.read_csv(metrics_path)
        if df.empty:
            continue
        last = df.iloc[-1]
        redundancy_df = df[df["round"] > 0] if "round" in df else df
        rows.append(
            {
                "run": run_dir.name,
                "dataset": dataset_cfg.get("name", ""),
                "availability_skew": fed_cfg.get("availability_skew", ""),
                "seed": config.get("seed", ""),
                "strategy": last["strategy"],
                "rounds": len(df),
                "final_accuracy": last["accuracy"],
                "best_accuracy": df["accuracy"].max(),
                "final_macro_f1": last["macro_f1"],
                "best_macro_f1": df["macro_f1"].max(),
                "final_query_gini": last["query_gini"],
                "best_query_gini": df["query_gini"].min(),
                "final_jain": last["jain_index"],
                "best_jain": df["jain_index"].max(),
                "final_query_rate_gini": last_metric(last, "query_rate_gini"),
                "best_query_rate_gini": best_metric(df, "query_rate_gini", "min"),
                "final_query_rate_jain": last_metric(last, "query_rate_jain"),
                "best_query_rate_jain": best_metric(df, "query_rate_jain", "max"),
                "final_low_avail_share": last["low_availability_query_share"],
                "best_low_avail_share": df["low_availability_query_share"].max(),
                "final_redundancy": last["mean_redundancy"],
                "best_redundancy": redundancy_df["mean_redundancy"].min(),
                "final_labeled_total": last["labeled_total"],
                "num_clients": fed_cfg.get("num_clients", ""),
                "participation_rate": fed_cfg.get("participation_rate", ""),
                "round_order": fed_cfg.get("round_order", ""),
                "warmup_rounds": fed_cfg.get("warmup_rounds", ""),
                "query_budget": al_cfg.get("query_budget", ""),
                "lambda_q": al_cfg.get("lambda_q", ""),
                "lambda_r": al_cfg.get("lambda_r", ""),
                "debt_target_mode": al_cfg.get("debt_target_mode", ""),
                "config": str(config_path) if config_path.exists() else "",
            }
        )

    summary = pd.DataFrame(rows)
    if summary.empty:
        print("No runs found.")
        return

    if args.latest_per_strategy:
        group_cols = ["dataset", "availability_skew", "seed", "strategy", "lambda_q", "lambda_r"]
        summary = summary.sort_values("run").groupby(group_cols, as_index=False).tail(1)

    if args.aggregate_seeds:
        metrics = [
            "final_accuracy",
            "best_accuracy",
            "final_macro_f1",
            "best_macro_f1",
            "final_query_gini",
            "best_query_gini",
            "final_jain",
            "best_jain",
            "final_query_rate_gini",
            "best_query_rate_gini",
            "final_query_rate_jain",
            "best_query_rate_jain",
            "final_low_avail_share",
            "final_redundancy",
            "best_redundancy",
        ]
        summary = (
            summary.groupby(["dataset", "availability_skew", "strategy", "lambda_q", "lambda_r"], as_index=False)
            .agg(
                seeds=("seed", "nunique"),
                **{f"{metric}_mean": (metric, "mean") for metric in metrics},
                **{f"{metric}_std": (metric, "std") for metric in metrics},
            )
            .sort_values(["dataset", "availability_skew", "strategy"])
        )
    else:
        summary = summary.sort_values(["dataset", "availability_skew", "seed", "strategy", "run"])

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(output, index=False)
        print(f"wrote {output}")

    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
