"""Experiments for association rules generated from Apriori frequent itemsets.

This module loads baskets once, fixes the minimum support value, runs rule
search for several minimum confidence values, measures runtime with
time.perf_counter(), saves CSV reports, builds charts, and prints a summary
table with several interesting rules.
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib_cache").resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from apriori import FrequentItemset, apriori, format_itemset, load_transactions
from rules import (
    AssociationRule,
    find_association_rules,
    get_total_items,
    sort_rules_by_confidence_and_support,
)


MIN_SUPPORT = 0.01
CONFIDENCE_VALUES = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]


def run_single_rules_experiment(
    frequent_itemsets: list[FrequentItemset],
    min_confidence: float,
) -> dict[str, float | int]:
    """Run rule search once and collect experiment statistics.

    Parameters:
        frequent_itemsets: Frequent itemsets found with the fixed support.
        min_confidence: Minimum confidence threshold for the current run.

    Returns:
        Dictionary with min_support, min_confidence, execution_time, and
        total_rules.
    """
    start_time = time.perf_counter()
    rules = find_association_rules(frequent_itemsets, min_confidence)
    end_time = time.perf_counter()

    return {
        "min_support": MIN_SUPPORT,
        "min_confidence": min_confidence,
        "execution_time": end_time - start_time,
        "total_rules": len(rules),
    }


def run_rules_experiments(
    frequent_itemsets: list[FrequentItemset],
    confidence_values: list[float],
) -> list[dict[str, float | int]]:
    """Run rule search for all confidence values.

    Parameters:
        frequent_itemsets: Frequent itemsets found with the fixed support.
        confidence_values: List of minimum confidence thresholds.

    Returns:
        List of dictionaries with experiment results.
    """
    results: list[dict[str, float | int]] = []

    for min_confidence in confidence_values:
        print(f"Запуск поиска правил для confidence {min_confidence * 100:.0f}%...")
        result = run_single_rules_experiment(frequent_itemsets, min_confidence)
        results.append(result)

    return results


def save_experiment_results(
    results: list[dict[str, float | int]],
    output_path: Path,
) -> None:
    """Save rule experiment results to a CSV file.

    Parameters:
        results: List of dictionaries with experiment results.
        output_path: Path where the CSV file must be saved.

    Returns:
        None.
    """
    fieldnames = ["min_support", "min_confidence", "execution_time", "total_rules"]

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def plot_rules_runtime(
    results: list[dict[str, float | int]],
    output_path: Path,
) -> None:
    """Build and save the rule search runtime chart.

    Parameters:
        results: List of dictionaries with experiment results.
        output_path: Path where the chart image must be saved.

    Returns:
        None.
    """
    confidence_percent = [float(result["min_confidence"]) * 100 for result in results]
    execution_time = [float(result["execution_time"]) for result in results]

    plt.figure(figsize=(8, 5))
    plt.plot(confidence_percent, execution_time, marker="o")
    plt.title("Зависимость времени поиска ассоциативных правил от минимальной достоверности")
    plt.xlabel("Минимальная достоверность, %")
    plt.ylabel("Время выполнения, секунды")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_rules_count(
    results: list[dict[str, float | int]],
    output_path: Path,
) -> None:
    """Build and save the chart with association rule counts.

    Parameters:
        results: List of dictionaries with experiment results.
        output_path: Path where the chart image must be saved.

    Returns:
        None.
    """
    confidence_percent = [float(result["min_confidence"]) * 100 for result in results]
    total_rules = [int(result["total_rules"]) for result in results]

    plt.figure(figsize=(8, 5))
    plt.plot(confidence_percent, total_rules, marker="o")
    plt.title("Количество найденных ассоциативных правил при изменении минимальной достоверности")
    plt.xlabel("Минимальная достоверность, %")
    plt.ylabel("Количество правил")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def filter_rules_with_max_items(
    rules: list[AssociationRule],
    max_items: int,
) -> list[AssociationRule]:
    """Keep rules whose total number of objects is not greater than max_items.

    Parameters:
        rules: List of association rules.
        max_items: Maximum allowed sum of antecedent and consequent lengths.

    Returns:
        Filtered list of association rules.
    """
    filtered_rules: list[AssociationRule] = []

    for rule in rules:
        if get_total_items(rule) <= max_items:
            filtered_rules.append(rule)

    return filtered_rules


def save_rules_to_csv(
    rules: list[AssociationRule],
    output_path: Path,
) -> None:
    """Save association rules to a CSV file.

    Parameters:
        rules: List of association rules.
        output_path: Path where the CSV file must be saved.

    Returns:
        None.
    """
    fieldnames = ["antecedent", "consequent", "support", "confidence", "total_items"]

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for rule in rules:
            writer.writerow(
                {
                    "antecedent": format_itemset(rule.antecedent),
                    "consequent": format_itemset(rule.consequent),
                    "support": rule.support,
                    "confidence": rule.confidence,
                    "total_items": get_total_items(rule),
                }
            )


def print_summary_table(results: list[dict[str, float | int]]) -> None:
    """Print a summary table after all rule experiments.

    Parameters:
        results: List of dictionaries with experiment results.

    Returns:
        None.
    """
    print("\nИтоговая таблица результатов")
    print(f"{'confidence':>12} {'time':>12} {'total_rules':>14}")
    print("-" * 40)

    for result in results:
        print(
            f"{float(result['min_confidence']) * 100:>11.0f}% "
            f"{float(result['execution_time']):>12.4f} "
            f"{int(result['total_rules']):>14}"
        )


def print_interesting_rules(rules: list[AssociationRule], limit: int = 5) -> None:
    """Print several interesting rules and explain their meaning.

    Parameters:
        rules: Sorted list of association rules.
        limit: Maximum number of rules to print.

    Returns:
        None.
    """
    if not rules:
        print("\nПравила для объяснения не найдены.")
        return

    print("\nНесколько интересных правил")
    print("-" * 40)

    for rule in rules[:limit]:
        antecedent_text = format_itemset(rule.antecedent)
        consequent_text = format_itemset(rule.consequent)
        print(
            f"{antecedent_text} -> {consequent_text}, "
            f"support = {rule.support * 100:.2f}%, "
            f"confidence = {rule.confidence * 100:.2f}%"
        )
        print(
            f"Это означает: в {rule.confidence * 100:.2f}% транзакций, "
            f"содержащих {antecedent_text}, также встречается {consequent_text}."
        )


def prepare_rules_max_7_items(
    frequent_itemsets: list[FrequentItemset],
) -> list[AssociationRule]:
    """Prepare sorted rules with no more than seven objects.

    Parameters:
        frequent_itemsets: Frequent itemsets found with the fixed support.

    Returns:
        Rules filtered by total item count and sorted by confidence and support.
    """
    rules = find_association_rules(frequent_itemsets, min(CONFIDENCE_VALUES))
    rules = filter_rules_with_max_items(rules, 7)
    return sort_rules_by_confidence_and_support(rules)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for association rule experiments.

    Parameters:
        None.

    Returns:
        Namespace with dataset path.
    """
    parser = argparse.ArgumentParser(
        description="Эксперименты с ассоциативными правилами на основе Apriori."
    )
    parser.add_argument(
        "--dataset",
        default="baskets.csv",
        help="Путь к CSV-файлу с корзинами. По умолчанию используется baskets.csv.",
    )

    return parser.parse_args()


def main() -> None:
    """Start rule experiments and save all output files.

    Parameters:
        None.

    Returns:
        None.
    """
    try:
        arguments = parse_arguments()
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)

        transactions = load_transactions(arguments.dataset)
        frequent_itemsets = apriori(transactions, MIN_SUPPORT)

        results = run_rules_experiments(frequent_itemsets, CONFIDENCE_VALUES)
        save_experiment_results(results, results_dir / "rules_experiment_results.csv")
        plot_rules_runtime(results, results_dir / "rules_runtime.png")
        plot_rules_count(results, results_dir / "rules_count.png")

        max_7_rules = prepare_rules_max_7_items(frequent_itemsets)
        save_rules_to_csv(max_7_rules, results_dir / "rules_max_7_items.csv")

        print_summary_table(results)
        print_interesting_rules(max_7_rules)

        print("\nФайлы сохранены в каталоге results/")
    except (FileNotFoundError, ValueError) as error:
        print(f"Ошибка: {error}")


if __name__ == "__main__":
    main()
