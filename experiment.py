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

from apriori import FrequentItemset, apriori, load_transactions


DEFAULT_SUPPORT_VALUES = [0.01, 0.03, 0.05, 0.10, 0.15]


def count_itemsets_by_length(frequent_itemsets: list[FrequentItemset]) -> dict[int, int]:
    counts: dict[int, int] = {}

    for itemset, _support in frequent_itemsets:
        itemset_length = len(itemset)
        counts[itemset_length] = counts.get(itemset_length, 0) + 1

    return counts


def run_single_experiment(
    transactions: list[frozenset[str]], min_support: float
) -> dict[str, float | int]:
    start_time = time.perf_counter()
    frequent_itemsets = apriori(transactions, min_support)
    end_time = time.perf_counter()

    counts_by_length = count_itemsets_by_length(frequent_itemsets)

    result: dict[str, float | int] = {
        "min_support": min_support,
        "execution_time": end_time - start_time,
        "total_itemsets": len(frequent_itemsets),
    }

    for itemset_length, count in sorted(counts_by_length.items()):
        result[f"length_{itemset_length}"] = count

    return result


def run_experiments(
    dataset_path: str,
    support_values: list[float],
) -> list[dict[str, float | int]]:
    transactions = load_transactions(dataset_path)
    results: list[dict[str, float | int]] = []

    for min_support in support_values:
        print(f"Запуск эксперимента для поддержки {min_support * 100:.0f}%...")
        result = run_single_experiment(transactions, min_support)
        results.append(result)

    return results


def get_length_columns(results: list[dict[str, float | int]]) -> list[str]:
    columns: set[str] = set()

    for result in results:
        for key in result:
            if key.startswith("length_"):
                columns.add(key)

    return sorted(columns, key=lambda column: int(column.split("_")[1]))


def save_results_to_csv(
    results: list[dict[str, float | int]],
    output_path: Path,
) -> None:
    length_columns = get_length_columns(results)
    fieldnames = ["min_support", "execution_time", "total_itemsets"] + length_columns

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for result in results:
            row = {fieldname: result.get(fieldname, 0) for fieldname in fieldnames}
            writer.writerow(row)


def plot_runtime(results: list[dict[str, float | int]], output_path: Path) -> None:
    support_percent = [float(result["min_support"]) * 100 for result in results]
    execution_time = [float(result["execution_time"]) for result in results]

    plt.figure(figsize=(8, 5))
    plt.plot(support_percent, execution_time, marker="o")
    plt.title("Зависимость времени работы алгоритма Apriori от минимального порога поддержки")
    plt.xlabel("Минимальная поддержка, %")
    plt.ylabel("Время выполнения, секунды")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_itemset_counts(
    results: list[dict[str, float | int]],
    output_path: Path,
) -> None:
    support_percent = [float(result["min_support"]) * 100 for result in results]
    length_columns = get_length_columns(results)

    plt.figure(figsize=(8, 5))

    for column in length_columns:
        counts = [int(result.get(column, 0)) for result in results]
        itemset_length = column.split("_")[1]
        plt.plot(support_percent, counts, marker="o", label=f"Длина {itemset_length}")

    plt.title("Количество частых наборов различной длины при изменении минимальной поддержки")
    plt.xlabel("Минимальная поддержка, %")
    plt.ylabel("Количество частых наборов")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def print_summary_table(results: list[dict[str, float | int]]) -> None:
    length_columns = get_length_columns(results)
    headers = ["min_support", "time", "total"] + length_columns
    widths = [12, 12, 10] + [10 for _column in length_columns]

    header_line = " ".join(
        f"{header:>{width}}" for header, width in zip(headers, widths)
    )
    print("\nИтоговая таблица результатов")
    print(header_line)
    print("-" * len(header_line))

    for result in results:
        values: list[str] = [
            f"{float(result['min_support']) * 100:.0f}%",
            f"{float(result['execution_time']):.4f}",
            str(int(result["total_itemsets"])),
        ]

        for column in length_columns:
            values.append(str(int(result.get(column, 0))))

        print(" ".join(f"{value:>{width}}" for value, width in zip(values, widths)))


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Эксперименты с алгоритмом Apriori для лабораторной работы."
    )
    parser.add_argument(
        "--dataset",
        default="baskets.csv",
        help="Путь к CSV-файлу с корзинами. По умолчанию используется baskets.csv.",
    )

    return parser.parse_args()


def main() -> None:
    try:
        arguments = parse_arguments()
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)

        results = run_experiments(arguments.dataset, DEFAULT_SUPPORT_VALUES)

        save_results_to_csv(results, results_dir / "experiment_results.csv")
        plot_runtime(results, results_dir / "runtime.png")
        plot_itemset_counts(results, results_dir / "frequent_itemsets.png")
        print_summary_table(results)

        print("\nФайлы сохранены в каталоге results/")
    except (FileNotFoundError, ValueError) as error:
        print(f"Ошибка: {error}")


if __name__ == "__main__":
    main()
