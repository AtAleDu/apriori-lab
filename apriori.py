from __future__ import annotations

import argparse
import csv
from itertools import combinations
from pathlib import Path


Transaction = frozenset[str]
Itemset = frozenset[str]
FrequentItemset = tuple[Itemset, float]


def load_transactions(file_path: str, encoding: str | None = None) -> list[Transaction]:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {file_path}")

    encodings = [encoding] if encoding else ["utf-8-sig", "cp1251"]
    last_error: UnicodeDecodeError | None = None

    for current_encoding in encodings:
        try:
            transactions = _read_transactions_with_encoding(path, current_encoding)
            if not transactions:
                raise ValueError("Набор данных пуст.")
            return transactions
        except UnicodeDecodeError as error:
            last_error = error

    raise ValueError(
        "Не удалось прочитать файл. Попробуйте указать кодировку явно."
    ) from last_error


def _read_transactions_with_encoding(path: Path, encoding: str) -> list[Transaction]:
    transactions: list[Transaction] = []

    with path.open("r", encoding=encoding, newline="") as file:
        reader = csv.reader(file)

        for row in reader:
            # Strip spaces and ignore empty cells, because real CSV files can
            # contain extra commas or spaces after item names.
            items = [item.strip() for item in row if item.strip()]
            if items:
                transactions.append(frozenset(items))

    return transactions


def validate_min_support(min_support: float) -> None:
    if min_support <= 0 or min_support > 1:
        raise ValueError("Минимальная поддержка должна быть больше 0 и не больше 1.")


def calculate_support(itemset: Itemset, transactions: list[Transaction]) -> float:
    containing_count = 0

    for transaction in transactions:
        if itemset.issubset(transaction):
            containing_count += 1

    return containing_count / len(transactions)


def find_frequent_single_items(
    transactions: list[Transaction], min_support: float
) -> dict[Itemset, float]:
    item_counts: dict[str, int] = {}

    for transaction in transactions:
        for item in transaction:
            item_counts[item] = item_counts.get(item, 0) + 1

    transaction_count = len(transactions)
    frequent_itemsets: dict[Itemset, float] = {}

    for item, count in item_counts.items():
        support = count / transaction_count
        if support >= min_support:
            frequent_itemsets[frozenset([item])] = support

    return frequent_itemsets


def has_frequent_subsets(candidate: Itemset, previous_frequent: set[Itemset]) -> bool:
    subset_length = len(candidate) - 1

    for subset in combinations(candidate, subset_length):
        if frozenset(subset) not in previous_frequent:
            return False

    return True


def generate_candidates(previous_frequent: set[Itemset], itemset_length: int) -> set[Itemset]:
    candidates: set[Itemset] = set()
    previous_list = sorted(previous_frequent, key=lambda itemset: sorted(itemset))

    for first_index in range(len(previous_list)):
        for second_index in range(first_index + 1, len(previous_list)):
            union_candidate = previous_list[first_index] | previous_list[second_index]

            if len(union_candidate) != itemset_length:
                continue

            if has_frequent_subsets(union_candidate, previous_frequent):
                candidates.add(union_candidate)

    return candidates


def filter_frequent_candidates(
    candidates: set[Itemset],
    transactions: list[Transaction],
    min_support: float,
) -> dict[Itemset, float]:
    frequent_itemsets: dict[Itemset, float] = {}

    for candidate in candidates:
        support = calculate_support(candidate, transactions)
        if support >= min_support:
            frequent_itemsets[candidate] = support

    return frequent_itemsets


def apriori(transactions: list[Transaction], min_support: float) -> list[FrequentItemset]:
    validate_min_support(min_support)
    if not transactions:
        raise ValueError("Набор данных пуст.")

    all_frequent_itemsets: list[FrequentItemset] = []

    current_frequent = find_frequent_single_items(transactions, min_support)
    all_frequent_itemsets.extend(current_frequent.items())

    itemset_length = 2

    while current_frequent:
        previous_frequent_set = set(current_frequent.keys())
        candidates = generate_candidates(previous_frequent_set, itemset_length)

        if not candidates:
            break

        current_frequent = filter_frequent_candidates(
            candidates,
            transactions,
            min_support,
        )
        all_frequent_itemsets.extend(current_frequent.items())
        itemset_length += 1

    return all_frequent_itemsets


def sort_results(
    frequent_itemsets: list[FrequentItemset], sort_method: str
) -> list[FrequentItemset]:
    if sort_method == "support":
        return sorted(
            frequent_itemsets,
            key=lambda result: (-result[1], len(result[0]), sorted(result[0])),
        )

    if sort_method == "lex":
        return sorted(
            frequent_itemsets,
            key=lambda result: (sorted(result[0]), len(result[0])),
        )

    raise ValueError('Неизвестный способ сортировки. Используйте "support" или "lex".')


def format_itemset(itemset: Itemset) -> str:
    return ", ".join(sorted(itemset))


def print_results(frequent_itemsets: list[FrequentItemset]) -> None:
    if not frequent_itemsets:
        print("Частые наборы не найдены.")
        return

    print(f"{'Набор объектов':<60} {'Длина':>7} {'Поддержка':>12} {'%':>10}")
    print("-" * 95)

    for itemset, support in frequent_itemsets:
        support_percent = support * 100
        print(
            f"{format_itemset(itemset):<60} "
            f"{len(itemset):>7} "
            f"{support:>12.4f} "
            f"{support_percent:>9.2f}%"
        )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Поиск частых наборов объектов с помощью алгоритма Apriori."
    )
    parser.add_argument("dataset", help="Путь к файлу baskets.csv.")
    parser.add_argument(
        "min_support",
        type=float,
        help="Минимальная поддержка, например 0.05 для 5%%.",
    )
    parser.add_argument(
        "sort_method",
        choices=["support", "lex"],
        help='Способ сортировки результата: "support" или "lex".',
    )

    return parser.parse_args()


def main() -> None:
    try:
        arguments = parse_arguments()
        transactions = load_transactions(arguments.dataset)
        frequent_itemsets = apriori(transactions, arguments.min_support)
        sorted_itemsets = sort_results(frequent_itemsets, arguments.sort_method)
        print_results(sorted_itemsets)
    except (FileNotFoundError, ValueError) as error:
        print(f"Ошибка: {error}")


if __name__ == "__main__":
    main()
