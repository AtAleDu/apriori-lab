"""Association rule search based on frequent itemsets found by Apriori.

This module uses the existing functions load_transactions() and apriori() from
apriori.py. It generates rules of the form "antecedent -> consequent",
calculates support and confidence, filters rules by minimum confidence, sorts
them, and provides a command-line interface for a separate rule search.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import combinations

from apriori import Itemset, FrequentItemset, apriori, format_itemset, load_transactions


@dataclass(frozen=True)
class AssociationRule:
    """Store one association rule and its calculated measures.

    Attributes:
        antecedent: Left part of the rule.
        consequent: Right part of the rule.
        support: Support of the full itemset antecedent union consequent.
        confidence: Confidence of the rule.
    """

    antecedent: Itemset
    consequent: Itemset
    support: float
    confidence: float


def validate_min_confidence(min_confidence: float) -> None:
    """Check that the minimum confidence value is correct.

    Parameters:
        min_confidence: Minimum confidence threshold as a number from 0 to 1.

    Returns:
        None.

    Raises:
        ValueError: If the confidence value is outside the allowed interval.
    """
    if min_confidence <= 0 or min_confidence > 1:
        raise ValueError("Минимальная достоверность должна быть больше 0 и не больше 1.")


def build_support_map(frequent_itemsets: list[FrequentItemset]) -> dict[Itemset, float]:
    """Build a dictionary for quick support lookup.

    Parameters:
        frequent_itemsets: Frequent itemsets returned by apriori().

    Returns:
        Dictionary where the key is an itemset and the value is its support.
    """
    support_map: dict[Itemset, float] = {}

    for itemset, support in frequent_itemsets:
        support_map[itemset] = support

    return support_map


def get_itemset_support(itemset: Itemset, support_map: dict[Itemset, float]) -> float:
    """Get support of one frequent itemset.

    Parameters:
        itemset: Frequent itemset whose support is needed.
        support_map: Dictionary with supports of frequent itemsets.

    Returns:
        Support value of the itemset.

    Raises:
        ValueError: If the itemset is not present in the support map.
    """
    if itemset not in support_map:
        raise ValueError("Поддержка набора не найдена среди частых наборов.")

    return support_map[itemset]


def calculate_confidence(
    full_itemset_support: float,
    antecedent_support: float,
) -> float:
    """Calculate confidence for an association rule.

    Parameters:
        full_itemset_support: Support of antecedent union consequent.
        antecedent_support: Support of the rule antecedent.

    Returns:
        Confidence value calculated as support(A union B) divided by support(A).
    """
    return full_itemset_support / antecedent_support


def generate_rules_from_itemset(
    itemset: Itemset,
    support_map: dict[Itemset, float],
) -> list[AssociationRule]:
    """Generate all possible association rules from one frequent itemset.

    Parameters:
        itemset: Frequent itemset with length at least 2.
        support_map: Dictionary with supports of frequent itemsets.

    Returns:
        List of rules generated from the itemset.
    """
    rules: list[AssociationRule] = []

    if len(itemset) < 2:
        return rules

    full_itemset_support = get_itemset_support(itemset, support_map)

    # Every non-empty proper subset can be the left part of a rule.
    for antecedent_length in range(1, len(itemset)):
        for antecedent_tuple in combinations(itemset, antecedent_length):
            antecedent = frozenset(antecedent_tuple)
            consequent = itemset - antecedent

            antecedent_support = get_itemset_support(antecedent, support_map)
            confidence = calculate_confidence(full_itemset_support, antecedent_support)

            rules.append(
                AssociationRule(
                    antecedent=antecedent,
                    consequent=consequent,
                    support=full_itemset_support,
                    confidence=confidence,
                )
            )

    return rules


def generate_all_rules(frequent_itemsets: list[FrequentItemset]) -> list[AssociationRule]:
    """Generate association rules from all frequent itemsets.

    Parameters:
        frequent_itemsets: Frequent itemsets returned by apriori().

    Returns:
        List of all rules generated from itemsets of length 2 and more.
    """
    support_map = build_support_map(frequent_itemsets)
    rules: list[AssociationRule] = []

    for itemset, _support in frequent_itemsets:
        rules.extend(generate_rules_from_itemset(itemset, support_map))

    return rules


def filter_rules_by_confidence(
    rules: list[AssociationRule],
    min_confidence: float,
) -> list[AssociationRule]:
    """Keep only rules that satisfy the minimum confidence threshold.

    Parameters:
        rules: List of generated association rules.
        min_confidence: Minimum confidence threshold.

    Returns:
        Filtered list of association rules.
    """
    validate_min_confidence(min_confidence)
    filtered_rules: list[AssociationRule] = []

    for rule in rules:
        if rule.confidence >= min_confidence:
            filtered_rules.append(rule)

    return filtered_rules


def find_association_rules(
    frequent_itemsets: list[FrequentItemset],
    min_confidence: float,
) -> list[AssociationRule]:
    """Generate and filter association rules.

    Parameters:
        frequent_itemsets: Frequent itemsets returned by apriori().
        min_confidence: Minimum confidence threshold.

    Returns:
        List of rules whose confidence is not lower than min_confidence.
    """
    all_rules = generate_all_rules(frequent_itemsets)
    return filter_rules_by_confidence(all_rules, min_confidence)


def sort_rules(
    rules: list[AssociationRule],
    sort_method: str,
) -> list[AssociationRule]:
    """Sort association rules before output.

    Parameters:
        rules: List of association rules.
        sort_method: Sorting method. Use "support" for descending support or
            "lex" for lexicographic sorting.

    Returns:
        Sorted list of association rules.

    Raises:
        ValueError: If the sorting method is unknown.
    """
    if sort_method == "support":
        return sorted(
            rules,
            key=lambda rule: (
                -rule.support,
                -rule.confidence,
                sorted(rule.antecedent),
                sorted(rule.consequent),
            ),
        )

    if sort_method == "lex":
        return sorted(
            rules,
            key=lambda rule: (sorted(rule.antecedent), sorted(rule.consequent)),
        )

    raise ValueError('Неизвестный способ сортировки. Используйте "support" или "lex".')


def sort_rules_by_confidence_and_support(
    rules: list[AssociationRule],
) -> list[AssociationRule]:
    """Sort rules by descending confidence and then descending support.

    Parameters:
        rules: List of association rules.

    Returns:
        Sorted list of association rules.
    """
    return sorted(
        rules,
        key=lambda rule: (
            -rule.confidence,
            -rule.support,
            sorted(rule.antecedent),
            sorted(rule.consequent),
        ),
    )


def get_total_items(rule: AssociationRule) -> int:
    """Count total number of objects in a rule.

    Parameters:
        rule: Association rule.

    Returns:
        Sum of antecedent length and consequent length.
    """
    return len(rule.antecedent) + len(rule.consequent)


def print_rules(rules: list[AssociationRule]) -> None:
    """Print association rules as a readable table.

    Parameters:
        rules: List of association rules.

    Returns:
        None.
    """
    if not rules:
        print("Ассоциативные правила не найдены.")
        return

    print(
        f"{'Правило':<80} {'Объектов':>8} "
        f"{'Поддержка':>12} {'%':>10} {'Confidence':>12} {'%':>10}"
    )
    print("-" * 138)

    for rule in rules:
        rule_text = f"{format_itemset(rule.antecedent)} -> {format_itemset(rule.consequent)}"
        print(
            f"{rule_text:<80} "
            f"{get_total_items(rule):>8} "
            f"{rule.support:>12.4f} "
            f"{rule.support * 100:>9.2f}% "
            f"{rule.confidence:>12.4f} "
            f"{rule.confidence * 100:>9.2f}%"
        )


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for association rule search.

    Parameters:
        None.

    Returns:
        Namespace with dataset path, minimum support, minimum confidence, and
        sorting method.
    """
    parser = argparse.ArgumentParser(
        description="Поиск ассоциативных правил на основе алгоритма Apriori."
    )
    parser.add_argument("dataset", help="Путь к файлу baskets.csv.")
    parser.add_argument(
        "min_support",
        type=float,
        help="Минимальная поддержка, например 0.01 для 1%%.",
    )
    parser.add_argument(
        "min_confidence",
        type=float,
        help="Минимальная достоверность, например 0.30 для 30%%.",
    )
    parser.add_argument(
        "sort_method",
        choices=["support", "lex"],
        help='Способ сортировки результата: "support" или "lex".',
    )

    return parser.parse_args()


def main() -> None:
    """Start association rule search from the command line.

    Parameters:
        None.

    Returns:
        None.
    """
    try:
        arguments = parse_arguments()
        validate_min_confidence(arguments.min_confidence)

        transactions = load_transactions(arguments.dataset)
        frequent_itemsets = apriori(transactions, arguments.min_support)
        rules = find_association_rules(frequent_itemsets, arguments.min_confidence)
        sorted_rules = sort_rules(rules, arguments.sort_method)

        print_rules(sorted_rules)
        print(f"\nВсего правил: {len(sorted_rules)}")
    except (FileNotFoundError, ValueError) as error:
        print(f"Ошибка: {error}")


if __name__ == "__main__":
    main()
