"""Simple plant recognition utilities for trees and flowers."""

from __future__ import annotations

from typing import Iterable

TREE_KEYWORDS = {
    "tree",
    "oak",
    "pine",
    "maple",
    "birch",
    "cedar",
    "willow",
    "palm",
    "spruce",
}

FLOWER_KEYWORDS = {
    "flower",
    "rose",
    "tulip",
    "daisy",
    "lily",
    "orchid",
    "sunflower",
    "hibiscus",
    "marigold",
}


def recognize_plant(name: str) -> str:
    """Recognize whether a plant name is a tree, flower, or unknown."""
    normalized = name.strip().lower()
    if not normalized:
        return "unknown"

    if any(keyword in normalized for keyword in TREE_KEYWORDS):
        return "tree"

    if any(keyword in normalized for keyword in FLOWER_KEYWORDS):
        return "flower"

    return "unknown"


def recognize_plants(names: Iterable[str]) -> dict[str, str]:
    """Recognize categories for multiple plant names."""
    return {name: recognize_plant(name) for name in names}


if __name__ == "__main__":
    import sys

    if len(sys.argv) <= 1:
        print("Usage: python plants_recognition.py <plant_name> [<plant_name> ...]")
        sys.exit(1)

    results = recognize_plants(sys.argv[1:])
    for plant, category in results.items():
        print(f"{plant}: {category}")
