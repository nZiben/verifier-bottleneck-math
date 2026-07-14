"""Tiny solver/checker self-check for local smoke testing."""

from .checker import check_game24
from .generate_data import split_tuple_pools
from .solver import enumerate_solvable_tuples, solve_game24
from .symbols import NUMBER_TO_SYMBOL


def main():
    numbers = [8, 3, 3, 8]
    expression = solve_game24(numbers)
    assert expression is not None
    assert check_game24(f"<answer>{expression}</answer>", numbers=numbers)["is_correct"]
    assert not check_game24("<answer>8 + 3 + 3 + 8</answer>", numbers=numbers)["is_correct"]

    symbols = [NUMBER_TO_SYMBOL[number] for number in numbers]
    assert check_game24(f"<answer>{expression}</answer>", numbers=numbers, symbols=symbols)["is_correct"]
    assert check_game24("<answer>nari / (zup - nari / zup)</answer>", symbols=symbols)["is_correct"]

    solvable = enumerate_solvable_tuples()
    train, _, test = split_tuple_pools(solvable)
    assert set(train).isdisjoint(test)
    print("self-check passed")


if __name__ == "__main__":
    main()
