"""Tiny solver/checker self-check for local smoke testing."""

from .checker import check_game24
from .collect_accepted_ab import collect_accepted_ab
from .generate_data import split_tuple_pools
from .noisy_checker_eval import score_noise
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

    examples = [
        {
            "id": "ab_0",
            "task_type": "AB_symbolic_game24",
            "question": "q",
            "answer": "<answer>old</answer>",
            "metadata": {"numbers": [8, 3, 3, 8]},
        }
    ]
    generations = [
        {"id": "ab_0", "is_correct": True, "sample_index": 2, "checker": {"extracted_expression": expression}},
        {"id": "ab_0", "is_correct": False, "sample_index": 3},
    ]
    accepted, stats = collect_accepted_ab(examples, generations, max_per_task=1)
    assert len(accepted) == 1
    assert accepted[0]["answer"] == f"<answer>{expression}</answer>"
    assert stats["tasks_with_accepted_examples"] == 1

    clean = score_noise(generations, alpha=0.0, beta=0.0, seed=42)
    assert clean["accepted_correct"] == 1
    assert clean["accepted_wrong"] == 0
    print("self-check passed")


if __name__ == "__main__":
    main()
