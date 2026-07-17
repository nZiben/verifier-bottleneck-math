from __future__ import annotations

import math
from pathlib import Path

import pytest

from verifier_bottleneck.data.countdown_composition import (
    COUNTDOWN_VALUES,
    make_countdown_composition_split,
    make_symbol_codebook,
    verify_postfix,
)
from verifier_bottleneck.evaluation.countdown import PostfixState, sample_index_at_temperature
from verifier_bottleneck.experiments.countdown_composition import (
    describe_countdown_composition,
    load_countdown_composition_config,
)
from verifier_bottleneck.experiments.countdown_protocol import (
    distribute_replay_examples,
    replay_example_count,
)
from verifier_bottleneck.experiments.qwen_countdown_composition import (
    describe_qwen_countdown_composition,
    load_qwen_countdown_composition_config,
    numeric_answer_token_sequences,
)


class _FakeQwenTokenizer:
    eos_token_id = 99

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert not add_special_tokens
        return {
            " 1": [10],
            " 25": [20, 25],
            " 50": [20, 50],
            " 100": [10, 0, 0],
        }[text]


def test_qwen_numeric_answers_allow_multiple_tokens() -> None:
    sequences = numeric_answer_token_sequences(_FakeQwenTokenizer(), [1, 25, 50, 100])

    assert sequences == {
        1: (10,),
        25: (20, 25),
        50: (20, 50),
        100: (10, 0, 0),
    }


def test_replay_allocation_is_ten_percent_of_effective_batch() -> None:
    assert replay_example_count(256, 0.1) == 26
    counts = distribute_replay_examples(
        micro_batch_size=8,
        accumulation_steps=32,
        replay_fraction=0.1,
    )

    assert len(counts) == 32
    assert sum(counts) == 26
    assert set(counts) <= {0, 1}


def test_temperature_sampling_is_greedy_at_zero_and_uses_softmax_above_zero() -> None:
    logits = [0.0, math.log(3.0)]

    assert sample_index_at_temperature(logits, temperature=0.0, random_value=0.99) == 1
    assert sample_index_at_temperature(logits, temperature=1.0, random_value=0.24) == 0
    assert sample_index_at_temperature(logits, temperature=1.0, random_value=0.25) == 1
    assert sample_index_at_temperature(logits, temperature=0.5, random_value=0.09) == 0
    assert sample_index_at_temperature(logits, temperature=0.5, random_value=0.10) == 1


def test_postfix_state_enforces_the_shared_countdown_grammar() -> None:
    state = PostfixState()

    assert state.legal_actions(3) == ["P0", "P1", "P2"]
    state.apply("P0")
    state.apply("P1")
    assert state.legal_actions(3) == ["P2", "+", "-", "*", "/"]
    state.apply("+")
    assert state.legal_actions(3) == ["P2", "<EOS>"]
    state.apply("<EOS>")
    assert state.finished
    assert state.tokens == ["P0", "P1", "+"]


def test_postfix_state_rejects_illegal_transitions() -> None:
    state = PostfixState()

    for action in ("+", "<EOS>"):
        with pytest.raises(ValueError):
            state.apply(action)

    state.apply("P0")
    with pytest.raises(ValueError):
        state.apply("P0")


def test_codebook_is_seeded_bijective_and_non_monotonic() -> None:
    first = make_symbol_codebook(seed=20260717)
    second = make_symbol_codebook(seed=20260717)

    assert first == second
    assert {example.value for example in first} == set(COUNTDOWN_VALUES)
    assert [example.value for example in first] != list(COUNTDOWN_VALUES)
    assert len({example.symbol for example in first}) == len(COUNTDOWN_VALUES)


def test_composition_split_is_disjoint_verified_and_paired() -> None:
    split = make_countdown_composition_split(
        train_examples=200,
        validation_examples=30,
        test_examples=20,
        minimum_target=101,
        maximum_target=999,
        seed=42,
        symbol_seed=20260717,
    )
    train = {puzzle.key() for puzzle in split.numeric_train}
    validation = {puzzle.key() for puzzle in split.numeric_validation}
    test = {puzzle.key() for puzzle in split.numeric_test}

    assert not train & validation
    assert not train & test
    assert not validation & test
    assert split.symbolic_test == split.numeric_test
    assert all(
        verify_postfix(puzzle, puzzle.solution)
        for puzzle in (
            *split.numeric_train,
            *split.numeric_validation,
            *split.numeric_test,
        )
    )


def test_scratch_and_qwen_configs_share_the_composition_protocol() -> None:
    scratch = load_countdown_composition_config(
        Path("configs/arithmetic/countdown_symbolic_composition_scratch_single_seed.yaml")
    )
    qwen = load_qwen_countdown_composition_config(
        Path("configs/arithmetic/countdown_symbolic_composition_qwen_single_seed.yaml")
    )
    scratch_description = describe_countdown_composition(scratch)
    qwen_description = describe_qwen_countdown_composition(qwen)

    assert scratch.seed == qwen.seed
    assert scratch.task.countdown_train_examples == qwen.task.countdown_train_examples
    assert scratch.task.countdown_validation_examples == qwen.task.countdown_validation_examples
    assert scratch.task.countdown_minimum_target == qwen.task.countdown_minimum_target
    assert scratch.task.countdown_maximum_target == qwen.task.countdown_maximum_target
    assert scratch.task.symbol_seed == qwen.task.symbol_seed
    assert scratch.evaluation.temperatures == qwen.evaluation.temperatures
    assert scratch.evaluation.pass_k == qwen.evaluation.pass_k
    assert scratch.evaluation.seed == qwen.evaluation.seed
    assert scratch.training.skill_a_replay_fraction == 0.1
    assert qwen.training.skill_a_replay_fraction == 0.1
    assert scratch.model.layers == 4
    assert scratch.model.d_model == 256
    assert scratch.model.heads == 8
    assert scratch.model.d_ff == 1024
    assert scratch.model.block_size == 32
    assert scratch.training.steps == 4000
    assert scratch_description["estimated_parameter_count"] == 3_431_424
    assert scratch_description["maximum_generated_proposals"] == 458_752
    assert qwen_description["maximum_generated_proposals"] == 229_376
    assert qwen_description["effective_batch_size"] == 128
    assert qwen_description["optimizer_steps"] == 600
    assert scratch_description["unseen_test"] == "A+B_symbolic_countdown"
    assert qwen_description["downloads_models"] is True
    assert scratch_description["downloads_models"] is False
