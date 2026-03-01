from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from decoding_amplifies_bias.cache import compute_generation_cache_key
from decoding_amplifies_bias.generation import GenerationRunner
from decoding_amplifies_bias.models import GeneratedText, GenerationConfig
from decoding_amplifies_bias.prompt_bank import (
    DEFAULT_PROMPT_BANK_PATH,
    load_prompt_bank,
    prompt_bank_digest,
)


class FakeGreedyBackend:
    def __init__(self) -> None:
        self.model_name = "fake-gpt2"
        self.device = "cpu"
        self.calls: list[tuple[str, int, int]] = []

    def generate(self, prompt_text: str, max_new_tokens: int, seed: int) -> GeneratedText:
        self.calls.append((prompt_text, max_new_tokens, seed))
        completion_text = f" completion-for-seed-{seed}"
        return GeneratedText(
            raw_text=f"{prompt_text}{completion_text}",
            completion_text=completion_text,
        )


def test_generation_cache_key_and_outputs_are_stable(tmp_path: Path) -> None:
    prompt_records = load_prompt_bank(DEFAULT_PROMPT_BANK_PATH)
    digest = prompt_bank_digest(prompt_records)
    config = GenerationConfig(
        prompt_bank_path=DEFAULT_PROMPT_BANK_PATH,
        output_dir=tmp_path,
        model_name="fake-gpt2",
        max_new_tokens=8,
        n_samples_per_prompt=2,
        seeds=(11, 29),
    )

    key_once = compute_generation_cache_key(config, digest)
    key_twice = compute_generation_cache_key(config, digest)
    assert key_once == key_twice

    runner = GenerationRunner()
    first_backend = FakeGreedyBackend()
    first_result = runner.run(config, backend=first_backend)

    assert first_result.cache_key == key_once
    assert first_result.from_cache is False
    assert len(first_backend.calls) == len(prompt_records) * len(config.seeds)

    frame = pd.read_parquet(first_result.generations_path)
    assert (
        int(frame.shape[0]) == len(prompt_records) * len(config.seeds) * config.n_samples_per_prompt
    )
    assert set(frame["sample_index"].tolist()) == {0, 1}

    manifest = json.loads(first_result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["cache_key"] == key_once
    assert manifest["decoding"] == {"strategy": "greedy", "do_sample": False}
    assert manifest["seeds"] == [11, 29]

    second_backend = FakeGreedyBackend()
    second_result = runner.run(config, backend=second_backend)
    assert second_result.cache_key == key_once
    assert second_result.from_cache is True
    assert len(second_backend.calls) == 0
