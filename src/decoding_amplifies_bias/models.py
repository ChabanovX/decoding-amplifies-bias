from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from decoding_amplifies_bias.paths import DEFAULT_OUTPUT_DIR, DEFAULT_PROMPT_BANK_PATH


@dataclass(frozen=True, slots=True)
class PromptRecord:
    prompt_id: str
    template_id: str
    prompt_type: str
    demographic: str
    prompt_text: str

    def to_dict(self) -> dict[str, str]:
        return {
            "prompt_id": self.prompt_id,
            "template_id": self.template_id,
            "prompt_type": self.prompt_type,
            "demographic": self.demographic,
            "prompt_text": self.prompt_text,
        }


@dataclass(frozen=True, slots=True)
class GreedyDecodingConfig:
    strategy: Literal["greedy"] = "greedy"
    do_sample: bool = False

    def to_dict(self) -> dict[str, str | bool]:
        return {
            "strategy": self.strategy,
            "do_sample": self.do_sample,
        }


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    prompt_bank_path: Path = field(default_factory=lambda: DEFAULT_PROMPT_BANK_PATH)
    output_dir: Path = field(default_factory=lambda: DEFAULT_OUTPUT_DIR)
    model_name: str = "gpt2"
    max_new_tokens: int = 40
    n_samples_per_prompt: int = 50
    seeds: tuple[int, ...] = (0, 1, 2)
    device: str | None = None
    decoding: GreedyDecodingConfig = field(default_factory=GreedyDecodingConfig)

    def __post_init__(self) -> None:
        object.__setattr__(self, "prompt_bank_path", Path(self.prompt_bank_path).expanduser())
        object.__setattr__(self, "output_dir", Path(self.output_dir).expanduser())
        object.__setattr__(self, "seeds", tuple(self.seeds))

        if not self.model_name.strip():
            raise ValueError("model_name must not be blank.")
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be at least 1.")
        if self.n_samples_per_prompt < 1:
            raise ValueError("n_samples_per_prompt must be at least 1.")
        if not self.seeds:
            raise ValueError("At least one seed is required.")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("Seeds must be unique.")
        if self.decoding.strategy != "greedy" or self.decoding.do_sample:
            raise ValueError("Week 1 only supports greedy decoding.")


@dataclass(frozen=True, slots=True)
class GeneratedText:
    raw_text: str
    completion_text: str


@dataclass(frozen=True, slots=True)
class GenerationRecord:
    cache_key: str
    model_name: str
    prompt_id: str
    template_id: str
    prompt_type: str
    demographic: str
    prompt_text: str
    decoding_strategy: str
    do_sample: bool
    seed: int
    max_new_tokens: int
    sample_index: int
    raw_text: str
    completion_text: str

    def to_dict(self) -> dict[str, str | bool | int]:
        return {
            "cache_key": self.cache_key,
            "model_name": self.model_name,
            "prompt_id": self.prompt_id,
            "template_id": self.template_id,
            "prompt_type": self.prompt_type,
            "demographic": self.demographic,
            "prompt_text": self.prompt_text,
            "decoding_strategy": self.decoding_strategy,
            "do_sample": self.do_sample,
            "seed": self.seed,
            "max_new_tokens": self.max_new_tokens,
            "sample_index": self.sample_index,
            "raw_text": self.raw_text,
            "completion_text": self.completion_text,
        }


@dataclass(frozen=True, slots=True)
class GenerationArtifactPaths:
    generations_path: Path
    manifest_path: Path


@dataclass(frozen=True, slots=True)
class GenerationRunResult:
    cache_key: str
    generations_path: Path
    manifest_path: Path
    record_count: int
    from_cache: bool
