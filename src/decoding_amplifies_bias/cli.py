from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from decoding_amplifies_bias.generation import GenerationRunner
from decoding_amplifies_bias.models import GenerationConfig
from decoding_amplifies_bias.paths import DEFAULT_OUTPUT_DIR, DEFAULT_PROMPT_BANK_PATH


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Week 1 GPT-2 greedy baseline with caching and manifests."
    )
    parser.add_argument("--prompt-bank", type=Path, default=DEFAULT_PROMPT_BANK_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-name", default="gpt2")
    parser.add_argument("--max-new-tokens", type=int, default=40)
    parser.add_argument("--n-samples", type=int, default=50)
    parser.add_argument("--seed", action="append", type=int, dest="seeds")
    parser.add_argument("--device", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    seeds = tuple(args.seeds) if args.seeds is not None else (0, 1, 2)

    config = GenerationConfig(
        prompt_bank_path=args.prompt_bank,
        output_dir=args.output_dir,
        model_name=args.model_name,
        max_new_tokens=args.max_new_tokens,
        n_samples_per_prompt=args.n_samples,
        seeds=seeds,
        device=args.device,
    )
    result = GenerationRunner().run(config)

    summary = {
        "cache_key": result.cache_key,
        "from_cache": result.from_cache,
        "record_count": result.record_count,
        "generations_path": str(result.generations_path),
        "manifest_path": str(result.manifest_path),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0
