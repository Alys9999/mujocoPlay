from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmark.core.config.loader import build_config_from_preset, load_benchmark_config
from benchmark.core.runtime.scheduler import BenchmarkScheduler


def main() -> None:
    """CLI entrypoint for the config-driven benchmark runner."""
    parser = argparse.ArgumentParser(description="Run the config-driven MuJoCo benchmark.")
    parser.add_argument("--config", type=Path, default=None, help="Path to a YAML/JSON benchmark config.")
    parser.add_argument("--preset", type=str, default=None, help="Built-in preset name when no config file is supplied.")
    parser.add_argument("--episodes", type=int, default=None, help="Optional episode override.")
    parser.add_argument("--max-steps", type=int, default=None, help="Optional benchmark-side step cap.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Optional output directory override.")
    parser.add_argument("--policy", type=str, default=None, help="Optional policy adapter override.")
    parser.add_argument("--policy-kwargs", type=str, default="{}", help="JSON object of policy kwargs overrides.")
    args = parser.parse_args()

    if args.config is not None:
        config = load_benchmark_config(args.config)
    else:
        preset_name = args.preset or "both_random_pick_place"
        overrides = {}
        benchmark_overrides = {}
        runtime_overrides = {}
        policy_overrides = {}
        if args.episodes is not None:
            benchmark_overrides["episodes"] = args.episodes
        if args.output_dir is not None:
            benchmark_overrides["output_dir"] = args.output_dir
        if benchmark_overrides:
            overrides["benchmark"] = benchmark_overrides
        if args.max_steps is not None:
            runtime_overrides["max_steps"] = args.max_steps
        if runtime_overrides:
            overrides["runtime"] = runtime_overrides
        if args.policy is not None:
            policy_overrides["adapter"] = args.policy
        policy_kwargs = json.loads(args.policy_kwargs)
        if policy_kwargs:
            policy_overrides["kwargs"] = policy_kwargs
        if policy_overrides:
            overrides["policy"] = policy_overrides
        config = build_config_from_preset(preset_name, overrides=overrides or None)

    result = BenchmarkScheduler().run(config)
    print(json.dumps(result.model_dump(mode="json"), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
