# MuJoCo Pick-and-Place Benchmark

This repo contains the benchmark code only.

In scope:

- the pick-and-place environment
- the benchmark runner
- four benchmark pipelines:
  - `normal_pick_place`
  - `body_random_pick_place`
  - `object_random_pick_place`
  - `both_random_pick_place`
- a `phase1.pi05_policy.PI05SequentialPolicy` adapter

Not guaranteed by this repo:

- any validated checkpoint source
- any validated Linux setup flow
- any validated GPU setup flow
- any validated quantized inference path

## Install Scripts

These scripts pin the runtime to the versions currently expected by this repo:

- Python `3.12`
- Torch `2.10.0`
- TorchVision `0.25.0`
- CUDA wheel runtime `cu128`

Files:

- `scripts/install_env_conda.sh`
- `scripts/install_env_conda.ps1`
- `scripts/install_env_native_linux.sh`
- `requirements.txt`

Linux/macOS with conda:

```bash
bash scripts/install_env_conda.sh
```

Windows with conda:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\install_env_conda.ps1
```

Native Linux with system Python and venv:

```bash
bash scripts/install_env_native_linux.sh
```

All install scripts finish by running:

```bash
python verify_env.py
```

## Benchmark CLI

The runner is `python -m phase1.policy_benchmark`.

Builtin policy support is intentionally limited to:

- `random`

All non-random policies must be passed explicitly as an import path:

```bash
python -m phase1.policy_benchmark \
  --policies package.module:ClassName \
  --policy-kwargs '{"key":"value"}'
```

## Pi 0.5 adapter

The adapter lives in `phase1/pi05_policy.py`.

It requires explicit configuration. It does not assume:

- model path
- device
- quantization mode
- dtype
- whether missing wrist cameras should be replaced by duplicated overview images

Example:

```python
from phase1.pi05_policy import PI05SequentialPolicy

policy = PI05SequentialPolicy(
    model_path="/absolute/path/to/model",
    device="cpu",
    quantization="none",
    dtype="float32",
    duplicate_overview_to_all_cameras=True,
)
```

Example benchmark invocation:

```bash
python -m phase1.policy_benchmark \
  --task pick_place \
  --pipeline all \
  --policies phase1.pi05_policy:PI05SequentialPolicy \
  --policy-kwargs '{"model_path":"/absolute/path/to/model","device":"cpu","quantization":"none","dtype":"float32","duplicate_overview_to_all_cameras":true}'
```

## Notes

- `int8_dynamic` is implemented as a CPU-only mode in the adapter.
- GPU execution depends on the torch build and the target runtime.
- If you need a validated deployment path, it has to be specified and tested against the exact checkpoint and target machine.
