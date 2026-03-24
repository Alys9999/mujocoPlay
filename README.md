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
- any validated GPU setup flow beyond wheel installation and import checks
- any validated quantized inference path

## Install Scripts

The native Linux installer is now intended to work on headless Ubuntu-class servers as well as desktop Linux. It uses:

- Python `3.12+`
- Torch `2.10.0`
- TorchVision `0.25.0`
- a torch wheel tag chosen automatically:
  - `cpu` when no working NVIDIA driver is detected
  - `cu128` when a working NVIDIA driver is detected
- a MuJoCo GL backend chosen automatically:
  - `osmesa` on headless CPU/server installs
  - `egl` when a display, render node, or NVIDIA device is detected

Files:

- `scripts/install_env_conda.sh`
- `scripts/install_env_conda.ps1`
- `scripts/download_paligemma_tokenizer.sh`
- `scripts/install_env_native_linux.sh`
- `scripts/repair_ubuntu_apt_sources.sh`
- `requirements.txt`

Linux/macOS with conda:

```bash
bash scripts/install_env_conda.sh
```

On Linux, the conda installer also ensures the MuJoCo runtime GL packages are present and will repair a mismatched Ubuntu archive codename before installing them. Set `SKIP_APT=1` only when those host packages are already in place.

Windows with conda:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\install_env_conda.ps1
```

Native Linux with system Python and venv:

```bash
bash scripts/install_env_native_linux.sh
```

Native Linux now requires a Python `3.12` interpreter. If your base distro does not ship `python3.12` in its normal apt repositories, use the conda installer instead or install Python `3.12` separately and pass `PYTHON_BIN=/path/to/python3.12`.

If the host reports one Ubuntu release in `/etc/os-release` but `apt` is pinned to another codename, preview a repair with:

```bash
bash scripts/repair_ubuntu_apt_sources.sh
```

Apply the repair and refresh package indexes with:

```bash
bash scripts/repair_ubuntu_apt_sources.sh --apply
```

Useful overrides for native Linux:

```bash
PYTHON_BIN=python3.12 bash scripts/install_env_native_linux.sh
TORCH_WHL_TAG=cpu bash scripts/install_env_native_linux.sh
TORCH_WHL_TAG=cu128 VERIFY_ENV_REQUIRE_CUDA=1 bash scripts/install_env_native_linux.sh
MUJOCO_GL_BACKEND=osmesa bash scripts/install_env_native_linux.sh
SKIP_APT=1 bash scripts/install_env_native_linux.sh
```

All install scripts finish by running:

```bash
python verify_env.py
```

On CPU-only or headless servers, `verify_env.py` treats missing CUDA as informational by default. Set `VERIFY_ENV_REQUIRE_CUDA=1` when you want the verification step to fail unless CUDA is actually available.

For Ubuntu `22.04`, the recommended path is usually:

```bash
bash scripts/install_env_conda.sh
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
    tokenizer_name_or_path="/absolute/path/to/paligemma_tokenizer_or_hf_repo",
)
```

`tokenizer_name_or_path` is optional. When provided, it is passed to
`transformers.AutoTokenizer.from_pretrained(...)`, so it can be either:

- a local tokenizer directory
- a Hugging Face repo id

Example benchmark invocation:

```bash
python -m phase1.policy_benchmark \
  --task pick_place \
  --pipeline all \
  --policies phase1.pi05_policy:PI05SequentialPolicy \
  --policy-kwargs '{"model_path":"/absolute/path/to/model","device":"cpu","quantization":"none","dtype":"float32","duplicate_overview_to_all_cameras":true,"tokenizer_name_or_path":"/absolute/path/to/paligemma_tokenizer_or_hf_repo"}'
```

If you have access to the gated PaliGemma repo, download a local tokenizer directory with:

```bash
bash scripts/download_paligemma_tokenizer.sh
```

By default this writes to:

```bash
/root/models/paligemma-3b-pt-224
```

## Notes

- `int8_dynamic` is implemented as a CPU-only mode in the adapter.
- GPU execution depends on the torch build and the target runtime.
- If you need a validated deployment path, it has to be specified and tested against the exact checkpoint and target machine.
