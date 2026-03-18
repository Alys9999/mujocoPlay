# MuJoCo Pick-and-Place Benchmark

This repo is reduced to the benchmark-only path:

- `normal_pick_place`
- `body_random_pick_place`
- `object_random_pick_place`
- `both_random_pick_place`

All four modes evaluate the same pick-and-place task and differ only by which hidden physics sources are randomized.

## Layout

- `phase1/`: benchmark environment, benchmark runner, and Pi 0.5 policy adapter.
- `configs/pi05_int8_linux.json`: ready-to-use Pi 0.5 INT8 CPU settings.
- `configs/pi05_gpu_linux.json`: ready-to-use Pi 0.5 CUDA settings.
- `scripts/setup_native_linux.sh`: native Linux environment bootstrap.
- `scripts/setup_native_linux_gpu.sh`: native Linux GPU bootstrap.
- `scripts/download_pi05.sh`: download `lerobot/pi05_base` into `external/models/pi05_base`.
- `scripts/run_pi05_int8_benchmark.sh`: run all four benchmark pipelines with the Linux INT8 preset.
- `scripts/run_pi05_gpu_benchmark.sh`: run all four benchmark pipelines with the Linux GPU preset.

## Native Linux setup

```bash
bash scripts/setup_native_linux.sh
source .venv-linux/bin/activate
export MUJOCO_GL=egl
```

## Native Linux GPU setup

This path expects a working NVIDIA driver on the machine already.

```bash
bash scripts/setup_native_linux_gpu.sh
source .venv-linux-gpu/bin/activate
export MUJOCO_GL=egl
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no-gpu')"
```

## Model setup

```bash
source .venv-linux/bin/activate
bash scripts/download_pi05.sh
```

## Run the benchmark

```bash
source .venv-linux/bin/activate
bash scripts/run_pi05_int8_benchmark.sh --episodes 8 --family all
```

GPU:

```bash
source .venv-linux-gpu/bin/activate
bash scripts/run_pi05_gpu_benchmark.sh --episodes 8 --family all
```

The benchmark writes:

- `benchmark_results/pi05_int8_benchmark.md`
- `benchmark_results/pi05_int8_benchmark.json`
- `benchmark_results/pi05_gpu_benchmark.md`
- `benchmark_results/pi05_gpu_benchmark.json`

## Direct CLI examples

Random baseline:

```bash
python -m phase1.policy_benchmark \
  --task pick_place \
  --pipeline all \
  --policies random \
  --episodes 8
```

Pi 0.5 INT8 CPU:

```bash
python -m phase1.policy_benchmark \
  --task pick_place \
  --pipeline all \
  --policies pi05-int8 \
  --policy-kwargs "$(tr -d '\n' < configs/pi05_int8_linux.json)"
```

Pi 0.5 GPU:

```bash
python -m phase1.policy_benchmark \
  --task pick_place \
  --pipeline all \
  --policies pi05 \
  --policy-kwargs "$(tr -d '\n' < configs/pi05_gpu_linux.json)"
```
