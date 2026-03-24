# MuJoCoPlay Guide: PI05 and SmolVLA

This guide is the practical runbook for this repo on a Linux server.

It covers:

- creating the Python environment
- repairing Ubuntu apt sources if the machine is misconfigured
- logging into Hugging Face
- downloading the PI05 checkpoint and tokenizer
- downloading the SmolVLA checkpoint
- running smoke tests first
- running the full benchmark commands

All commands below assume you are starting from the repo root:

```bash
cd /root/mujocoBenchmark/mujocoPlay
```

## 1. Recommended host setup

Use:

- Ubuntu 22.04 LTS (`jammy`)
- Python 3.12
- conda env `mujocoplay`

This repo currently works best with the conda installer. The older native `.venv-native*` paths are not the recommended path for LeRobot `0.5.0`.

If you already switched to conda and still have old native venv folders from earlier attempts, it is safe to remove them:

```bash
rm -rf /root/mujocoBenchmark/mujocoPlay/.venv-native
rm -rf /root/mujocoBenchmark/mujocoPlay/.venv-native-jammy
```

## 2. Optional: repair apt sources first

Only do this if the machine says Ubuntu 22.04 but `apt` is still pointed at the wrong release, such as `focal`.

Preview only:

```bash
bash scripts/repair_ubuntu_apt_sources.sh
```

Apply the repair:

```bash
bash scripts/repair_ubuntu_apt_sources.sh --apply
```

## 3. Create the conda environment

Create the repo environment:

```bash
bash scripts/install_env_conda.sh
```

On Linux, this installer now also ensures the MuJoCo runtime GL packages are present. If the host says Ubuntu `jammy` but `apt` still points at `focal`, the script will repair the archive codename before installing those runtime packages. Use `SKIP_APT=1` only if the host-side packages are already correct.

Activate it:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mujocoplay
```

Verify the environment:

```bash
python verify_env.py
```

If you prefer to avoid activating the shell env each time, all later commands can also be run with `conda run -n mujocoplay ...`.

## 4. Hugging Face login

Some checkpoints are public. Some tokenizer/model dependencies are gated.

Normal login:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mujocoplay
hf auth login
hf auth whoami
```

If you are on a remote machine and do not want to paste your token directly into the terminal, use a hidden prompt:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mujocoplay
python - <<'PY'
import getpass
from huggingface_hub import login

token = getpass.getpass("HF token: ")
login(token=token)
print("Hugging Face login saved.")
PY
```

## 5. Recommended local model layout

These paths are used in the examples below:

- PI05 checkpoint: `/root/models/pi05_base`
- Official gated Paligemma tokenizer: `/root/models/paligemma-3b-pt-224`
- Public Gemma fallback tokenizer: `/root/models/gemma-tokenizer-public`
- SmolVLA checkpoint: `/root/models/smolvla_base`

## 6. Running PI05

### 6.1 What PI05 uses in this repo

PI05 is wired through the sequential benchmark CLI:

```bash
python -m phase1.policy_benchmark
```

For PI05 you must provide:

- a real local checkpoint directory for `model_path`
- an explicit `device`
- an explicit `quantization`
- an explicit `dtype`
- an explicit `duplicate_overview_to_all_cameras`

Recommended GPU settings on this server:

- `device="cuda"`
- `quantization="none"`
- `dtype="float32"`

Use `float32` on CUDA here. The adapter currently falls back away from CUDA `bfloat16`.

### 6.2 Download the PI05 checkpoint

Download the public checkpoint into a real local directory:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda run -n mujocoplay hf download lerobot/pi05_base --local-dir /root/models/pi05_base
```

Quick check:

```bash
ls /root/models/pi05_base/config.json
ls /root/models/pi05_base/model.safetensors
```

### 6.3 Download the PI05 tokenizer

PI05 in this repo supports two practical tokenizer setups:

- the official gated PaliGemma tokenizer at `/root/models/paligemma-3b-pt-224`
- the public Gemma fallback tokenizer at `/root/models/gemma-tokenizer-public`

Important:

- A directory that exists but only contains `.cache` is incomplete and will not work as `tokenizer_name_or_path`.
- The PI05 adapter can fall back to the public Gemma tokenizer automatically, but using an explicit real local directory is easier to debug and reproduce.

#### Option A: official gated Paligemma tokenizer

Preferred path: use the official gated Paligemma tokenizer.

First, request access for your Hugging Face account:

- `https://huggingface.co/google/paligemma-3b-pt-224`

Then download the tokenizer files:

```bash
cd /root/mujocoBenchmark/mujocoPlay
bash scripts/download_paligemma_tokenizer.sh
```

By default this writes to:

```bash
/root/models/paligemma-3b-pt-224
```

Quick check:

```bash
ls /root/models/paligemma-3b-pt-224/tokenizer_config.json
ls /root/models/paligemma-3b-pt-224/tokenizer.json
```

Notes:

- `401` means you are not logged in.
- `403` means you are logged in but your account has not been approved for the gated repo.
- If you see only `/root/models/paligemma-3b-pt-224/.cache/...lock` files, the download did not produce a usable tokenizer directory.

#### Option B: public Gemma fallback tokenizer

Use this when you do not have gated PaliGemma access, or when you want a deterministic local tokenizer path that works with the PI05 adapter in this repo.

Scripted download:

```bash
cd /root/mujocoBenchmark/mujocoPlay
bash scripts/download_public_gemma_tokenizer.sh
```

Manual `hf` command:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda run -n mujocoplay hf download pcuenq/gemma-tokenizer --local-dir /root/models/gemma-tokenizer-public
```

Quick check:

```bash
ls /root/models/gemma-tokenizer-public/tokenizer_config.json
ls /root/models/gemma-tokenizer-public/tokenizer.json
```

Notes:

- This is the tokenizer used by the PI05 adapter's automatic fallback path.
- If you pass `tokenizer_name_or_path`, point it at a real directory such as `/root/models/gemma-tokenizer-public`.
- If you omit `tokenizer_name_or_path`, the adapter will try to download and cache the public fallback automatically.

### 6.4 PI05 short smoke test

Run a short smoke test first. This keeps the run quick while still exercising model load, camera input, rollout, and video writing.

```bash
cd /root/mujocoBenchmark/mujocoPlay
conda run -n mujocoplay python -m phase1.policy_benchmark \
  --family block \
  --task pick_place \
  --pipeline normal_pick_place \
  --episodes 1 \
  --max-steps 20 \
  --policies phase1.pi05_policy:PI05SequentialPolicy \
  --policy-kwargs '{"model_path":"/root/models/pi05_base","device":"cuda","quantization":"none","dtype":"float32","action_chunk_size":10,"duplicate_overview_to_all_cameras":true,"tokenizer_name_or_path":"/root/models/gemma-tokenizer-public"}' \
  --output benchmark_results/pi05_smoke_short.md \
  --video-dir benchmark_results/pi05_smoke_short_videos
```

If you successfully downloaded the official gated tokenizer, replace `/root/models/gemma-tokenizer-public` with `/root/models/paligemma-3b-pt-224`.

### 6.5 PI05 full pipeline benchmark

This is the full benchmark command for PI05 across all configured pipelines.

```bash
cd /root/mujocoBenchmark/mujocoPlay
conda run -n mujocoplay python -m phase1.policy_benchmark \
  --family all \
  --family-split all \
  --task pick_place \
  --split unseen \
  --pipeline all \
  --episodes 24 \
  --policies phase1.pi05_policy:PI05SequentialPolicy \
  --policy-kwargs '{"model_path":"/root/models/pi05_base","device":"cuda","quantization":"none","dtype":"float32","action_chunk_size":10,"duplicate_overview_to_all_cameras":true,"tokenizer_name_or_path":"/root/models/gemma-tokenizer-public"}' \
  --output benchmark_results/pi05_full.md \
  --video-dir benchmark_results/pi05_full_videos
```

If you want to rely on the adapter's automatic fallback instead of an explicit tokenizer path, remove the `tokenizer_name_or_path` key from the JSON.

### 6.6 PI05 outputs

PI05 benchmark outputs go to:

- Markdown summary: `benchmark_results/pi05_full.md`
- JSON summary: `benchmark_results/pi05_full.json`
- Videos: `benchmark_results/pi05_full_videos/`

## 7. Running SmolVLA

### 7.1 Important caveat

`lerobot/smolvla_base` is the public SmolVLA checkpoint currently available from the `lerobot` account.

In this repo, SmolVLA runs through the config-driven benchmark CLI:

```bash
python -m benchmark
```

Important: `lerobot/smolvla_base` is an SO100-style joint-space checkpoint, while this benchmark uses a Franka Cartesian action packet. The current adapter bridges this heuristically so the run can proceed, but the resulting numbers should be treated as exploratory rather than directly comparable to native Franka or PI05 results.

### 7.2 Download the SmolVLA checkpoint

Download it locally:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda run -n mujocoplay hf download lerobot/smolvla_base --local-dir /root/models/smolvla_base
```

Quick check:

```bash
ls /root/models/smolvla_base/config.json
ls /root/models/smolvla_base/model.safetensors
```

SmolVLA also depends on the public VLM backbone `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`. If it is not already cached, it will be downloaded automatically on first load.

### 7.3 SmolVLA smoke test

```bash
cd /root/mujocoBenchmark/mujocoPlay
conda run -n mujocoplay python -m benchmark \
  --preset normal_pick_place \
  --episodes 2 \
  --max-steps 150 \
  --output-dir benchmark_results/smolvla_smoke \
  --policy lerobot.smolvla \
  --policy-kwargs '{"model_path":"/root/models/smolvla_base","device":"cuda","dtype":"float32","duplicate_overview_to_all_cameras":true}'
```

### 7.4 SmolVLA full benchmark

```bash
cd /root/mujocoBenchmark/mujocoPlay
conda run -n mujocoplay python -m benchmark \
  --preset both_random_pick_place \
  --output-dir benchmark_results/smolvla_full \
  --policy lerobot.smolvla \
  --policy-kwargs '{"model_path":"/root/models/smolvla_base","device":"cuda","dtype":"float32","duplicate_overview_to_all_cameras":true}'
```

### 7.5 SmolVLA outputs

SmolVLA benchmark outputs go under the chosen output directory, for example:

- `benchmark_results/smolvla_full/benchmark-result.json`
- `benchmark_results/smolvla_full/trace.jsonl`

## 8. Suggested run order

Use this order on a fresh server:

1. Repair apt sources only if Ubuntu and apt releases are mismatched.
2. Create the conda env with `bash scripts/install_env_conda.sh`.
3. Activate `mujocoplay` and run `python verify_env.py`.
4. Log into Hugging Face.
5. Download `/root/models/pi05_base`.
6. Download either `/root/models/gemma-tokenizer-public` or `/root/models/paligemma-3b-pt-224`.
7. Run the PI05 smoke test.
8. Run the PI05 full benchmark.
9. Download `/root/models/smolvla_base`.
10. Run the SmolVLA smoke test.
11. Run the SmolVLA full benchmark.

## 9. Common failures and what they mean

`FileNotFoundError: pi05 model path does not exist`

- Your `model_path` is not a real local checkpoint directory.
- Fix by downloading `lerobot/pi05_base` into `/root/models/pi05_base` and using that exact path.

`401 Client Error` from Hugging Face

- You are not logged in.
- Fix with `hf auth login`.

`403 Client Error` for `google/paligemma-3b-pt-224`

- You are logged in, but the account has not been granted access to the gated repo yet.
- Visit the model page and request access with the same account shown by `hf auth whoami`.
- Or use the public fallback tokenizer instead:
  `bash scripts/download_public_gemma_tokenizer.sh`

`ValueError: Failed to instantiate processor step 'tokenizer_processor' ... You need to have sentencepiece or tiktoken installed`

- In this setup, this usually means the tokenizer path exists but is incomplete, often with only `.cache` and lock files under `/root/models/paligemma-3b-pt-224`.
- Fix by downloading a real tokenizer directory with either:
  `bash scripts/download_paligemma_tokenizer.sh`
- Or:
  `bash scripts/download_public_gemma_tokenizer.sh`
- You can also remove `tokenizer_name_or_path` and let `phase1.pi05_policy` use its public fallback logic.

`PI05 tokenizer override path ... exists but is incomplete; falling back to a public Gemma tokenizer`

- This warning is safe.
- It means the adapter detected an unusable tokenizer directory and switched to the public fallback tokenizer automatically.

`ImportError: Cannot initialize a EGL device display` followed by `AttributeError: 'GLContext' object has no attribute '_context'`

- MuJoCo auto-selected `MUJOCO_GL=egl`, but the host cannot actually create an EGL renderer.
- First run `python verify_env.py` and make sure `mujoco_renderer` passes before starting the benchmark smoke test.
- On Ubuntu `22.04`, check whether `apt` is pointed at the wrong release:
  `bash scripts/repair_ubuntu_apt_sources.sh`
- If `/etc/os-release` says `jammy` but the preview still shows `focal` package sources, apply the repair:
  `bash scripts/repair_ubuntu_apt_sources.sh --apply`
- The conda installer does not repair system GL runtime packages for you. After fixing apt sources, make sure the host has the runtime libraries used by MuJoCo rendering, especially `libegl1`, `libgl1`, and `libosmesa6`.
- Re-run `python verify_env.py` after fixing the host so the renderer check goes green.

`dtype=bfloat16` problems on PI05 CUDA

- Use `dtype="float32"` in the benchmark command.

`int8_dynamic` quantization error on GPU

- `int8_dynamic` is CPU-only here.
- Use `quantization="none"` on CUDA.

SmolVLA scores look strange

- This is expected if you use `lerobot/smolvla_base`.
- The current adapter is a heuristic bridge from SmolVLA's native action space into this benchmark's Franka Cartesian control packet.
