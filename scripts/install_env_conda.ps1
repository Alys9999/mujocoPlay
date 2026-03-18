$ErrorActionPreference = "Stop"

$EnvName = if ($env:ENV_NAME) { $env:ENV_NAME } else { "mujocoplay" }
$PythonVersion = if ($env:PYTHON_VERSION) { $env:PYTHON_VERSION } else { "3.12" }
$TorchVersion = if ($env:TORCH_VERSION) { $env:TORCH_VERSION } else { "2.10.0" }
$TorchVisionVersion = if ($env:TORCHVISION_VERSION) { $env:TORCHVISION_VERSION } else { "0.25.0" }
$CudaWhlTag = if ($env:CUDA_WHL_TAG) { $env:CUDA_WHL_TAG } else { "cu128" }
$RootDir = Split-Path -Parent $PSScriptRoot

$condaCmd = Get-Command conda -ErrorAction SilentlyContinue
if (-not $condaCmd) {
    throw "conda is required but was not found in PATH."
}

conda create -y -n $EnvName "python=$PythonVersion"

$pipInstallTorch = @"
python -m pip install --upgrade pip
python -m pip install torch==$TorchVersion+$CudaWhlTag torchvision==$TorchVisionVersion+$CudaWhlTag --index-url https://download.pytorch.org/whl/$CudaWhlTag
python -m pip install -r `"$RootDir\requirements.txt`"
python `"$RootDir\verify_env.py`"
"@

conda run -n $EnvName powershell -NoProfile -Command $pipInstallTorch
