# Setup script for Python Virtual Environment

Write-Host "--- Enemy Detection: Python Environment Setup ---" -ForegroundColor Cyan

$pytorchCudaChannel = "cu128"
$pytorchTorchVersion = "2.9.1"
$pytorchTorchvisionVersion = "0.24.1"
$pytorchTorchaudioVersion = "2.9.1"

# 1. Check for Python
$pythonExists = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonExists) {
    Write-Host "Error: Python is not installed or not in your PATH." -ForegroundColor Red
    Write-Host "Please install Python from https://www.python.org/ (ensure 'Add to PATH' is checked)."
    exit 1
}

$pythonVersion = python --version
Write-Host "Found Python: $pythonVersion" -ForegroundColor Green

# 2. Create Virtual Environment
$venvDir = ".venv"
if (Test-Path $venvDir) {
    Write-Host "Updating existing virtual environment in $venvDir..." -ForegroundColor Yellow
} else {
    Write-Host "Creating virtual environment in $venvDir..." -ForegroundColor Yellow
    python -m venv $venvDir
}

# 3. Path to python executable inside venv
$venvPython = Join-Path (Join-Path $venvDir "Scripts") "python.exe"

# 4. Upgrade pip
Write-Host "Upgrading pip..." -ForegroundColor Green
& $venvPython -m pip install --upgrade pip

# 5. Install requirements
$requirementsFile = "requirements.txt"
if (Test-Path $requirementsFile) {
    Write-Host "Installing dependencies from $requirementsFile..." -ForegroundColor Green
    & $venvPython -m pip install -r $requirementsFile
} else {
    Write-Host "Warning: $requirementsFile not found. Skipping dependency installation." -ForegroundColor Yellow
}

# 6. Prefer CUDA-enabled PyTorch when an NVIDIA GPU is available
$nvidiaSmi = Get-Command "nvidia-smi" -ErrorAction SilentlyContinue
if ($nvidiaSmi) {
    Write-Host "NVIDIA GPU detected. Installing CUDA-enabled PyTorch ($pytorchCudaChannel)..." -ForegroundColor Green
    & $venvPython -m pip install --upgrade --force-reinstall `
        "torch==$pytorchTorchVersion" `
        "torchvision==$pytorchTorchvisionVersion" `
        "torchaudio==$pytorchTorchaudioVersion" `
        --index-url "https://download.pytorch.org/whl/$pytorchCudaChannel"
} else {
    Write-Host "No NVIDIA GPU detected. Keeping the default CPU PyTorch build from requirements.txt." -ForegroundColor Yellow
}

# 7. Show the final torch runtime so GPU/CPU state is obvious
Write-Host "Verifying final PyTorch runtime..." -ForegroundColor Green
& $venvPython -c "import torch; print(f'torch={torch.__version__}'); print(f'cuda_available={torch.cuda.is_available()}'); print(f'torch_cuda={torch.version.cuda}'); print(f'device_count={torch.cuda.device_count()}')"

Write-Host "`n--- Setup Complete! ---" -ForegroundColor Cyan
Write-Host "The Electron app should now be able to run Python scripts."
Write-Host "To manually activate the environment in PowerShell, run:"
Write-Host ".\.venv\Scripts\Activate.ps1" -ForegroundColor Gray
