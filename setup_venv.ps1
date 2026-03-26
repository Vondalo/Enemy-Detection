param(
    [string]$PythonExe
)

$ErrorActionPreference = "Stop"

Write-Host "--- Enemy Detection: Python Environment Setup ---" -ForegroundColor Cyan

$pytorchCudaChannel = "cu128"
$pytorchTorchVersion = "2.9.1"
$pytorchTorchvisionVersion = "0.24.1"
$pytorchTorchaudioVersion = "2.9.1"

function Invoke-NativeCommand {
    param(
        [string]$Executable,
        [string[]]$Arguments,
        [string]$FailureMessage
    )

    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw $FailureMessage
    }
}

function Test-PythonExecutable {
    param(
        [string]$Executable
    )

    if ([string]::IsNullOrWhiteSpace($Executable)) {
        return $false
    }

    try {
        & $Executable -c "import sys; print(sys.executable)" *> $null
        return $LASTEXITCODE -eq 0
    } catch {
        return $false
    }
}

function Get-CommandPath {
    param(
        [string]$CommandName
    )

    try {
        $command = Get-Command $CommandName -ErrorAction Stop
        if ($command.Source) {
            return $command.Source
        }
        if ($command.Path) {
            return $command.Path
        }
    } catch {
    }

    return $null
}

function Get-PythonFromPyLauncher {
    $pyLauncher = Get-CommandPath "py"
    if (-not $pyLauncher) {
        return $null
    }

    try {
        $resolved = (& $pyLauncher -3 -c "import sys; print(sys.executable)" 2>$null | Select-Object -First 1)
        if ($LASTEXITCODE -eq 0 -and $resolved) {
            $candidate = $resolved.ToString().Trim()
            if (Test-PythonExecutable $candidate) {
                return $candidate
            }
        }
    } catch {
    }

    return $null
}

function Get-CommonPythonCandidates {
    $patterns = @()

    if ($env:LocalAppData) {
        $patterns += (Join-Path $env:LocalAppData "Programs\Python\Python*\python.exe")
    }
    if ($env:ProgramFiles) {
        $patterns += (Join-Path $env:ProgramFiles "Python*\python.exe")
        $patterns += (Join-Path $env:ProgramFiles "Python\Python*\python.exe")
    }
    if (${env:ProgramFiles(x86)}) {
        $patterns += (Join-Path ${env:ProgramFiles(x86)} "Python*\python.exe")
        $patterns += (Join-Path ${env:ProgramFiles(x86)} "Python\Python*\python.exe")
    }

    $results = @()
    foreach ($pattern in $patterns) {
        try {
            $results += Get-ChildItem -Path $pattern -File -ErrorAction SilentlyContinue |
                Select-Object -ExpandProperty FullName
        } catch {
        }
    }

    return $results | Where-Object { $_ } | Sort-Object -Descending -Unique
}

function Find-BasePython {
    param(
        [string]$PreferredPython
    )

    $candidates = New-Object System.Collections.Generic.List[string]

    if ($PreferredPython) {
        $candidates.Add($PreferredPython)
    }

    $pythonCommand = Get-CommandPath "python"
    if ($pythonCommand) {
        $candidates.Add($pythonCommand)
    }

    $pyResolved = Get-PythonFromPyLauncher
    if ($pyResolved) {
        $candidates.Add($pyResolved)
    }

    foreach ($candidate in Get-CommonPythonCandidates) {
        $candidates.Add($candidate)
    }

    foreach ($candidate in ($candidates | Where-Object { $_ } | Select-Object -Unique)) {
        if (Test-PythonExecutable $candidate) {
            return $candidate
        }
    }

    return $null
}

$venvDir = ".venv"
$venvPython = Join-Path (Join-Path $venvDir "Scripts") "python.exe"
$needsRebuild = $false

if (Test-Path $venvPython) {
    if (Test-PythonExecutable $venvPython) {
        Write-Host "Using existing virtual environment in $venvDir..." -ForegroundColor Yellow
    } else {
        Write-Host "Existing virtual environment is broken. Recreating $venvDir..." -ForegroundColor Yellow
        $needsRebuild = $true
    }
} elseif (Test-Path $venvDir) {
    Write-Host "Existing virtual environment is incomplete. Recreating $venvDir..." -ForegroundColor Yellow
    $needsRebuild = $true
} else {
    Write-Host "Creating virtual environment in $venvDir..." -ForegroundColor Yellow
    $needsRebuild = $true
}

if ($needsRebuild) {
    $basePython = Find-BasePython $PythonExe
    if (-not $basePython) {
        Write-Host "Error: Could not find a usable base Python interpreter." -ForegroundColor Red
        Write-Host "Re-run this script with an explicit interpreter, for example:" -ForegroundColor Red
        Write-Host ".\setup_venv.ps1 -PythonExe C:\Path\To\python.exe" -ForegroundColor Gray
        exit 1
    }

    Write-Host "Using base Python: $basePython" -ForegroundColor Green

    if (Test-Path $venvDir) {
        Remove-Item $venvDir -Recurse -Force
    }

    Invoke-NativeCommand -Executable $basePython `
        -Arguments @("-m", "venv", $venvDir) `
        -FailureMessage "Failed to create the virtual environment."

    if (-not (Test-PythonExecutable $venvPython)) {
        throw "Virtual environment creation succeeded, but '$venvPython' is not runnable."
    }
}

Write-Host "Upgrading pip..." -ForegroundColor Green
Invoke-NativeCommand -Executable $venvPython `
    -Arguments @("-m", "pip", "install", "--upgrade", "pip") `
    -FailureMessage "Failed to upgrade pip."

$requirementsFile = "requirements.txt"
if (Test-Path $requirementsFile) {
    Write-Host "Installing dependencies from $requirementsFile..." -ForegroundColor Green
    Invoke-NativeCommand -Executable $venvPython `
        -Arguments @("-m", "pip", "install", "-r", $requirementsFile) `
        -FailureMessage "Failed to install project dependencies."
} else {
    Write-Host "Warning: $requirementsFile not found. Skipping dependency installation." -ForegroundColor Yellow
}

$nvidiaSmi = Get-Command "nvidia-smi" -ErrorAction SilentlyContinue
if ($nvidiaSmi) {
    Write-Host "NVIDIA GPU detected. Installing CUDA-enabled PyTorch ($pytorchCudaChannel)..." -ForegroundColor Green
    Invoke-NativeCommand -Executable $venvPython `
        -Arguments @(
            "-m", "pip", "install", "--upgrade", "--force-reinstall",
            "torch==$pytorchTorchVersion",
            "torchvision==$pytorchTorchvisionVersion",
            "torchaudio==$pytorchTorchaudioVersion",
            "--index-url", "https://download.pytorch.org/whl/$pytorchCudaChannel"
        ) `
        -FailureMessage "Failed to install the CUDA-enabled PyTorch wheel."
} else {
    Write-Host "No NVIDIA GPU detected. Keeping the default CPU PyTorch build from requirements.txt." -ForegroundColor Yellow
}

$torchCheck = @'
import torch
print('torch=' + str(torch.__version__))
print('cuda_available=' + str(torch.cuda.is_available()))
print('torch_cuda=' + str(torch.version.cuda))
print('device_count=' + str(torch.cuda.device_count()))
if torch.cuda.is_available() and torch.cuda.device_count():
    print('device_name=' + str(torch.cuda.get_device_name(0)))
'@

Write-Host "Verifying final PyTorch runtime..." -ForegroundColor Green
Invoke-NativeCommand -Executable $venvPython `
    -Arguments @("-c", $torchCheck) `
    -FailureMessage "Failed to verify the final PyTorch runtime."

Write-Host "`n--- Setup Complete! ---" -ForegroundColor Cyan
Write-Host "The Electron app should now be able to run Python scripts."
Write-Host "To manually activate the environment in PowerShell, run:"
Write-Host ".\.venv\Scripts\Activate.ps1" -ForegroundColor Gray
