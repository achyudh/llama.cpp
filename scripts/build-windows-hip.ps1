# Build the fork on Windows with the AMD HIP SDK (ROCm for Windows).
#
# Needs: AMD HIP SDK for Windows 7.2 or later (sets HIP_PATH), and Visual Studio 2022/2026 or its Build Tools with
# "Desktop development with C++" and the MSVC 14.44 toolset (the HIP SDK's clang does not build with MSVC 14.51).
#
#   powershell -ExecutionPolicy Bypass -File scripts\build-windows-hip.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\build-windows-hip.ps1 -Target gfx1201 -Clean

param(
    [string] $Target   = "gfx1201",     # RX 9070 / 9070 XT; gfx1200 for RX 9060 XT
    [string] $BuildDir = "build-rocm",
    [string] $Msvc     = "14.44",       # MSVC toolset version prefix
    [switch] $Clean
)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

if (-not $env:HIP_PATH) {
    throw "HIP_PATH is not set: install the AMD HIP SDK for Windows (7.2 or later) and open a new terminal"
}
Write-Host "HIP SDK: $env:HIP_PATH"

# The MSVC environment (linker, Windows SDK libraries, CMake, Ninja) with the chosen toolset
$vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
if (-not (Test-Path $vswhere)) {
    throw "Visual Studio 2022/2026 (or its Build Tools) with 'Desktop development with C++' is required"
}
$installs = @(& $vswhere -all -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath)
if ($installs.Count -eq 0) {
    throw "No Visual Studio installation with the MSVC x64 tools found"
}

# the install that has the toolset, and its exact version
$vs = $null
$vcvars = $null
foreach ($i in $installs) {
    $tool = Get-ChildItem (Join-Path $i "VC\Tools\MSVC") -Directory -ErrorAction SilentlyContinue |
            Where-Object { $_.Name -like "$Msvc*" } | Sort-Object Name -Descending | Select-Object -First 1
    if ($tool) { $vs = $i; $vcvars = $tool.Name; break }
}
if (-not $vcvars) {
    throw "MSVC $Msvc not found: in the Visual Studio Installer, add 'MSVC v143 - VS 2022 C++ x64/x86 build tools (v14.44)' under Individual components"
}

if ($env:VCToolsVersion -ne $vcvars) {
    $devArgs = "-arch=x64 -host_arch=x64 -vcvars_ver=$vcvars"
    Import-Module (Join-Path $vs "Common7\Tools\Microsoft.VisualStudio.DevShell.dll")
    Enter-VsDevShell -VsInstallPath $vs -SkipAutomaticLocation -DevCmdArguments $devArgs | Out-Null
}
Write-Host "MSVC: $env:VCToolsVersion ($vs)"

# The HIP SDK's clang compiles everything (CMake has no HIP language on Windows)
$env:PATH = "$env:HIP_PATH\bin;$env:PATH"

if ($Clean -and (Test-Path $BuildDir)) {
    Remove-Item -Recurse -Force $BuildDir
}

cmake -S . -B $BuildDir -G Ninja `
    -DCMAKE_C_COMPILER=clang `
    -DCMAKE_CXX_COMPILER=clang++ `
    -DCMAKE_BUILD_TYPE=Release `
    -DGGML_HIP=ON `
    -DGPU_TARGETS="$Target" `
    -DGGML_HIP_NO_VMM=ON `
    -DGGML_CUDA_FA_QUANTS="q4_0-q4_0,q8_0-q8_0,f16-f16,bf16-bf16,q8_0-q4_0" `
    -DGGML_BACKEND_DL=OFF `
    -DGGML_CCACHE=OFF `
    -DLLAMA_BUILD_TESTS=OFF
if ($LASTEXITCODE -ne 0) { throw "CMake configure failed" }

cmake --build $BuildDir --config Release -j $env:NUMBER_OF_PROCESSORS
if ($LASTEXITCODE -ne 0) { throw "Build failed" }

Write-Host ""
Write-Host "Built: $BuildDir\bin\llama-server.exe"
Write-Host "It needs %HIP_PATH%\bin on PATH (rocBLAS/hipBLAS DLLs)."
