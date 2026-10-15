param(
    [switch]$Force
)

$root = Split-Path $PSScriptRoot -Parent
$onnxPath = Join-Path $root 'nonescape-mini-v0.onnx'
$safetensorsPath = Join-Path $root 'nonescape-mini-v0.safetensors'

# Upstream ships both formats, but the ONNX CDN is currently 403 Forbidden.
# The original script fell back to the safetensors URL yet saved it under the
# ".onnx" name, producing a file onnxruntime cannot parse. We now fetch the
# safetensors under its real name and convert it to ONNX locally.
$onnxUrls = @(
    "https://nonescape.sfo2.cdn.digitaloceanspaces.com/nonescape-mini-v0.onnx"
)
$safetensorsUrls = @(
    "https://huggingface.co/e3ntity/nonescape-v0/resolve/main/nonescape-mini-v0.safetensors"
)

function Test-ValidOnnx($path) {
    if (-not (Test-Path $path)) { return $false }
    & python -c "import sys, onnxruntime as ort; ort.InferenceSession(sys.argv[1])" $path 2>$null | Out-Null
    return ($LASTEXITCODE -eq 0)
}

function Download-File($name, $urls, $outPath) {
    if ((Test-Path $outPath) -and -not $Force) {
        Write-Host "EXISTS: $name (use -Force to re-download)"
        return $true
    }
    foreach ($url in $urls) {
        Write-Host "Trying: $url"
        try {
            Invoke-WebRequest -Uri $url -OutFile $outPath -ErrorAction Stop
            $size = (Get-Item $outPath).Length
            Write-Host "OK: $name ($([math]::Round($size/1MB,1)) MB)"
            return $true
        } catch {
            Write-Host "FAIL: $($_.Exception.Message)"
        }
    }
    return $false
}

Write-Host "=== Downloading ML model files ==="

# 1) Prefer a ready-to-use ONNX if the upstream CDN still serves one.
if ((Test-Path $onnxPath) -and (Test-ValidOnnx $onnxPath) -and -not $Force) {
    Write-Host "EXISTS: nonescape-mini-v0.onnx (valid)"
} else {
    $gotOnnx = $false
    if (Download-File "nonescape-mini-v0.onnx" $onnxUrls $onnxPath) {
        if (Test-ValidOnnx $onnxPath) {
            $gotOnnx = $true
        } else {
            Write-Host "WARN: downloaded file is not a valid ONNX - discarding."
            Remove-Item $onnxPath -Force
        }
    }

    # 2) Fall back to the safetensors checkpoint and convert it locally.
    if (-not $gotOnnx) {
        if (-not (Download-File "nonescape-mini-v0.safetensors" $safetensorsUrls $safetensorsPath)) {
            Write-Host "`nFAILED to obtain a model checkpoint."
            Write-Host "  Manual link: https://huggingface.co/e3ntity/nonescape-v0"
            exit 1
        }
        Write-Host "Converting safetensors -> ONNX (offline; needs torch + torchvision)..."
        & python (Join-Path $PSScriptRoot 'convert-nonescape.py') --input $safetensorsPath --output $onnxPath
        if ($LASTEXITCODE -ne 0 -or -not (Test-ValidOnnx $onnxPath)) {
            Write-Host "`nONNX conversion failed."
            Write-Host "  Install deps: pip install torch torchvision onnx onnxruntime safetensors"
            exit 1
        }
    }
}

Write-Host "`nAll models ready."
