param(
    [switch]$Force
)

$root = Split-Path $PSScriptRoot -Parent
$files = @{
    "nonescape-mini-v0.onnx" = @(
        "https://nonescape.sfo2.cdn.digitaloceanspaces.com/nonescape-mini-v0.onnx",
        "https://huggingface.co/e3ntity/nonescape-v0/resolve/main/nonescape-mini-v0.safetensors"
    )
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
$ok = $true
foreach ($name in $files.Keys) {
    $outPath = Join-Path $root $name
    if (-not (Download-File $name $files[$name] $outPath)) {
        Write-Host "FAILED: $name"
        $ok = $false
    }
}

if (-not $ok) {
    Write-Host "`nSome downloads failed. Manual download links:"
    Write-Host "  nonescape-mini-v0.onnx: https://github.com/e3ntity/nonescape"
    Write-Host "  Place the file(s) in: $root"
    exit 1
}

Write-Host "`nAll models downloaded."
