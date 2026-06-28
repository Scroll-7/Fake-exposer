# All 32 known face images already exist in known_faces/.
# This script is kept as a reference for re-downloading if needed.
# Usage: .\download_faces.ps1

$knownDir = Join-Path $PSScriptRoot "known_faces"
New-Item -ItemType Directory -Force -Path $knownDir | Out-Null

# Football players (already downloaded, kept for completeness)
$players = @{
    "lionel_messi"       = "https://upload.wikimedia.org/wikipedia/commons/c/c1/Lionel_Messi_20180626.jpg"
    "cristiano_ronaldo"  = "https://upload.wikimedia.org/wikipedia/commons/d/d7/Cristiano_Ronaldo_playing_for_Al_Nassr_FC_in_2023.jpg"
    "neymar"             = "https://upload.wikimedia.org/wikipedia/commons/b/b4/Neymar_Jr._with_PSG_in_2019.jpg"
    "kylian_mbappe"      = "https://upload.wikimedia.org/wikipedia/commons/8/86/Kylian_Mbapp%C3%A9_%C3%A9chauffe_avant_RCSA-PSG_%28cropped%29.jpg"
    "mohamed_salah"      = "https://upload.wikimedia.org/wikipedia/commons/f/f2/Mohamed_Salah%2C_Liverpool_FC.jpg"
}

# Celebrities for deepfake detection coverage
$celebrities = @{
    "elon_musk"          = "https://upload.wikimedia.org/wikipedia/commons/4/49/Elon_Musk_2015.jpg"
    "taylor_swift"       = "https://upload.wikimedia.org/wikipedia/commons/4/49/Taylor_Swift_2_-_2019_by_Glenn_Francis.jpg"
    "donald_trump"       = "https://upload.wikimedia.org/wikipedia/commons/1/1e/January_2025_Official_Presidential_Portrait_of_Donald_J._Trump.jpg"
    "tom_cruise"         = "https://upload.wikimedia.org/wikipedia/commons/3/33/Tom_Cruise_by_Gage_Skidmore_2.jpg"
    "joe_biden"          = "https://upload.wikimedia.org/wikipedia/commons/9/97/Joe_Biden_official_portrait.jpg"
    "barack_obama"       = "https://upload.wikimedia.org/wikipedia/commons/9/9d/Official_portrait_of_Barack_Obama.jpg"
    "the_rock"           = "https://upload.wikimedia.org/wikipedia/commons/7/7e/Dwayne_The_Rock_Johnson_2009_portrait.jpg"
}

Write-Host "=== Checking existing football player images ==="
foreach ($name in $players.Keys) {
    $path = Join-Path $knownDir "$name.jpg"
    if (Test-Path $path) {
        Write-Host "EXISTS: $name (skipping)"
    } else {
        try {
            $url = $players[$name]
            Write-Host "DOWNLOADING: $name ..."
            Invoke-WebRequest -Uri $url -OutFile $path -ErrorAction Stop
            Write-Host "OK: $name"
            Start-Sleep -Seconds 3
        } catch {
            Write-Host "FAIL: $name - $($_.Exception.Message)"
        }
    }
}

Write-Host ""
Write-Host "=== Downloading celebrity faces (deepfake detection) ==="
foreach ($name in $celebrities.Keys) {
    $path = Join-Path $knownDir "$name.jpg"
    if (Test-Path $path) {
        Write-Host "EXISTS: $name (skipping)"
    } else {
        try {
            $url = $celebrities[$name]
            Write-Host "DOWNLOADING: $name ..."
            Invoke-WebRequest -Uri $url -OutFile $path -ErrorAction Stop
            Write-Host "OK: $name"
            Start-Sleep -Seconds 5
        } catch {
            Write-Host "FAIL: $name - $($_.Exception.Message)"
        }
    }
}

Write-Host ""
Write-Host "=== Final count ==="
$files = Get-ChildItem -Path $knownDir -Filter "*.jpg"
Write-Host "$($files.Count) face images in known_faces/"
