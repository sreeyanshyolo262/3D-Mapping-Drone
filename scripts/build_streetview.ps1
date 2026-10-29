# Runs ON WINDOWS, after a stop-and-snap session on the Jetson (jetson_snap.sh).
# Fetches the newest snap session, joins the snaps into one 3D map, and opens the walk-through page.
#   .\scripts\build_streetview.ps1                 # fetch newest session from the Jetson, then build
#   .\scripts\build_streetview.ps1 -NoFetch        # rebuild from the newest session already in .\snaps
#   .\scripts\build_streetview.ps1 -Session .\snaps\snaps_20261020_150000
#   .\scripts\build_streetview.ps1 -NoLoops        # skip the drift correction
param([string]$JetsonIp = "172.16.56.49", [switch]$NoFetch, [string]$Session = "", [switch]$NoLoops)

$key  = "$HOME\.ssh\id_ed25519_jetson"
$here = $PSScriptRoot
$dest = Join-Path $here "..\snaps"

# One-time: OpenCV is needed to match the snaps against each other.
python -c "import cv2" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Installing OpenCV (one time)..."
    python -m pip install --quiet opencv-python-headless
}

if (-not $Session) {
    if (-not $NoFetch) {
        $remote = (ssh -i $key -o BatchMode=yes jetson@$JetsonIp "ls -dt ~/jetson-work/snaps/snaps_* 2>/dev/null | head -n 1") | Select-Object -First 1
        if (-not $remote) { Write-Host "No snap sessions found on the Jetson."; exit 1 }
        $remote = $remote.Trim()
        Write-Host "Fetching $remote ..."
        New-Item -ItemType Directory -Force $dest | Out-Null
        scp -i $key -o BatchMode=yes -r "jetson@${JetsonIp}:$remote" $dest
    }
    $newest = Get-ChildItem $dest -Directory -Filter "snaps_*" -ErrorAction SilentlyContinue |
              Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if (-not $newest) { Write-Host "No session folder in $dest"; exit 1 }
    $Session = $newest.FullName
}
Write-Host "Building from $Session"

$extra = @()
if ($NoLoops) { $extra += "--no-loops" }
python "$here\make_streetview.py" $Session @extra
if ($LASTEXITCODE -ne 0) { exit 1 }

Write-Host ""
Write-Host "Opening the walk-through:  $Session\streetview.html"
Start-Process (Join-Path $Session "streetview.html")
