# Runs ON WINDOWS. Takes ONE normal-camera image + ONE depth image from the D455 on the Jetson,
# copies them into .\shots, builds a smooth depth map you can MEASURE on, and opens it.
#   .\scripts\capture.ps1                 # hold the camera steady first, ~1 m from the object
#   .\scripts\capture.ps1 -Name drone
param([string]$JetsonIp = "172.16.56.49", [string]$Name = "", [string]$Res = "640x480", [int]$Fps = 10)

$key  = "$HOME\.ssh\id_ed25519_jetson"
$here = $PSScriptRoot
$dest = Join-Path $here "..\shots"
New-Item -ItemType Directory -Force $dest | Out-Null
if (-not $Name) { $Name = "shot_" + (Get-Date -Format "HHmmss") }

# Stop the live preview first if it is running: the camera can only be opened once.
scp -i $key -o BatchMode=yes "$here\jetson_capture.sh" "$here\jetson_capture.py" "jetson@${JetsonIp}:jetson-work/" | Out-Null
ssh -i $key -o BatchMode=yes jetson@$JetsonIp "sed -i 's/\r$//' ~/jetson-work/jetson_capture.sh ~/jetson-work/jetson_capture.py; chmod +x ~/jetson-work/jetson_capture.sh"

Write-Host "Capturing (about 15 seconds, keep the camera still)..."
ssh -i $key -o BatchMode=yes jetson@$JetsonIp "bash ~/jetson-work/jetson_capture.sh $Name $Res $Fps"

foreach ($suffix in "color.png", "depth.png", "depth_filled.png", "depth_mm.npy", "intrinsics.json") {
    scp -i $key -o BatchMode=yes "jetson@${JetsonIp}:jetson-work/shots/${Name}_${suffix}" $dest 2>$null
}

$prefix = Join-Path $dest $Name
if ((Test-Path "${prefix}_color.png") -and (Test-Path "${prefix}_depth_mm.npy")) {
    python "$here\make_measure_view.py" $prefix
    Write-Host "Saved to $dest :"
    Write-Host "  ${Name}_color.png         normal camera view"
    Write-Host "  ${Name}_depth.png         raw depth (near = bright, black = no reading)"
    Write-Host "  ${Name}_depth_filled.png  depth with gaps filled"
    Write-Host "  ${Name}_measure.html      smooth depth map: click two points to measure"
    Start-Process "${prefix}_measure.html"
} else {
    Write-Host "No images came back. Check the output above."
}
