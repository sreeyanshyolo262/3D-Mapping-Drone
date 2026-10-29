# Runs ON WINDOWS, after the Jetson has recorded a map (jetson_map.sh).
# One command: fetch the newest map from the Jetson -> build the 3D world + a depth image -> open it.
#   .\scripts\build_world.ps1                    # fetch from the Jetson, then build
#   .\scripts\build_world.ps1 -NoFetch           # just rebuild from the newest map already in .\maps
#   .\scripts\build_world.ps1 -Ply .\maps\map_xxx_cloud.ply
param([string]$JetsonIp = "172.16.56.49", [switch]$NoFetch, [string]$Ply = "", [int]$Points = 2000000)

$here = $PSScriptRoot
$maps = Join-Path $here "..\maps"

if (-not $Ply) {
    if (-not $NoFetch) { & "$here\fetch_map.ps1" -JetsonIp $JetsonIp }
    $newest = Get-ChildItem $maps -Filter "*_cloud.ply" -ErrorAction SilentlyContinue |
              Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if (-not $newest) { Write-Host "No *_cloud.ply found in $maps"; exit 1 }
    $Ply = $newest.FullName
}
Write-Host "Building from $Ply"

python "$here\make_bw_world.py" $Ply $Points        # 3D world (orbit, zoom, click-to-measure)
python "$here\make_depth_map.py" $Ply               # smooth depth image from the same cloud

$base = $Ply -replace "_cloud\.ply$", ""
Write-Host ""
Write-Host "Done:"
Write-Host "  ${base}_world.html       3D world in real colour: drag = orbit, wheel = zoom, Measure button = distances"
Write-Host "  ${base}_depth.png        smooth depth image (near = bright)"
Write-Host "  ${base}_depth_color.png  same, colour-coded"
Start-Process "${base}_world.html"
