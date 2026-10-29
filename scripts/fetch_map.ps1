# Runs ON WINDOWS. Copies the newest map (.db and .ply) from the Jetson into .\maps
param([string]$JetsonIp = "172.16.56.49")

$key  = "$HOME\.ssh\id_ed25519_jetson"
$dest = Join-Path $PSScriptRoot "..\maps"
New-Item -ItemType Directory -Force $dest | Out-Null

$latest = ssh -i $key -o BatchMode=yes jetson@$JetsonIp "ls -t ~/jetson-work/maps/*.ply 2>/dev/null | head -n 1; ls -t ~/jetson-work/maps/*.db 2>/dev/null | head -n 1"
foreach ($f in $latest) {
    if ($f) { scp -i $key -o BatchMode=yes "jetson@${JetsonIp}:$f" $dest; Write-Host "fetched $f" }
}
Write-Host "Saved to $dest. Open the .ply in MeshLab or CloudCompare."
