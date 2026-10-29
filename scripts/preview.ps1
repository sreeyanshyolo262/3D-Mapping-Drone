# Runs ON WINDOWS. Starts the live camera preview on the Jetson and opens it in your browser.
# Press Ctrl+C in this window to stop the preview (and the camera).
param([string]$JetsonIp = "172.16.56.49", [string]$ColorRes = "640x480", [string]$DepthRes = "640x480", [int]$Fps = 10)

$key = "$HOME\.ssh\id_ed25519_jetson"
$here = $PSScriptRoot

# Copy the latest scripts to the Jetson so they are always in sync with this folder.
scp -i $key -o BatchMode=yes "$here\jetson_preview.sh" "$here\jetson_preview.py" "jetson@${JetsonIp}:jetson-work/" | Out-Null
ssh -i $key -o BatchMode=yes jetson@$JetsonIp "sed -i 's/\r$//' ~/jetson-work/jetson_preview.sh ~/jetson-work/jetson_preview.py; chmod +x ~/jetson-work/jetson_preview.sh"

# Open the browser after the camera has had time to start.
Start-Job -ScriptBlock { param($ip) Start-Sleep -Seconds 14; Start-Process "http://${ip}:8080" } -ArgumentList $JetsonIp | Out-Null

# -t gives a terminal so Ctrl+C reaches the script on the Jetson and shuts the camera down cleanly.
ssh -t -i $key jetson@$JetsonIp "bash ~/jetson-work/jetson_preview.sh $ColorRes $DepthRes $Fps"
