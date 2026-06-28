$log = "$env:TEMP\opencode\server_out.txt"
$proc = Start-Process -FilePath "node" -ArgumentList "server.js" -WorkingDirectory "$PSScriptRoot" -PassThru -NoNewWindow -RedirectStandardOutput $log -RedirectStandardError $log
$proc.Id | Out-File "$env:TEMP\opencode\server_pid.txt"
Write-Output "Started node PID: $($proc.Id)"
