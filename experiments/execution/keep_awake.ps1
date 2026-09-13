# A task-scoped idle-sleep request; does not change the machine power plan.
# Remove the supplied marker to release it; also exits after 12 hours.
param([Parameter(Mandatory=$true)][string]$Marker)
Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
public static class RoboPADExperimentAwake {
    [DllImport("kernel32.dll", SetLastError = true)]
    public static extern uint SetThreadExecutionState(uint flags);
}
'@
$previousState = [RoboPADExperimentAwake]::SetThreadExecutionState([uint32]2147483649)
if ($previousState -eq 0) { throw 'Could not request task-scoped idle-sleep prevention.' }
$deadline = (Get-Date).AddHours(12)
Write-Output 'RoboPAD idle-sleep request active; display policy unchanged.'
try {
    while ((Test-Path -LiteralPath $Marker) -and (Get-Date) -lt $deadline) {
        Start-Sleep -Seconds 15
    }
} finally {
    [void][RoboPADExperimentAwake]::SetThreadExecutionState([uint32]2147483648)
    Write-Output 'RoboPAD idle-sleep request released.'
}
