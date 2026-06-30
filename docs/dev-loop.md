# Dev loop

Optional agent loop for local verification:

```text
/loop 30m run cargo test --workspace && pytest
```

PowerShell equivalent:

```powershell
while ($true) {
  Start-Sleep -Seconds 1800
  Write-Output 'AGENT_LOOP_TICK_mmn {"prompt":"Run cargo test --workspace and pytest for MagicMindNet"}'
}
```
