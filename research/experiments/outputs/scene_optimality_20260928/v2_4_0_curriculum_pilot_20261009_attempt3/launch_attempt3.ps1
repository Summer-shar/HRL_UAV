$ErrorActionPreference = 'Stop'
$cpil = 'D:/HRL_UAV_V2/outputs/scene_optimality_20260928/v2_4_0_curriculum_pilot_20261009_attempt3'
$wt = 'D:/HRL_UAV_V2_speedup'
$py = 'D:/HRL_UAV_V2/.venv/Scripts/python.exe'
if (Test-Path -LiteralPath (Join-Path $cpil 'start.json')) { throw 'Already launched; refuse duplicate launch.' }
if (Test-Path -LiteralPath (Join-Path $cpil 'failure.json')) { throw 'Failure exists; do not auto-restart.' }
foreach ($name in @('cp_contract.json','cp_contract.sha256','step0_facts/facts.json','output_guard_tests.json','rp_reuse_manifest.json')) { if (-not (Test-Path -LiteralPath (Join-Path $cpil $name))) { throw ('Missing launch prerequisite: '+$name) } }
$facts = Get-Content -LiteralPath (Join-Path $cpil 'step0_facts/facts.json') -Raw | ConvertFrom-Json
if ($facts.status -ne 'PASS') { throw 'Step0 is not PASS.' }
$guard = Get-Content -LiteralPath (Join-Path $cpil 'output_guard_tests.json') -Raw | ConvertFrom-Json
if ($guard.status -ne 'PASS') { throw 'Output path guard tests are not PASS.' }
$contractHash = (Get-FileHash -LiteralPath (Join-Path $cpil 'cp_contract.json') -Algorithm SHA256).Hash.ToLowerInvariant()
$sidecar = (Get-Content -LiteralPath (Join-Path $cpil 'cp_contract.sha256') -Raw).Trim().Split(' ')[0]
if ($contractHash -ne $sidecar) { throw 'Contract sidecar mismatch.' }
$running = @(Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -like '*v2_4_0_curriculum_pilot_20261009_attempt3*' })
if ($running.Count -ne 0) { throw 'An attempt3 Python process already exists.' }
$env:PYTHONPATH = Join-Path $wt 'src'
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:MPLCONFIGDIR = Join-Path $cpil 'mplconfig'
$env:GIT_OPTIONAL_LOCKS = '0'
$stdout = Join-Path $cpil 'logs/main.stdout.log'
$stderr = Join-Path $cpil 'logs/main.stderr.log'
$script = Join-Path $cpil 'cp_main.py'
$launchTime = Get-Date -Format o
$child = Start-Process -FilePath $py -ArgumentList @('-u',$script) -WorkingDirectory $wt -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
$priorityResult = 'BelowNormal'
try { $child.PriorityClass = 'BelowNormal' } catch { $priorityResult = $_.Exception.Message }
[pscustomobject]@{started_at=$launchTime;pid=$child.Id;script=$script;contract_sha256=$contractHash;stdout=$stdout;stderr=$stderr;priority=$priorityResult;seed=4104;monitor_automation_id='seed-4104-attempt3';previous_failed_attempt='D:/HRL_UAV_V2/outputs/scene_optimality_20260928/v2_4_0_curriculum_pilot_20261008_attempt2';expected_sequence='preflight,verify RP reuse,DG,S1,D-a,S2,S3,H,S4,S5,F,D-b,report'} | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $cpil 'start.json') -Encoding UTF8
Get-Content -LiteralPath (Join-Path $cpil 'start.json') -Raw
