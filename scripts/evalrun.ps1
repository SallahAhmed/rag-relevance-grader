# Waits for T4, then evaluates the published adapter and pushes results back.
# ASCII only (PowerShell 5.1 reads as ANSI).
#
# The test split is UPLOADED from data/processed/test.parquet, not rebuilt
# remotely: rebuilding re-runs the query-level split and any drift would grade
# the adapter on a different query set than the baseline. eval_remote.py's
# verify_test() re-checks the counts and the qid set before scoring anything.
$ErrorActionPreference = "Continue"
$colab = "/home/salla/miniconda3/bin/colab"
$sess  = "pb-eval"
$log   = "F:\finetune\logs\evalrun.log"

function Say($m) {
  Add-Content -Path $log -Value ("[eval {0}] {1}" -f (Get-Date -Format "HH:mm:ss"), $m)
}

Say "waiting for T4 capacity"
$got = $false
for ($i = 1; $i -le 60; $i++) {
  $out = wsl $colab new -s $sess --gpu T4 2>&1 | Out-String
  if ($out -match "READY") { Say "T4 READY (attempt $i)"; $got = $true; break }
  if ($out -match "400") { Say "no T4 entitlement (400) - abort"; exit 3 }
  Say ("no capacity (attempt $i); sleeping 90s")
  Start-Sleep -Seconds 90
}
if (-not $got) { Say "gave up"; exit 4 }

# mkdir BEFORE upload - uploading into a missing dir is a silent 500
$mk = @"
import pathlib
for p in ['/content/pb04/src','/content/pb04/config','/content/pb04/in','/content/pb04/out','/content/pb04/adapter']:
    pathlib.Path(p).mkdir(parents=True, exist_ok=True)
print('DIRS_OK')
"@
$m = wsl $colab exec -s $sess --timeout 90 $mk 2>&1 | Out-String
if ($m -match "DIRS_OK") { Say "dirs created" } else { Say "DIR FAIL"; Say $m; exit 5 }

$files = @(
  @("/mnt/f/finetune/src/__init__.py",          "/content/pb04/src/__init__.py"),
  @("/mnt/f/finetune/src/config.py",            "/content/pb04/src/config.py"),
  @("/mnt/f/finetune/src/dataset_utils.py",     "/content/pb04/src/dataset_utils.py"),
  @("/mnt/f/finetune/src/evaluation_utils.py",  "/content/pb04/src/evaluation_utils.py"),
  @("/mnt/f/finetune/src/mmlu_utils.py",        "/content/pb04/src/mmlu_utils.py"),
  @("/mnt/f/finetune/src/grader_interface.py",  "/content/pb04/src/grader_interface.py"),
  @("/mnt/f/finetune/config/sft_config.yaml",   "/content/pb04/config/sft_config.yaml"),
  @("/mnt/f/finetune/notebooks/04_evaluation_comparison.py","/content/pb04/job04.py"),
  @("/mnt/f/finetune/data/processed/test.parquet",     "/content/pb04/in/test.parquet"),
  @("/mnt/f/finetune/results/predictions_baseline.json","/content/pb04/baseline_predictions.json")
)
foreach ($f in $files) {
  $r = wsl $colab upload -s $sess $f[0] $f[1] 2>&1 | Out-String
  if ($r -match "Uploaded") { Say ("up " + $f[1]) } else { Say ("UPFAIL " + $f[1] + " :: " + $r) }
}

$r = wsl $colab upload -s $sess "/mnt/f/finetune/.hfkey.tmp" "/content/.hfkey" 2>&1 | Out-String
if ($r -match "Uploaded") { Say "token up" } else { Say "TOKEN FAIL"; exit 6 }

Say "running eval: fetch adapter from Hub, grade held-out queries, run MMLU check"
$a = @($colab,"exec","-s",$sess,"-f","/mnt/f/finetune/scripts/eval_remote.py","--timeout","14400")
$p = Start-Process -FilePath "wsl" -ArgumentList $a -RedirectStandardOutput "F:\finetune\logs\eval.out" -RedirectStandardError "F:\finetune\logs\eval.err" -NoNewWindow -PassThru
Say "eval pid $($p.Id)"
$p.WaitForExit()
Say "eval exit $($p.ExitCode)"
Say "DONE"