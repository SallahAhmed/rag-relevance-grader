# Autonomous launcher: waits for T4 capacity, then runs the full pipeline.
# Safe to fire-and-forget. Uploads the HF token BEFORE training starts, because
# an adapter that cannot be pushed is an adapter that dies with the VM.
# ASCII only: PowerShell 5.1 reads this file as ANSI.
$ErrorActionPreference = "Continue"
$colab = "/home/salla/miniconda3/bin/colab"
$sess  = "pb-run2"
$log   = "F:\finetune\logs\autorun.log"

function Say($m) {
  $line = "[autorun {0}] {1}" -f (Get-Date -Format "HH:mm:ss"), $m
  Add-Content -Path $log -Value $line
}

Say "start; waiting for T4 capacity"

$got = $false
for ($i = 1; $i -le 45; $i++) {
  $out = wsl $colab new -s $sess --gpu T4 2>&1 | Out-String
  if ($out -match "READY") { Say "T4 READY (attempt $i)"; $got = $true; break }
  if ($out -match "400") { Say "no T4 entitlement (400) - aborting"; exit 3 }
  Say "no capacity (attempt $i); sleeping 100s"
  Start-Sleep -Seconds 100
}
if (-not $got) { Say "gave up waiting for capacity"; exit 4 }

$files = @(
  @("/mnt/f/finetune/src/__init__.py",             "/content/pb03/src/__init__.py"),
  @("/mnt/f/finetune/src/config.py",               "/content/pb03/src/config.py"),
  @("/mnt/f/finetune/src/dataset_utils.py",        "/content/pb03/src/dataset_utils.py"),
  @("/mnt/f/finetune/src/evaluation_utils.py",     "/content/pb03/src/evaluation_utils.py"),
  @("/mnt/f/finetune/src/grader_interface.py",     "/content/pb03/src/grader_interface.py"),
  @("/mnt/f/finetune/src/rag_pipeline.py",         "/content/pb03/src/rag_pipeline.py"),
  @("/mnt/f/finetune/config/sft_config.yaml",      "/content/pb03/config/sft_config.yaml"),
  @("/mnt/f/finetune/notebooks/03_fine_tuning.py","/content/pb03/job03.py"),
  @("/mnt/f/finetune/notebooks/04_evaluation_comparison.py","/content/pb03/job04.py"),
  @("/mnt/f/finetune/scripts/colab_pipeline_v2.py","/content/pb03/pipe.py")
)
foreach ($f in $files) {
  $r = wsl $colab upload -s $sess $f[0] $f[1] 2>&1 | Out-String
  if ($r -match "Uploaded") { Say ("uploaded " + $f[1]) } else { Say ("UPLOAD FAIL " + $f[1]) }
}

$r = wsl $colab upload -s $sess "/mnt/f/finetune/.hfkey.tmp" "/content/.hfkey" 2>&1 | Out-String
if ($r -match "Uploaded") { Say "TOKEN uploaded to /content/.hfkey" } else { Say "TOKEN UPLOAD FAILED"; exit 5 }

Say "launching pipeline: data -> train -> push -> eval"
$argl = @($colab,"exec","-s",$sess,"-f","/mnt/f/finetune/scripts/colab_pipeline_v2.py","--timeout","14400")
$p = Start-Process -FilePath "wsl" -ArgumentList $argl -RedirectStandardOutput "F:\finetune\logs\pipeline2.out" -RedirectStandardError "F:\finetune\logs\pipeline2.err" -NoNewWindow -PassThru
Say "pipeline pid $($p.Id)"
$p.WaitForExit()
Say "pipeline exited with $($p.ExitCode)"
Say "DONE"
