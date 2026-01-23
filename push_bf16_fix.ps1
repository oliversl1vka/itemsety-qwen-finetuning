# Push bfloat16 fix to HF Space

Write-Host "🔧 Pushing bf16=False fix to HF Space..." -ForegroundColor Cyan

$SPACE_DIR = "hf_space_testrun2"

# Check if space directory exists
if (-not (Test-Path $SPACE_DIR)) {
    Write-Host "❌ Space directory not found. Cloning..." -ForegroundColor Red
    git clone https://huggingface.co/spaces/OliverSlivka/testrun2 $SPACE_DIR
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Clone failed!" -ForegroundColor Red
        exit 1
    }
}

# Navigate to space directory
Set-Location $SPACE_DIR

# Copy updated training scripts
Copy-Item ..\run_sft_test.py . -Force
Copy-Item ..\run_sft_full.py . -Force

Write-Host "✅ Files copied" -ForegroundColor Green

# Git operations
git add run_sft_test.py run_sft_full.py

$status = git status --porcelain
if ([string]::IsNullOrWhiteSpace($status)) {
    Write-Host "⚠️ No changes to commit" -ForegroundColor Yellow
    Set-Location ..
    exit 0
}

git commit -m "fix: Explicitly disable bf16 for T4 GPU compatibility"

Write-Host "📤 Pushing to HF Space..." -ForegroundColor Cyan
git push

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Push successful! Space will rebuild in ~2-3 minutes" -ForegroundColor Green
    Write-Host "🔗 Check status: https://huggingface.co/spaces/OliverSlivka/testrun2" -ForegroundColor Cyan
} else {
    Write-Host "❌ Push failed!" -ForegroundColor Red
}

Set-Location ..
