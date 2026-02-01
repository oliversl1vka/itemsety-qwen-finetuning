# Push updated README to HF Space

Write-Host "📝 Pushing README update (T4 GPU info)..." -ForegroundColor Cyan

$SPACE_DIR = "hf_space_testrun2"

Set-Location $SPACE_DIR

# Copy updated README
Copy-Item ..\README_SPACE.md README.md -Force

Write-Host "✅ README copied" -ForegroundColor Green

# Git operations
git add README.md

$status = git status --porcelain
if ([string]::IsNullOrWhiteSpace($status)) {
    Write-Host "⚠️ No changes to commit" -ForegroundColor Yellow
    Set-Location ..
    exit 0
}

git commit -m "docs: Update README to reflect T4 GPU (not Zero GPU)"

Write-Host "📤 Pushing to HF Space..." -ForegroundColor Cyan
git push

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ README updated on HF Space" -ForegroundColor Green
    Write-Host "🔗 https://huggingface.co/spaces/OliverSlivka/testrun2" -ForegroundColor Cyan
} else {
    Write-Host "❌ Push failed!" -ForegroundColor Red
}

Set-Location ..
