# Deploy to HuggingFace Space
# Run this script to deploy training app to Zero GPU Space

$ErrorActionPreference = "Stop"

Write-Host "🚀 Deploying to HuggingFace Space..." -ForegroundColor Green

# 1. Setup paths
$projectRoot = "C:\Users\slivk\Desktop\itemsety_real_training"
$itemsetyDir = Join-Path $projectRoot "itemsety"
$spaceDir = Join-Path $projectRoot "hf_space"

# 2. Clone Space if not exists
if (-not (Test-Path $spaceDir)) {
    Write-Host "`n📥 Cloning HuggingFace Space..." -ForegroundColor Cyan
    Set-Location $projectRoot
    git clone https://huggingface.co/spaces/OliverSlivka/testrun2 hf_space
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Failed to clone Space" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "`n✅ Space already cloned" -ForegroundColor Green
}

# 3. Copy files
Write-Host "`n📄 Copying files..." -ForegroundColor Cyan
Set-Location $spaceDir

Copy-Item (Join-Path $itemsetyDir "app_v2.py") .\app.py -Force
Write-Host "   ✓ app.py (from app_v2.py)"

Copy-Item (Join-Path $itemsetyDir "run_sft_test.py") . -Force
Write-Host "   ✓ run_sft_test.py"

Copy-Item (Join-Path $itemsetyDir "run_sft_full.py") . -Force
Write-Host "   ✓ run_sft_full.py"

Copy-Item (Join-Path $itemsetyDir "requirements.txt") . -Force
Write-Host "   ✓ requirements.txt"

Copy-Item (Join-Path $itemsetyDir "README_SPACE.md") .\README.md -Force
Write-Host "   ✓ README.md"

# 4. Git add, commit, push
Write-Host "`n📦 Committing changes..." -ForegroundColor Cyan
git add app.py run_sft_test.py run_sft_full.py requirements.txt README.md

$gitStatus = git status --short
if ($gitStatus) {
    Write-Host "Changes to commit:"
    Write-Host $gitStatus
    
    git commit -m "feat: Add Qwen2.5-3B training (test + full modes)"
    
    Write-Host "`n🔼 Pushing to HuggingFace..." -ForegroundColor Cyan
    git push
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "`n🎉 Deployment successful!" -ForegroundColor Green
        Write-Host "`nSpace URL: https://huggingface.co/spaces/OliverSlivka/testrun2" -ForegroundColor Yellow
        Write-Host "`nNext steps:" -ForegroundColor Cyan
        Write-Host "1. Wait for Space to rebuild (~2-3 minutes)"
        Write-Host "2. Open the Space URL above"
        Write-Host "3. Click 'Submit' to start test training"
        Write-Host "4. Monitor logs for ~10-20 minutes"
    } else {
        Write-Host "`n❌ Push failed!" -ForegroundColor Red
        Write-Host "Check your HuggingFace credentials and try again."
        exit 1
    }
} else {
    Write-Host "`n✅ No changes to commit (already up-to-date)" -ForegroundColor Green
}

Write-Host "`n✨ Done!" -ForegroundColor Green
