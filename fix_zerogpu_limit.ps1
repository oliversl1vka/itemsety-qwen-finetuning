# Quick fix for ZeroGPU duration limit
$ErrorActionPreference = "Stop"

Write-Host "🔧 Fixing ZeroGPU duration limit..." -ForegroundColor Cyan

cd C:\Users\slivk\Desktop\itemsety_real_training\hf_space

# Fix app.py - change 7200 to 120
$content = Get-Content app.py -Raw
$content = $content -replace 'duration=7200', 'duration=120'
$content = $content -replace '# Set a 2-hour duration', '# ZeroGPU max: 120 seconds'
Set-Content app.py $content -NoNewline

Write-Host "✓ Updated app.py duration to 120s" -ForegroundColor Green

# Show diff
Write-Host "`n📝 Changes:" -ForegroundColor Yellow
git diff app.py | Select-Object -First 20

# Commit and push
git add app.py
git commit -m "fix: Set ZeroGPU duration to 120s (max allowed limit)"
git push

Write-Host "`n✅ Fixed and pushed!" -ForegroundColor Green
Write-Host "`n⚠️  WARNING: 120s is only good for SMOKE TESTS" -ForegroundColor Yellow
Write-Host "   Real training needs 10-60 minutes → Use PAID GPU!" -ForegroundColor Yellow
Write-Host "`n📖 See PAID_GPU_SETUP.md for next steps" -ForegroundColor Cyan
