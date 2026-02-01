# Push bitsandbytes fix to HF Space
Write-Host "🔧 Pushing bitsandbytes fix..." -ForegroundColor Cyan

cd C:\Users\slivk\Desktop\itemsety_real_training\hf_space

# Copy updated requirements
Copy-Item ..\itemsety\requirements.txt . -Force
Write-Host "✓ Copied requirements.txt" -ForegroundColor Green

# Commit and push
git add requirements.txt
git commit -m "fix: Add bitsandbytes>=0.41.0 to requirements.txt"
git push

Write-Host "`n✅ Pushed! Space will rebuild (~2-3 minutes)" -ForegroundColor Green
Write-Host "`n⏳ Wait for rebuild, then retry Test Training" -ForegroundColor Yellow
