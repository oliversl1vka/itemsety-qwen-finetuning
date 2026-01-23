# 💰 Setup Paid GPU for Training

## Why Paid GPU?

ZeroGPU has a **120-second limit** which is insufficient for:
- Test training: 10-15 minutes needed
- Full training: 40-60 minutes needed

**Solution**: Use persistent paid GPU on your Space.

## Steps

### 1. Update Space Hardware

1. Go to: https://huggingface.co/spaces/OliverSlivka/testrun2/settings
2. Scroll to **Hardware** section
3. Select one of:
   - **T4 small** - $0.60/hour (cheapest, sufficient for 3B model)
   - **A10G small** - $1.10/hour (faster, recommended)
4. Click **Update**
5. Wait for Space to restart (~1-2 minutes)

### 2. Cost Estimates

| Training | Duration | T4 small | A10G small |
|----------|----------|----------|------------|
| Test (50 examples) | ~15 min | $0.15 | $0.28 |
| Full (439 examples, 3 epochs) | ~60 min | $0.60 | $1.10 |
| **Total** | ~75 min | **$0.75** | **$1.38** |

### 3. Code Changes

The `app_v2.py` has been updated to remove `@spaces.GPU` decorator since we're using persistent GPU.

### 4. Billing

- HuggingFace charges per minute of GPU usage
- Billing starts when Space is active
- **Stop the Space** when not training to save money (Settings → Sleep)

## Training Workflow

1. **Update hardware** to T4/A10G small
2. **Wait for rebuild** (~2 min)
3. **Run test training** (15 min, verify everything works)
4. **Run full training** (60 min, get production model)
5. **Stop the Space** or downgrade hardware

## Alternative: HuggingFace Training Jobs

If you prefer not to use paid GPU Space, you can use HF Training Jobs:
- Same pricing
- Better monitoring
- Requires code restructuring

Let me know if you want guidance on HF Jobs instead.

## Current Status

✅ Code updated for paid GPU
✅ Deployment script ready
❌ Hardware not upgraded yet (manual step required)

**Next**: Go to Space settings and upgrade hardware!
