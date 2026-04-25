# 📊 Evaluation Report: Qwen2.5-3B Itemset Extractor

**Model:** `OliverSlivka/qwen2.5-3b-itemset-extractor`  
**Base:** `Qwen/Qwen2.5-3B-Instruct`  
**Date:** January 19, 2026  
**Hardware:** Google Colab T4 GPU (15GB), 4-bit quantization

---

## 🎯 Executive Summary

| Metric | Value |
|--------|-------|
| **Datasets Tested** | 5 |
| **Avg Precision** | 20.0% |
| **Avg Recall** | 15.0% |
| **Avg F1 Score** | 17.1% |
| **Exact Matches** | 0/5 (0%) |
| **Successful JSON Parse** | 1/5 (20%) |
| **Avg Inference Time** | ~120s per dataset |

### ⚠️ Overall Assessment: **Model Needs Improvement**

---

## 📋 Detailed Results

| Dataset | Rows | Apriori | Model | TP | FP | FN | Precision | Recall | F1 | Time |
|---------|------|---------|-------|----|----|----|-----------| -------|-----|------|
| eval_01_tiny_grocery_5x4 | 5 | 1 | 0 | 0 | 0 | 1 | 0% | 0% | 0% | 52s |
| eval_02_tiny_letters_5x4 | 5 | 3 | 0 | 0 | 0 | 3 | 0% | 0% | 0% | 55s |
| eval_03_small_grocery_10x5 | 10 | 8 | 6 | 6 | 0 | 2 | **100%** | **75%** | **85.7%** | 161s |
| eval_04_small_fruits_10x5 | 10 | 8 | 0 | 0 | 0 | 8 | 0% | 0% | 0% | 164s |
| eval_05_medium_colors_15x6 | 15 | 17 | 0 | 0 | 0 | 17 | 0% | 0% | 0% | 164s |

---

## 🔍 Problem Analysis

### 1. **Output Format Issues** (Critical)
Model adds explanatory text before/after JSON:
```
Extracting frequent itemsets from the transactional data with minimum support count of 3.

Itemset in Row 1:
- items: [juice]
...

Resulting frequent itemsets:
[{"itemset": ["juice"], ...}]
```

**Expected:** Pure JSON array without any surrounding text.

### 2. **JSON Truncation** (Critical)
Responses are cut off mid-JSON, causing parse failures:
```json
{
  "itemset": ["banana"],
  "support": 2,
  "rows": ["Row 10"    <-- TRUNCATED HERE
```

**Cause:** MAX_NEW_TOKENS (512) may be insufficient, or generation stops prematurely.

### 3. **Hallucination/Repetition** (Severe)
Dataset 5 showed infinite repetition:
```
itemset:["green", "green", "green", "green", "green"...
```

**Cause:** Model enters repetition loop, common with smaller models.

### 4. **Extremely Slow Inference** (Major)
- 52-164 seconds per small dataset (5-15 rows)
- Expected: 5-15 seconds
- 10-30x slower than expected

**Possible causes:**
- 4-bit quantization overhead
- Inefficient generation parameters
- Model generating too much reasoning text

### 5. **Inconsistent Performance**
- Dataset 3 (grocery, 10 rows): 85.7% F1 ✅
- Dataset 4 (fruits, 10 rows): 0% F1 ❌
- Same size, different items → inconsistent behavior

---

## 📈 Comparison: Training vs Evaluation

| Metric | Training | Evaluation |
|--------|----------|------------|
| Loss | 1.36 → 0.35 ✅ | - |
| Token Accuracy | 93.6% ✅ | - |
| JSON Parse Rate | - | 20% ❌ |
| Avg F1 | - | 17% ❌ |
| Inference Speed | - | Very Slow ❌ |

**Gap Analysis:** Training metrics looked good but don't translate to real-world performance.

---

## 🔧 Root Causes

### 1. **Training Data Mismatch**
- Training used complex system prompts with XML tags
- Evaluation used simple prompts
- Model learned to generate explanatory text, not just JSON

### 2. **3B Model Limitations**
- 3B parameters may be insufficient for:
  - Understanding itemset mining
  - Generating structured JSON consistently
  - Handling varied item names

### 3. **Quantization Impact**
- 4-bit quantization reduces quality
- May contribute to hallucinations and repetition

### 4. **Insufficient Training Data Diversity**
- Training datasets had specific patterns
- Model struggles with unseen item distributions

---

## 💡 Recommendations

### Immediate Fixes
1. **Improve JSON Extraction**
   - Strip all text before first `[` and after last `]`
   - Handle truncated JSON gracefully

2. **Increase MAX_NEW_TOKENS**
   - Set to 2048 or higher
   - Add early stopping on `]` detection

3. **Add Repetition Penalty**
   ```python
   outputs = model.generate(
       ...,
       repetition_penalty=1.2,
       no_repeat_ngram_size=3,
   )
   ```

### Model Improvements
4. **Retrain with Simpler Prompts**
   - Remove XML system prompt complexity
   - Train on exact inference prompt format

5. **Consider Larger Base Model**
   - Qwen2.5-7B or 14B would perform better
   - Requires A100/H100 GPU for training

6. **Data Augmentation**
   - Add more diverse item types
   - Include edge cases (empty results, single items)

### Alternative Approaches
7. **Use GPT-4/Claude for Evaluation**
   - Compare fine-tuned model vs API models
   - Establish performance baseline

8. **Hybrid Approach**
   - Use fine-tuned model for initial extraction
   - Post-process with rule-based validation

---

## 📊 Visualization

```
Performance by Dataset Size:

5 rows:   ████░░░░░░░░░░░░░░░░ 0% F1 (2/2 failed)
10 rows:  ████████░░░░░░░░░░░░ 43% F1 (1/2 worked)
15 rows:  ░░░░░░░░░░░░░░░░░░░░ 0% F1 (0/1 worked)

JSON Parse Success:
█░░░░ 20% (1/5)

Inference Speed (seconds):
eval_01: ████████████████████████████████████████████████████ 52s
eval_02: ███████████████████████████████████████████████████████ 55s
eval_03: ████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████ 161s
eval_04: ████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████ 164s
eval_05: ████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████ 164s
```

---

## 🎓 Lessons Learned

1. **Training metrics ≠ Real performance**
   - Low loss doesn't guarantee good inference
   - Test during training, not just at the end

2. **Prompt format consistency is critical**
   - Training and inference prompts must match exactly
   - Small differences cause major output issues

3. **3B models have limits**
   - Complex structured output is challenging
   - Consider task complexity when choosing model size

4. **Quantization has tradeoffs**
   - Saves memory but impacts quality
   - Test quantized model specifically

---

## 🚀 Next Steps

1. [ ] Fix JSON extraction in evaluation code
2. [ ] Add repetition penalty and increase token limit
3. [ ] Re-run evaluation with fixes
4. [ ] If still failing: retrain with simpler prompts
5. [ ] Consider 7B model if 3B insufficient

---

*Report generated from evaluation run on January 19, 2026*
