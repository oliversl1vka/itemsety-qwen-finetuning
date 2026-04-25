# Evaluation Results - Fine-tuned Qwen2.5-0.5B on Itemset Extraction

## Test Configuration
- **Model**: Qwen/Qwen2.5-0.5B-Instruct + LoRA adapter (OliverSlivka/qwen-itemsety-qlora)
- **Training**: 88 train / 10 validation examples, 3 epochs, 4-bit QLoRA
- **Test Environment**: CPU (Windows), FP16
- **Date**: December 17, 2025

## Datasets Tested (First 9)

| Dataset | Rows | Cols | Status | Time | Valid JSON | Itemsets | Notes |
|---------|------|------|--------|------|------------|----------|-------|
| ds_0001_5x53 | 5 | 53 | ✅ Tested | 74.3s | ✓ YES | 2 | Hallucinated items |
| ds_0002_50x93 | 50 | 93 | ⏭️ Skipped | - | - | - | Too large (CPU) |
| ds_0003_80x43 | 80 | 43 | ⏭️ Skipped | - | - | - | Too large (CPU) |
| ds_0004_31x19 | 31 | 19 | ⏭️ Skipped | - | - | - | Too large (CPU) |
| ds_0005_87x93 | 87 | 93 | ⏭️ Skipped | - | - | - | Too large (CPU) |
| ds_0006_28x100 | 28 | 100 | ✅ Tested | 973.2s | ✗ NO | 0 | Invalid JSON (typos) |
| ds_0007_74x99 | 74 | 99 | ⏭️ Skipped | - | - | - | Too large (CPU) |
| ds_0008_69x45 | 69 | 45 | ⏭️ Skipped | - | - | - | Too large (CPU) |
| ds_0009_82x27 | 82 | 27 | ⏭️ Skipped | - | - | - | Too large (CPU) |

**Summary**: 2/9 tested (7 skipped due to CPU limitation), 1/2 valid JSON

---

## Key Findings

### ✅ Positive Observations

1. **JSON Structure Learned**
   - Model outputs JSON array format consistently
   - Understands object structure with itemset, count, evidence/explanation fields
   - No markdown wrappers or extra text (mostly)

2. **Format Compliance**
   - Attempts to follow system prompt instructions
   - Uses lowercase canonicalization in some cases
   - Tries to provide explanations

3. **Inference Speed**
   - Small dataset (5 rows): ~74s on CPU
   - Medium dataset (28 rows): ~973s on CPU
   - Shows model can generate responses (not completely broken)

### ❌ Critical Issues

#### 1. **Hallucination of Items** (ds_0001)
**Problem**: Model generates item names that don't exist in the CSV
```json
{
  "itemset": ["niwere", "wali", "kawo", "gaba", "bogbo", "mabudu", "daga", "tumama", "zakawa", "sagabo", "pembe"],
  "proof": "1"
}
```
- **Real columns**: `attr_1_float`, `attr_2_int`, `attr_3_text`, ... `attr_53_int`
- **Model output**: Made-up words like "niwere", "wali", "kawo" (not in dataset)

**Root Cause**: Model doesn't understand CSV structure - treats all text as potential items instead of parsing column headers/values

#### 2. **Invalid JSON Syntax** (ds_0006)
**Problem**: Typos in field names break JSON parsing
```json
{
  "vidence": "",          // Should be "evidence"
  "explaination": "..."   // Should be "explanation"
}
```
- Model learned approximate structure but not exact field names
- Missing commas/formatting issues in some outputs

#### 3. **Incorrect Field Names**
**Expected**: `evidence_rows` or `evidence_transactions`
**Got**: `proof`, `vidence`, `observed_support_count`

Model learned concept but not precise schema from training data

#### 4. **Repetitive/Nonsensical Itemsets** (ds_0006)
```json
{
  "itemset": ["jobezu", "jobezu", "jobezu", "jobezu"],
  "observed_support_count": 3
}
```
- Repeats same item multiple times (invalid for itemset)
- Doesn't understand set semantics (unique items only)

#### 5. **Wrong Support Count Logic**
```json
{
  "observed_support_count": 2,
  "explaination": "...is not explicitly observed more than once; hence it does not count towards the count criterion."
}
```
- Count is 2 (below threshold of 3) but still reports it
- Contradictory explanation acknowledges it shouldn't count

---

## Analysis & Insights

### Why Model Fails

1. **Too Small Model (0.5B parameters)**
   - Insufficient capacity to learn CSV parsing + itemset logic simultaneously
   - Can't hold complex prompt context (2000+ tokens) effectively

2. **Insufficient Training Data**
   - Only 88 examples may not be enough for generalization
   - Synthetic data (random column names) might have confused the model
   - Model memorized output format but not reasoning

3. **Task Complexity**
   - Multi-step reasoning: parse CSV → identify items → count occurrences → filter by support → format JSON
   - Small model struggles with this pipeline

4. **Real-world Context Not Applied**
   - Despite dataset enhancement (grocery/electronics mapping), model still hallucinates
   - Enhancement may not have been strong enough or examples too few

### What Model DID Learn

✅ **Output structure**: JSON array with objects
✅ **Field concepts**: itemset (array), count (number), explanation (string)
✅ **Canonicalization attempt**: Lowercase item names
✅ **Following system prompt**: No extra markdown/text

❌ **CSV parsing**: Completely fails
❌ **Item identification**: Hallucinates non-existent items
❌ **Support counting**: Incorrect logic
❌ **Set semantics**: Allows duplicate items
❌ **Schema compliance**: Wrong field names (proof vs evidence_rows)

---

## Recommendations for Improvement

### Immediate Actions

1. **Use Larger Model**
   - Try Qwen2.5-1.5B or Qwen2.5-3B (still fits on A10G with 4-bit)
   - More parameters = better reasoning capacity

2. **More Training Data**
   - Increase from 88 to 200-500 examples
   - Include more variety in dataset sizes (5-30 rows)

3. **Better Examples**
   - Use real CSVs with actual item names (not randomized attr_N)
   - Show explicit CSV parsing in chain-of-thought examples
   - Include more validation examples (current 10 → 50)

### Training Improvements

4. **Curriculum Learning**
   - Start with simple 2-3 row CSVs
   - Gradually increase complexity
   - Fine-tune in stages

5. **Enhanced Prompt Engineering**
   - Add step-by-step reasoning in training examples
   - Include "thinking" section showing CSV parsing process
   - Emphasize: "ONLY use column values from CSV, do NOT invent items"

6. **Validation During Training**
   - Check if model outputs contain items from input CSV
   - Add constraint: hallucination detection in validation

### Alternative Approaches

7. **Two-Stage Model**
   - Stage 1: Fine-tune for CSV parsing → JSON conversion
   - Stage 2: Fine-tune for itemset mining on structured JSON
   - Separate concerns

8. **RAG Enhancement**
   - Provide CSV schema in separate context
   - Use retrieval to show similar examples
   - Ground model in actual data

9. **Hybrid Approach**
   - Use model for formatting/explanation only
   - Keep Apriori algorithm for actual mining
   - LLM just converts Apriori output to natural language

---

## Comparison to Base Model (Needed)

**Next Steps**:
1. Test same datasets on **base Qwen2.5-0.5B** (no fine-tuning)
2. Test on **GPT-4** (baseline from original pipeline)
3. Compare:
   - JSON format compliance
   - Item hallucination rate
   - Support count accuracy
   - Processing time

This will reveal if fine-tuning helped at all or made things worse.

---

## Conclusion

**Current Status**: Fine-tuned model **partially learned output format** but **completely fails at task logic**

**Key Issue**: Model hallucinates items and doesn't parse CSV correctly

**Path Forward**:
1. ✅ Test base model for comparison
2. ✅ Try larger model (1.5B+)
3. ✅ Generate 200+ better training examples
4. ✅ Add explicit CSV parsing in prompts
5. ⚠️ Consider hybrid approach (Apriori + LLM for formatting only)

**Verdict**: Current 0.5B model with 88 examples is **insufficient** for this task. Need larger model + more/better data.
