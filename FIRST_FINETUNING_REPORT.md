# Prvý Fine-tuning Experiment - Itemset Extraction z CSV

**Autor:** Oliver Slivka  
**Dátum:** 17. december 2025  
**Model:** Qwen2.5-0.5B-Instruct + QLoRA adapter  
**Úloha:** Extrahovanie frequent itemsetov z CSV dát pomocou LLM

---

## 🎯 Cieľ Experimentu

Natrénovať malý LLM (0.5B parametrov) na automatickú extrakciu častých kombinácií položiek (frequent itemsets) z CSV datasetu. Model má:
1. Načítať CSV vstup
2. Identifikovať položky (items) zo stľpcov
3. Nájsť frequent itemsety s podporou ≥ 3
4. Vygenerovať štruktúrovaný JSON výstup s dôkazmi

---

## 📊 Tréningová Konfigurácia

### Dataset
- **Tréningové príklady:** 88
- **Validačné príklady:** 10
- **Zdroj:** Syntetické CSV datasety (rôzne veľkosti, schémy)
- **Format:** System prompt + CSV input → JSON array output

### Model & Tréning
- **Base model:** Qwen/Qwen2.5-0.5B-Instruct (494M parametrov)
- **Metóda:** 4-bit QLoRA (NF4 quantization)
- **Trénovateľné parametre:** 8.65M (1.75% z celku)
- **Hardware:** Hugging Face A10G GPU
- **Tréningový čas:** ~10-15 minút
- **Epochy:** 3
- **LoRA rank:** 16, alpha: 32

### Fine-tuned Model
- **Hub:** `OliverSlivka/qwen-itemsety-qlora`
- **Veľkosť adaptéra:** 17.6 MB
- **Optimizer:** paged_adamw_8bit

---

## 🧪 Evaluačný Setup

**Testovanie na 15 datasetoch:**
- Veľkosti: 5-17 riadkov, 19-99 stĺpcov
- Prostredie: Windows CPU (FP16, bez quantization)
- Všetky odpovede uložené pre analýzu

---

## 📈 Výsledky

### Kvantitatívne Metriky

| Metrika | Hodnota |
|---------|---------|
| **Otestované datasety** | 15/15 ✅ |
| **Úspešné behy** | 15/15 (žiadne crashe) |
| **Valid JSON výstupy** | 1/15 (6.7%) ❌ |
| **Celkový počet itemsetov** | 2 |
| **Priemerný čas generovania** | 21.6 min (5-260 min) |
| **Priemerná dĺžka outputu** | 1,094 znakov |

### Detailná Tabuľka

| Dataset | Riadky | Stĺpce | Čas (s) | Valid JSON | Itemsety |
|---------|--------|--------|---------|------------|----------|
| ds_0001_5x53 | 5 | 53 | 75 | ✓ | 2 |
| ds_0096_6x99 | 6 | 99 | 199 | ✗ | 0 |
| ds_0029_7x66 | 7 | 66 | 151 | ✗ | 0 |
| ds_0010_9x24 | 9 | 24 | 115 | ✗ | 0 |
| ds_0024_9x88 | 9 | 88 | 263 | ✗ | 0 |
| ds_0025_9x90 | 9 | 90 | 15,654 | ✗ | 0 |
| ds_0032_10x20 | 10 | 20 | 99 | ✗ | 0 |
| ds_0048_10x86 | 10 | 86 | 314 | ✗ | 0 |
| ds_0056_11x30 | 11 | 30 | 67 | ✗ | 0 |
| ds_0058_11x91 | 11 | 91 | 238 | ✗ | 0 |
| ds_0012_13x29 | 13 | 29 | 251 | ✗ | 0 |
| ds_0017_13x94 | 13 | 94 | 567 | ✗ | 0 |
| ds_0023_13x87 | 13 | 87 | 622 | ✗ | 0 |
| ds_0076_14x71 | 14 | 71 | 608 | ✗ | 0 |
| ds_0068_17x19 | 17 | 19 | 223 | ✗ | 0 |

---

## 🔍 Analýza Výsledkov

### ✅ Čo Model Zvládol

1. **Formát outputu**
   - Model konzistentne generuje JSON array štruktúru
   - Rozumie objektom s poľami `itemset`, `proof`, `explanation`
   - Žiadne extra markdown wrappery (väčšinou)

2. **Stabilita**
   - Žiadne crashe ani výnimky pri generovaní
   - Model dokončil všetky 15 testov

3. **Inference**
   - Funguje na CPU (FP16)
   - Relatívne rýchle pre malé datasety

### ❌ Kritické Problémy

#### 1. **Halucinácia Položiek** (Hlavný problém)
Model vymýšľa názvy itemov, ktoré neexistujú v CSV:

**Príklad (ds_0001):**
```json
{
  "itemset": ["niwere", "wali", "kawo", "gaba", "bogbo", ...],
  "proof": "1"
}
```
**Realita:** CSV obsahuje stĺpce `attr_1_float`, `attr_2_int`, ... nie `niwere`, `wali`

**→ Model nečíta CSV správne, generuje náhodné slová**

#### 2. **Nevalidný JSON Syntax** (14/15 prípadov)
- Preklepy vo field names: `vidence` namiesto `evidence`, `explaination`, `explainer`
- Chýbajúce uvodzovky, čiarky
- Markdown code blocky (`````json```) v outpute

#### 3. **Duplikáty v Itemsetoch**
```json
{
  "itemset": ["jos", "jos", "jos", "jos", ...]  // 40+ krát to isté slovo
}
```
**→ Model nechápe množinovú sémantiku (set = unikátne prvky)**

#### 4. **Nesprávne Support Counts**
- Reportuje `count=2` ale vysvetľuje že threshold je `≥3` (matematická nekonzistencia)
- Halusinuje počty riadkov (napr. "12" keď dataset má 7 riadkov)

#### 5. **CSV Parsing Zlyhanie**
- Nerozlišuje medzi header (názvy stĺpcov) a dátovými hodnotami
- Vkladá numerické tuples namiesto item names
- Ignoruje skutočné stĺpce CSV

---

## 💡 Root Cause Analýza

### Prečo Model Zlyhal?

1. **Model príliš malý (0.5B parametrov)**
   - Úloha vyžaduje viacstupňové uvažovanie:
     - Parsovať CSV → Identifikovať items → Spočítať podporu → Filtrovať → Formátovať JSON
   - 0.5B model nemá kapacitu na komplexnú reasoning úlohu

2. **Nedostatok tréningových dát (88 príkladov)**
   - Model sa naučil formát, ale nie logiku
   - Potrebné stovky príkladov pre generalizáciu

3. **Syntetické dáta matú model**
   - Náhodné názvy stĺpcov (`attr_N_type`) vs. skutočné hodnoty
   - Model si mýli čo sú položky vs. čo sú metadáta

4. **Instruction Following vs. Reasoning**
   - Model sa naučil **formát outputu** (JSON štruktúra)
   - Model **NEzvládol logiku úlohy** (parsovanie, počítanie, filtrovanie)

---

## 📋 Odporúčania pre Ďalšie Iterácie

### 1. **Väčší Model** (Priorita: VYSOKÁ)
- Vyskúšať **Qwen2.5-1.5B** alebo **3B**
- Viac parametrov = lepšie reasoning schopnosti
- Stále sa zmestí na A10G s 4-bit QLoRA

### 2. **Viac Tréningových Dát** (Priorita: VYSOKÁ)
- Zvýšiť z 88 na **200-500 príkladov**
- Použiť reálne CSV príklady (nie len syntetické)
- Pridať explicit chain-of-thought do tréningových dát:
  ```
  1. Identify columns: [col1, col2, col3]
  2. Scan rows for co-occurrences
  3. Count support for each itemset
  4. Filter by threshold ≥3
  5. Format as JSON
  ```

### 3. **Lepšie Prompt Engineering** (Priorita: STREDNÁ)
- Explicitne zakázať vymýšľanie položiek:
  > "ONLY use items from CSV column headers. DO NOT invent new words."
- Pridať few-shot examples do system promptu
- Štruktúrovať prompt: Parse → Count → Filter → Output

### 4. **Curriculum Learning** (Priorita: STREDNÁ)
Trénovať postupne:
- **Fáza 1:** Jednoduché CSV (3-5 riadkov, 5-10 stĺpcov)
- **Fáza 2:** Stredné CSV (10-15 riadkov, 20-30 stĺpcov)
- **Fáza 3:** Komplexné CSV (20-30 riadkov, 50+ stĺpcov)

### 5. **Hybridný Prístup** (Alternatíva)
- Použiť **Apriori algoritmus** pre mining (správny, deterministický)
- LLM použiť **len na vysvetľovanie** (natural language generation)
- Sidestepne problém halucinácie

### 6. **Validácia počas Tréningu** (Priorita: NÍZKA)
- Implementovať custom eval callback
- Merať správnosť itemsetov (nie len loss)
- Early stopping ak validation accuracy klesá

---

## 🎓 Závery

### Poučenia z Prvého Fine-tuningu

1. **Format ≠ Understanding**
   - Model sa dokáže naučiť štruktúru outputu (JSON syntax)
   - To **neznamená**, že chápe logiku úlohy

2. **Small Models Have Limits**
   - 0.5B model nie je dostatočný pre multi-step reasoning
   - Fine-tuning nevyrieši fundamentálne kapacitné limity

3. **Data Quality > Quantity**
   - 88 syntetických príkladov nestačí
   - Lepšie 100 reálnych príkladov než 500 syntetických

4. **Evaluation je kľúčová**
   - Systematické testovanie odhalilo konkrétne failure modes
   - Bez evaluácie by sme nevedeli čo zlepšovať

### Ďalšie Kroky

1. ✅ **Porovnať s base modelom** (bez fine-tuningu) - overenie že FT pomohol/uškodil
2. 🔄 **Trénovať Qwen2.5-1.5B** s rovnakým datasetom - test kapacity
3. 📊 **Generovať 200+ lepších príkladov** - s reálnymi CSV a CoT
4. 🧪 **A/B test**: Apriori vs. Fine-tuned LLM vs. Hybrid

---

## 📚 Technické Detaily

### Reprodukcia

**Model na HuggingFace Hub:**
```
OliverSlivka/qwen-itemsety-qlora
```

**Test script:**
```bash
python test_extended_evaluation.py
```

**Výsledky:**
- `model_test_results/` - Individuálne odpovede
- `summary_extended_evaluation.json` - Agregované metriky

### Citácie

- **Qwen Team**: Qwen2.5 Technical Report (2024)
- **HuggingFace**: TRL Library (Supervised Fine-tuning)
- **Dettmers et al.**: QLoRA: Efficient Finetuning of Quantized LLMs (2023)

---

**Status:** ✅ Prvý experiment ukončený, dáta pripravené na porovnanie  
**Ďalší milestone:** Base model comparison + väčší model fine-tuning
