# Fine-tuning V2 - Plán Zlepšenia

**Cieľ:** Natrénovať LLM schopné extrahovať frequent itemsety s presnosťou porovnateľnou alebo lepšou ako Apriori algoritmus.

---

## 📊 Analýza Zlyhania V1

### Čo model robil zle (15 testov, 1/15 = 6.7% úspešnosť)

| Problém | Príčina | Dopad |
|---------|---------|-------|
| **Halucinácia položiek** | Model nerozumie CSV formátu | Výstup obsahuje slová, ktoré v datasete neexistujú |
| **Nevalidný JSON** | Nedostatok príkladov, malý model | 14/15 výstupov s chybným JSON |
| **Duplikáty v itemsetoch** | Nechápe množinovú sémantiku | `["a", "a", "a"]` namiesto `["a"]` |
| **Zlé support counts** | Nerobí výpočty, háda čísla | Matematicky nekonzistentné |
| **CSV parsing** | Nezíval tréning na parsovanie | Mieša headers s hodnotami |

### Root Causes (čo musíme zmeniť)

1. **Model 0.5B je príliš malý** pre multi-step reasoning
2. **88 príkladov nestačí** na naučenie logiky úlohy
3. **Tréningové dáta nemali chain-of-thought** - model videl len input→output
4. **Syntetické dáta** (attr_N_type) sú mätúce - model si myslí že items sú náhodné slová

---

## 🎯 Stratégia V2: 5 Kľúčových Zmien

### 1. Väčší Model (KRITICKÉ)

**Pred:** Qwen2.5-0.5B-Instruct (494M parametrov)  
**Teraz:** Qwen2.5-3B-Instruct (3B parametrov) alebo Qwen2.5-7B-Instruct

**Prečo:**
- Frequent itemset mining vyžaduje:
  - CSV parsing (porozumenie štruktúre)
  - Počítanie (koľkokrát sa item objavuje)
  - Kombinatoriku (ktoré položky sa vyskytujú spolu)
  - Filtrovanie (support ≥ 3)
  - JSON formátovanie
- Toto je **5-krokový reasoning task** - malé modely to nezvládnu

**Ako:**
```python
model_id = "Qwen/Qwen2.5-3B-Instruct"  # alebo 7B
# S 4-bit QLoRA sa 7B zmestí na A10G (24GB)
```

---

### 2. Chain-of-Thought Tréning (KRITICKÉ)

**Pred:** Input (CSV) → Output (JSON) - žiadne medzikroky  
**Teraz:** Input (CSV) → **Reasoning Steps** → Output (JSON)

**Príklad nového tréningového formátu:**

```json
{
  "input": "attr_1,attr_2,attr_3\napple,banana,cherry\napple,banana,date\napple,banana,cherry\norange,banana,cherry",
  "output": "## Step 1: Parse CSV\nColumns: attr_1, attr_2, attr_3\nRows: 4 transactions\n\n## Step 2: Extract items per row\nRow 1: [apple, banana, cherry]\nRow 2: [apple, banana, date]\nRow 3: [apple, banana, cherry]\nRow 4: [orange, banana, cherry]\n\n## Step 3: Count item occurrences\n- apple: 3 rows (1, 2, 3)\n- banana: 4 rows (1, 2, 3, 4)\n- cherry: 3 rows (1, 3, 4)\n- date: 1 row (2)\n- orange: 1 row (4)\n\n## Step 4: Find 2-itemsets\n- {apple, banana}: rows 1, 2, 3 → count=3 ✓\n- {apple, cherry}: rows 1, 3 → count=2 ✗\n- {banana, cherry}: rows 1, 3, 4 → count=3 ✓\n\n## Step 5: Filter (support ≥ 3)\nValid itemsets: {apple, banana}, {banana, cherry}, {apple}, {banana}, {cherry}\n\n## Final Output:\n[\n  {\"itemset\": [\"apple\", \"banana\"], \"count\": 3, \"evidence_rows\": [1, 2, 3], \"explanation\": \"Co-occurs in 3 transactions\"},\n  {\"itemset\": [\"banana\", \"cherry\"], \"count\": 3, \"evidence_rows\": [1, 3, 4], \"explanation\": \"Co-occurs in 3 transactions\"}\n]"
}
```

**Výhoda:** Model sa naučí **proces**, nie len **výsledok**

---

### 3. Viac a Lepšie Tréningové Dáta (KRITICKÉ)

**Pred:** 88 syntetických príkladov  
**Teraz:** 300-500 príkladov s reálnymi dátami

#### Typy dát, ktoré potrebujeme:

| Typ | Počet | Účel |
|-----|-------|------|
| **Jednoduché CSV** (3-5 riadkov) | 100 | Základné pochopenie |
| **Stredné CSV** (10-15 riadkov) | 150 | Hlavný use case |
| **Ťažké CSV** (20-30 riadkov) | 100 | Škálovanie |
| **Edge cases** | 50 | Robustnosť |
| **Prázdne/žiadne itemsety** | 50 | Naučiť kedy vrátiť `[]` |

#### Reálne mená položiek (nie syntetické):

```csv
# ZLE (V1) - mätúce náhodné slová
attr_1_text,attr_2_int,attr_3_float
werecomi,123,456.78
gava,789,012.34

# DOBRE (V2) - zrozumiteľné položky
product,category,store
milk,dairy,supermarket
bread,bakery,supermarket
eggs,dairy,supermarket
```

#### Diverzita scenárov:
- Transakčné dáta (nákupné košíky)
- Binárne matice (0/1 prítomnosť)
- Long format (transaction_id, item)
- Wide format (row = transakcia)

---

### 4. Explicitné Pravidlá v Prompte (VYSOKÁ PRIORITA)

**Nový system prompt - pridať:**

```markdown
## CRITICAL RULES:

1. **ONLY use items from the CSV columns/values** - NEVER invent new words
2. **Items are the VALUES in the CSV, not column headers** (unless dataset is one-hot encoded)
3. **Count = number of DISTINCT ROWS containing the itemset**
4. **Itemset is a SET** - no duplicates allowed (["a", "a"] is INVALID)
5. **Output ONLY valid JSON** - no markdown, no code blocks
6. **If no itemsets have count ≥ 3, return []**

## Example of WRONG output:
{"itemset": ["niwere", "wali"]} ← WRONG: These words don't exist in the CSV

## Example of CORRECT output:
{"itemset": ["milk", "bread"]} ← CORRECT: Both "milk" and "bread" are actual values in the CSV
```

---

### 5. Validácia počas Tréningu (STREDNÁ PRIORITA)

**Implementovať eval callback:**

```python
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    
    # Parse JSON outputs
    valid_json_rate = count_valid_json(predictions) / len(predictions)
    
    # Check if items exist in input CSV
    hallucination_rate = count_hallucinations(predictions, inputs) / len(predictions)
    
    # Check itemset correctness vs Apriori ground truth
    f1_score = compute_itemset_f1(predictions, apriori_outputs)
    
    return {
        "valid_json_rate": valid_json_rate,
        "hallucination_rate": hallucination_rate,
        "f1_score": f1_score
    }
```

---

## 📋 Implementačný Checklist

### Fáza 1: Príprava Dát (1-2 dni)

- [ ] **Generovať 500 nových tréningových príkladov**
  - Použiť reálne názvy položiek (produkty, farby, mestá, ...)
  - Rôzne veľkosti (5-30 riadkov)
  - Rôzne formáty (wide, long, one-hot)

- [ ] **Pridať Chain-of-Thought do každého príkladu**
  - Step 1: Parse CSV
  - Step 2: Extract items
  - Step 3: Count occurrences  
  - Step 4: Find itemsets
  - Step 5: Filter by support
  - Final: JSON output

- [ ] **Vytvoriť edge case príklady**
  - Žiadne frequent itemsety → `[]`
  - Veľa singletonov, málo 2-itemsetov
  - Numerické hodnoty (ignorovať vs. použiť)
  - Prázdne bunky

### Fáza 2: Setup Tréningu (0.5 dňa)

- [ ] **Vybrať väčší model**
  - Qwen2.5-3B alebo 7B
  - Test či sa zmestí na A10G s QLoRA

- [ ] **Upraviť tréningový script**
  - Nový model ID
  - Dlhší max_length (pre CoT)
  - Custom eval metrics

- [ ] **Nahrať dataset na HF Hub**
  - Nový repo pre V2 dáta
  - Train/Val split 90/10

### Fáza 3: Tréning (2-4 hodiny)

- [ ] **Spustiť tréning** (Gemini CLI alebo HF Jobs)
  - 3-5 epoch (väčší dataset = menej epoch)
  - Learning rate: 2e-4
  - Batch size: auto

- [ ] **Monitorovať metriky**
  - Training loss
  - Eval loss
  - Valid JSON rate
  - Hallucination rate

### Fáza 4: Evaluácia (0.5 dňa)

- [ ] **Testovať na 15 pôvodných datasetoch**
  - Porovnať s V1 výsledkami
  - Merať: valid JSON, hallucination, F1 vs Apriori

- [ ] **Testovať na held-out datasetoch**
  - 10 datasetov, ktoré model nikdy nevidel
  - Merať generalizáciu

- [ ] **Benchmark vs Apriori**
  - Precision: koľko LLM itemsetov je správnych
  - Recall: koľko Apriori itemsetov LLM našlo
  - F1 score

---

## 🛠️ Nové Scripty na Vytvorenie

### 1. `generate_training_data_v2.py`

```python
"""
Generuje 500 tréningových príkladov s:
- Reálnymi názvami položiek
- Chain-of-thought reasoning
- Diverzitou veľkostí a formátov
- Ground truth z Apriori algoritmu
"""

def generate_example(num_rows, num_items, item_vocabulary):
    # 1. Generuj CSV s reálnymi názvami
    csv_content = generate_csv(num_rows, num_items, item_vocabulary)
    
    # 2. Spusti Apriori pre ground truth
    apriori_result = run_apriori(csv_content, min_support=3)
    
    # 3. Generuj Chain-of-Thought reasoning
    cot = generate_cot_steps(csv_content, apriori_result)
    
    # 4. Formátuj ako tréningový príklad
    return {
        "input": csv_content,
        "output": cot + "\n\n## Final Output:\n" + json.dumps(apriori_result)
    }
```

### 2. `evaluate_model_v2.py`

```python
"""
Komplexná evaluácia s metrikami:
- Valid JSON rate
- Hallucination rate
- F1 vs Apriori
- Per-size breakdown (singletons, 2-itemsets, 3-itemsets)
"""

def evaluate(model, datasets):
    results = []
    for ds in datasets:
        prediction = model.generate(ds.csv)
        apriori_truth = run_apriori(ds.csv)
        
        results.append({
            "valid_json": is_valid_json(prediction),
            "hallucination": check_hallucination(prediction, ds.csv),
            "precision": compute_precision(prediction, apriori_truth),
            "recall": compute_recall(prediction, apriori_truth),
            "f1": compute_f1(prediction, apriori_truth)
        })
    return aggregate_results(results)
```

---

## 📈 Očakávané Zlepšenie

| Metrika | V1 (0.5B, 88 examples) | V2 (3B+, 500 examples, CoT) |
|---------|------------------------|----------------------------|
| Valid JSON | 6.7% (1/15) | **80-90%** |
| Hallucination Rate | ~100% | **<10%** |
| F1 vs Apriori | ~0% | **60-80%** |
| Inference Time (CPU) | 1-20 min | 2-30 min (väčší model) |

---

## ⚠️ Riziká a Mitigácie

| Riziko | Pravdepodobnosť | Mitigácia |
|--------|-----------------|-----------|
| 3B/7B príliš veľký pre GPU | Stredná | Použiť 4-bit QLoRA, alebo menší 1.5B |
| CoT predĺži inference | Vysoká | Skrátiť CoT, použiť "thinking" tag na skrytie |
| Apriori ground truth chyby | Nízka | Validovať Apriori output pred použitím |
| Model kopíruje CoT ale nerozumie | Stredná | Testovať na out-of-distribution dátach |

---

## 🏁 Ďalšie Kroky

1. **DNES:** Rozhodnúť sa pre 3B vs 7B model (testovať memory)
2. **ZAJTRA:** Začať generovať 500 tréningových príkladov
3. **ZA 2 DNI:** Spustiť V2 tréning
4. **ZA 3 DNI:** Evaluácia a porovnanie s V1

---

**Otázka:** Chceš začať generovaním nových tréningových dát, alebo najprv otestovať či sa 3B/7B model zmestí na dostupný hardware?
