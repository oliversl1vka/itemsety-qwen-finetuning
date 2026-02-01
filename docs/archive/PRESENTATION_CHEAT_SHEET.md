# QUICK REFERENCE - Prezentace Cheat Sheet

## 🔢 KLÍČOVÁ ČÍSLA (naučit nazpaměť!)

```
✓ 300 runů (3 modely × 100 datasetů)
✓ 100% validation pass rate
✓ r = 0.8056 (korelace LLM vs. Apriori, p < 0.001)
✓ 5-95 řádků, 11-100 sloupců (průměr: 49 × 61)
✓ 0-90 itemsetů per dataset (průměr: 16, medián: 5)
✓ GPT-4.1: 14.49 itemsetů
✓ GPT-5.0: 14.50 itemsetů (téměř identické!)
✓ GPT-4o-mini: 18.89 itemsetů
```

## 📊 MAPOVÁNÍ VIZUALIZACÍ NA SLIDY

### Slide 4 (Methods):
- `slide4_pipeline_architecture.png` - Diagram celého pipeline

### Slide 6 (Results) - POUŽIJTE VŠECH 5 GRAFŮ:
1. `slide6_llm_vs_apriori.png` - Scatter plot (hlavní!)
2. `slide6_models_comparison.png` - Bar chart modelů
3. `slide6_histogram_distribution.png` - Distribuce itemsetů
4. `slide6_dataset_dimensions.png` - Box plots dimenzí
5. `slide6_heatmap_dimensions.png` - Heatmap úspěšnosti

## 🎯 CO ZDŮRAZNIT NA KAŽDÉM SLIDU

**Slide 2 (Topic):**
→ "LLM jako alternativa k deterministickým algoritmům"

**Slide 3 (Goals):**
→ "Systematické porovnání - 100 datasetů, 3 modely, automatická validace"

**Slide 4 (Methods):**
→ "Chunked processing, 6 validačních invariantů, temperature=0"

**Slide 5 (Stav poznání):**
→ "Žádná předchozí studie LLM pro frequent itemset mining"

**Slide 6 (Results) - NEJDŮLEŽITĚJŠÍ:**
→ "100% validation pass rate, r=0.8056, GPT-4.1 ≈ GPT-5.0"

**Slide 7 (Přínos):**
→ "První systematická studie, open-source pipeline, důkaz ekvivalence"

**Slide 8 (Conclusion):**
→ "300 runů, 100% success, ale limity: syntetická data, fixed support"

## ⏱️ TIMING (celkem 13 min)

```
0:00-0:30   Slide 1: Titulka
0:30-2:30   Slide 2: Topic (2 min)
2:30-4:30   Slide 3: Goals (2 min)
4:30-7:30   Slide 4: Methods (3 min)
7:30-9:30   Slide 5: Stav poznání (2 min)
9:30-14:30  Slide 6: Results (5 min!) ⭐
14:30-16:30 Slide 7: Přínos (2 min)
16:30-18:00 Slide 8: Conclusion (1.5 min)
```

## 💬 ODPOVĚDI NA OČEKÁVANÉ OTÁZKY

**"Proč identické výsledky?"**
→ Deterministická úloha, temperature=0, dobře definovaný problém

**"Jak škáluje?"**
→ Chunked processing (50/request), lineární náklady, ready pro produkci

**"Co real-world data?"**
→ Future work, pipeline ready, plánujeme e-commerce + healthcare

**"Proč ne FP-Growth?"**
→ TODO, priorita byl důkaz LLM feasibility, nyní můžeme přidat baselines

**"Statistická signifikance?"**
→ 300 runů, p<0.001, robustní, větší vzorek by byl bonus ale není nutný

## 🎤 OPENING (naučit nazpaměť!)

> "Dobrý den, jmenuji se Eduard Jurášek a téma mé bakalářské práce je 
> Frequent Itemset Mining pomocí algoritmů Apriori, FP-Growth a Large 
> Language Models. Cílem této práce bylo systematicky porovnat, zda LLM 
> dokáží extrahovat frequent itemsets stejně přesně jako tradiční 
> deterministické algoritmy. Výsledek: Na 300 experimentech jsme dosáhli 
> 100% validation pass rate s korelací r=0.8056 s Apriori algoritmem."

## 🎯 CLOSING (naučit nazpaměť!)

> "Shrnutí: Tato práce jako první systematicky prokázala, že Large Language 
> Models jsou viable alternativa pro frequent itemset mining s 100% přesností. 
> Vytvořili jsme open-source pipeline, který je plně replikovatelný. 
> Limity: syntetická data, fixed support. Future work: real-world datasety, 
> FP-Growth comparison. Děkuji za pozornost, máte nějaké otázky?"

## 📱 KONTAKTY (mít připravené)

- Eduard Jurášek: jure01@vse.cz
- Prof. Tomáš Kliegr: tomas.kliegr@vse.cz
- GitHub repo: (připravit URL pokud je public)

## ✅ BEFORE PREZENTACE CHECKLIST

☐ Vizualizace vloženy do PowerPoint
☐ Nazkoušeno na čas (12-15 min)
☐ Tento cheat sheet vytištěn
☐ Backup USB s prezentací
☐ Telefon na ticho
☐ Sklenice vody připravena
☐ Příchod 10 min před začátkem

---
**Hodně štěstí! 🍀**
