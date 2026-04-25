# Štruktúra bakalárskej práce
**Pracovný názov:** Fine-tuning malého LLM modelu pre úlohu hľadania frequent itemsetov  
**Výskumná otázka:** Je možné pomocou fine-tuningu natrénovať malý LLM model tak, aby sa v úlohe hľadania frequent itemsetov priblížil výstupu Apriori algoritmu bližšie ako veľký nedoladený base model?  
**Cieľový rozsah:** ~50 strán  
**Dátum:** 2026-03-16

---

## 1. Úvod (~3 strany)

### 1.1 Motivácia
- Prečo je ťaženie asociačných pravidiel stále relevantné (e-commerce, odporúčacie systémy, medicína)
- Apriori je spoľahlivý deterministický algoritmus — ale vyžaduje explicitné spustenie nad dátami
- Otázka: čo ak by jazykový model „vedel" hľadať itemsety priamo z CSV bez volania algoritmu?
- Boom malých inštrukcií doladených LLM (Qwen, Mistral, Phi) — sú tieto modely schopné symbolického uvažovania?
- Praktická motivácia: embedded inference, offline prostredia, integrácia do chat agentov

### 1.2 Výskumná otázka
- **Hlavná otázka:** Je možné pomocou fine-tuningu natrénovať malý LLM model tak, aby sa v úlohe hľadania frequent itemsetov priblížil výstupu Apriori algoritmu bližšie ako veľký nedoladený base model?
- **Čiastková otázka 1:** Akú rolu hrá Chain-of-Thought reasoning pri presnosti extrakcie?
- **Čiastková otázka 2:** Je DPO trénované na reálnych zlyhaniach LLM efektívnejšie ako syntetické odmietnutia?
- **Čiastková otázka 3:** Kde leží hranica — aké datasety sú príliš veľké/komplexné pre fine-tuned malý model?

### 1.3 Prínosy práce
- Open-source pipeline kombinujúca Apriori ako deterministický oracle s LLM extraktorom
- Verejný trénovací dataset na HuggingFace Hub (sft/dpo/grpo konfigurácie)
- Doladený model Qwen2.5-7B dostupný na HuggingFace Hub
- Evaluačná metodológia: P/R/F1 vs. Apriori, JSON parse rate, inference time
- Agentúrny workflow pre iteratívny ML výskum ako vedľajší príspevok

---

## 2. Teoretické základy a príbuzné práce (~10 strán)

### 2.1 Frequent itemset mining
- Definícia transakčnej databázy, transakcie, itemsetu
- Formálna definícia frequent itemsetu: $\text{support}(X) = \frac{|\{t \in T : X \subseteq t\}|}{|T|} \geq \text{min\_support}$
- Asociačné pravidlá (Agrawal & Srikant, 1993) — historický kontext a dôležitosť
- Prehľad algoritmov:
  - **Apriori** — level-by-level generovanie kandidátov, anti-monotónna vlastnosť, výhody a nevýhody
  - **FP-Growth** — compact FP-tree štruktúra, bez generovania kandidátov, výhody pre veľké datasety
  - **Eclat** — vertikálna reprezentácia, intersection-based
- Prečo Apriori ako základ tejto práce: deterministický, dobre definovaný výstup, jednoducho verifikovateľný

### 2.2 Veľké jazykové modely ako extraktory znalostí
- Rozlíšenie: LLM ako generátor voľného textu vs. LLM ako štruktúrovaný extraktor
- Transformer architektúra (Vaswani et al., 2017) — stručný prehľad relevantný pre pochopenie limitácií
- Relevantné práce o LLM na tabulárnych dátach a symbolickom uvažovaní
- Chain-of-Thought prompting (Wei et al., 2022) — prečo pomáha pri viacstupňových výpočtoch
- Obmedzenia base modelov na tejto úlohe:
  - Halucinácie (vymyslené riadky v evidence)
  - Nesprávne počítanie (off-by-one, vynechané riadky)
  - Nekonzistentný JSON výstup
  - Závislosť od poradia v prompte

### 2.3 Fine-tuning malých LLM
- Prehľad metód:
  - **Full fine-tuning** — vysoké nároky na VRAM, riziko catastrophic forgetting
  - **LoRA** (Hu et al., 2021) — low-rank adapter vrstvy, parametre $r$ a $\alpha$
  - **QLoRA** (Dettmers et al., 2023) — 4-bit kvantizácia + LoRA, umožňuje tréning na spotrebnej GPU
- Unsloth optimalizácia — custom CUDA kernely, 2× rýchlejší tréning, 70 % menej VRAM
- **SFT (Supervised Fine-Tuning)** — imitácia demonštrácií, `train_on_responses_only`
- **DPO (Direct Preference Optimization)** (Rafailov et al., 2023):
  - Princíp: priame trénanie na pároch (chosen, rejected) bez reward modelu
  - Výhody oproti RLHF: stabilnejší tréning, bez potreby separátneho reward modelu
  - Beta parameter ako regularizácia (KL divergencia od ref. modelu)
- **GRPO (Group Relative Policy Optimization)** — online RL s reward funkciami, zmienka pre kontext

### 2.4 Medzera v literatúre
- Neexistuje práca kombinujúca Apriori ako oracle s LLM fine-tuningom na structured extraction
- Väčšina DPO prác používa syntetické odmietnutia (GPT-4 generuje slabšie odpovede) — reálne zlyhania LLM sú nevyužitý zdroj signálu
- Chýba systematické porovnanie: malý fine-tuned model vs. veľký base model na deterministickej kombinatorickej úlohe
- Táto práca tieto medzery vypĺňa

---

## 3. Navrhované riešenie (~4 strany)

### 3.1 Celková architektúra systému
- Schéma pipeline: CSV datasety → Apriori (ground truth) → LLM extractor (zlyhania) → trénovacie dáta → fine-tuning → evaluácia
- Voľba základného modelu: **Qwen2.5-7B-Instruct**
  - Open weights, komerčne použiteľná licencia
  - Instruct variant — lepší štartovací bod pre štruktúrované výstupy
  - 7B parametrov — zvládnuteľné na školskom GPU serveri s Unsloth + 4-bit kvantizáciou
  - Porovnávaná s GPT-4.1-mini (veľký base model bez fine-tuningu)
- Prečo 3-fázový tréning: SFT-CoT → DPO-Real (→ GRPO ako voliteľná fáza v4)

### 3.2 Kľúčové rozhodnutia dizajnu a ich odôvodnenie
- **Apriori ako deterministický oracle** — nie ground truth z ľudských anotácií; výhodou je 100 % opakovateľnosť a verifikovateľnosť
- **Reálne zlyhania LLM ako rejected samples** — GPT-4.1-mini, GPT-4.1-nano, o4-mini, GPT-4o produkovali 606 reálnych zlyhaní; 99.5 % chyba = hallucinated evidence rows
- **Chain-of-Thought vo formáte `<think>` tagov** — model sa naučí krok po kroku skenovať stĺpce pred výstupom JSON
- **Concise CoT formát (v3)** — column-grouped scanning namiesto verbose evidence-row formátu; kratší = menej tokenov = vyšší throughput
- **Agentúrny workflow** namiesto monolitického skriptu — odôvodnenie potreby organizácie pri dlhodobom iteratívnom výskume

---

## 4. Dátová pipeline (~8 strán)

### 4.1 Generátor syntetických datasetov
- Popis `generate_datasets_v2.py` — parametrizácia počtu riadkov, stĺpcov, domén položiek
- 500 CSV datasetov, formát `ds_{ID:04d}_{rows}x{cols}.csv`
- Domény: potraviny, školské predmety, e-commerce kategórie, medicínske symptómy
- Stratégia pokrytia: rôzne veľkosti (5–25 riadkov), rôzne hustoty itemsetov
- Normalizácia: `Row N` labeling, lowercase/trim/sort pre konzistentnosť
- Príklad datasetu a jeho expected output

### 4.2 Apriori extrakcia (ground truth)
- Implementácia `apriori_frequent_itemsets` — level-by-level generovanie
- Anti-monotónna vlastnosť: ak $X$ nie je frequent, žiadna nadmnožina $X$ nie je frequent
- Parametrizácia: `--min-support` (absolútny count), `--max-size` (max veľkosť itemsetu)
- Výstupný formát: `{itemset: [...], count: N, rows: ["Row 1", ...], support: float}`
- Hash-based pomenovanie artefaktov (SHA256, prvých 12 znakov) — reprodukovateľnosť a deduplikácia

### 4.3 LLM extrakcia
- `llm_extract_full` — chunked invocation cez OpenAI API s LangChain
- Vývoj systémového promptu: od verbose k compact verzii (~150 tokenov)
- Modely použité v produkcii: GPT-4.1-mini, GPT-4.1-nano, o4-mini, GPT-4o
- Spracovanie odpovedí: deduplikácia, normalizácia, JSON parsing, retry logika
- Chybové stavy a exit kódy: 0 (úspech), 1 (všetky zlyhali), 2 (zlý data-dir), 3 (chýbajúce credentials)

### 4.4 Validačný subsystém
- 13 invariantov, ktoré každý výstup musí splniť:
  - Matematická správnosť support hodnôt
  - Prítomnosť všetkých citovaných items v transakciách
  - Integrita row label formátu (`Row N`)
  - Zhoda count vs. počet unique riadkov v evidence
  - Absencia duplikátnych itemsetov
  - Správnosť kanonickej formy (sorted, lowercase)
  - (a ďalšie)
- `validate_all` — postupnosť kontroly, fail-fast vs. collect-all režim
- Štatistiky validácie zo 500 datasetov: pass rate, distribúcia typov chýb

### 4.5 Perzistencia a sledovateľnosť
- SQLite databáza `runs.db` — schéma: `dataset_id`, `timestamp`, `validation_passed`, `llm_itemsets_count`, `apriori_itemsets_count`, `llm_model`
- Auto-migrácia nových stĺpcov pri ALTER TABLE — spätná kompatibilita
- Generačné logy v `logs/` — rozdiel medzi reálnymi (gpt_4_1) a syntetickými (gpt_5_mini) logmi
- Štruktúra artefaktov v `artifacts/`: `apriori_outputs/`, `extractor_outputs/`, `validation_reports/`, `db_prepared/`
- Prečo hash-based naming: ak rovnaký dataset spustíme znovu, výstup je deterministicky rovnaký a nový beh ho neprepíše

---

## 5. Počiatočné pokusy a pivot (~4 strany)

### 5.1 Pokus o fine-tuning s Gemini CLI a HuggingFace MCP
- Popis prístupu: Gemini CLI pre generovanie trénovacích dát, HuggingFace MCP tools pre upload a správu modelov
- Čo fungovalo: rýchle prototypovanie, jednoduché generovanie promptov
- Čo nefungovalo:
  - Absencia systematického verziovania dát a experimentov
  - Ťažkosť reprodukovať výsledky (manuálne kroky, žiadny audit trail)
  - Nemožnosť škálovať na 500 datasetov bez pipeline
  - Strata kontextu medzi sessions — opakované chyby

### 5.2 Poučenia a zmena paradigmy
- Od manuálnych experimentov k systematickej pipeline s SQLite perzistenciou
- Potreba deterministického oracle (Apriori) namiesto manuálneho labellovania
- Potreba verziovania dát, modelov a experimentov — motivácia pre `runs.db` a hash-based artifakty
- Obmedzenia prístupu „notebook + manuálne kroky" pri iteratívnom ML výskume
- Rozhodnutie vybudovať agentúrny workflow — prechod k systematickej práci

---

## 6. Agentúrny workflow (~6 strán)

### 6.1 Motivácia pre multi-agentný prístup
- Komplexnosť projektu: 9 rôznych zodpovedností (dataset generácia, pipeline, tréning, deployment, monitoring, evaluácia, čistenie, údržba, orchestrácia)
- Problémy monolitického prístupu: strata kontextu, opakovanie rovnakých chýb, nekonzistentná dokumentácia
- Potreba externej pamäte agentov — motivácia pre Obsidian vault
- Inšpirácia z multi-agent systems literatúry — špecializácia vs. generalizácia

### 6.2 Architektúra agentského systému
- 9 agentov v `.github/agents/`: Orchestrator, Dataset, Pipeline, Training, Deployment, Monitoring, Evaluation, Cleanup, Maintainer
- Workflow state (`workflow_state.json`) — koordinácia sekvenčného vykonávania, 8 stage stavov
- **Memory-First rule** — každý agent číta pamäťový súbor z `obsidian-brain/Agents/` pred akýmkoľvek úkonom
- Obsidian vault ako externá pamäť: `Logs/`, `Experiments/`, `Decisions/`, `References/`
- Slash commands pre každého agenta (napr. `/datasets`, `/pipeline`, `/export`)

### 6.3 Kľúčové agenty a ich úlohy
- **Orchestrator** — inicializácia a finalizácia workflowu, globálny stav
- **Dataset Agent** — generácia 500 trénovacích + fixných eval datasetov
- **Pipeline Agent** — Apriori + LLM extrakcia, validácia, SQLite perzistencia
- **Training Agent** — export trénovacích dát, generovanie verzionovaných notebookov, validácia výsledkov po tréningu
- **Deployment Agent** — push datasetu a notebooku na HuggingFace Hub
- **Monitoring Agent** — porovnávacie vizualizácie (base vs. fine-tuned vs. Apriori)

### 6.4 Workflow — kompletný postup
- Stage 1 (Orchestrator `/organize`): inicializácia
- Stage 2 (Dataset `/datasets`): generácia datasetov
- Stage 3 (Pipeline `/pipeline`): extrakcia + validácia
- Stage 4 (Training `/export`): export trénovacích dát + notebook
- Stage 5 (Deployment `/push`): push na HuggingFace Hub
- **⏸️ PAUZA** — tréning prebieha na externom Jupyter GPU serveri
- Stage 6 (Training `/validate`): prijatie eval výsledkov, zápis poznatkov
- Stage 7 (Monitoring `/visualize`): porovnávacie vizualizácie
- Stage 8 (Orchestrator `/finalize`): finalizácia, záverečný report

### 6.5 Interaktívny vs. automatizovaný workflow
- Prečo manuálne prepínanie agentov (nie plne automatizované)
- Výhody: plná kontrola, možnosť zásahu pri chybe, viditeľnosť každého kroku
- Obmedzenia: závislosť od ľudského operátora
- Porovnanie s plne automatizovanými ML pipeline frameworkmi (MLflow, Kubeflow)

---

## 7. Fine-tuning modelu (~8 strán)

### 7.1 Príprava trénovacích dát
- SFT-CoT dáta (`sft_cot_v3.json`): 348 príkladov, ChatML formát s `<think>` tagmi
- Porovnanie v2 (verbose) vs. v3 (concise) CoT formátu:
  - v2: explicitné evidence_rows v `<think>` bloku → dlhé sekvencie
  - v3: column-grouped scanning s `✓`/`✗` markermi → ~40 % kratšie
  - Konkrétny príklad oboch formátov
- DPO dáta (`dpo_real_v2.json`): 606 párov z reálnych zlyhaní LLM
  - Distribúcia zdrojových modelov: GPT-4.1-mini, GPT-4.1-nano, o4-mini, GPT-4o
  - 99.5 % chybovosť = hallucinated evidence rows (vymyslené row labels)
  - Štruktúra páru: `prompt` (messages), `chosen` (Apriori CoT), `rejected` (reálny LLM výstup)
- Build HuggingFace datasetu (`build_hf_dataset_v2.py`): 3 konfigurácie sft/dpo/grpo, train/validation split

### 7.2 Konfigurácia modelu a LoRA
- Základný model: `unsloth/Qwen2.5-7B-Instruct-bnb-4bit`
- LoRA hyperparametre (v3, po LLM Council analýze):
  - `r=32` (bolo 64) — menší rank, nižšia kapacita, menej overfitting
  - `alpha=64` (bolo 16) — ratio alpha/r = 2.0 pre silnejší gradient signál
  - `dropout=0.05` (bolo 0) — regularizácia
  - Target modules: `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`
- Počet trénovateľných parametrov vs. celkový počet parametrov
- VRAM využitie na rôznych GPU (T4 vs. A100 vs. H200)

### 7.3 Fáza 1 — SFT s Chain-of-Thought
- `train_on_responses_only` — maskovanie system+user tokenov, tréning len na assistant odpovediach
- Hyperparametre v3:
  - 3 epochy pri `lr=1e-4` (bolo 2 × `2e-4`) — pomalšie, stabilnejšie učenie
  - `warmup_ratio=0.10`, `weight_decay=0.01`
  - Cosine LR scheduler
  - Eval každých 50 krokov, `load_best_model_at_end=True`
- Adapter-only save namiesto `merged_4bit_forced`:
  - Prečo `merged_4bit_forced` zlyhá na 4-bit modeloch: dequant → merge → requant → LoRA deltas ≈ 0
  - Adapter save: ~65 MB vs. ~5 GB merged, a zachováva štruktúru pre DPO fázu

### 7.4 Fáza 2 — DPO s reálnymi zlyhaniami
- Načítanie SFT adaptéra ako **trainable** (`is_trainable=True`) — nie merge
- Prečo nie `merge_and_unload()`: zničilo by adapter na NF4 modeli
- DPOTrainer automaticky používa base model (adapter disabled) ako referenčný model
- Hyperparametre v3: 1 epocha (bolo 2), `beta=0.1`, `lr=5e-5`, `warmup_ratio=0.10`
- Výsledok: jeden adapter checkpoint obsahujúci SFT + DPO znalosti
- Výhoda reálnych vs. syntetických rejected samples:
  - Syntetické: GPT-4 generuje zámerne horšiu odpoveď → umelá distribúcia
  - Reálne: skutočná chybová distribúcia LLM → model sa učí vyhnúť sa reálnym pascám

### 7.5 Verziovanie a iterácia (v1 → v2 → v3)
- **v1** — prvý pokus: verbose CoT, r=64/alpha=16, `merged_4bit_forced` → zlyhanie pri DPO fáze
- **v2** — opravená architektúra: SFT adapter-only save, DPO trainable load, 2 epochy
- **v3** — LLM Council analýza: concise CoT, r=32/alpha=64, 1 DPO epocha, GRPO preskočené
- Rola **LLM Council** (multi-model analýza cez OpenRouter):
  - 4 frontier modely (GPT-4o, Gemini 2.0, DeepSeek v3, Claude Sonnet) súčasne analyzujú hyperparametre
  - Výsledok: jednomyseľné odporúčania v3 konfigurácie
  - Poznatky uložené v Obsidian vaulte (`obsidian-brain/Experiments/`, `Decisions/`)

---

## 8. Evaluácia a výsledky (~8 strán)

### 8.1 Evaluačná metodológia
- Fixné eval datasety — verzionované, nemenia sa medzi modelmi, garantujú porovnateľnosť
- Metriky:
  - **Precision** = $\frac{|predikované \cap správne|}{|predikované|}$
  - **Recall** = $\frac{|predikované \cap správne|}{|správne|}$
  - **F1** = harmonický priemer P a R vs. Apriori ground truth
  - **JSON parse rate** — % výstupov, ktoré sú validný JSON so správnou schémou
  - **Inference time** — sekundy na dataset
- Porovnávaná trojica: **Qwen2.5-7B base** vs. **fine-tuned model** vs. **Apriori** (oracle)

### 8.2 Výsledky SFT checkpointu
- F1 skóre po SFT fáze (konkrétne čísla)
- Kvalita `<think>` reasoning — štruktúrovanosť, správnosť krokov skenovania
- Typické chyby SFT modelu: nesprávne counts, chýbajúce singles, off-by-one v row labels
- Porovnanie s base modelom — prvé zlepšenie

### 8.3 Výsledky po DPO
- Zlepšenie oproti SFT (delta F1, ak merateľné)
- Pokles halucinácií evidence rows — hlavný cieľ DPO fázy
- JSON parse rate pred a po DPO
- Príklady: kde DPO pomohlo, kde nestačilo

### 8.4 Porovnanie s base modelom
- Kvantitatívne: F1 fine-tuned vs. F1 Qwen2.5-7B bez fine-tuningu
- Kvantitatívne: F1 fine-tuned vs. F1 GPT-4.1-mini (veľký base model)
- Kvalitatívne príklady:
  - Dataset kde base model halucinuje, fine-tuned uspeje
  - Dataset kde ani fine-tuned nestačí (limit modelu)
- Vizualizácie z `visualization.py`: histogramy F1, scatter plot P vs. R

### 8.5 Limity a ohraničenia
- Distribučný shift: syntetické datasety (tréning) vs. reálne dáta (produkcia)
- Závislosť od kvality Apriori ako oracle — Apriori je správny len pre definovaný `min_support` a `max_size`
- Veľkosť trénovacej sady: 348 SFT príkladov je relatívne malé
- Maximálna veľkosť datasetu: `seq_length=2048` tokenov limituje veľké CSV
- Čo model neumí: datasety s viac ako ~20 riadkami a ~15 stĺpcami (exponenciálna explózia kandidátov)

---

## 9. Diskusia (~3 strany)

### 9.1 Odpoveď na výskumnú otázku
- **Áno, s podmienkami** — fine-tuned Qwen2.5-7B sa priblíži Apriori výraznejšie ako base model, avšak len na datasetoch nepresahujúcich určitú veľkosť
- Kvantifikácia zlepšenia: base model F1 X → fine-tuned F1 Y → Apriori F1 1.0
- Čo fine-tuning naučil model: systematické stĺpcové skenovanie, dodržiavanie JSON schémy, vyhýbanie sa halucináciám row labels
- Čo fine-tuning nenaučil: presné počítanie pri veľkých kombinatoriách (model stále aproximuje)

### 9.2 Zaujímavé vedľajšie zistenia
- DPO z reálnych zlyhaní: konkrétny dopad na halucinácie (99.5 % → ?)
- Vplyv concise CoT formátu: kratší `<think>` blok vs. presnosť — trade-off
- LLM Council ako výskumná metodológia: 4 modely súčasne = rýchlejší konsenzus ako iteratívne pokusy
- `merged_4bit_forced` anti-pattern: praktické zistenie s priamym dopadom na reprodukovateľnosť výskumu
- Agentúrny workflow: prínosy pre dlhodobý iteratívny ML výskum

### 9.3 Možné vylepšenia a budúca práca
- **GRPO fáza (v4)**: podmienky kedy by mala zmysel — ak F1 > 0.60 po SFT+DPO
- Väčšia trénovacia sada (500+ SFT príkladov) — predpokladaný dopad
- Reálne (nie syntetické) datasety pre evaluáciu — väčšia externá validita
- Iné základné modely: Mistral-7B, Phi-4-mini, Llama-3.2-3B — porovnanie
- FP-Growth namiesto Apriori pre generáciu ground truth na väčších datasetoch
- Rozšírenie na asociačné pravidlá (nielen frequent itemsets)

---

## 10. Záver (~2 strany)

- Rekapitulácia problému: LLM ako potenciálny náhradník deterministického algoritmu pre frequent itemset mining
- Kľúčové výsledky:
  - F1 fine-tuned modelu: X (vs. base model Y, Apriori 1.0)
  - JSON parse rate: Z %
  - Hlavný prínos DPO: pokles halucinácií evidence rows
- Metodologický prínos: 3-fázový tréning pipeline (SFT-CoT → DPO-Real) ako opakovateľný postup
- Dostupnosť artefaktov:
  - Kód: [GitHub](https://github.com/oliversl1vka/itemsety-qwen-finetuning)
  - Dataset: [HuggingFace Hub — OliverSlivka/itemset-extraction-v2](https://huggingface.co/datasets/OliverSlivka/itemset-extraction-v2)
  - Model: [HuggingFace Hub — OliverSlivka/qwen2.5-7b-itemset-extractor](https://huggingface.co/OliverSlivka/qwen2.5-7b-itemset-extractor)
- Záverečné zhrnutie: fine-tuning malého LLM je realizovateľnou cestou k priblíženiu sa deterministickému algoritmu, avšak nie jeho náhradou — ide o komplementárny prístup s vlastnými silnými stránkami (prirodzený jazyk, integrácia do chatbotov) aj limitmi (presnosť na veľkých kombinatoriách)

---

## Literatúra (výber)

- Agrawal, R., & Srikant, R. (1994). Fast algorithms for mining association rules.
- Vaswani, A., et al. (2017). Attention is all you need.
- Wei, J., et al. (2022). Chain-of-thought prompting elicits reasoning in large language models.
- Hu, E., et al. (2021). LoRA: Low-rank adaptation of large language models.
- Dettmers, T., et al. (2023). QLoRA: Efficient finetuning of quantized LLMs.
- Rafailov, R., et al. (2023). Direct preference optimization: Your language model is secretly a reward model.
- Han, J., Pei, J., & Yin, Y. (2000). Mining frequent patterns without candidate generation (FP-Growth).

---

## Poznámky k rozsahu

| Kapitola | Odhadovaný rozsah |
|----------|-------------------|
| 1. Úvod | ~3 strany |
| 2. Teoretické základy | ~10 strán |
| 3. Navrhované riešenie | ~4 strany |
| 4. Dátová pipeline | ~8 strán |
| 5. Počiatočné pokusy | ~4 strany |
| 6. Agentúrny workflow | ~6 strán |
| 7. Fine-tuning | ~8 strán |
| 8. Evaluácia | ~8 strán |
| 9. Diskusia | ~3 strany |
| 10. Záver | ~2 strany |
| **Spolu** | **~56 strán** |

> Rozsah je orientačný. Kapitoly 2 a 4 môžeš skrátiť ak bude práca príliš dlhá. Kapitola 8 môže byť dlhšia ak budeš mať bohaté výsledky.

---

## Odporúčané poradie písania

1. **Kapitola 4** — dátová pipeline (poznáš ju najlepšie, máš kód)
2. **Kapitola 7** — fine-tuning (máš notebook a záznamy z Obsidian)
3. **Kapitola 6** — agentúrny workflow (máš `.github/agents/` ako základ)
4. **Kapitola 5** — počiatočné pokusy (krátka, píšeš z pamäte)
5. **Kapitola 3** — navrhované riešenie (prehľad pred detailmi)
6. **Kapitola 8** — výsledky (až keď budeš mať finálne čísla)
7. **Kapitola 9** — diskusia (po výsledkoch)
8. **Kapitola 1** — úvod (najlepšie písať posledný, keď vieš čo píšeš)
9. **Kapitola 10** — záver (po úvode)
10. **Kapitola 2** — príbuzné práce (najnáročnejšia, nechaj na záver)
