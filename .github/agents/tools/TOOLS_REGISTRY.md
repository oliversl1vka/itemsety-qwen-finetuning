# Agent Tools Registry

Centrálny register nástrojov pre všetkých agentov v itemsety-qwen-finetuning projekte.

## Filozofia

- **Žiadna redundancia** — jeden tool má jedného vlastníka, ostatní ho používajú cez referenciu
- **Efektivita** — len nástroje, ktoré aktívne potrebujeme
- **Funkcionalita pred štruktúrou** — menej súborov, viac akcie

---

## Tool Ownership Matrix

| Tool | Vlastník | Používatelia | Kategória |
|------|----------|--------------|-----------|
| `sqlite_query` | pipeline-agent | monitoring, evaluation, training | Database |
| `csv_loader` | dataset-agent | pipeline | File Operations |
| `json_reader` | orchestrator | všetci | File Operations |
| `json_writer` | orchestrator | všetci | File Operations |
| `shell_exec` | orchestrator | deployment, cleanup | Code Interpreter |
| `python_exec` | orchestrator | training, evaluation | Code Interpreter |
| `git_ops` | deployment-agent | cleanup, maintainer | Productivity |
| `hf_hub_api` | deployment-agent | training | External API |
| `openai` | pipeline-agent | - | External API |
| `file_search` | maintainer-agent | cleanup | File Operations |
| `md_writer` | monitoring-agent | maintainer | File Operations |
| `log_writer` | orchestrator | všetci | File Operations |
| `memory_writer` | orchestrator | všetci | File Operations |

---

## 1. Orchestrator Agent Tools

**Rola:** Master koordinátor — vlastní základné nástroje zdieľané všetkými agentmi.

```yaml
toolkit:
  name: orchestrator-toolkit
  description: Workflow coordination and shared utilities
  
  tools:
    # === Core Execution ===
    - name: shell_exec
      description: Execute shell commands in zsh/bash
      params: {command: str, cwd?: str, timeout?: int}
      returns: {stdout: str, stderr: str, exit_code: int}
      
    - name: python_exec
      description: Execute Python script or code snippet
      params: {script?: str, code?: str, args?: list}
      returns: {output: str, error?: str, exit_code: int}
      
    # === File Operations ===
    - name: json_reader
      description: Read and parse JSON file
      params: {path: str}
      returns: {data: dict | list}
      
    - name: json_writer
      description: Write data to JSON file
      params: {path: str, data: dict | list, indent?: int}
      returns: {success: bool}
      
    - name: log_writer
      description: Write activity log to obsidian-brain/Logs/
      params: {agent: str, action: str, details: dict}
      returns: {log_path: str}
      
    - name: memory_writer
      description: Append insight to obsidian-brain/Agents/
      params: {agent: str, title: str, context: str, insight: str, application: str}
      returns: {success: bool}
      
    # === Workflow Control ===
    - name: checkpoint_save
      description: Save workflow checkpoint for recovery
      params: {workflow_id: str, stage: str, state: dict}
      returns: {checkpoint_path: str}
      
    - name: checkpoint_load
      description: Load workflow checkpoint
      params: {workflow_id: str}
      returns: {stage: str, state: dict} | null
      
    # === Agent Communication ===
    - name: agent_invoke
      description: Invoke another agent with task
      params: {agent: str, task: str, params: dict}
      returns: {result: any, status: str}
      
    - name: broadcast
      description: Send message to all agents
      params: {message_type: str, payload: dict}
      returns: {delivered_to: list[str]}
```

---

## 2. Dataset Agent Tools

**Rola:** Generovanie a správa CSV datasetov.

```yaml
toolkit:
  name: dataset-toolkit
  description: CSV dataset generation and management
  
  tools:
    # === Core Generation ===
    - name: csv_generator
      description: Generate synthetic CSV dataset
      params: {rows: int, cols: int, seed?: int, strategy?: str}
      returns: {path: str, hash: str, metadata: dict}
      script: generate_datasets_v2.py
      
    - name: csv_loader
      description: Load and parse CSV file (auto-detect format)
      params: {path: str}
      returns: {transactions: list[set], rows: int, cols: int, items: list}
      script: pipeline.py::load_transactions_csv
      
    - name: csv_validator
      description: Validate dataset quality metrics
      params: {path: str}
      returns: {valid: bool, metrics: dict, issues: list}
      
    # === Metadata ===
    - name: hash_dataset
      description: Compute SHA256 hash (first 12 chars)
      params: {path: str}
      returns: {hash: str}
      
    - name: log_generation
      description: Log dataset metadata to generation_log.json
      params: {dataset_id: str, path: str, metadata: dict}
      returns: {success: bool}
      
  uses_from_orchestrator:
    - json_writer
    - log_writer
```

---

## 3. Pipeline Agent Tools

**Rola:** Apriori + LLM extraction + validácia.

```yaml
toolkit:
  name: pipeline-toolkit
  description: Frequent itemset extraction pipeline
  
  tools:
    # === Mining ===
    - name: apriori_mine
      description: Run Apriori algorithm on transactions
      params: {transactions: list, min_support: int, max_size: int}
      returns: {itemsets: list[dict], duration_s: float}
      script: pipeline.py::apriori_frequent_itemsets
      
    # === LLM Extraction ===
    - name: openai_call
      description: Call OpenAI API for itemset extraction
      params: {prompt: str, model?: str, chunk_size?: int}
      returns: {response: str, itemsets: list, tokens_used: int, duration_s: float}
      requires: openai.env
      
    - name: llm_parse_response
      description: Parse LLM JSON response with error handling
      params: {response: str}
      returns: {itemsets: list | null, parse_error?: str}
      
    # === Validation ===
    - name: validate_itemsets
      description: Run 13 validation invariants
      params: {apriori: list, llm: list, transactions: list}
      returns: {passed: bool, failures: list[dict], report: dict}
      script: pipeline.py::validate_all
      
    # === Database ===
    - name: sqlite_query
      description: Execute SQL query on runs.db
      params: {query: str, params?: list}
      returns: {rows: list[dict], affected?: int}
      
    - name: sqlite_insert_run
      description: Insert run metadata to runs.db
      params: {run_data: dict}
      returns: {run_id: int}
      script: pipeline.py::persist_run_to_sqlite
      
    # === Artifacts ===
    - name: artifact_writer
      description: Write artifact JSON with hash-based naming
      params: {stage: str, dataset_stem: str, hash: str, data: dict}
      returns: {path: str}
      
  uses_from_orchestrator:
    - json_reader
    - json_writer
    - log_writer
    
  uses_from_dataset:
    - csv_loader
```

---

## 4. Training Agent Tools

**Rola:** Export training dát a fine-tuning modelov.

```yaml
toolkit:
  name: training-toolkit
  description: Model fine-tuning and data preparation
  
  tools:
    # === Data Export ===
    - name: export_training_examples
      description: Export validated runs as training examples
      params: {db_path?: str, min_itemsets?: int, output_dir?: str}
      returns: {count: int, output_path: str}
      script: src/training/export_training_data.py
      
    - name: create_hf_dataset
      description: Convert training data to HuggingFace Dataset format
      params: {input_dir: str, output_dir: str, train_split?: float}
      returns: {train_count: int, val_count: int, path: str}
      script: src/training/create_hf_dataset.py
      
    # === Training ===
    - name: sft_train
      description: Run SFT training with LoRA/QLoRA
      params: {model_name: str, dataset_path: str, config?: dict}
      returns: {model_path: str, metrics: dict, duration_s: float}
      script: src/training/run_sft_full.py
      
    - name: training_monitor
      description: Monitor training progress and metrics
      params: {run_dir: str}
      returns: {step: int, loss: float, lr: float, eta: str}
      
    # === Model Management ===
    - name: model_merge
      description: Merge LoRA adapter with base model
      params: {base_model: str, adapter_path: str, output_path: str}
      returns: {merged_path: str}
      
  uses_from_orchestrator:
    - python_exec
    - json_writer
    - log_writer
    
  uses_from_pipeline:
    - sqlite_query
    
  uses_from_deployment:
    - hf_hub_upload
```

---

## 5. Evaluation Agent Tools

**Rola:** Testovanie a vyhodnocovanie modelov.

```yaml
toolkit:
  name: evaluation-toolkit
  description: Model evaluation and benchmarking
  
  tools:
    # === Inference ===
    - name: model_inference
      description: Run inference on single dataset
      params: {model_path: str, csv_path: str, max_tokens?: int}
      returns: {response: str, itemsets: list, duration_s: float}
      script: eval_finetuned_model.py
      
    - name: batch_inference
      description: Run inference on multiple datasets
      params: {model_path: str, dataset_dir: str, count?: int}
      returns: {results: list[dict], total_duration_s: float}
      
    # === Metrics ===
    - name: compute_metrics
      description: Compute P/R/F1 vs Apriori ground truth
      params: {predicted: list, ground_truth: list}
      returns: {precision: float, recall: float, f1: float, exact_match: bool}
      
    - name: aggregate_metrics
      description: Aggregate metrics across multiple evaluations
      params: {results: list[dict]}
      returns: {avg_precision: float, avg_recall: float, avg_f1: float, 
                parse_rate: float, hallucination_rate: float, avg_time_s: float}
      
    # === Analysis ===
    - name: failure_analyzer
      description: Analyze common failure patterns
      params: {results: list[dict]}
      returns: {patterns: list[dict], recommendations: list[str]}
      
    - name: comparison_report
      description: Compare two models side by side
      params: {model_a_results: dict, model_b_results: dict}
      returns: {comparison: dict, winner: str, diff: dict}
      
  uses_from_orchestrator:
    - json_writer
    - log_writer
    
  uses_from_pipeline:
    - apriori_mine
    - sqlite_query
    
  uses_from_dataset:
    - csv_loader
```

---

## 6. Deployment Agent Tools

**Rola:** Publikovanie modelov a datasetov na HuggingFace Hub.

```yaml
toolkit:
  name: deployment-toolkit
  description: HuggingFace Hub deployment and management
  
  tools:
    # === HuggingFace Hub ===
    - name: hf_hub_upload
      description: Upload model or dataset to HF Hub
      params: {local_path: str, repo_id: str, repo_type: str, private?: bool}
      returns: {url: str, commit: str}
      requires: HF_TOKEN env
      
    - name: hf_hub_download
      description: Download model or dataset from HF Hub
      params: {repo_id: str, local_path: str}
      returns: {path: str}
      
    - name: hf_model_card
      description: Generate and push model card (README.md)
      params: {repo_id: str, template: str, metadata: dict}
      returns: {success: bool}
      
    # === Gradio Space ===
    - name: space_deploy
      description: Deploy Gradio app to HF Spaces
      params: {space_id: str, files: list[str], requirements?: list[str]}
      returns: {url: str, status: str}
      script: deploy_to_hf_space.ps1
      
    - name: space_status
      description: Check HF Space health and status
      params: {space_id: str}
      returns: {status: str, runtime: str, last_updated: str}
      
    # === Git Operations ===
    - name: git_ops
      description: Git operations (commit, push, tag)
      params: {operation: str, args?: dict}
      returns: {success: bool, output: str}
      operations: [commit, push, pull, tag, status]
      
  uses_from_orchestrator:
    - shell_exec
    - log_writer
```

---

## 7. Monitoring Agent Tools

**Rola:** Vizualizácie, metriky a reporting.

```yaml
toolkit:
  name: monitoring-toolkit
  description: Observability, metrics, and reporting
  
  tools:
    # === Visualization ===
    - name: plot_generator
      description: Generate matplotlib charts
      params: {chart_type: str, data: dict, output_path: str, title?: str}
      returns: {path: str}
      chart_types: [bar, line, scatter, histogram, heatmap, comparison]
      script: visualization.py
      
    - name: dashboard_update
      description: Update metrics dashboard
      params: {metrics: dict}
      returns: {dashboard_path: str}
      
    # === Metrics Collection ===
    - name: db_metrics
      description: Compute metrics from runs.db
      params: {time_range?: str, group_by?: str}
      returns: {metrics: dict}
      
    - name: system_metrics
      description: Collect system resource usage
      params: {}
      returns: {cpu: float, memory: float, disk: float, gpu?: dict}
      
    # === Reporting ===
    - name: md_writer
      description: Generate markdown report
      params: {template: str, data: dict, output_path: str}
      returns: {path: str}
      
    - name: alert_sender
      description: Send alert notification
      params: {severity: str, title: str, message: str, channel?: str}
      returns: {sent: bool}
      severities: [info, warning, error, critical]
      
  uses_from_orchestrator:
    - json_reader
    - log_writer
    
  uses_from_pipeline:
    - sqlite_query
```

---

## 8. Maintainer Agent Tools

**Rola:** Audit a údržba agent súborov.

```yaml
toolkit:
  name: maintainer-toolkit
  description: Agent file maintenance and accuracy
  
  tools:
    # === Audit ===
    - name: file_search
      description: Search for files by pattern
      params: {pattern: str, directory?: str}
      returns: {files: list[str]}
      
    - name: content_grep
      description: Search content in files
      params: {pattern: str, files: list[str], is_regex?: bool}
      returns: {matches: list[dict]}
      
    - name: reference_checker
      description: Verify file references exist
      params: {agent_file: str}
      returns: {valid: list, invalid: list, missing: list}
      
    # === Validation ===
    - name: command_tester
      description: Test if documented command works
      params: {command: str}
      returns: {success: bool, output?: str, error?: str}
      
    - name: code_example_validator
      description: Validate code examples are syntactically correct
      params: {agent_file: str}
      returns: {valid: int, invalid: int, issues: list}
      
    # === Reporting ===
    - name: drift_reporter
      description: Generate drift detection report
      params: {audit_results: dict}
      returns: {report_path: str, drift_count: int}
      
  uses_from_orchestrator:
    - shell_exec
    - json_writer
    - log_writer
    
  uses_from_monitoring:
    - md_writer
```

---

## 9. Cleanup Agent Tools

**Rola:** Organizácia a čistenie repozitára.

```yaml
toolkit:
  name: cleanup-toolkit
  description: Repository organization and hygiene
  
  tools:
    # === Analysis ===
    - name: file_inventory
      description: Generate complete file inventory
      params: {exclude?: list[str]}
      returns: {files: list[dict], total_count: int, total_size: int}
      
    - name: duplicate_finder
      description: Find duplicate files by content hash
      params: {directory?: str}
      returns: {duplicates: list[list[str]]}
      
    - name: orphan_detector
      description: Find unreferenced files
      params: {file_type: str}
      returns: {orphans: list[str]}
      
    # === Operations ===
    - name: file_archiver
      description: Move file to archive with metadata
      params: {source: str, reason: str}
      returns: {archive_path: str, meta_path: str}
      
    - name: file_mover
      description: Move file to new location
      params: {source: str, destination: str}
      returns: {success: bool}
      
    - name: doc_consolidator
      description: Merge multiple docs into one
      params: {sources: list[str], destination: str}
      returns: {path: str, lines_merged: int}
      
    # === Safety ===
    - name: backup_creator
      description: Create backup before major changes
      params: {scope?: str}
      returns: {backup_path: str}
      
  uses_from_orchestrator:
    - shell_exec
    - log_writer
    
  uses_from_deployment:
    - git_ops
    
  uses_from_maintainer:
    - file_search
```

---

## Tool Implementation Priority

### Phase 1 — Core (Immediate)
1. `json_reader` / `json_writer` — základné I/O
2. `log_writer` / `memory_writer` — logging infraštruktúra
3. `sqlite_query` — prístup k runs.db
4. `shell_exec` — spúšťanie skriptov

### Phase 2 — Workflow (Week 1-2)
5. `checkpoint_save` / `checkpoint_load` — recovery
6. `agent_invoke` — inter-agent komunikácia
7. `csv_loader` — dataset handling
8. `apriori_mine` — mining wrapper

### Phase 3 — Specialized (Week 3-4)
9. `hf_hub_upload` — deployment
10. `plot_generator` — visualization
11. `file_search` / `content_grep` — audit
12. `duplicate_finder` — cleanup

---

## Použitie v Agent Files

Každý agent má v svojom súbore sekciu `# Tools` s odkazom na tento register:

```markdown
# Tools

See [tools/TOOLS_REGISTRY.md](../tools/TOOLS_REGISTRY.md) for full tool definitions.

## Primary Tools (owned by this agent)
- `tool_name_1` — brief description
- `tool_name_2` — brief description

## Shared Tools (from other agents)
- `orchestrator::json_reader` — read JSON files
- `pipeline::sqlite_query` — query runs.db
```

---

**Last Updated:** 2026-02-01
