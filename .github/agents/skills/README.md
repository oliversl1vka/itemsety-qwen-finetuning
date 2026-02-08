# Agent Skills

Skills are folders of instructions that agents use to complete specialized tasks. Each skill is self-contained with a `SKILL.md` file containing the instructions and metadata.

## Available Skills

| Skill | Description | Primary Agent |
|-------|-------------|---------------|
| [csv-dataset-generation](csv-dataset-generation/) | Generate synthetic CSV datasets for training | Dataset Agent |
| [apriori-extraction](apriori-extraction/) | Extract frequent itemsets using Apriori algorithm | Pipeline Agent |
| [llm-itemset-extraction](llm-itemset-extraction/) | Extract itemsets using OpenAI | Pipeline Agent |
| [validation-pipeline](validation-pipeline/) | Validate itemsets against 13 invariants | Pipeline Agent |
| [sqlite-persistence](sqlite-persistence/) | Store and query run metadata | Pipeline Agent |
| [training-data-export](training-data-export/) | Export validated runs as training examples | Training Agent |
| [qwen-finetuning](qwen-finetuning/) | Fine-tune Qwen models with LoRA/QLoRA | Training Agent |
| [model-evaluation](model-evaluation/) | Evaluate model performance vs Apriori | Evaluation Agent |
| [huggingface-deployment](huggingface-deployment/) | Deploy to HuggingFace Hub | Deployment Agent |
| [metrics-visualization](metrics-visualization/) | Generate charts and dashboards | Monitoring Agent |
| [agent-logging](agent-logging/) | Record activities and insights | All Agents |

## Skill Format

Each skill follows the [Anthropic Skills](https://github.com/anthropics/skills) format:

```markdown
---
name: skill-name
description: Clear description of what this skill does and when to use it
---

# Skill Title

Instructions, examples, and guidelines that the agent follows.

## Quick Start
...

## Parameters
...

## Examples
...

## Troubleshooting
...
```

## Usage

### In Agent Files
Reference skills in agent definition files:
```markdown
# Skills

Use these skills for specialized tasks:
- `csv-dataset-generation` — when generating new datasets
- `apriori-extraction` — when running Apriori algorithm
- `validation-pipeline` — when validating outputs
```

### During Task Execution
Agents load skills dynamically when performing specialized tasks:
1. Identify task type
2. Load relevant skill(s)
3. Follow skill instructions
4. Log completion to obsidian-brain/Logs/

## Creating New Skills

1. Create folder: `agents/skills/{skill-name}/`
2. Add `SKILL.md` with frontmatter and instructions
3. Reference in relevant agent files
4. Test skill execution

### Template
```markdown
---
name: new-skill-name
description: What this skill does and when to use it
---

# Skill Title

## Overview
Brief description of the skill's purpose.

## Quick Start
Minimal example to get started.

## Detailed Instructions
Step-by-step guide.

## Parameters/Options
Configuration options if applicable.

## Output
What the skill produces.

## Troubleshooting
Common issues and solutions.
```

## Skill Dependencies

Some skills depend on others:

```
csv-dataset-generation
    └── (none)

apriori-extraction
    └── csv-dataset-generation (for input)

llm-itemset-extraction
    └── csv-dataset-generation (for input)

validation-pipeline
    ├── apriori-extraction (for ground truth)
    └── llm-itemset-extraction (for predictions)

sqlite-persistence
    └── validation-pipeline (for validated data)

training-data-export
    └── sqlite-persistence (for query)

qwen-finetuning
    └── training-data-export (for dataset)

model-evaluation
    ├── apriori-extraction (for ground truth)
    └── qwen-finetuning (for model)

huggingface-deployment
    └── qwen-finetuning (for model)

metrics-visualization
    └── sqlite-persistence (for data)

agent-logging
    └── (used by all)
```

## Directory Structure

```
agents/skills/
├── README.md                      # This file
├── TOOLS_REGISTRY.md              # Tool definitions
├── csv-dataset-generation/
│   └── SKILL.md
├── apriori-extraction/
│   └── SKILL.md
├── llm-itemset-extraction/
│   └── SKILL.md
├── validation-pipeline/
│   └── SKILL.md
├── sqlite-persistence/
│   └── SKILL.md
├── training-data-export/
│   └── SKILL.md
├── qwen-finetuning/
│   └── SKILL.md
├── model-evaluation/
│   └── SKILL.md
├── huggingface-deployment/
│   └── SKILL.md
├── metrics-visualization/
│   └── SKILL.md
└── agent-logging/
    └── SKILL.md
```

## Related Resources

- [TOOLS_REGISTRY.md](TOOLS_REGISTRY.md) — Tool definitions for agents
- [../AGENTS.md](../../AGENTS.md) — Main agent documentation
- [Agent definition files](../) — Individual agent specifications
