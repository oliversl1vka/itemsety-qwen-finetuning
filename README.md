# Itemset Extraction via Fine-Tuned Qwen2.5-7B

Code repository for a bachelor's thesis on frequent itemset extraction with Apriori-generated supervision and fine-tuned Qwen2.5-7B models.

This repo is intentionally kept lightweight and reusable. Generated datasets, training exports, HuggingFace dataset shards, local databases, logs, artifacts, and evaluation outputs are not committed. See [data/README.md](data/README.md) for the data policy.

The detailed thesis journey, methodology, experiments, decisions, and results are documented in [docs/index.md](docs/index.md).

## Quick Start

```bash
pip install -r requirements.txt
cp openai.env.template openai.env
python pipeline.py --data path/to/dataset.csv --min-support 3 --max-size 3
```

For full usage, see [docs/quickstart.md](docs/quickstart.md).

## License

See [LICENSE](LICENSE).
