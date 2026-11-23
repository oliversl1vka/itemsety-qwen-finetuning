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

## Erratum (23 November 2026)

Later work by the author (the C26 follow-up campaign, 2026) re-examined the provenance of the 30 evaluation datasets used in this thesis. It found that 24 of the 30 cases share a source file with the training data. The statements in [docs/methodology/evaluation.md](docs/methodology/evaluation.md) that the 30 evaluation datasets are held out are therefore weaker than stated: the cases are not held out at the level of source files. The thesis results themselves are unchanged. The provenance finding is documented in the C26 provenance review and summarised in the follow-up C27 technical report (O. Slivka, 2026).
