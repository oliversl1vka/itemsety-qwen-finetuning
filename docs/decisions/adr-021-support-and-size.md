# ADR-021: min_support=3, max_size=3

**Status:** Accepted  
**Date:** 2026-02

## Context

The Apriori algorithm requires two parameters: minimum support count and maximum itemset size. These parameters define the difficulty and scope of the extraction task.

## Options Considered

| min_support | Effect on Small Datasets (5-10 rows) | Effect on Large Datasets (15-25 rows) |
|-------------|--------------------------------------|--------------------------------------|
| 2 | Combinatorial explosion (nearly everything qualifies) | Many itemsets |
| **3** | Meaningful patterns without noise | Good coverage |
| 4 | Prunes almost everything on 5-row datasets | Some patterns |
| 5 | Nothing qualifies on small datasets | Sparse patterns |

| max_size | Complexity | Interpretability |
|----------|-----------|-----------------|
| 2 | Low (only singles + pairs) | High |
| **3** | Moderate (singles + pairs + triples) | Good |
| 4 | High (exponential growth) | Lower |

## Decision

**min_support=3, max_size=3** as the default parameters.

## Rationale

For datasets with 5--25 rows:

- **min_support=2** causes combinatorial explosion: in a 10-row dataset with 8 columns, nearly every pair of items co-occurs at least twice. This makes the task trivially easy and the output impractically large.
- **min_support=4** on a 5-row dataset prunes nearly everything, leaving too few itemsets for meaningful training.
- **min_support=3** is the empirical sweet spot: it produces enough itemsets for training while filtering noise.

For max_size:

- **max_size=3** covers singles, pairs, and triples -- the most common itemset sizes in practice
- **max_size=4** causes exponential growth in the search space with diminishing returns (size-4 itemsets are rare in 3--15 column datasets)

## Source Evidence

- `pipeline.py:595-596` -- argparse defaults
- `pipeline.py:401-466` -- `apriori_frequent_itemsets` function
