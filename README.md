# LangPro

LangPro is a language-guided framework for conditional protein sequence
generation from functional descriptions and ligand context. This repository
contains the data preparation, model, generation, and evaluation code.

## Repository layout

- `dataset/construction`: construction of the base protein-ligand-instruction dataset
- `dataset/expansion`: chemical, conservation, and structural feature expansion
- `model/langpro/architecture`: model components and structure-token planning
- `model/langpro/training`: supervised and preference-training entry points
- `model/langpro/generation`: sequence generation and output cleaning
- `model/langpro/evaluation`: ESMFold, DiffDock, and Chai-1 evaluation workflows

## Installation

Create a Python environment with the PyTorch build appropriate for your
hardware, then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

Chai-1, DiffDock, ESMFold, and DSSP are optional and are only required for
their corresponding evaluation or feature-extraction stages.

## Quick start

All commands below are run from the repository root.

```bash
python -m dataset.construction.run --list
python -m dataset.expansion.run --help
python -m model.langpro.training.train --help
python -m model.langpro.generation.generate --help
```

See `dataset/README.md` and `model/README.md` for the available workflows.

## Citation

Citation information will be added with the publication record.
