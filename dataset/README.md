# Dataset pipeline

The dataset code is divided into base construction and feature expansion.
Run commands from the repository root so that relative paths resolve
consistently.

## Base construction

The construction stages are kept as separate scripts because each stage
produces an auditable intermediate table. A single runner provides the public
interface:

```bash
python -m dataset.construction.run --list
python -m dataset.construction.run --step 13_generate_ligand_3d_rdkit
python -m dataset.construction.run --group structure --dry-run
```

Arguments following a single `--step` are forwarded to that stage. Generated
tables and downloaded source files are written outside the tracked source tree.

## Feature expansion

The expansion pipeline adds RDKit descriptors, ESM-2 conservation scores, and
AlphaFold-derived structural annotations:

```bash
python -m dataset.expansion.run \
  --input-csv data/dataset.csv \
  --output-dir outputs/dataset \
  --phases 1 2 3 \
  --download-af
```

Useful auxiliary operations are also exposed through this entry point:

```bash
python -m dataset.expansion.run \
  --input-csv data/dataset.csv \
  --export-af-ids outputs/uniprot_ids.txt

python -m dataset.expansion.run --check-alphafold P00533
```
