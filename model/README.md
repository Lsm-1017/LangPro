# LangPro model code

The model package is organized by responsibility:

- `architecture`: multimodal encoders, Perceiver routing, decoder, fold planner,
  and structure-token planner
- `training`: supervised training and preference optimization
- `generation`: conditional sequence generation and output cleaning
- `evaluation`: structure prediction, docking, complex prediction, and analysis

Run the entry points from the repository root:

```bash
python -m model.langpro.training.train --help
python -m model.langpro.training.train_preference --help
python -m model.langpro.generation.generate --help
python -m model.langpro.evaluation.stages.esmfold --help
python -m model.langpro.evaluation.stages.diffdock --help
python -m model.langpro.evaluation.stages.chai1 --help
python -m model.langpro.evaluation.analysis.summarize --help
```

The internal `DockingWizard*` class identifiers are retained to preserve
compatibility with existing checkpoints. The public package and framework name
are LangPro.
