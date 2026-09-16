# TEG-FND

Temporal Evidence-Graph Fusion Network for Fake News Detection.

## Current data contract
The supplied Twitter15/Twitter16 files contain `label.txt` and `source_tweets.txt`. They provide source-event text and four labels, but not propagation edges/timestamps. The final implementation therefore **does not fabricate graph data**. The graph interface is isolated in `graph_utils.py` for a future dataset acquisition.

## Active model
DeBERTa encoder → attention pooling → semantic/context projection → cross-view attention → adaptive mixture of experts → 4/6-class veracity head + uncertainty head.

## Evaluation contract
- Train/validation/test split: 70/15/15, stratified, event-level.
- Model selection: validation Macro-F1 only.
- Test set: evaluated only after the checkpoint is frozen.
- Primary metrics: Accuracy and Macro-F1.
- Secondary metrics: macro precision/recall and MCC.
- Twitter15/16: four-class classification.
- PolitiFact: six-class classification.

## Run
```bash
python main_train.py --dataset twitter15 --path data/twitter15
python main_train.py --dataset twitter16 --path data/twitter16
python main_train.py --dataset politifact --path data/politifact/politifact_factcheck_data.json
```

Do not report a benchmark number until the experiment is actually run.
