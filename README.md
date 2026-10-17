# Freight Rate Prediction Challenge

This repository contains a reproducible solution for the Spotter freight-rate assessment.

## Contents

- `train_predict.py` performs a chronological holdout comparison, fits the selected model, creates the 12,000 validation predictions, and fills the 31 December chart inputs.
- `data/train_test.csv` is the labeled development dataset. The prediction target is `posted_rate`.
- `data/validation.csv` contains the 12,000 loads for final predictions.
- `data/validation_predictions_template.csv` is the required ID template.
- `data/december_chart_inputs.csv` contains the 31 fixed-route December inputs; its `predicted_rate` column is filled by the pipeline.
- `score.py` validates the output files and creates `scorer_results/candidate_december.png`.
- `validation_metrics.json` records the October holdout metrics and selected model.
- `freight_rate_prediction_report.docx` summarizes the validation split, model comparison, data checks, and December prediction chart.

## Run

From this repository directory, install dependencies and run:

```powershell
python -m pip install -r requirements.txt
python train_predict.py
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
```

Before running, place the provided training, validation, and prediction-template CSVs in `data/` using the filenames above. Those raw input files are excluded from GitHub; the completed prediction outputs remain in the repository.

The prediction model uses pandas and NumPy. The supplied scorer uses Matplotlib to make the December chart.

## Validation approach

The labeled data runs from January through October 2025. January through September are used for fitting, and October is held out as a future-month validation set. Two feature sets are compared on that holdout. The selected model is refit on all labeled rows before predicting the final validation loads. The December chart model uses fields shared with its input file because that file does not contain coordinates, `market_index`, or `quote_signal`.

The scorer checks file structure, IDs, row counts, dates, fixed December inputs, and positive predictions. It does not calculate prediction accuracy; Spotter calculates final validation metrics after submission.
