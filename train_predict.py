from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
TRAIN_PATH = DATA / "train_test.csv"
VALIDATION_PATH = DATA / "validation.csv"
TEMPLATE_PATH = DATA / "validation_predictions_template.csv"
DECEMBER_PATH = DATA / "december_chart_inputs.csv"
OUTPUT_PATH = ROOT / "validation_predictions.csv"
DECEMBER_OUTPUT_PATH = DATA / "december_chart_inputs.csv"
METRICS_PATH = ROOT / "validation_metrics.json"
COMMON_NUMERIC = ["distance", "weight"]
EXTRA_NUMERIC = [
    "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon",
    "market_index", "quote_signal",
]
CATEGORICAL = ["pickup", "delivery", "equipment"]


def _date_features(frame: pd.DataFrame) -> pd.DataFrame:
    dates = pd.to_datetime(frame["date"], errors="raise")
    day = (dates - pd.Timestamp("2025-01-01")).dt.days.to_numpy(dtype=float)
    day_of_year = dates.dt.dayofyear.to_numpy(dtype=float)
    return pd.DataFrame(
        {
            "day_trend": day / 365.0,
            "day_sin": np.sin(2.0 * np.pi * day_of_year / 365.25),
            "day_cos": np.cos(2.0 * np.pi * day_of_year / 365.25),
        },
        index=frame.index,
    )


def _rate_encodings(
    fit_frame: pd.DataFrame,
    fit_target: np.ndarray,
    frame: pd.DataFrame,
    leave_one_out: bool,
) -> pd.DataFrame:
    # Target is rate per mile. Leave-one-out values prevent a training row
    # from directly revealing its own target through its category average.
    result: dict[str, np.ndarray] = {}
    global_mean = float(np.average(fit_target))
    for name, columns in {
        "route": ["pickup", "delivery", "equipment"],
        "pickup": ["pickup"],
        "delivery": ["delivery"],
        "equipment": ["equipment"],
    }.items():
        group_key = fit_frame[columns].astype("string").fillna("<missing>").agg("|".join, axis=1)
        mapping_frame = pd.DataFrame({"key": group_key.to_numpy(), "target": fit_target})
        stats = mapping_frame.groupby("key")["target"].agg(["sum", "count"])
        smooth = 20.0
        if leave_one_out:
            current_key = frame[columns].astype("string").fillna("<missing>").agg("|".join, axis=1)
            own = pd.Series(fit_target, index=frame.index)
            sums = current_key.map(stats["sum"]).fillna(0.0).to_numpy(dtype=float) - own.to_numpy(dtype=float)
            counts = current_key.map(stats["count"]).fillna(0.0).to_numpy(dtype=float) - 1.0
            counts = np.maximum(counts, 0.0)
        else:
            current_key = frame[columns].astype("string").fillna("<missing>").agg("|".join, axis=1)
            sums = current_key.map(stats["sum"]).fillna(0.0).to_numpy(dtype=float)
            counts = current_key.map(stats["count"]).fillna(0.0).to_numpy(dtype=float)
        result[f"target_rate_{name}"] = (sums + smooth * global_mean) / (counts + smooth)
    return pd.DataFrame(result, index=frame.index)


def _design(
    fit_frame: pd.DataFrame,
    fit_target: np.ndarray,
    frame: pd.DataFrame,
    full_features: bool,
    leave_one_out: bool,
) -> tuple[np.ndarray, list[str]]:
    numeric_names = COMMON_NUMERIC + (EXTRA_NUMERIC if full_features else [])
    fit_numeric = fit_frame[numeric_names].apply(pd.to_numeric, errors="coerce")
    medians = fit_numeric.median().fillna(0.0)
    numeric = frame[numeric_names].apply(pd.to_numeric, errors="coerce").fillna(medians)
    numeric = numeric.copy()
    numeric["distance_sqrt"] = np.sqrt(np.maximum(numeric["distance"], 0.0)) / 30.0
    numeric["distance_log"] = np.log1p(np.maximum(numeric["distance"], 0.0))
    numeric["distance_squared"] = (numeric["distance"] / 1000.0) ** 2
    numeric["weight_scaled"] = numeric["weight"] / 10000.0
    if full_features:
        numeric["market_x_distance"] = numeric["market_index"] * numeric["distance"] / 1000.0
        numeric["quote_x_distance"] = numeric["quote_signal"] * numeric["distance"] / 1000.0
        for col in ["weight", "market_index", "quote_signal"]:
            numeric[f"{col}_missing"] = frame[col].isna().astype(float)

    categorical = {}
    for col in CATEGORICAL:
        levels = sorted(fit_frame[col].dropna().astype(str).unique().tolist())
        categories = [*levels, "<unknown>", "<missing>"]
        values = frame[col].astype("string").fillna("<missing>")
        values = values.where(values.isin(levels + ["<missing>"]), "<unknown>")
        categorical[col] = pd.Categorical(values, categories=categories)
    dummy = pd.get_dummies(pd.DataFrame(categorical, index=frame.index), dtype=float)
    dates = _date_features(frame)
    encodings = _rate_encodings(fit_frame, fit_target, frame, leave_one_out)
    parts = [numeric.reset_index(drop=True), dates.reset_index(drop=True), dummy.reset_index(drop=True), encodings.reset_index(drop=True)]
    matrix_frame = pd.concat(parts, axis=1)
    values = matrix_frame.to_numpy(dtype=float)
    values = np.column_stack([np.ones(len(values), dtype=float), values])
    names = ["intercept", *matrix_frame.columns.astype(str).tolist()]
    return values, names


def _fit_predict(
    fit_frame: pd.DataFrame,
    fit_target: np.ndarray,
    predict_frame: pd.DataFrame,
    full_features: bool,
) -> np.ndarray:
    x_train, _ = _design(fit_frame, fit_target, fit_frame, full_features, leave_one_out=True)
    x_predict, _ = _design(fit_frame, fit_target, predict_frame, full_features, leave_one_out=False)
    # Tiny ridge penalty stabilizes sparse and correlated category columns.
    penalty = np.eye(x_train.shape[1], dtype=float) * 0.02
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(x_train.T @ x_train + penalty, x_train.T @ fit_target)
    return x_predict @ coefficients


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    predicted = np.maximum(predicted, 0.01)
    error = predicted - actual
    return {
        "mae_dollars": float(np.mean(np.abs(error))),
        "rmse_dollars": float(np.sqrt(np.mean(error**2))),
        "mape_percent": float(np.mean(np.abs(error) / actual) * 100.0),
        "r2": float(1.0 - np.sum(error**2) / np.sum((actual - actual.mean()) ** 2)),
    }


def main() -> None:
    train = pd.read_csv(TRAIN_PATH)
    validation = pd.read_csv(VALIDATION_PATH)
    template = pd.read_csv(TEMPLATE_PATH)
    december = pd.read_csv(DECEMBER_PATH)
    if "posted_rate" not in train or train["posted_rate"].isna().any():
        raise ValueError("Training data must have a complete posted_rate target.")
    if train["load_id"].duplicated().any() or validation["load_id"].duplicated().any():
        raise ValueError("load_id values must be unique in train and validation data.")

    train["date"] = pd.to_datetime(train["date"], errors="raise")
    validation["date"] = pd.to_datetime(validation["date"], errors="raise")
    december["date"] = pd.to_datetime(december["date"], errors="raise")
    labels = train["posted_rate"].to_numpy(dtype=float)
    target_per_mile = labels / train["distance"].to_numpy(dtype=float)

    # The latest complete month in the labeled data is held out to mimic
    # predicting future loads. No random row split is used.
    holdout_start = pd.Timestamp("2025-10-01")
    fit_mask = train["date"] < holdout_start
    dev_fit = train.loc[fit_mask].reset_index(drop=True)
    dev_holdout = train.loc[~fit_mask].reset_index(drop=True)
    dev_target = (dev_fit["posted_rate"] / dev_fit["distance"]).to_numpy(dtype=float)
    actual_holdout = dev_holdout["posted_rate"].to_numpy(dtype=float)
    candidates = {}
    for full in (False, True):
        pred_per_mile = _fit_predict(dev_fit, dev_target, dev_holdout, full)
        candidates["extended" if full else "shared_features"] = _metrics(
            actual_holdout,
            pred_per_mile * dev_holdout["distance"].to_numpy(dtype=float),
        )
    chosen = min(candidates, key=lambda key: candidates[key]["mae_dollars"])
    use_full = chosen == "extended"

    validation_per_mile = _fit_predict(train, target_per_mile, validation, use_full)
    validation_rates = np.maximum(
        validation_per_mile * validation["distance"].to_numpy(dtype=float), 0.01
    )
    template_ids = template["load_id"].astype(str)
    predictions = pd.DataFrame({"load_id": validation["load_id"].astype(str), "predicted_rate": validation_rates})
    predictions = template[["load_id"]].astype({"load_id": "string"}).merge(
        predictions, on="load_id", how="left", validate="one_to_one"
    )
    if predictions["predicted_rate"].isna().any() or set(template_ids) != set(predictions["load_id"]):
        raise ValueError("Could not match every validation prediction to the supplied template.")
    predictions.to_csv(OUTPUT_PATH, index=False, float_format="%.2f")

    # December inputs omit coordinates and market/quote signals, so a model
    # trained only on shared columns is used for this fixed-route series.
    december_per_mile = _fit_predict(train, target_per_mile, december, full_features=False)
    december["predicted_rate"] = np.maximum(
        december_per_mile * december["distance"].to_numpy(dtype=float), 0.01
    )
    december["date"] = december["date"].dt.strftime("%Y-%m-%d")
    december.to_csv(DECEMBER_OUTPUT_PATH, index=False, float_format="%.2f")

    result = {
        "time_split": {
            "training_rows": int(fit_mask.sum()),
            "holdout_rows": int((~fit_mask).sum()),
            "training_period": f"{train.loc[fit_mask, 'date'].min().date()} to {train.loc[fit_mask, 'date'].max().date()}",
            "holdout_period": f"{train.loc[~fit_mask, 'date'].min().date()} to {train.loc[~fit_mask, 'date'].max().date()}",
        },
        "holdout_metrics": candidates,
        "selected_model": chosen,
        "final_validation_rows": int(len(predictions)),
        "december_rows": int(len(december)),
    }
    METRICS_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Wrote {OUTPUT_PATH}")
    print(f"Completed December predictions in {DECEMBER_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
