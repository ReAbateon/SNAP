# 
# Copyright (c) 2026 Lorenzo Abate <lorenzo.abate@unina.it>.
# 
# This program is free software: you can redistribute it and/or modify  
# it under the terms of the GNU General Public License as published by  
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful, but 
# WITHOUT ANY WARRANTY; without even the implied warranty of 
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU 
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License 
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump, load

from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

def compute_columnwise_scale(feature_matrix: np.ndarray) -> np.ndarray:
    max_vals = np.max(np.abs(feature_matrix), axis=0)
    max_vals = np.where(max_vals == 0, 1.0, max_vals)
    return 32766.0 / max_vals


def quantize_columnwise(feature_matrix: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return np.round(feature_matrix * scale).astype(np.int16)


def load_joblib_bundle(joblib_path: Path):
    """
    Supports:
      - new format: dict with keys {model, feature_names, label_name, x_scale, ...}
      - legacy format: DecisionTreeClassifier only
    Returns: (model, bundle_dict_or_none)
    """
    loaded_obj = load(joblib_path)
    if isinstance(loaded_obj, dict) and "model" in loaded_obj:
        return loaded_obj["model"], loaded_obj
    return loaded_obj, None


def save_quantized_test_csv(
    original_test_path: Path,
    quantized_features: np.ndarray,
    labels: np.ndarray,
    feature_names: list[str],
    label_name: str,
) -> Path:
    quantized_path = original_test_path.with_name(original_test_path.stem + "_quantized.csv")
    quantized_df = pd.DataFrame(quantized_features, columns=feature_names)
    quantized_df[label_name] = labels
    quantized_df.to_csv(quantized_path, index=False)
    return quantized_path


def dt_test_stats(model: DecisionTreeClassifier, test_features: np.ndarray, test_labels: np.ndarray, weight: int = 8) -> dict:
    predicted_labels = model.predict(test_features)
    accuracy = float(accuracy_score(test_labels, predicted_labels))

    n_nodes = int(model.tree_.node_count)
    size_bytes = int(n_nodes * weight)
    size_kb = float(size_bytes / 1024.0)

    return {
        "accuracy": accuracy,
        "n_nodes": n_nodes,
        "size_bytes": size_bytes,
        "size_kb": size_kb,
        "y_pred": predicted_labels,
    }


def training_dt_with_external_test(
    train_csv_path,
    test_csv_path,
    max_depth: int,
    random_seed: int,
    weight: int = 8,
):
    train_csv_path = Path(train_csv_path)
    test_csv_path = Path(test_csv_path)

    output_dir = train_csv_path.parent
    output_stem = train_csv_path.stem
    joblib_path = output_dir / f"{output_stem}_DT_D{max_depth}_RS{random_seed}.joblib"

    # ---- Load train split
    train_df = pd.read_csv(train_csv_path)
    label_name = str(train_df.columns[-1])
    feature_names = [str(column_name) for column_name in train_df.columns[:-1]]

    train_feature_df = train_df.iloc[:, :-1]
    train_labels = train_df.iloc[:, -1].to_numpy(copy=False)
    feature_count = int(train_feature_df.shape[1])

    # Quantize train split (scale computed only from train).
    train_features = train_feature_df.to_numpy(dtype=np.float32, copy=False)
    feature_scale = compute_columnwise_scale(train_features)
    train_features_quantized = quantize_columnwise(train_features, feature_scale)

    # ---- Load test split
    test_df = pd.read_csv(test_csv_path)
    test_label_name = str(test_df.columns[-1])
    test_feature_names = [str(column_name) for column_name in test_df.columns[:-1]]

    # Sanity check: feature names and count.
    if len(test_feature_names) != feature_count:
        raise ValueError(f"Test features mismatch: train has {feature_count}, test has {len(test_feature_names)}")
    if test_feature_names != feature_names:
        # Fail early if feature names/order differ between train and test.
        raise ValueError(
            "Feature names/order mismatch between train and test.\n"
            f"Train first cols: {feature_names[:5]}...\n"
            f"Test  first cols: {test_feature_names[:5]}..."
        )
    if test_label_name != label_name:
        # Fail early if the label column name differs.
        raise ValueError(f"Label column mismatch: train label='{label_name}', test label='{test_label_name}'")

    test_features = test_df.iloc[:, :-1].to_numpy(dtype=np.float32, copy=False)
    test_labels = test_df.iloc[:, -1].to_numpy(copy=False)

    # Quantize test split using TRAIN scale.
    test_features_quantized = quantize_columnwise(test_features, feature_scale)

    # Save quantized test CSV with original column names.
    quantized_test_path = save_quantized_test_csv(
        original_test_path=test_csv_path,
        quantized_features=test_features_quantized,
        labels=test_labels,
        feature_names=feature_names,
        label_name=label_name,
    )

    # ---- If joblib exists -> load + compute stats.
    if joblib_path.exists():
        model, bundle = load_joblib_bundle(joblib_path)

        stats = dt_test_stats(model, test_features_quantized, test_labels, weight=weight)

        # If a bundle exists, enrich it; for legacy models build minimal metadata.
        if bundle is None:
            bundle = {
                "model": model,
                "feature_names": feature_names,
                "label_name": label_name,
                "x_scale": feature_scale,
                "meta": {
                    "train_csv": str(train_csv_path),
                    "test_csv": str(test_csv_path),
                    "max_depth": max_depth,
                    "random_seed": random_seed,
                },
            }

        stats.update({
            "quantized_test_path": quantized_test_path,
            "bundle_keys": sorted(list(bundle.keys())),
        })
        return joblib_path, feature_count, output_stem, stats

    # ---- Train with grid search on TRAIN only.
    base_model = DecisionTreeClassifier(max_depth=max_depth, random_state=random_seed)

    param_grid = {
        "min_samples_split": list(range(2, 11)),
        "min_samples_leaf": list(range(1, 11)),
    }

    grid_search = GridSearchCV(base_model, param_grid, cv=5, scoring="accuracy", n_jobs=-1, verbose=0)
    grid_search.fit(train_features_quantized, train_labels)
    best_params = grid_search.best_params_

    # Final model
    model = DecisionTreeClassifier(
        max_depth=max_depth,
        random_state=random_seed,
        min_samples_split=best_params["min_samples_split"],
        min_samples_leaf=best_params["min_samples_leaf"],
    )
    model.fit(train_features_quantized, train_labels)

    # Save bundle (model + feature names + scale + metadata).
    bundle = {
        "model": model,
        "feature_names": feature_names,
        "label_name": label_name,
        "x_scale": feature_scale,
        "best_params": best_params,
        "best_cv_score": float(grid_search.best_score_),
        "meta": {
            "train_csv": str(train_csv_path),
            "test_csv": str(test_csv_path),
            "max_depth": max_depth,
            "random_seed": random_seed,
            "weight_bytes_per_node": int(weight),
            "quantization": {"type": "columnwise_int16", "range": "[-32766, 32766]"},
        },
    }
    dump(bundle, joblib_path)

    # Stats on provided test split.
    stats = dt_test_stats(model, test_features_quantized, test_labels, weight=weight)
    stats.update({
        "quantized_test_path": quantized_test_path,
        "best_params": best_params,
        "best_cv_score": float(grid_search.best_score_),
    })

    return joblib_path, feature_count, output_stem, stats

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train a single Decision Tree (classification) from train CSV and evaluate on external test CSV. "
                    "Last column must be the label. Saves a quantized copy of test CSV with _quantized suffix."
    )
    p.add_argument("--train", required=True, type=Path, help="Path to train CSV (last column = label).")
    p.add_argument("--test", required=True, type=Path, help="Path to test CSV (last column = label).")
    p.add_argument("--max-depth", required=True, type=int, help="DecisionTree max_depth.")
    p.add_argument("--seed", default=0, type=int, help="Random seed (default: 0).")
    p.add_argument("--weight", default=8, type=int, help="Bytes per node for size estimate (default: 8).")
    return p

def main():
    args = build_argparser().parse_args()

    joblib_path, _, _, stats = training_dt_with_external_test(
        train_csv_path=args.train,
        test_csv_path=args.test,
        max_depth=args.max_depth,
        random_seed=args.seed,
        weight=args.weight,
    )

    # Minimal output.
    print(f"Joblib: {joblib_path}")
    print(f"Accuracy: {stats['accuracy']:.6f}")
    print(f"Quantized test saved at: {stats['quantized_test_path']}")


if __name__ == "__main__":
    main()
