"""
Pipeline principal de Machine Learning para el Proyecto Final de MLOps.

Este script:
1. Carga un dataset externo desde UCI.
2. Realiza preprocesamiento básico.
3. Entrena un modelo de clasificación.
4. Evalúa el modelo con varias métricas.
5. Registra parámetros, métricas, firma, input_example y modelo en MLflow.
6. Guarda artefactos locales para GitHub Actions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def load_config(config_path: str = "config.yaml") -> dict[str, Any]:
    """Carga el archivo de configuración YAML."""
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_data(config: dict[str, Any]) -> pd.DataFrame:
    """Carga el dataset externo desde la URL configurada."""
    dataset_cfg = config["dataset"]
    return pd.read_csv(dataset_cfg["url"], sep=dataset_cfg["separator"])


def prepare_data(
    df: pd.DataFrame, config: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Prepara los datos para entrenamiento.

    La variable original quality es convertida a una tarea de clasificación:
    - 1: vino de calidad aceptable/alta, si quality >= positive_threshold.
    - 0: vino de calidad baja, si quality < positive_threshold.
    """
    target_column = config["dataset"]["target_column"]
    positive_threshold = config["dataset"]["positive_threshold"]

    df = df.copy()
    df["target"] = (df[target_column] >= positive_threshold).astype(int)

    x = df.drop(columns=[target_column, "target"])
    y = df["target"]

    split_cfg = config["split"]

    stratify = y if split_cfg.get("stratify", True) else None

    return train_test_split(
        x,
        y,
        test_size=split_cfg["test_size"],
        random_state=split_cfg["random_state"],
        stratify=stratify,
    )


def build_pipeline(config: dict[str, Any]) -> Pipeline:
    """Construye el pipeline de preprocesamiento y modelo."""
    model_cfg = config["model"]

    numeric_preprocessor = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_preprocessor, list(range(11))),
        ]
    )

    model = RandomForestClassifier(
        n_estimators=model_cfg["n_estimators"],
        max_depth=model_cfg["max_depth"],
        random_state=model_cfg["random_state"],
        class_weight=model_cfg["class_weight"],
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def evaluate_model(model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    """Evalúa el modelo con métricas de clasificación."""
    predictions = model.predict(x_test)

    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
    }


def save_local_artifacts(
    model: Pipeline, metrics: dict[str, float], config: dict[str, Any]
) -> None:
    """Guarda modelo y métricas como artefactos locales del proyecto."""
    model_dir = Path(config["artifacts"]["model_dir"])
    metrics_path = Path(config["artifacts"]["metrics_path"])

    model_dir.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, model_dir / "model.joblib")

    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)


def train_pipeline(config_path: str = "config.yaml") -> dict[str, float]:
    """Ejecuta el pipeline completo de entrenamiento, evaluación y tracking."""
    config = load_config(config_path)

    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    df = load_data(config)
    x_train, x_test, y_train, y_test = prepare_data(df, config)

    pipeline = build_pipeline(config)

    with mlflow.start_run(run_name="random-forest-wine-quality"):
        pipeline.fit(x_train, y_train)
        metrics = evaluate_model(pipeline, x_test, y_test)

        model_cfg = config["model"]

        # Registro de parámetros en MLflow
        mlflow.log_param("dataset_url", config["dataset"]["url"])
        mlflow.log_param("target_column", config["dataset"]["target_column"])
        mlflow.log_param("positive_threshold", config["dataset"]["positive_threshold"])
        mlflow.log_param("test_size", config["split"]["test_size"])
        mlflow.log_param("random_state", config["split"]["random_state"])
        mlflow.log_param("model_type", model_cfg["type"])
        mlflow.log_param("n_estimators", model_cfg["n_estimators"])
        mlflow.log_param("max_depth", model_cfg["max_depth"])
        mlflow.log_param("class_weight", model_cfg["class_weight"])

        # Registro de métricas en MLflow
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)

        input_example = x_train.head(5)
        signature = infer_signature(input_example, pipeline.predict(input_example))

        # Registro del modelo con firma e input_example
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            signature=signature,
            input_example=input_example,
            registered_model_name=config["mlflow"]["registered_model_name"],
        )

        save_local_artifacts(pipeline, metrics, config)

        mlflow.log_artifact(config["artifacts"]["metrics_path"])

    print("Entrenamiento finalizado correctamente.")
    print(json.dumps(metrics, indent=4))

    return metrics


if __name__ == "__main__":
    train_pipeline()
