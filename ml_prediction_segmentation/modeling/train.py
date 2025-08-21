"""
Entrenamiento de Modelos
-----------------------------------------------
Este script entrena modelos supervisados y no supervisados
usando las features servidas desde Feast.
Registra experimentos en MLflow.
"""

import logging
import yaml
# import mlflow
# import mlflow.sklearn
from feast import FeatureStore
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
# from sklearn.metrics import r2_score, silhouette_score
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.cluster import KMeans
# import numpy as np

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

from pathlib import Path

# from loguru import logger
# from tqdm import tqdm
# import typer

from ml_prediction_segmentation.config import PROJ_ROOT


def load_config(config_path: str = "config.yaml") -> dict:
    """Carga parámetros desde un archivo YAML de configuración."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def get_training_data(fs: FeatureStore, view_features: list) -> pd.DataFrame:
    """Obtiene features históricas de Feast para entrenamiento."""
    # Entity DataFrame
    entity_df = pd.read_parquet(PROJ_ROOT / "data" / "processed" / "favorita_clean.parquet")

    training_df = fs.get_historical_features(
        entity_df=entity_df[["store_nbr", "family", "date"]].sample(n=1000, random_state=42),
        features=view_features
    ).to_df()

    logger.info("✅ Dataset de entrenamiento obtenido de Feast con shape %s", training_df.shape)
    return training_df


def preprocessor_features(categorical_cols: list, numeric_cols: list) -> ColumnTransformer:
    """Separar columnas categóricas y numéricas (encodering e imputación)"""
    transformer = ColumnTransformer(
        transformers=[
            ("cat", Pipeline(steps=[
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
                ("fillna", "passthrough")  # Rellenar valores nulos con "None"
            ]), categorical_cols),
            ("num", Pipeline(steps=[
                ("fillna", SimpleImputer(strategy="constant", fill_value=0)),  # Rellenar valores nulos con 0
            ]), numeric_cols),
        ]
    )
    return transformer


def train_supervised(X_train, X_test, y_train, y_test, preprocessor: ColumnTransformer):
    """Entrenamiento de modelos supervisados ensambles"""
    model = BaggingRegressor(
        estimator=DecisionTreeRegressor(),
        n_estimators=10,
        random_state=42
    )

    # Definir modelo con pipeline
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])
    # print("y_train-->:", y_train.info())
    # print("X_train-->:", X_train.info())
    # Entrenar el modelo
    pipeline.fit(X_train, y_train)

    # Evaluar el modelo
    score = pipeline.score(X_test, y_test)
    logger.info(f"R2 Score={score:.4f}")
    return model


def main():
    logger.info("Entrenamiento de modelos iniciado...")

    # Cargar el Feature Store
    config = load_config(str(PROJ_ROOT / "config.yaml"))
    fs_path = config["feast"]["repo_path"]
    fs = FeatureStore(repo_path=str(PROJ_ROOT / fs_path))

    # Obtener dataset
    training_df = get_training_data(
        fs,
        view_features=["sales_store_view:sales", "sales_store_view:onpromotion", "sales_store_view:transactions", "sales_store_view:dcoilwtico", "sales_store_view:type_y"],
    )

    # Separar features y target
    y = training_df["sales"]
    X = training_df.drop(columns=["sales", "date"])  # eliminamos target y timestamp

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Definir columnas categóricas y numéricas
    categorical_cols = ["family", "type_y"]
    numeric_cols = [col for col in X.columns if col not in categorical_cols]

    # Crear el preprocesador
    preprocessor = preprocessor_features(categorical_cols, numeric_cols)

    # Entrenar supervisado
    train_supervised(X_train, X_test, y_train, y_test, preprocessor)

    # Aquí iría el código para entrenar el modelo usando las features cargadas
    # Por ejemplo, un modelo de regresión o clasificación

    logger.info("Entrenamiento de modelos completado.")


if __name__ == "__main__":
    main()
