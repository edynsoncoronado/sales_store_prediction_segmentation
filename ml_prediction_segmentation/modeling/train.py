"""
Entrenamiento de Modelos
-----------------------------------------------
Este script entrena modelos supervisados y no supervisados
usando las features servidas desde Feast.
Registra experimentos en MLflow.
"""

import logging
import yaml
# import mlflow.sklearn
from feast import FeatureStore
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.cluster import KMeans # Importa KMeans para clustering
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, silhouette_score, davies_bouldin_score, calinski_harabasz_score
# from sklearn.ensemble import RandomForestRegressor
import numpy as np

import dagshub  # Importa la librería dagshub para integrar el seguimiento de experimentos con DagsHub
import mlflow   # Importa la librería mlflow para el seguimiento de experimentos de machine learning


import xgboost as xgb

import joblib   # Para guardar y cargar modelos
import matplotlib.pyplot as plt # Para visualización de clusters
from sklearn.decomposition import PCA   # Para reducción de dimensionalidad y visualización

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

from pathlib import Path
from functools import wraps

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


def preprocessor_features(categorical_cols: list, numeric_cols: list, type: str = "supervised") -> ColumnTransformer:
    """Separar columnas categóricas y numéricas (encodering e imputación)"""
    if type == "unsupervised":
        # Para clustering, solo se imputan valores nulos
        transformer = ColumnTransformer(
            transformers=[
                ("cat", Pipeline(steps=[
                    ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ("fillna", "passthrough")  # Rellenar valores nulos con "None"
                ]), categorical_cols),
                ("num", Pipeline(steps=[
                    ("scaler", StandardScaler()),  # Escalado
                    ("fillna", SimpleImputer(strategy="constant", fill_value=0)),  # Rellenar valores nulos con 0
                ]), numeric_cols),
            ]
        )
    else:
        # Para modelos supervisados, se imputan valores nulos y se encodifican categóricas
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


def with_mlflow_autolog(func):
    """Decorador para habilitar y deshabilitar mlflow.autolog automáticamente."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        mlflow.autolog()
        try:
            result = func(*args, **kwargs)  # ejecuta la función decorada
            for model_name in result:
                logger.info(f"💯🚀🎯 Modelo registrado en MLflow: {model_name}")
        finally:
            mlflow.autolog(disable=True)   # siempre se deshabilita al final
        return result
    return wrapper


@with_mlflow_autolog
def train_supervised(X_train, X_test, y_train, y_test, preprocessor: ColumnTransformer) -> list:
    """Entrenamiento de modelos supervisados ensambles y registro en MLflow."""

    list_models_name = []

    def model_bagging():
        """Modelo de regresión con Bagging"""
        return BaggingRegressor(
            estimator=DecisionTreeRegressor(),
            n_estimators=10,
            random_state=42
        )
    
    def model_xgboost():
        """Modelo de regresión con XGBoost"""
        model = xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
        return model

    ensamble_models = [model_bagging(), model_xgboost()]
    for model in ensamble_models:
        model_name = f"{model.__class__.__name__}_{model.get_params()['n_estimators']}"
        with mlflow.start_run(run_name=model_name) as run:
            logger.info(f"🤖🧠👾 Entrenando modelo: {model_name}")

            # Definir modelo con pipeline
            pipeline = Pipeline(steps=[
                ("preprocessor", preprocessor),
                ("model", model)
            ])
            # Entrenar el modelo
            pipeline.fit(X_train, y_train)

            # Predicciones
            y_pred = pipeline.predict(X_test)

            # Métricas
            r2 = pipeline.score(X_test, y_test)    # r2 = 1 - sum((y_test - y_pred) ** 2) / sum((y_test - np.mean(y_test)) ** 2)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))  # rmse = ((y_test - y_pred) ** 2).mean() ** 0.5
            mape = mean_absolute_percentage_error(y_test, y_pred)   # mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

            logger.info(f"R2 Score = {r2:.4f} | RMSE = {rmse:.4f} | MAPE = {mape:.4f}")

            # Registra las métricas calculadas en MLflow
            mlflow.log_metrics(
                {
                    "R2 Score": r2,
                    "RMSE": rmse,
                    "MAPE": mape
                }
            )

            # Guarda el modelo entrenado localmente
            modelpkl_path = f"{PROJ_ROOT / "models"}/{model_name}.pkl"
            logger.info(f"Guardando modelo en: {modelpkl_path}")
            joblib.dump(pipeline, modelpkl_path)

            mlflow.log_artifact(
                modelpkl_path,
                # artifact_path=model_name
            )
        list_models_name.append(model_name)

    return list_models_name


@with_mlflow_autolog
def train_unsupervised(X: pd.DataFrame, preprocessor: ColumnTransformer) -> list:
    """Entrena un modelo no supervisado (clustering) y registra en MLflow."""

    model = KMeans(n_clusters=4, random_state=42)
    model_name = f"{model.__class__.__name__}_{model.n_clusters}"

    with mlflow.start_run(run_name=model_name) as run:
        # Pipeline con preprocesamiento + KMeans
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("cluster", model)
        ])
        # Entrenar el modelo
        pipeline.fit(X)

        # Extraer labels asignados
        labels = pipeline.named_steps["cluster"].labels_

        # Convertir a array denso para Davies-Bouldin y Calinski-Harabasz
        X_transformed = pipeline.named_steps["preprocessor"].transform(X).toarray()

        # Métricas de clustering
        sil_score = silhouette_score(pipeline.named_steps["preprocessor"].transform(X), labels)
        db_score = davies_bouldin_score(X_transformed, labels)
        ch_score = calinski_harabasz_score(X_transformed, labels)

        logger.info(f"Silhouette Score: {sil_score:.4f}")
        logger.info(f"Davies-Bouldin Index: {db_score:.4f}")
        logger.info(f"Calinski-Harabasz Index: {ch_score:.4f}")

        # Registra las métricas calculadas en MLflow
        mlflow.log_metrics(
            {
                "Silhouette Score": sil_score,
                "Davies-Bouldin Index": db_score,
                "Calinski-Harabasz Index": ch_score
            }
        )

        # Guarda el modelo entrenado localmente
        modelpkl_path = f"{PROJ_ROOT / "models"}/{model_name}.pkl"
        logger.info(f"Guardando modelo en: {modelpkl_path}")
        joblib.dump(pipeline, modelpkl_path)

        mlflow.log_artifact(
            modelpkl_path,
            # artifact_path=model_name
        )

        # Reducir a 2 dimensiones para visualizar
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X_transformed)

        # Crear la figura
        fig, ax = plt.subplots(figsize=(8,6))
    
        # Graficar los puntos
        scatter = ax.scatter(X_pca[:,0], X_pca[:,1], c=labels, cmap="viridis", alpha=0.6)
    
        # Graficar centroides (proyectados a PCA)
        ax.scatter(
            pca.transform(pipeline.named_steps["cluster"].cluster_centers_)[:,0],
            pca.transform(pipeline.named_steps["cluster"].cluster_centers_)[:,1],
            c="red", marker="X", s=200, label="Centroides"
        )
    
        # Personalización
        ax.set_title("Clusters con KMeans (visualización PCA)")
        ax.set_xlabel("Componente Principal 1")
        ax.set_ylabel("Componente Principal 2")
        ax.legend()
    
        # Registrar en MLflow
        logger.info("📊 Guardando figura de clusters en MLflow")
        mlflow.log_figure(
            fig,
            "plots/kmeans_clusters_pca.png"
        )
    
        plt.close(fig)  # cerrar la figura para evitar duplicados en notebooks

    return [model_name]


def mlflow_setup(repo_name: str):
    """Configuración inicial de MLflow"""
    # Inicializa la integración con DagsHub, especificando el propietario y nombre del repositorio,
    # y habilita la integración con MLflow para registrar experimentos en DagsHub
    dagshub.init(
        repo_owner='edynsoncoronado',
        repo_name=f'ml_{repo_name}',
        mlflow=True
    )

    # Establece la URI de seguimiento de MLflow para que apunte al servidor remoto de DagsHub,
    # permitiendo así registrar y visualizar experimentos de MLflow en esa plataforma.
    mlflow.set_tracking_uri(f"https://dagshub.com/edynsoncoronado/{repo_name}.mlflow")

    # Configura MLflow para registrar experimentos
    mlflow.set_experiment(f"ml_{repo_name}")
    return True
    

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
    categorical_cols = X.select_dtypes(include=["object"]).columns
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns


    # Crear el preprocesador para modelos supervisados y no supervisados
    preprocessor_super = preprocessor_features(categorical_cols, numeric_cols, "supervised")
    preprocessor_nosuper = preprocessor_features(categorical_cols, numeric_cols, "unsupervised")

    # Configurar MLflow
    mlflow_setup("sales_store_prediction_segmentation")
    
    # Entrenar supervisado
    train_supervised(X_train, X_test, y_train, y_test, preprocessor_super)

    # Entrenar no supervisado
    train_unsupervised(X, preprocessor_nosuper)

    logger.info("Entrenamiento de modelos completado.")


if __name__ == "__main__":
    main()
