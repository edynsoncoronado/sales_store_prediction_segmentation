"""
Script de Preprocesamiento
-------------------------------------------------
Este script realiza la ingesta, limpieza y unión de los datasets
del proyecto, generando un dataset integrado listo para Feature Engineering.
"""

import logging
import os
import pandas as pd
import yaml
from pathlib import Path
from ml_prediction_segmentation.config import PROJ_ROOT


# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> dict:
    """Carga parámetros desde un archivo YAML de configuración."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def load_datasets(data_path: str):
    """Carga los datasets originales desde la carpeta data_path."""
    logger.info("📥 Cargando datasets desde %s", data_path)

    sales = pd.read_csv(os.path.join(data_path, "train.csv"), parse_dates=["date"])
    stores = pd.read_csv(os.path.join(data_path, "stores.csv"))
    oil = pd.read_csv(os.path.join(data_path, "oil.csv"), parse_dates=["date"])
    holidays = pd.read_csv(os.path.join(data_path, "holidays_events.csv"), parse_dates=["date"])
    transactions = pd.read_csv(os.path.join(data_path, "transactions.csv"), parse_dates=["date"])

    logger.info("✅ Datasets cargados correctamente")
    return sales, stores, oil, holidays, transactions


def clean_and_merge(sales, stores, oil, holidays, transactions):
    """Limpia valores nulos y une todos los datasets en un DataFrame."""
    logger.info("🧹 Limpiando datos y uniendo tablas...")

    # Rellenar nulos en precios de petróleo
    oil["dcoilwtico"].fillna(method="ffill", inplace=True)

    # Merge ventas con tiendas
    df = sales.merge(stores, on="store_nbr", how="left")

    # Merge con transacciones (por tienda y fecha)
    df = df.merge(transactions, on=["date", "store_nbr"], how="left")

    # Merge con petróleo
    df = df.merge(oil, on="date", how="left")

    # Merge con feriados
    df = df.merge(holidays, on="date", how="left")

    logger.info("✅ Dataset integrado, tamaño final: %s", df.shape)
    return df


def save_dataset(df, output_path: str):
    """Guarda el dataset limpio en formato parquet."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.info("💾 Dataset limpio guardado en %s", output_path)


def main():
    """Pipeline principal de preprocesamiento."""
    # Leer configuración
    config = load_config(f"{PROJ_ROOT}/config.yaml")
    raw_path = f"{PROJ_ROOT}/" + config["data"]["raw"]
    processed_path = f"{PROJ_ROOT}/" + config["data"]["processed"]
    sales, stores, oil, holidays, transactions = load_datasets(raw_path)
    df = clean_and_merge(sales, stores, oil, holidays, transactions)
    save_dataset(df, os.path.join(processed_path, "favorita_clean.parquet"))


if __name__ == "__main__":
    main()
