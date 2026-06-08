"""
automate_Alfin-Himawan-Santosa.py
Script otomatisasi preprocessing dataset Iris
Konversi dari proses eksperimen notebook ke fungsi terstruktur.
"""

import pandas as pd
import os
import joblib
import argparse
import logging
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_dataset(data_path: str = None) -> pd.DataFrame:
    """
    Memuat dataset Iris dari file CSV atau dari sklearn.

    Args:
        data_path: Path ke file CSV. Jika None, load dari sklearn.

    Returns:
        DataFrame berisi dataset Iris raw.
    """
    if data_path and os.path.exists(data_path):
        logger.info(f"Memuat dataset dari: {data_path}")
        df = pd.read_csv(data_path)
    else:
        logger.info("Memuat dataset dari sklearn.datasets.load_iris()")
        iris = load_iris()
        df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
        df["species"] = iris.target
        df["species_name"] = df["species"].map(
            {0: "setosa", 1: "versicolor", 2: "virginica"}
        )

    logger.info(
        f"Dataset berhasil dimuat: {df.shape[0]} baris, {df.shape[1]} kolom"
    )  # noqa: E501
    return df


def handle_missing_values(
    df: pd.DataFrame, feature_cols: list
) -> pd.DataFrame:  # noqa: E501
    """
    Menangani missing values dengan mengisi menggunakan median.

    Args:
        df: DataFrame input
        feature_cols: List nama kolom fitur

    Returns:
        DataFrame tanpa missing values
    """
    df_clean = df.copy()
    total_missing = df_clean[feature_cols].isnull().sum().sum()

    if total_missing > 0:
        logger.info(
            f"Ditemukan {total_missing} missing values. Mengisi dengan median..."  # noqa: E501
        )
        for col in feature_cols:
            if df_clean[col].isnull().sum() > 0:
                median_val = df_clean[col].median()
                df_clean[col].fillna(median_val, inplace=True)
                logger.info(
                    f"  Kolom '{col}': diisi dengan median = {median_val:.4f}"
                )  # noqa: E501
    else:
        logger.info("Tidak ada missing values ditemukan.")

    return df_clean


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Menghapus baris duplikat dari dataset.

    Args:
        df: DataFrame input

    Returns:
        DataFrame tanpa duplikat
    """
    n_before = len(df)
    df_clean = df.drop_duplicates()
    n_after = len(df_clean)
    n_removed = n_before - n_after

    if n_removed > 0:
        logger.info(f"Menghapus {n_removed} baris duplikat.")
    else:
        logger.info("Tidak ada duplikat ditemukan.")

    return df_clean


def handle_outliers_iqr(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """
    Mendeteksi dan menangani outlier menggunakan metode IQR (clipping).

    Args:
        df: DataFrame input
        feature_cols: List nama kolom fitur

    Returns:
        DataFrame dengan outlier yang sudah di-clip
    """
    df_clean = df.copy()
    total_outliers = 0

    for col in feature_cols:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        n_outliers = ((df_clean[col] < lower) | (df_clean[col] > upper)).sum()
        total_outliers += n_outliers

        if n_outliers > 0:
            df_clean[col] = df_clean[col].clip(lower=lower, upper=upper)
            logger.info(
                f"  Kolom '{col}': {n_outliers} outlier di-clip ke [{lower:.4f}, {upper:.4f}]"  # noqa: E501
            )

    if total_outliers == 0:
        logger.info("Tidak ada outlier ditemukan.")
    else:
        logger.info(f"Total {total_outliers} outlier berhasil ditangani.")

    return df_clean


def scale_features(
    X_train: pd.DataFrame, X_test: pd.DataFrame, output_dir: str
):  # noqa: E501
    """
    Melakukan standardisasi fitur menggunakan StandardScaler.

    Args:
        X_train: DataFrame fitur training
        X_test: DataFrame fitur testing
        output_dir: Direktori untuk menyimpan scaler

    Returns:
        Tuple (X_train_scaled, X_test_scaled, scaler)
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_scaled_df = pd.DataFrame(
        X_train_scaled, columns=X_train.columns, index=X_train.index
    )
    X_test_scaled_df = pd.DataFrame(
        X_test_scaled, columns=X_test.columns, index=X_test.index
    )

    # Simpan scaler
    scaler_path = os.path.join(output_dir, "scaler.pkl")
    joblib.dump(scaler, scaler_path)
    logger.info(f"Scaler disimpan ke: {scaler_path}")

    return X_train_scaled_df, X_test_scaled_df, scaler


def preprocess_data(
    data_path: str = None,
    output_dir: str = "iris_preprocessing",
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """
    Fungsi utama untuk melakukan preprocessing dataset Iris secara otomatis.

    Tahapan:
    1. Load dataset
    2. Handle missing values
    3. Remove duplicates
    4. Handle outliers (IQR)
    5. Train-test split
    6. Feature scaling
    7. Simpan hasil

    Args:
        data_path: Path ke file CSV (opsional)
        output_dir: Direktori output untuk dataset preprocessed
        test_size: Proporsi data test (default: 0.2)
        random_state: Random seed (default: 42)

    Returns:
        Dictionary berisi dataset yang sudah diproses
    """
    logger.info("=" * 50)
    logger.info("MEMULAI PROSES PREPROCESSING DATASET IRIS")
    logger.info("=" * 50)

    # Buat output directory
    os.makedirs(output_dir, exist_ok=True)

    # Kolom fitur
    feature_cols = [
        "sepal length (cm)",
        "sepal width (cm)",
        "petal length (cm)",
        "petal width (cm)",
    ]

    # Step 1: Load Dataset
    logger.info("\n[Step 1] Loading Dataset...")
    df = load_dataset(data_path)

    # Step 2: Handle Missing Values
    logger.info("\n[Step 2] Handling Missing Values...")
    df = handle_missing_values(df, feature_cols)

    # Step 3: Remove Duplicates
    logger.info("\n[Step 3] Removing Duplicates...")
    df = remove_duplicates(df)

    # Step 4: Handle Outliers
    logger.info("\n[Step 4] Handling Outliers (IQR Method)...")
    df = handle_outliers_iqr(df, feature_cols)

    # Step 5: Feature & Target Split
    logger.info("\n[Step 5] Splitting Features and Target...")
    X = df[feature_cols].copy()
    y = df["species"].copy()
    logger.info(f"Fitur shape: {X.shape}, Target shape: {y.shape}")

    # Step 6: Train-Test Split
    logger.info(
        f"\n[Step 6] Train-Test Split (test_size={test_size}, random_state={random_state})..."  # noqa: E501
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    logger.info(f"X_train: {X_train.shape}, X_test: {X_test.shape}")

    # Step 7: Feature Scaling
    logger.info("\n[Step 7] Feature Scaling (StandardScaler)...")
    X_train_scaled, X_test_scaled, scaler = scale_features(
        X_train, X_test, output_dir
    )  # noqa: E501

    # Step 8: Simpan Dataset
    logger.info("\n[Step 8] Saving Preprocessed Dataset...")

    # Train set
    train_df = X_train_scaled.copy()
    train_df["species"] = y_train.values
    train_path = os.path.join(output_dir, "iris_train.csv")
    train_df.to_csv(train_path, index=False)

    # Test set
    test_df = X_test_scaled.copy()
    test_df["species"] = y_test.values
    test_path = os.path.join(output_dir, "iris_test.csv")
    test_df.to_csv(test_path, index=False)

    # Full preprocessed
    full_df = pd.concat([train_df, test_df], ignore_index=True)
    full_path = os.path.join(output_dir, "iris_preprocessed.csv")
    full_df.to_csv(full_path, index=False)

    logger.info(f"  Train set disimpan ke: {train_path} ({train_df.shape})")
    logger.info(f"  Test set disimpan ke: {test_path} ({test_df.shape})")
    logger.info(
        f"  Full preprocessed disimpan ke: {full_path} ({full_df.shape})"
    )  # noqa: E501

    logger.info("\n" + "=" * 50)
    logger.info("PREPROCESSING SELESAI!")
    logger.info("=" * 50)

    return {
        "X_train": X_train_scaled,
        "X_test": X_test_scaled,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler,
        "feature_cols": feature_cols,
        "output_dir": output_dir,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Automate Iris Dataset Preprocessing"
    )  # noqa: E501
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path ke file CSV dataset (opsional, default: load dari sklearn)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="iris_preprocessing",
        help="Direktori output untuk dataset preprocessed",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proporsi data test (default: 0.2)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )

    args = parser.parse_args()

    result = preprocess_data(
        data_path=args.data_path,
        output_dir=args.output_dir,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    print("\nHasil Preprocessing:")
    print(f"  X_train shape: {result['X_train'].shape}")
    print(f"  X_test shape: {result['X_test'].shape}")
    print(f"  Output dir: {result['output_dir']}")
