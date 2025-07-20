
from __future__ import annotations

import ipaddress
from pathlib import Path
from typing import Dict

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from tqdm import tqdm


class FraudDetector:
    """Train and evaluate unsupervised models for fraud detection."""

    def __init__(self, train_file: str | Path, test_file: str | Path):
        self.train_file = Path(train_file)
        self.test_file = Path(test_file)
        self.scaler = StandardScaler()
        self.models: Dict[str, object] = {}

    @staticmethod
    def _load_dataset(path: Path) -> pd.DataFrame:
        df = pd.read_csv(path)
        df = df.drop(
            columns=[
                "cardowner_firstname",
                "cardowner_lastname",
                "card_number",
                "realbank_issuer",
                "transaction_id",
            ]
        )
        df["transaction_time"] = pd.to_datetime(df["transaction_time"]).astype("int64")
        df["cardowner_dateofbirth"] = pd.to_datetime(df["cardowner_dateofbirth"]).astype(
            "int64"
        )
        df["3D_SecureTransaction(yes/no)"] = df["3D_SecureTransaction(yes/no)"].map(
            {"yes": 1, "no": 0}
        )
        df["ip_address"] = df["ip_address"].apply(lambda x: int(ipaddress.ip_address(x)))
        for col in df.select_dtypes(include="object").columns:
            df[col] = pd.factorize(df[col])[0]
        return df

    def _prepare_features(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        X = df.values
        if fit:
            X = self.scaler.fit_transform(X)
            joblib.dump(self.scaler, "scaler.joblib")
        else:
            if not Path("scaler.joblib").exists():
                raise RuntimeError("Scaler not found. Train models first.")
            self.scaler = joblib.load("scaler.joblib")
            X = self.scaler.transform(X)
        return X

    def _get_algorithms(self) -> Dict[str, object]:
        return {
            "isolation_forest": IsolationForest(random_state=42),
            "local_outlier_factor": LocalOutlierFactor(novelty=True),
            "one_class_svm": OneClassSVM(gamma="auto"),
        }

    def train(self) -> None:
        df = self._load_dataset(self.train_file)
        X = self._prepare_features(df, fit=True)
        for name, model in tqdm(self._get_algorithms().items(), desc="Training models"):
            model.fit(X)
            joblib.dump(model, f"{name}.joblib")
            self.models[name] = model

    def test(self) -> Dict[str, int]:
        df = self._load_dataset(self.test_file)
        X = self._prepare_features(df, fit=False)
        results: Dict[str, int] = {}
        algorithms = self._get_algorithms()
        for name in algorithms.keys():
            model = self.models.get(name)
            if model is None:
                if not Path(f"{name}.joblib").exists():
                    raise RuntimeError("Model not trained: %s" % name)
                model = joblib.load(f"{name}.joblib")
                self.models[name] = model
        for name, model in tqdm(self.models.items(), desc="Testing models"):
            preds = model.predict(X)
            results[name] = int((preds == -1).sum())
        self._last_results = results
        return results

    def visualize(self, results: Dict[str, int] | None = None) -> Path:
        if results is None:
            if not hasattr(self, "_last_results"):
                raise RuntimeError("No results to visualize. Run test first.")
            results = self._last_results
        names = list(results.keys())
        counts = [results[n] for n in names]
        plt.figure(figsize=(8, 4))
        plt.bar(names, counts)
        plt.ylabel("Anomalies detected")
        plt.title("Model comparison")
        plt.tight_layout()
        out = Path("model_comparison.png")
        plt.savefig(out)
        plt.close()
        return out


if __name__ == "__main__":
    detector = FraudDetector("train_dataset.csv", "test_dataset.csv")
    detector.train()
    results = detector.test()
    print("Anomalies detected:", results)
    img = detector.visualize(results)
    print(f"Saved comparison chart to {img}")