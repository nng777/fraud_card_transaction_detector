from __future__ import annotations

import ipaddress
from pathlib import Path
from typing import Dict, Iterable

import joblib
import matplotlib.pyplot as plt
import seaborn as sns
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
        self.bad_ip_list = self._load_ip_list(Path("bad_reputation_ips.csv"))
        self.blacklisted_countries = self._load_country_list(
            Path("blacklisted_countries.csv")
        )

    @staticmethod
    def _load_ip_list(path: Path) -> set[int]:
        """Load a set of bad reputation IP addresses from a CSV file."""
        if not path.exists():
            return set()
        df = pd.read_csv(path)
        ips: set[int] = set()
        for ip in df.iloc[:, 0].dropna().astype(str):
            try:
                ips.add(int(ipaddress.ip_address(ip.strip())))
            except ValueError:
                continue
        return ips

    @staticmethod
    def _load_country_list(path: Path) -> set[str]:
        """Load a set of blacklisted countries from a CSV file."""
        if not path.exists():
            return set()
        df = pd.read_csv(path)
        return {str(code).strip().upper() for code in df.iloc[:, 0].dropna()}

    def is_bad_ip(self, ip: int | str) -> bool:
        """Return True if IP address is in the bad reputation list."""
        ip_int = int(ipaddress.ip_address(ip))
        return ip_int in self.bad_ip_list

    def is_blacklisted_country(self, country: str | int | float) -> bool:
        """Return True if the country is blacklisted.

        The dataset may contain encoded values or missing data. Cast the input to
        string and ignore NaN values to avoid ``AttributeError`` during
        ``str.upper``.
        """
        if pd.isna(country):
            return False
        return str(country).upper() in self.blacklisted_countries

    @staticmethod
    def _load_dataset(path: Path) -> pd.DataFrame:
        df = pd.read_csv(path)
        df = df.drop(
            columns=[
                "cardowner_firstname",
                "cardowner_lastname",
                "card_number",
                "realbank_issuer",
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

        # explicitly ensure additional columns are present for anomaly detection
        required = [
            "card_level",
            "card_expirationdate",
            "merchant_category",
            "device_info",
            "transaction_country",
            "transaction_city",
            "location_latitude",
            "location_longitude",
        ]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        return df

    def _prepare_features(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        df_enc = df.copy()
        # ``transaction_id`` is used only for identifying transactions and
        # should not influence the models, so exclude it from the feature set.
        if "transaction_id" in df_enc.columns:
            df_enc = df_enc.drop(columns=["transaction_id"])
        for col in df_enc.select_dtypes(include="object").columns:
            df_enc[col] = pd.factorize(df_enc[col])[0]
        X = df_enc.values
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
            # ``contamination`` must be set when using ``novelty=True`` otherwise
            # the model may consider every sample normal after training.
            "local_outlier_factor": LocalOutlierFactor(
                novelty=True,
                contamination=0.05,
            ),
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
        predictions: Dict[str, Iterable[int]] = {}
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
            predictions[name] = preds
            results[name] = int((preds == -1).sum())

        # additional heuristic based detections
        df["bad_ip"] = df["ip_address"].apply(self.is_bad_ip).astype(int)
        df["blacklisted_country"] = df["transaction_country"].apply(
            self.is_blacklisted_country
        ).astype(int)
        results["bad_ip"] = int(df["bad_ip"].sum())
        results["blacklisted_country"] = int(df["blacklisted_country"].sum())
        self._last_results = results
        self._last_predictions = predictions
        self._test_df = df
        return results

    def save_anomalies(self) -> list[Path]:
        """Save detected anomalies from the most recent test to CSV files.

        One file is created for each model and each heuristic check. Filenames
        follow the pattern ``<model>_anomalies.csv``.
        """

        if not hasattr(self, "_test_df") or not hasattr(self, "_last_predictions"):
            raise RuntimeError("Run test first.")

        df = self._test_df.copy()
        for name, preds in self._last_predictions.items():
            df[f"{name}_flag"] = (preds == -1).astype(int)

        # ip address back to readable form for easier identification
        df["ip_address"] = df["ip_address"].apply(
            lambda x: str(ipaddress.ip_address(int(x)))
        )

        saved: list[Path] = []
        for name in self._last_predictions.keys():
            path = Path(f"{name}_anomalies.csv")
            df[df[f"{name}_flag"] == 1].to_csv(path, index=False)
            saved.append(path)

        if "bad_ip" in df.columns:
            path = Path("bad_ip_anomalies.csv")
            df[df["bad_ip"] == 1].to_csv(path, index=False)
            saved.append(path)

        if "blacklisted_country" in df.columns:
            path = Path("blacklisted_country_anomalies.csv")
            df[df["blacklisted_country"] == 1].to_csv(path, index=False)
            saved.append(path)

        return saved

    def visualize(self, results: Dict[str, int] | None = None) -> Path:
        """Create a bar chart comparing anomaly counts across models."""
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

    def visualize_scatter(self, model: str = "isolation_forest") -> Path:
        """Plot amount vs. time colored by anomaly flag for a given model."""
        if not hasattr(self, "_test_df") or not hasattr(self, "_last_predictions"):
            raise RuntimeError("Run test first.")

        if model not in self._last_predictions:
            raise ValueError(f"Unknown model: {model}")

        df = self._test_df.copy()
        df["anomaly"] = (self._last_predictions[model] == -1)
        times = pd.to_datetime(df["transaction_time"])
        plt.figure(figsize=(8, 6))
        plt.scatter(times, df["amount"], c=df["anomaly"], cmap="coolwarm", s=10, alpha=0.7)
        plt.xlabel("Transaction time")
        plt.ylabel("Amount")
        plt.title(f"Anomaly scatter plot - {model}")
        plt.tight_layout()
        out = Path(f"{model}_scatter.png")
        plt.savefig(out)
        plt.close()
        return out

    def visualize_heatmap(self) -> Path:
        """Show a heatmap of feature correlations."""
        if not hasattr(self, "_test_df"):
            raise RuntimeError("Run test first.")

        corr = self._test_df.corr(numeric_only=True)
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, cmap="coolwarm", annot=False)
        plt.title("Feature correlation heatmap")
        plt.tight_layout()
        out = Path("feature_heatmap.png")
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