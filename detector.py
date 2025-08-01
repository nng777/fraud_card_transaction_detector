from __future__ import annotations

import ipaddress
from pathlib import Path
from typing import Dict, Iterable
from math import radians, sin, cos, sqrt, atan2
import time

import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from tqdm import tqdm


class FraudDetector:
    """Train and evaluate unsupervised models for fraud detection."""

    def __init__(self, train_file: str | Path, test_file: str | Path):
        self.train_file = Path(train_file)
        self.test_file = Path(test_file)
        self.models: Dict[str, object] = {}
        self.bad_ip_list = self._load_ip_list(Path("bad_reputation_ips.csv"))
        self.blacklisted_countries = self._load_country_list(
            Path("blacklisted_countries.csv")
        )
        self.train_times: Dict[str, float] = {}
        self.test_times: Dict[str, float] = {}
        self.precision: Dict[str, float] = {}
        self.recall: Dict[str, float] = {}
        self.f1: Dict[str, float] = {}

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

    @staticmethod
    def _haversine(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
        """Return the great-circle distance between two coordinates in kilometers."""
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        return 6371.0 * c


    def is_bad_ip(self, ip: int | str) -> bool:
        """Return True if IP address is in the bad reputation list."""
        ip_int = int(ipaddress.ip_address(ip))
        return ip_int in self.bad_ip_list

    def is_blacklisted_country(self, country: str | int | float) -> bool:
        """Return True if the country is blacklisted."""
        if pd.isna(country):
            return False
        return str(country).upper() in self.blacklisted_countries

    def _load_dataset(self, path: Path) -> pd.DataFrame:
        df = pd.read_csv(path)
        df["transaction_time"] = pd.to_datetime(df["transaction_time"])
        df["cardowner_dateofbirth"] = pd.to_datetime(df["cardowner_dateofbirth"])


        df = df.drop(
            columns=[
                "cardowner_firstname",
                "cardowner_lastname",
                "card_number",
                "realbank_issuer",
            ]
        )
        df["transaction_time"] = df["transaction_time"].astype("int64")
        df["cardowner_dateofbirth"] = df["cardowner_dateofbirth"].astype("int64")
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

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df_enc = df.copy()
        # ``transaction_id`` is used only for identifying transactions and
        # should not influence the models, so exclude it from the feature set.
        if "transaction_id" in df_enc.columns:
            df_enc = df_enc.drop(columns=["transaction_id"])
        for col in df_enc.select_dtypes(include="object").columns:
            df_enc[col] = pd.factorize(df_enc[col])[0]
        X = df_enc.values
        return X


    def _get_algorithms(self) -> Dict[str, object]:
        """Return the anomaly detection models used by the detector."""

        algorithms = {
            "isolation_forest": IsolationForest(random_state=50),
            # ``contamination`` must be set when using ``novelty=True`` to avoid
            # the model marking all samples as normal after training.
            "local_outlier_factor": LocalOutlierFactor(
                novelty=True,
                contamination=0.05,
            ),
            "one_class_svm": OneClassSVM(gamma="auto"),
        }
        return algorithms

    def train(self) -> None:
        df = self._load_dataset(self.train_file)
        X = self._prepare_features(df)
        for name, model in tqdm(self._get_algorithms().items(), desc="Training models"):
            start = time.perf_counter()
            model.fit(X)
            self.train_times[name] = time.perf_counter() - start
            joblib.dump(model, f"{name}.joblib")
            self.models[name] = model


    def test(self) -> Dict[str, int]:
        df = self._load_dataset(self.test_file)
        X = self._prepare_features(df)
        results: Dict[str, int] = {}
        predictions: Dict[str, Iterable[int]] = {}
        algorithms = self._get_algorithms()
        for name in algorithms.keys():
            model = self.models.get(name)
            if model is None:
                if not Path(f"{name}.joblib").exists():
                    raise RuntimeError("Model not trained: %s" % name)
                model = joblib.load(f"{name}.joblib")
            # validate feature dimensionality to avoid mismatched models
            if hasattr(model, "n_features_in_") and model.n_features_in_ != X.shape[1]:
                raise RuntimeError(
                    f"Loaded model '{name}' expects {model.n_features_in_} features "
                    f"but dataset has {X.shape[1]}. Retrain models."
                )
            self.models[name] = model
        for name in tqdm(algorithms.keys(), desc="Testing models"):
            model = self.models[name]
            start = time.perf_counter()
            preds = model.predict(X)
            self.test_times[name] = time.perf_counter() - start
            predictions[name] = preds
            results[name] = int((preds == -1).sum())



        # additional heuristic based detections
        df["bad_ip"] = df["ip_address"].apply(self.is_bad_ip).astype(int)
        df["blacklisted_country"] = df["transaction_country"].apply(
            self.is_blacklisted_country
        ).astype(int)
        results["bad_ip"] = int(df["bad_ip"].sum())
        results["blacklisted_country"] = int(df["blacklisted_country"].sum())
        ground_truth = (df["bad_ip"] | df["blacklisted_country"]).astype(int)
        for name, preds in predictions.items():
            pred_series = (preds == -1).astype(int)
            tp = int(((pred_series == 1) & (ground_truth == 1)).sum())
            fp = int(((pred_series == 1) & (ground_truth == 0)).sum())
            fn = int(((pred_series == 0) & (ground_truth == 1)).sum())
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
            self.precision[name] = precision * 100
            self.recall[name] = recall * 100
            self.f1[name] = f1 * 100
        self._last_results = results
        self._last_predictions = predictions
        self._test_df = df
        return results

    def save_anomalies(self) -> list[Path]:
        """Save detected anomalies from the most recent test to CSV files."""

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
        """Bar chart comparing normal and anomalous predictions per model."""
        if results is None:
            if not hasattr(self, "_last_results") or not hasattr(self, "_test_df"):
                raise RuntimeError("No results to visualize. Run test first.")
            results = self._last_results
        names = [n for n in results.keys() if n not in {"bad_ip", "blacklisted_country"}]
        if not hasattr(self, "_test_df"):
            raise RuntimeError("Run test first")
        total = len(self._test_df)

        normal_counts = []
        anomaly_counts = []
        for n in names:
            anomalies = results[n]
            normal = total - anomalies
            normal_counts.append(normal)
            anomaly_counts.append(anomalies)

        x = np.arange(len(names))
        width = 0.35
        fig, ax = plt.subplots(figsize=(8, 4))
        bars_norm = ax.bar(x - width / 2, normal_counts, width, label="Normal")
        bars_anom = ax.bar(x + width / 2, anomaly_counts, width, label="Anomaly")

        ax.set_xticks(x)
        ax.set_xticklabels(names)
        ax.set_ylabel("Count")
        ax.set_title("Model comparison")
        ax.legend()

        ax.bar_label(bars_norm, padding=3)
        ax.bar_label(bars_anom, padding=3)

        fig.tight_layout()
        out = Path("model_comparison.png")
        fig.savefig(out)
        plt.close(fig)
        return out

    def visualize_heuristics(self, results: Dict[str, int] | None = None) -> Path:
        """Bar chart of heuristic anomaly counts."""
        if results is None:
            if not hasattr(self, "_last_results"):
                raise RuntimeError("No results to visualize. Run test first.")
            results = self._last_results
        names = [n for n in ["bad_ip", "blacklisted_country"] if n in results]
        counts = [results[n] for n in names]
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(names, counts)
        ax.set_ylabel("Anomalies detected")
        ax.set_title("Heuristic comparison")
        ax.bar_label(bars, padding=3)
        fig.tight_layout()
        out = Path("heuristic_comparison.png")
        fig.savefig(out)
        plt.close(fig)
        return out

    def visualize_metrics(self) -> Path:
        """Grouped bar chart of precision, recall and F1 score per model."""
        if not self.precision:
            raise RuntimeError("Run test first.")
        names = list(self.precision.keys())
        prec = np.array([self.precision[n] for n in names])
        rec = np.array([self.recall[n] for n in names])
        f1 = np.array([self.f1[n] for n in names])
        x = np.arange(len(names))
        width = 0.25
        fig, ax = plt.subplots(figsize=(8, 4))
        bars_p = ax.bar(x - width, prec, width, label="Precision")
        bars_r = ax.bar(x, rec, width, label="Recall")
        bars_f = ax.bar(x + width, f1, width, label="F1")
        ax.set_xticks(x)
        ax.set_xticklabels(names)
        ax.set_ylabel("Score (%)")
        ax.set_title("Model precision/recall/F1")
        ax.legend()

        ax.bar_label(bars_p, fmt="%.1f", padding=3)
        ax.bar_label(bars_r, fmt="%.1f", padding=3)
        ax.bar_label(bars_f, fmt="%.1f", padding=3)

        fig.tight_layout()
        out = Path("model_metrics.png")
        fig.savefig(out)
        plt.close(fig)
        return out

    def visualize_timings(self) -> Path:
        """Grouped bar chart of training and testing time per model."""
        if not self.train_times or not self.test_times:
            raise RuntimeError("Run train and test first.")
        names = list(self._get_algorithms().keys())
        train = [self.train_times.get(n, 0) for n in names]
        test = [self.test_times.get(n, 0) for n in names]
        x = np.arange(len(names))
        width = 0.35
        fig, ax = plt.subplots(figsize=(8, 4))
        bars_train = ax.bar(x - width / 2, train, width, label="Train")
        bars_test = ax.bar(x + width / 2, test, width, label="Test")
        ax.set_xticks(x)
        ax.set_xticklabels(names)
        ax.set_ylabel("Seconds")
        ax.set_title("Model processing time")
        ax.legend()

        ax.bar_label(bars_train, fmt="%.2f", padding=3)
        ax.bar_label(bars_test, fmt="%.2f", padding=3)

        fig.tight_layout()
        out = Path("model_timings.png")
        fig.savefig(out)
        plt.close(fig)
        return out

    def visualize_summary(self) -> Path:
        """Create a single figure with multiple comparison charts."""

        if not hasattr(self, "_last_results") or not hasattr(self, "_test_df"):
            raise RuntimeError("Run test first.")
        if not self.train_times or not self.test_times:
            raise RuntimeError("Run train first.")

        results = self._last_results
        names = [n for n in results.keys() if n not in {"bad_ip", "blacklisted_country"}]
        total = len(self._test_df)

        normal_counts = []
        anomaly_counts = []
        for n in names:
            anomalies = results[n]
            normal_counts.append(total - anomalies)
            anomaly_counts.append(anomalies)

        heuristics = [n for n in ["bad_ip", "blacklisted_country"] if n in results]
        heuristic_counts = [results[n] for n in heuristics]

        metrics_names = list(self.precision.keys())
        prec = np.array([self.precision[n] for n in metrics_names])
        rec = np.array([self.recall[n] for n in metrics_names])
        f1 = np.array([self.f1[n] for n in metrics_names])

        time_names = list(self._get_algorithms().keys())
        train = [self.train_times.get(n, 0) for n in time_names]
        test = [self.test_times.get(n, 0) for n in time_names]

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        # Model comparison
        x = np.arange(len(names))
        width = 0.35
        bars_norm = axes[0, 0].bar(x - width / 2, normal_counts, width, label="Normal")
        bars_anom = axes[0, 0].bar(x + width / 2, anomaly_counts, width, label="Anomaly")
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(names)
        axes[0, 0].set_ylabel("Count")
        axes[0, 0].set_title("Model comparison")
        axes[0, 0].legend()
        axes[0, 0].bar_label(bars_norm, padding=3)
        axes[0, 0].bar_label(bars_anom, padding=3)

        # Heuristic comparison
        bars_h = axes[0, 1].bar(heuristics, heuristic_counts)
        axes[0, 1].set_ylabel("Anomalies detected")
        axes[0, 1].set_title("Heuristic comparison")
        axes[0, 1].bar_label(bars_h, padding=3)

        # Metrics
        x_m = np.arange(len(metrics_names))
        width_m = 0.25
        bars_p = axes[1, 0].bar(x_m - width_m, prec, width_m, label="Precision")
        bars_r = axes[1, 0].bar(x_m, rec, width_m, label="Recall")
        bars_f = axes[1, 0].bar(x_m + width_m, f1, width_m, label="F1")
        axes[1, 0].set_xticks(x_m)
        axes[1, 0].set_xticklabels(metrics_names)
        axes[1, 0].set_ylabel("Score (%)")
        axes[1, 0].set_title("Model precision/recall/F1")
        axes[1, 0].legend()
        axes[1, 0].bar_label(bars_p, fmt="%.1f", padding=3)
        axes[1, 0].bar_label(bars_r, fmt="%.1f", padding=3)
        axes[1, 0].bar_label(bars_f, fmt="%.1f", padding=3)

        # Timings
        x_t = np.arange(len(time_names))
        width_t = 0.35
        bars_train = axes[1, 1].bar(x_t - width_t / 2, train, width_t, label="Train")
        bars_test = axes[1, 1].bar(x_t + width_t / 2, test, width_t, label="Test")
        axes[1, 1].set_xticks(x_t)
        axes[1, 1].set_xticklabels(time_names)
        axes[1, 1].set_ylabel("Seconds")
        axes[1, 1].set_title("Model processing time")
        axes[1, 1].legend()
        axes[1, 1].bar_label(bars_train, fmt="%.2f", padding=3)
        axes[1, 1].bar_label(bars_test, fmt="%.2f", padding=3)

        fig.tight_layout()
        out = Path("summary_view.png")
        fig.savefig(out)
        plt.close(fig)
        return out

    def _plot_heatmap(
        self,
        column: str,
        title: str,
        *,
        ax: plt.Axes | None = None,
        top_n: int | None = None,
        mapping: dict | None = None,
    ) -> None:
        """Plot a heatmap of anomaly counts for ``column`` by model."""

        if not hasattr(self, "_test_df") or not hasattr(self, "_last_predictions"):
            raise RuntimeError("Run test first.")

        df = self._test_df.copy()
        if mapping:
            df[column] = df[column].map(mapping)

        if top_n is not None:
            top_categories = df[column].value_counts().nlargest(top_n).index
            df = df[df[column].isin(top_categories)]

        categories = sorted(df[column].dropna().unique())
        models = list(self._last_predictions.keys())
        heat = pd.DataFrame(0, index=categories, columns=models)

        for model, preds in self._last_predictions.items():
            subset = df[preds == -1]
            counts = subset[column].value_counts()
            for cat, val in counts.items():
                heat.loc[cat, model] = val

        if ax is None:
            ax = plt.gca()
        sns.heatmap(heat, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Model")
        ax.set_ylabel(column.replace("_", " ").title())
        ax.set_title(title)

    def visualize_histograms(self) -> Path:
        """Create a grid of histograms for various categorical features."""

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        self._plot_heatmap(
            column="card_scheme",
            title="Card Scheme",
            ax=axes[0, 0],
        )
        self._plot_heatmap(
            column="card_level",
            title="Card Level",
            ax=axes[0, 1],
        )
        self._plot_heatmap(
            column="device_info",
            title="Device Info",
            ax=axes[1, 0],
        )
        self._plot_heatmap(
            column="3D_SecureTransaction(yes/no)",
            title="3D Secure",
            ax=axes[1, 1],
            mapping={1: "Yes", 0: "No"},
        )
        fig.tight_layout()
        out = Path("anomaly_histograms.png")
        fig.savefig(out)
        plt.close(fig)
        return out




if __name__ == "__main__":
    detector = FraudDetector("train_dataset.csv", "test_dataset.csv")
    detector.train()
    results = detector.test()
    print("Anomalies detected:", results)
    img = detector.visualize(results)
    print(f"Saved comparison chart to {img}")
