from pathlib import Path

from detector import FraudDetector


class Menu:
    def __init__(self) -> None:
        self.detector = FraudDetector("train_dataset.csv", "test_dataset.csv")
        self.trained = False
        self.results = None

    def run(self) -> None:
        while True:
            print(
                "\nMenu:\n"
                "1. Train models\n"
                "2. Test models\n"
                "3. Visualize comparison\n"
                "4. Save anomalies\n"
                "5. Amount box plot\n"
                "6. Heatmap correlation\n"
                "7. Exit"
            )
            choice = input("Select option: ").strip()
            if choice == "1":
                self.detector.train()
                self.trained = True
                print("Training finished")
            elif choice == "2":
                if not self.trained and not Path("isolation_forest.joblib").exists():
                    print("Train models first")
                    continue
                self.results = self.detector.test()
                for name, count in self.results.items():
                    print(f"{name}: {count} anomalies")
            elif choice == "3":
                if self.results is None:
                    print("Run test first")
                    continue
                img = self.detector.visualize(self.results)
                print(f"Chart saved to {img}")
            elif choice == "4":
                if self.results is None:
                    print("Run test first")
                    continue
                files = self.detector.save_anomalies()
                for f in files:
                    print(f"Saved anomalies to {f}")
            elif choice == "5":
                if self.results is None:
                    print("Run test first")
                    continue
                img = self.detector.visualize_boxplot()
                print(f"Box plot saved to {img}")
            elif choice == "6":
                if self.results is None:
                    print("Run test first")
                    continue
                img = self.detector.visualize_heatmap()
                print(f"Heatmap saved to {img}")
            elif choice == "7":
                break
            else:
                print("Invalid option")


if __name__ == "__main__":
    Menu().run()
