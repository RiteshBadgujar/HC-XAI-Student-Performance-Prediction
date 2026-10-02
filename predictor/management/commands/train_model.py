from pathlib import Path
import pandas as pd
from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from predictor.services.ml_service import FEATURES, MODEL_PATH


class Command(BaseCommand):
    help = "Train the Random Forest student performance classifier from a labeled CSV."

    def add_arguments(self, parser):
        parser.add_argument("csv_path")
        parser.add_argument("--target", default="performance_category")

    def handle(self, *args, **options):
        try:
            from sklearn.ensemble import RandomForestClassifier
        except ImportError as exc:
            raise CommandError("scikit-learn is not installed.") from exc
        try:
            import joblib
        except ImportError as exc:
            raise CommandError("joblib is not installed.") from exc

        path = Path(options["csv_path"])
        if not path.exists():
            raise CommandError(f"Dataset not found: {path}")
        df = pd.read_csv(path)
        target = options["target"]
        missing = [c for c in FEATURES + [target] if c not in df.columns]
        if missing:
            raise CommandError("Missing columns: " + ", ".join(missing))
        df = df[FEATURES + [target]].dropna()
        if len(df) < 10 or df[target].nunique() < 2:
            raise CommandError("Provide at least 10 complete rows with at least two target classes.")

        model = RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")
        model.fit(df[FEATURES], df[target])
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, MODEL_PATH)
        self.stdout.write(self.style.SUCCESS(f"Model trained and saved to {MODEL_PATH}"))
