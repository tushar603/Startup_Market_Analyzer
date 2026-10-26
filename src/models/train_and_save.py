"""
Module: src/models/train_and_save.py
Role: Trains the Out-of-Fold Stacking Ensemble on real Crunchbase records.
"""

import os
from src.data.build_dataset import DatasetBuilder
from src.data.crunchbase_adapter import prepare_crunchbase_cohort
from src.features.fusion import FeatureFusionPipeline
from src.models.stacking import OutOfFoldStackingClassifier
from src.nlp.sentiment_engine_ import HybridNLPEngine


def train_and_persist(output_dir: str = "models", artifact_name: str = "stacking_ensemble_v1.pkl") -> str:
    os.makedirs(output_dir, exist_ok=True)
    target_path = os.path.join(output_dir, artifact_name)

    print("[1/4] Processing raw Crunchbase historical dataset...")
    cohort = prepare_crunchbase_cohort(data_dir="data/dataset", max_samples=1500)
    print(f"      Loaded {len(cohort)} balanced historical startup records.")

    print("[2/4] Initializing Feature Fusion Pipeline...")
    nlp = HybridNLPEngine(use_transformer=False)
    pipeline = FeatureFusionPipeline(nlp_engine=nlp)
    builder = DatasetBuilder(pipeline=pipeline)

    print("[3/4] Compiling zero-leakage tabular features (X, y)...")
    X, y, report = builder.compile_tabular_dataset(cohort, drop_unlabeled=True)
    print(f"      Compiled: {X.shape[0]} rows × {X.shape[1]} features.")
    print(f"      Empirical Class Balance: {report.get('class_distribution')}")

    print("[4/4] Fitting Out-of-Fold Stacking Classifier (RF + XGBoost + Logistic Regression)...")
    model = OutOfFoldStackingClassifier(n_splits=5, random_state=42)
    model.fit(X, y["growth_category"])

    model.save_artifacts(target_path)
    print(f"[✓] Successfully serialized production model to: {target_path}")
    return target_path


if __name__ == "__main__":
    train_and_persist()