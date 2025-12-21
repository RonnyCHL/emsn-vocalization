#!/usr/bin/env python3
"""
EMSN 2.0 Vocalisatie Classifier - Random Forest Training

Train een classifier om vogelzang, roep en alarm te onderscheiden
op basis van BirdNET features.
"""

import json
import logging
import os
import pickle
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class TrainingResult:
    """Resultaten van model training."""
    accuracy: float
    f1_macro: float
    f1_weighted: float
    classification_report: str
    confusion_matrix: np.ndarray
    class_names: list[str]
    cv_scores: Optional[np.ndarray] = None
    feature_importances: Optional[np.ndarray] = None
    pca_variance_explained: Optional[float] = None


class VocalizationClassifier:
    """
    Classifier voor vocalisatie types (song/call/alarm).

    Gebruikt Random Forest op BirdNET features, optioneel met PCA.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        pca_components: Optional[int] = None,
        random_state: int = 42
    ):
        """
        Initialiseer de classifier.

        Args:
            n_estimators: Aantal decision trees
            pca_components: Aantal PCA componenten (None = geen PCA)
            random_state: Random seed voor reproduceerbaarheid
        """
        self.n_estimators = n_estimators
        self.pca_components = pca_components
        self.random_state = random_state

        self.scaler = StandardScaler()
        self.pca = None
        self.classifier = None
        self.class_names = None
        self.feature_names = None

        if pca_components is not None:
            self.pca = PCA(n_components=pca_components, random_state=random_state)

    def load_data(self, csv_path: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """
        Laad features en labels uit CSV.

        Args:
            csv_path: Pad naar features CSV

        Returns:
            Tuple van (X features, y labels, class_names)
        """
        logger.info(f"Laden data uit: {csv_path}")

        df = pd.read_csv(csv_path)
        logger.info(f"Dataset shape: {df.shape}")

        # Extract features (alle kolommen behalve metadata)
        # Ondersteunt zowel BirdNET features (feature_*) als akoestische features
        metadata_cols = ['file_path', 'segment_idx', 'vocalization_type', 'xeno_canto_id']
        feature_cols = [c for c in df.columns if c not in metadata_cols]
        self.feature_names = feature_cols

        X = df[feature_cols].values
        y = df['vocalization_type'].values

        # Class names
        self.class_names = sorted(list(set(y)))

        logger.info(f"Features shape: {X.shape}")
        logger.info(f"Feature namen: {feature_cols[:5]}... ({len(feature_cols)} totaal)")
        logger.info(f"Labels: {dict(zip(*np.unique(y, return_counts=True)))}")

        return X, y, self.class_names

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2,
        cv_folds: int = 5
    ) -> TrainingResult:
        """
        Train de classifier.

        Args:
            X: Feature matrix
            y: Labels
            test_size: Fractie voor test set
            cv_folds: Aantal cross-validation folds

        Returns:
            TrainingResult met alle metrics
        """
        logger.info("Splitsen in train/test sets...")

        # Stratified train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            stratify=y,
            random_state=self.random_state
        )

        logger.info(f"Train set: {X_train.shape[0]} samples")
        logger.info(f"Test set: {X_test.shape[0]} samples")

        # Scaling
        logger.info("Schalen van features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # PCA (optioneel)
        pca_variance = None
        if self.pca is not None:
            logger.info(f"Toepassen PCA ({self.pca_components} componenten)...")
            X_train_scaled = self.pca.fit_transform(X_train_scaled)
            X_test_scaled = self.pca.transform(X_test_scaled)
            pca_variance = sum(self.pca.explained_variance_ratio_)
            logger.info(f"PCA verklaarde variantie: {pca_variance:.2%}")

        # Train classifier
        logger.info(f"Trainen Random Forest ({self.n_estimators} trees)...")
        self.classifier = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            class_weight='balanced',  # Compenseer class imbalance
            n_jobs=-1,
            random_state=self.random_state,
            verbose=0
        )

        self.classifier.fit(X_train_scaled, y_train)

        # Predictions
        logger.info("Evalueren op test set...")
        y_pred = self.classifier.predict(X_test_scaled)

        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average='macro')
        f1_weighted = f1_score(y_test, y_pred, average='weighted')
        report = classification_report(y_test, y_pred, target_names=self.class_names)
        cm = confusion_matrix(y_test, y_pred, labels=self.class_names)

        logger.info(f"Accuracy: {accuracy:.2%}")
        logger.info(f"F1 (macro): {f1_macro:.2%}")
        logger.info(f"F1 (weighted): {f1_weighted:.2%}")

        # Cross-validation
        logger.info(f"Cross-validation ({cv_folds} folds)...")

        # Voor CV moeten we de hele pipeline opnieuw fitten
        X_scaled = self.scaler.fit_transform(X)
        if self.pca is not None:
            X_scaled = self.pca.fit_transform(X_scaled)

        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        cv_scores = cross_val_score(
            self.classifier, X_scaled, y,
            cv=cv, scoring='accuracy', n_jobs=-1
        )

        logger.info(f"CV Accuracy: {cv_scores.mean():.2%} (+/- {cv_scores.std()*2:.2%})")

        # Feature importances (alleen zonder PCA zinvol)
        feature_importances = None
        if self.pca is None:
            feature_importances = self.classifier.feature_importances_

        return TrainingResult(
            accuracy=accuracy,
            f1_macro=f1_macro,
            f1_weighted=f1_weighted,
            classification_report=report,
            confusion_matrix=cm,
            class_names=self.class_names,
            cv_scores=cv_scores,
            feature_importances=feature_importances,
            pca_variance_explained=pca_variance
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Voorspel vocalisatie types."""
        X_scaled = self.scaler.transform(X)
        if self.pca is not None:
            X_scaled = self.pca.transform(X_scaled)
        return self.classifier.predict(X_scaled)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Voorspel vocalisatie types met waarschijnlijkheden."""
        X_scaled = self.scaler.transform(X)
        if self.pca is not None:
            X_scaled = self.pca.transform(X_scaled)
        return self.classifier.predict_proba(X_scaled)

    def save(self, filepath: str):
        """Sla model op als pickle."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            'scaler': self.scaler,
            'pca': self.pca,
            'classifier': self.classifier,
            'class_names': self.class_names,
            'feature_names': self.feature_names,
            'n_estimators': self.n_estimators,
            'pca_components': self.pca_components,
            'random_state': self.random_state
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

        logger.info(f"Model opgeslagen: {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'VocalizationClassifier':
        """Laad model uit pickle."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        classifier = cls(
            n_estimators=model_data['n_estimators'],
            pca_components=model_data['pca_components'],
            random_state=model_data['random_state']
        )

        classifier.scaler = model_data['scaler']
        classifier.pca = model_data['pca']
        classifier.classifier = model_data['classifier']
        classifier.class_names = model_data['class_names']
        classifier.feature_names = model_data['feature_names']

        return classifier


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    output_path: str,
    title: str = "Confusion Matrix"
):
    """Plot en sla confusion matrix op."""
    plt.figure(figsize=(8, 6))

    # Normaliseer voor percentages
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Plot
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt='.1%',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        square=True,
        cbar_kws={'label': 'Percentage'}
    )

    # Voeg absolute aantallen toe
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            plt.text(
                j + 0.5, i + 0.7,
                f'(n={cm[i, j]})',
                ha='center', va='center',
                fontsize=8, color='gray'
            )

    plt.xlabel('Voorspeld')
    plt.ylabel('Werkelijk')
    plt.title(title)
    plt.tight_layout()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Confusion matrix opgeslagen: {output_path}")


def plot_feature_importance(
    importances: np.ndarray,
    feature_names: list[str],
    output_path: str,
    top_n: int = 30
):
    """Plot top N belangrijkste features."""
    # Sorteer op importance
    indices = np.argsort(importances)[::-1][:top_n]

    plt.figure(figsize=(10, 8))
    plt.barh(range(top_n), importances[indices][::-1])
    plt.yticks(range(top_n), [feature_names[i] for i in indices[::-1]])
    plt.xlabel('Feature Importance')
    plt.title(f'Top {top_n} Belangrijkste Features')
    plt.tight_layout()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Feature importance plot opgeslagen: {output_path}")


def main():
    """Train vocalisatie classifier."""
    import argparse

    parser = argparse.ArgumentParser(description='Train vocalisatie classifier')
    parser.add_argument('--input', default='data/embeddings/merel_features.csv',
                       help='Input features CSV')
    parser.add_argument('--output-model', default='data/models/merel_classifier_v1.pkl',
                       help='Output model bestand')
    parser.add_argument('--output-dir', default='logs',
                       help='Output directory voor rapporten en plots')
    parser.add_argument('--n-estimators', type=int, default=200,
                       help='Aantal decision trees')
    parser.add_argument('--pca', type=int, default=None,
                       help='PCA componenten (None = geen PCA)')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Test set fractie')
    parser.add_argument('--cv-folds', type=int, default=5,
                       help='Cross-validation folds')

    args = parser.parse_args()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    print(f"\n{'='*60}")
    print("EMSN 2.0 - Vocalisatie Classifier Training")
    print(f"{'='*60}")
    print(f"Input: {args.input}")
    print(f"Model output: {args.output_model}")
    print(f"N estimators: {args.n_estimators}")
    print(f"PCA componenten: {args.pca or 'Geen'}")
    print(f"Test size: {args.test_size}")
    print(f"CV folds: {args.cv_folds}")
    print(f"{'='*60}\n")

    # Maak output dirs
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialiseer classifier
    classifier = VocalizationClassifier(
        n_estimators=args.n_estimators,
        pca_components=args.pca
    )

    # Laad data
    X, y, class_names = classifier.load_data(args.input)

    # Train
    result = classifier.train(X, y, test_size=args.test_size, cv_folds=args.cv_folds)

    # Print resultaten
    print(f"\n{'='*60}")
    print("RESULTATEN")
    print(f"{'='*60}")
    print(f"\nAccuracy: {result.accuracy:.2%}")
    print(f"F1 (macro): {result.f1_macro:.2%}")
    print(f"F1 (weighted): {result.f1_weighted:.2%}")

    if result.pca_variance_explained:
        print(f"PCA variantie verklaard: {result.pca_variance_explained:.2%}")

    print(f"\nCross-validation ({args.cv_folds}-fold):")
    print(f"  Mean: {result.cv_scores.mean():.2%}")
    print(f"  Std:  {result.cv_scores.std():.2%}")
    print(f"  Min:  {result.cv_scores.min():.2%}")
    print(f"  Max:  {result.cv_scores.max():.2%}")

    print(f"\n{'-'*60}")
    print("Classification Report:")
    print(f"{'-'*60}")
    print(result.classification_report)

    print(f"\n{'-'*60}")
    print("Confusion Matrix:")
    print(f"{'-'*60}")
    print(f"Classes: {class_names}")
    print(result.confusion_matrix)

    # Sla model op
    classifier.save(args.output_model)

    # Plots
    cm_path = output_dir / f"confusion_matrix_{timestamp}.png"
    plot_confusion_matrix(
        result.confusion_matrix,
        class_names,
        str(cm_path),
        title=f"Merel Vocalisatie Classifier\nAccuracy: {result.accuracy:.1%}"
    )

    # Feature importance (alleen zonder PCA)
    if result.feature_importances is not None:
        fi_path = output_dir / f"feature_importance_{timestamp}.png"
        plot_feature_importance(
            result.feature_importances,
            classifier.feature_names,
            str(fi_path)
        )

    # Sla rapport op als JSON
    report_path = output_dir / f"training_report_{timestamp}.json"
    report_data = {
        'timestamp': timestamp,
        'input_file': args.input,
        'model_file': args.output_model,
        'parameters': {
            'n_estimators': args.n_estimators,
            'pca_components': args.pca,
            'test_size': args.test_size,
            'cv_folds': args.cv_folds
        },
        'results': {
            'accuracy': result.accuracy,
            'f1_macro': result.f1_macro,
            'f1_weighted': result.f1_weighted,
            'cv_mean': float(result.cv_scores.mean()),
            'cv_std': float(result.cv_scores.std()),
            'pca_variance_explained': result.pca_variance_explained
        },
        'class_distribution': dict(zip(*np.unique(y, return_counts=True))),
        'confusion_matrix': result.confusion_matrix.tolist()
    }

    with open(report_path, 'w') as f:
        json.dump(report_data, f, indent=2, default=str)

    logger.info(f"Training rapport opgeslagen: {report_path}")

    print(f"\n{'='*60}")
    print("OUTPUTS")
    print(f"{'='*60}")
    print(f"  Model: {args.output_model}")
    print(f"  Confusion matrix: {cm_path}")
    if result.feature_importances is not None:
        print(f"  Feature importance: {fi_path}")
    print(f"  Rapport: {report_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
