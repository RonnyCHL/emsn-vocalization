#!/usr/bin/env python3
"""
EMSN 2.0 - CNN Classifier voor Vocalisatie Types

Traint een Convolutional Neural Network op mel-spectrogrammen
voor classificatie van song/call/alarm.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# TensorFlow imports
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF warnings
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CNNVocalizationClassifier:
    """CNN classifier voor vocalisatie types op basis van spectrogrammen."""

    def __init__(
        self,
        input_shape: tuple = (128, 282, 1),
        num_classes: int = 3,
        learning_rate: float = 0.001
    ):
        """
        Initialiseer de classifier.

        Args:
            input_shape: Shape van input spectrogrammen (height, width, channels)
            num_classes: Aantal output klassen
            learning_rate: Learning rate voor optimizer
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.model = None
        self.label_encoder = LabelEncoder()
        self.history = None

    def build_model(self) -> keras.Model:
        """
        Bouw het CNN model.

        Returns:
            Compiled Keras model
        """
        model = keras.Sequential([
            # Input layer
            layers.Input(shape=self.input_shape),

            # Conv block 1
            layers.Conv2D(32, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            # Conv block 2
            layers.Conv2D(64, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            # Conv block 3
            layers.Conv2D(128, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            # Conv block 4
            layers.Conv2D(256, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.GlobalAveragePooling2D(),

            # Dense layers
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        self.model = model
        return model

    def load_data(self, data_dir: str) -> tuple:
        """
        Laad spectrogrammen en labels.

        Args:
            data_dir: Directory met X_spectrograms.npy en y_labels.npy

        Returns:
            Tuple van (X, y_encoded, class_names)
        """
        data_dir = Path(data_dir)

        logger.info(f"Laden data uit {data_dir}")

        X = np.load(data_dir / 'X_spectrograms.npy')
        y = np.load(data_dir / 'y_labels.npy')

        # Voeg channel dimensie toe
        X = X[..., np.newaxis]

        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        class_names = self.label_encoder.classes_.tolist()

        logger.info(f"Data shape: X={X.shape}, y={y_encoded.shape}")
        logger.info(f"Klassen: {class_names}")
        logger.info(f"Verdeling: {dict(zip(*np.unique(y, return_counts=True)))}")

        return X, y_encoded, class_names

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2,
        epochs: int = 100,
        batch_size: int = 32,
        patience: int = 15
    ) -> dict:
        """
        Train het model.

        Args:
            X: Input spectrogrammen
            y: Encoded labels
            test_size: Fractie voor test set
            epochs: Maximum aantal epochs
            batch_size: Batch size
            patience: Early stopping patience

        Returns:
            Dict met training resultaten
        """
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            stratify=y,
            random_state=42
        )

        logger.info(f"Train set: {X_train.shape[0]} samples")
        logger.info(f"Test set: {X_test.shape[0]} samples")

        # Bouw model indien nodig
        if self.model is None:
            self.input_shape = X_train.shape[1:]
            self.build_model()

        self.model.summary()

        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            )
        ]

        # Class weights voor imbalance
        class_counts = np.bincount(y_train)
        total = len(y_train)
        class_weights = {i: total / (len(class_counts) * count) for i, count in enumerate(class_counts)}
        logger.info(f"Class weights: {class_weights}")

        # Train
        logger.info(f"Starten training ({epochs} epochs max)...")
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=1
        )

        # Evalueer
        logger.info("Evalueren op test set...")
        y_pred_proba = self.model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)

        # Metrics
        accuracy = np.mean(y_pred == y_test)
        class_names = self.label_encoder.classes_.tolist()
        report = classification_report(y_test, y_pred, target_names=class_names)
        cm = confusion_matrix(y_test, y_pred)

        logger.info(f"Test Accuracy: {accuracy:.2%}")

        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm,
            'class_names': class_names,
            'history': self.history.history,
            'y_test': y_test,
            'y_pred': y_pred
        }

    def save(self, filepath: str):
        """Sla model op."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        self.model.save(filepath)

        # Sla label encoder ook op
        encoder_path = str(filepath).replace('.keras', '_encoder.npy')
        np.save(encoder_path, self.label_encoder.classes_)

        logger.info(f"Model opgeslagen: {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'CNNVocalizationClassifier':
        """Laad model."""
        classifier = cls()
        classifier.model = keras.models.load_model(filepath)

        # Laad label encoder
        encoder_path = str(filepath).replace('.keras', '_encoder.npy')
        if Path(encoder_path).exists():
            classifier.label_encoder.classes_ = np.load(encoder_path, allow_pickle=True)

        return classifier

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Voorspel klassen."""
        if X.ndim == 3:
            X = X[..., np.newaxis]
        proba = self.model.predict(X, verbose=0)
        return self.label_encoder.inverse_transform(np.argmax(proba, axis=1))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Voorspel waarschijnlijkheden."""
        if X.ndim == 3:
            X = X[..., np.newaxis]
        return self.model.predict(X, verbose=0)


def plot_training_history(history: dict, output_path: str):
    """Plot training curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Loss
    ax1.plot(history['loss'], label='Train')
    ax1.plot(history['val_loss'], label='Validation')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training en Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy
    ax2.plot(history['accuracy'], label='Train')
    ax2.plot(history['val_accuracy'], label='Validation')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training en Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    logger.info(f"Training history opgeslagen: {output_path}")


def plot_confusion_matrix(cm: np.ndarray, class_names: list, output_path: str, accuracy: float):
    """Plot confusion matrix."""
    plt.figure(figsize=(8, 6))

    # Normaliseer
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    sns.heatmap(
        cm_norm,
        annot=True,
        fmt='.1%',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        square=True
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
    plt.title(f'CNN Vocalisatie Classifier\nAccuracy: {accuracy:.1%}')
    plt.tight_layout()

    plt.savefig(output_path, dpi=150)
    plt.close()

    logger.info(f"Confusion matrix opgeslagen: {output_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Train CNN vocalisatie classifier')
    parser.add_argument('--data-dir', default='data/spectrograms',
                       help='Directory met spectrogrammen')
    parser.add_argument('--output-model', default='data/models/merel_cnn_v1.keras',
                       help='Output model bestand')
    parser.add_argument('--output-dir', default='logs',
                       help='Output directory voor rapporten')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Maximum epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--patience', type=int, default=15,
                       help='Early stopping patience')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Test set fractie')

    args = parser.parse_args()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    print(f"\n{'='*60}")
    print("EMSN 2.0 - CNN Vocalisatie Classifier Training")
    print(f"{'='*60}")
    print(f"Data: {args.data_dir}")
    print(f"Model output: {args.output_model}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Early stopping patience: {args.patience}")
    print(f"{'='*60}\n")

    # Maak output dirs
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialiseer classifier
    classifier = CNNVocalizationClassifier()

    # Laad data
    X, y, class_names = classifier.load_data(args.data_dir)

    # Train
    results = classifier.train(
        X, y,
        test_size=args.test_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience
    )

    # Print resultaten
    print(f"\n{'='*60}")
    print("RESULTATEN")
    print(f"{'='*60}")
    print(f"\nAccuracy: {results['accuracy']:.2%}")
    print(f"\n{'-'*60}")
    print("Classification Report:")
    print(f"{'-'*60}")
    print(results['classification_report'])
    print(f"\n{'-'*60}")
    print("Confusion Matrix:")
    print(f"{'-'*60}")
    print(f"Classes: {class_names}")
    print(results['confusion_matrix'])

    # Sla model op
    classifier.save(args.output_model)

    # Plots
    history_path = output_dir / f"cnn_training_history_{timestamp}.png"
    plot_training_history(results['history'], str(history_path))

    cm_path = output_dir / f"cnn_confusion_matrix_{timestamp}.png"
    plot_confusion_matrix(
        results['confusion_matrix'],
        class_names,
        str(cm_path),
        results['accuracy']
    )

    # Sla rapport op
    report_path = output_dir / f"cnn_training_report_{timestamp}.json"
    report_data = {
        'timestamp': timestamp,
        'data_dir': args.data_dir,
        'model_file': args.output_model,
        'parameters': {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'patience': args.patience,
            'test_size': args.test_size
        },
        'results': {
            'accuracy': results['accuracy'],
            'final_train_loss': results['history']['loss'][-1],
            'final_val_loss': results['history']['val_loss'][-1],
            'epochs_trained': len(results['history']['loss'])
        },
        'confusion_matrix': results['confusion_matrix'].tolist()
    }

    with open(report_path, 'w') as f:
        json.dump(report_data, f, indent=2)

    print(f"\n{'='*60}")
    print("OUTPUTS")
    print(f"{'='*60}")
    print(f"  Model: {args.output_model}")
    print(f"  Training history: {history_path}")
    print(f"  Confusion matrix: {cm_path}")
    print(f"  Rapport: {report_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
