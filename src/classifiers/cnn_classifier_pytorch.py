#!/usr/bin/env python3
"""
EMSN 2.0 - PyTorch CNN Classifier voor Vocalisatie Types

Traint een Convolutional Neural Network op mel-spectrogrammen
voor classificatie van song/call/alarm.

PyTorch versie - werkt op CPUs zonder AVX ondersteuning.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VocalizationCNN(nn.Module):
    """CNN model voor vocalisatie classificatie."""

    def __init__(self, input_channels=1, num_classes=3):
        super(VocalizationCNN, self).__init__()

        # Conv block 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25)
        )

        # Conv block 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25)
        )

        # Conv block 3
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25)
        )

        # Conv block 4
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))  # Global average pooling
        )

        # Dense layers
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.classifier(x)
        return x


class CNNVocalizationClassifier:
    """CNN classifier voor vocalisatie types op basis van spectrogrammen."""

    def __init__(
        self,
        num_classes: int = 3,
        learning_rate: float = 0.001,
        device: str = None
    ):
        self.num_classes = num_classes
        self.learning_rate = learning_rate

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        logger.info(f"Gebruik device: {self.device}")

        self.model = None
        self.label_encoder = LabelEncoder()
        self.history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}

    def build_model(self, input_channels: int = 1) -> VocalizationCNN:
        self.model = VocalizationCNN(
            input_channels=input_channels,
            num_classes=self.num_classes
        ).to(self.device)
        return self.model

    def load_data(self, data_dir: str) -> tuple:
        data_dir = Path(data_dir)
        logger.info(f"Laden data uit {data_dir}")

        X = np.load(data_dir / 'X_spectrograms.npy')
        y = np.load(data_dir / 'y_labels.npy')

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
        patience: int = 15,
        progress_callback=None
    ) -> dict:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=42
        )

        logger.info(f"Train set: {X_train.shape[0]} samples")
        logger.info(f"Test set: {X_test.shape[0]} samples")

        X_train = torch.FloatTensor(X_train).unsqueeze(1)
        X_test = torch.FloatTensor(X_test).unsqueeze(1)
        y_train = torch.LongTensor(y_train)
        y_test = torch.LongTensor(y_test)

        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)

        # DataLoaders - num_workers=0 voor stabiliteit op NAS
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            num_workers=0,
            pin_memory=False,
            
        )

        if self.model is None:
            self.build_model(input_channels=1)

        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Model parameters: {total_params:,}")

        class_counts = np.bincount(y_train.numpy())
        total = len(y_train)
        class_weights = torch.FloatTensor([total / (len(class_counts) * count) for count in class_counts])
        class_weights = class_weights.to(self.device)
        logger.info(f"Class weights: {class_weights.cpu().numpy()}")

        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        # Agressievere LR scheduling voor snellere convergentie
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6
        )

        best_val_loss = float('inf')
        best_model_state = None
        epochs_without_improvement = 0

        logger.info(f"Starten training ({epochs} epochs max)...")

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * batch_X.size(0)
                _, predicted = outputs.max(1)
                train_total += batch_y.size(0)
                train_correct += predicted.eq(batch_y).sum().item()

            train_loss /= train_total
            train_acc = train_correct / train_total

            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for batch_X, batch_y in test_loader:
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)

                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)

                    val_loss += loss.item() * batch_X.size(0)
                    _, predicted = outputs.max(1)
                    val_total += batch_y.size(0)
                    val_correct += predicted.eq(batch_y).sum().item()

            val_loss /= val_total
            val_acc = val_correct / val_total

            self.history['loss'].append(train_loss)
            self.history['accuracy'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_accuracy'].append(val_acc)

            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = self.model.state_dict().copy()
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            # Callback voor voortgang
            if progress_callback:
                progress_callback(epoch + 1, epochs, val_acc)

            if (epoch + 1) % 5 == 0 or epoch == 0:
                logger.info(
                    f"Epoch {epoch+1}/{epochs} - "
                    f"loss: {train_loss:.4f} - acc: {train_acc:.4f} - "
                    f"val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f}"
                )

            if epochs_without_improvement >= patience:
                logger.info(f"Early stopping na {epoch+1} epochs")
                break

        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            logger.info("Best model weights hersteld")

        logger.info("Evalueren op test set...")
        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X = batch_X.to(self.device)
                outputs = self.model(batch_X)
                _, predicted = outputs.max(1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(batch_y.numpy())

        y_pred = np.array(all_preds)
        y_test_np = np.array(all_labels)

        accuracy = np.mean(y_pred == y_test_np)
        class_names = self.label_encoder.classes_.tolist()
        report = classification_report(y_test_np, y_pred, target_names=class_names)
        cm = confusion_matrix(y_test_np, y_pred)

        logger.info(f"Test Accuracy: {accuracy:.2%}")

        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm,
            'class_names': class_names,
            'history': self.history,
            'y_test': y_test_np,
            'y_pred': y_pred
        }

    def save(self, filepath: str):
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        if filepath.endswith('.keras'):
            filepath = filepath.replace('.keras', '.pt')

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'label_encoder_classes': self.label_encoder.classes_,
            'num_classes': self.num_classes
        }, filepath)

        logger.info(f"Model opgeslagen: {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'CNNVocalizationClassifier':
        checkpoint = torch.load(filepath, map_location='cpu')
        classifier = cls(num_classes=checkpoint['num_classes'])
        classifier.build_model()
        classifier.model.load_state_dict(checkpoint['model_state_dict'])
        classifier.label_encoder.classes_ = checkpoint['label_encoder_classes']
        return classifier

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        if X.ndim == 3:
            X = X[:, np.newaxis, :, :]
        X_tensor = torch.FloatTensor(X).to(self.device)
        with torch.no_grad():
            outputs = self.model(X_tensor)
            _, predicted = outputs.max(1)
        return self.label_encoder.inverse_transform(predicted.cpu().numpy())

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        if X.ndim == 3:
            X = X[:, np.newaxis, :, :]
        X_tensor = torch.FloatTensor(X).to(self.device)
        with torch.no_grad():
            outputs = self.model(X_tensor)
            proba = torch.softmax(outputs, dim=1)
        return proba.cpu().numpy()


def plot_training_history(history: dict, output_path: str):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history['loss'], label='Train')
    ax1.plot(history['val_loss'], label='Validation')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training en Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

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
    plt.figure(figsize=(8, 6))
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    sns.heatmap(
        cm_norm, annot=True, fmt='.1%', cmap='Blues',
        xticklabels=class_names, yticklabels=class_names, square=True
    )

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            plt.text(j + 0.5, i + 0.7, f'(n={cm[i, j]})',
                ha='center', va='center', fontsize=8, color='gray')

    plt.xlabel('Voorspeld')
    plt.ylabel('Werkelijk')
    plt.title(f'PyTorch CNN Vocalisatie Classifier\nAccuracy: {accuracy:.1%}')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info(f"Confusion matrix opgeslagen: {output_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Train PyTorch CNN vocalisatie classifier')
    parser.add_argument('--data-dir', default='data/spectrograms', help='Directory met spectrogrammen')
    parser.add_argument('--output-model', default='data/models/merel_cnn_v1.pt', help='Output model bestand')
    parser.add_argument('--output-dir', default='logs', help='Output directory voor rapporten')
    parser.add_argument('--epochs', type=int, default=100, help='Maximum epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--patience', type=int, default=15, help='Early stopping patience')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test set fractie')

    args = parser.parse_args()

    if args.output_model.endswith('.keras'):
        args.output_model = args.output_model.replace('.keras', '.pt')

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    print(f"\n{'='*60}")
    print("EMSN 2.0 - PyTorch CNN Vocalisatie Classifier Training")
    print(f"{'='*60}")
    print(f"Data: {args.data_dir}")
    print(f"Model output: {args.output_model}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Early stopping patience: {args.patience}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"{'='*60}\n")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    classifier = CNNVocalizationClassifier()
    X, y, class_names = classifier.load_data(args.data_dir)

    results = classifier.train(
        X, y,
        test_size=args.test_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience
    )

    print(f"\n{'='*60}")
    print("RESULTATEN")
    print(f"{'='*60}")
    print(f"\nTest Accuracy: {results['accuracy']:.2%}")
    print(f"\n{'-'*60}")
    print("Classification Report:")
    print(f"{'-'*60}")
    print(results['classification_report'])
    print(f"\n{'-'*60}")
    print("Confusion Matrix:")
    print(f"{'-'*60}")
    print(f"Classes: {class_names}")
    print(results['confusion_matrix'])

    classifier.save(args.output_model)

    history_path = output_dir / f"cnn_training_history_{timestamp}.png"
    plot_training_history(results['history'], str(history_path))

    cm_path = output_dir / f"cnn_confusion_matrix_{timestamp}.png"
    plot_confusion_matrix(results['confusion_matrix'], class_names, str(cm_path), results['accuracy'])

    report_path = output_dir / f"cnn_training_report_{timestamp}.json"
    report_data = {
        'timestamp': timestamp,
        'data_dir': args.data_dir,
        'model_file': args.output_model,
        'framework': 'PyTorch',
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
