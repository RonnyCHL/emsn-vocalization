#!/usr/bin/env python3
"""
BirdNET Vocalization Classifier

Lightweight CNN classifier for bird vocalization types (song/call/alarm).
Works with BirdNET-Pi audio files and species-specific trained models.

Can be used standalone or integrated with display systems like AWTRIX/Ulanzi.

Usage:
    # Set models directory (or use VOCALIZATION_MODELS_DIR env var)
    classifier = VocalizationClassifier(models_dir="/path/to/models")

    # Classify a detection
    result = classifier.classify("Eurasian Blackbird", "/path/to/audio.wav")
    if result:
        print(f"{result['type']} ({result['confidence']:.0%})")
        # Output: song (87%)

Requirements:
    - PyTorch
    - librosa
    - scikit-image
    - numpy
"""

import re
import logging
from pathlib import Path

import numpy as np

# Lazy imports voor snellere startup
_torch = None
_librosa = None


def get_torch():
    """Lazy load torch."""
    global _torch
    if _torch is None:
        import torch
        _torch = torch
    return _torch


def get_librosa():
    """Lazy load librosa."""
    global _librosa
    if _librosa is None:
        import librosa
        _librosa = librosa
    return _librosa


# Configuration - override MODELS_DIR via environment or constructor
MODELS_DIR = Path(
    __import__('os').environ.get(
        'VOCALIZATION_MODELS_DIR',
        '/mnt/nas-docker/emsn-vocalization/data/models'  # Default for EMSN
    )
)
SAMPLE_RATE = 48000
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
FMIN = 500
FMAX = 8000
SEGMENT_DURATION = 3.0

# Vertaling naar Nederlands
VOC_TYPE_NL = {
    'song': 'zang',
    'call': 'roep',
    'alarm': 'alarm'
}

logger = logging.getLogger(__name__)


def create_cnn_model(num_classes=3):
    """Create CNN model als nn.Module."""
    torch = get_torch()
    nn = torch.nn

    class ColabVocalizationCNN(nn.Module):
        """CNN model zoals getraind in Google Colab."""

        def __init__(self, input_shape=(128, 128), num_classes=3):
            super().__init__()

            self.features = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Dropout2d(0.25),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Dropout2d(0.25),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Dropout2d(0.25),
            )

            h, w = input_shape[0] // 8, input_shape[1] // 8
            flatten_size = 128 * h * w

            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(flatten_size, 256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, num_classes)
            )

        def forward(self, x):
            x = self.features(x)
            x = self.classifier(x)
            return x

    return ColabVocalizationCNN(num_classes=num_classes)


class VocalizationClassifier:
    """
    Classifier voor vocalisatie types (song/call/alarm).

    Gebruiksvoorbeeld:
        classifier = VocalizationClassifier()
        result = classifier.classify("Merel", "/path/to/audio.wav")
        if result:
            print(f"{result['type_nl']} ({result['confidence']:.0%})")
    """

    def __init__(self, models_dir: Path = MODELS_DIR, max_cached_models: int = 5):
        self.models_dir = Path(models_dir)
        self.models_cache = {}
        self.cache_order = []  # LRU tracking
        self.max_cached_models = max_cached_models
        self.available_models = {}
        self._initialized = False

    def _init_lazy(self):
        """Lazy initialization - alleen laden als nodig."""
        if self._initialized:
            return
        self._scan_models()
        self._initialized = True

    def _scan_models(self):
        """Scan beschikbare modellen.

        Ondersteunt meerdere model formaten:
        - *_cnn_2025.pt (standaard)
        - *_cnn_2025_ultimate.pt (Colab A100 training - prioriteit)
        - *_ultimate.pt (alternatief formaat)
        """
        if not self.models_dir.exists():
            logger.warning(f"Models directory niet gevonden: {self.models_dir}")
            return

        # Eerst alle modellen scannen
        all_models = {}
        for model_file in self.models_dir.glob("*.pt"):
            name = model_file.stem

            # Bepaal prioriteit: ultimate modellen hebben voorrang
            is_ultimate = 'ultimate' in name.lower()

            # Extract species name uit verschillende formaten
            # Format: soort_cnn_2025_ultimate.pt of soort_cnn_2025.pt of soort_ultimate.pt
            species_name = re.sub(r'_cnn_\d{4}(_ultimate)?$', '', name)
            species_name = re.sub(r'_ultimate$', '', species_name)
            species_name = species_name.replace('_', ' ').title()
            key = species_name.lower()

            # Ultimate modellen overschrijven altijd standaard modellen
            if key not in all_models or is_ultimate:
                all_models[key] = (model_file, is_ultimate)

        # Sla alleen de paths op
        for key, (model_file, is_ultimate) in all_models.items():
            self.available_models[key] = model_file

        # Tel ultimate vs standaard
        ultimate_count = sum(1 for _, is_ult in all_models.values() if is_ult)
        standard_count = len(all_models) - ultimate_count

        logger.info(f"Vocalization classifier: {len(self.available_models)} modellen "
                   f"({ultimate_count} ultimate, {standard_count} standaard)")

    def _normalize_name(self, name: str) -> str:
        """Normaliseer soortnaam."""
        normalized = name.lower().strip()
        normalized = re.sub(r'[^\w\s]', '', normalized)
        normalized = re.sub(r'\s+', ' ', normalized)
        return normalized

    def _find_model(self, species_name: str) -> Path | None:
        """Vind model voor soort."""
        self._init_lazy()
        normalized = self._normalize_name(species_name)

        if normalized in self.available_models:
            return self.available_models[normalized]

        for key in self.available_models:
            if key in normalized or normalized in key:
                return self.available_models[key]

        return None

    def _load_model(self, model_path: Path):
        """Laad model met LRU caching. Returns (model, metadata)."""
        path_str = str(model_path)

        # Cache hit - verplaats naar einde (recent used)
        if path_str in self.models_cache:
            if path_str in self.cache_order:
                self.cache_order.remove(path_str)
            self.cache_order.append(path_str)
            return self.models_cache[path_str]

        try:
            torch = get_torch()
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

            num_classes = checkpoint.get('num_classes', 3)
            model = create_cnn_model(num_classes=num_classes)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()

            # Haal class_names uit checkpoint (voor modellen met <3 klassen)
            class_names = checkpoint.get('class_names', ['song', 'call', 'alarm'])

            # LRU cache cleanup - verwijder oudste model als cache vol is
            while len(self.models_cache) >= self.max_cached_models and self.cache_order:
                oldest = self.cache_order.pop(0)
                if oldest in self.models_cache:
                    del self.models_cache[oldest]
                    logger.debug(f"Model uit cache verwijderd: {Path(oldest).name}")

            self.models_cache[path_str] = (model, class_names)
            self.cache_order.append(path_str)
            return (model, class_names)

        except Exception as e:
            logger.error(f"Fout bij laden model {model_path}: {e}")
            return None

    def _audio_to_spectrogram(self, audio_path: Path) -> np.ndarray | None:
        """Converteer audio naar spectrogram."""
        try:
            librosa = get_librosa()
            audio, sr = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True)

            segment_samples = int(SEGMENT_DURATION * SAMPLE_RATE)
            if len(audio) < segment_samples:
                padded = np.zeros(segment_samples)
                padded[:len(audio)] = audio
                audio = padded
            else:
                audio = audio[:segment_samples]

            mel_spec = librosa.feature.melspectrogram(
                y=audio, sr=SAMPLE_RATE, n_mels=N_MELS,
                n_fft=N_FFT, hop_length=HOP_LENGTH,
                fmin=FMIN, fmax=FMAX
            )

            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)

            return mel_spec_norm

        except Exception as e:
            logger.error(f"Audio verwerking fout: {e}")
            return None

    def has_model(self, species_name: str) -> bool:
        """Check of er een model is voor deze soort."""
        self._init_lazy()
        return self._find_model(species_name) is not None

    def classify(self, species_name: str, audio_path: str | Path) -> dict | None:
        """
        Classificeer vocalisatie type.

        Args:
            species_name: Nederlandse soortnaam (bijv. "Merel")
            audio_path: Pad naar audiobestand

        Returns:
            Dict met type, type_nl, confidence, of None als niet mogelijk
        """
        model_path = self._find_model(species_name)
        if not model_path:
            return None

        audio_path = Path(audio_path)
        if not audio_path.exists():
            return None

        result = self._load_model(model_path)
        if result is None:
            return None

        model, class_names = result

        spectrogram = self._audio_to_spectrogram(audio_path)
        if spectrogram is None:
            return None

        try:
            torch = get_torch()

            # Resize naar 128x128
            if spectrogram.shape != (128, 128):
                from skimage.transform import resize
                spectrogram = resize(spectrogram, (128, 128), anti_aliasing=True)

            # Naar tensor
            x = torch.FloatTensor(spectrogram).unsqueeze(0).unsqueeze(0)

            with torch.no_grad():
                outputs = model(x)
                probas = torch.softmax(outputs, dim=1)[0]

            # Gebruik class_names uit model (ondersteunt 2 of 3 klassen)
            class_idx = probas.argmax().item()
            confidence = probas[class_idx].item()
            voc_type = class_names[class_idx]

            return {
                'type': voc_type,
                'type_nl': VOC_TYPE_NL.get(voc_type, voc_type),
                'confidence': confidence,
                'model': model_path.name,
                'probabilities': {
                    name: probas[i].item()
                    for i, name in enumerate(class_names)
                }
            }

        except Exception as e:
            logger.error(f"Classificatie fout: {e}")
            return None


# Singleton instance voor hergebruik
_classifier_instance = None


def get_classifier() -> VocalizationClassifier:
    """Get singleton classifier instance."""
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = VocalizationClassifier()
    return _classifier_instance


def classify_vocalization(species_name: str, audio_path: str | Path) -> dict | None:
    """
    Convenience functie voor snelle classificatie.

    Args:
        species_name: Nederlandse soortnaam
        audio_path: Pad naar audiobestand

    Returns:
        Dict met type_nl en confidence, of None
    """
    return get_classifier().classify(species_name, audio_path)
