#!/usr/bin/env python3
"""
EMSN 2.0 Vocalisatie Classifier - BirdNET Feature Extractor

Extraheert features uit audio bestanden met BirdNET.
Gebruikt de classificatie output (6522 logits) als feature vector.

Output: CSV met features per 3-seconde segment.
"""

import csv
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
import tensorflow as tf
from tqdm import tqdm

# Suppress TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class FeatureSegment:
    """Een feature segment met metadata."""
    filename: str
    segment_start: float
    segment_end: float
    vocalization_type: str
    features: np.ndarray


class BirdNETFeatureExtractor:
    """
    Extraheert features uit audio met BirdNET model.

    Het BirdNET model verwacht:
    - Sample rate: 48000 Hz
    - Segment lengte: 3 seconden
    - Input shape: (batch, 144000) = 3s * 48000Hz

    Output: 6522 classificatie logits die als features dienen.
    """

    SAMPLE_RATE = 48000
    SEGMENT_DURATION = 3.0  # seconden
    SEGMENT_SAMPLES = int(SAMPLE_RATE * SEGMENT_DURATION)  # 144000
    FEATURE_DIM = 6522  # BirdNET output logits

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialiseer de extractor.

        Args:
            model_path: Pad naar BirdNET TFLite model.
        """
        if model_path is None:
            import birdnetlib
            model_path = os.path.join(
                os.path.dirname(birdnetlib.__file__),
                'models', 'analyzer',
                'BirdNET_GLOBAL_6K_V2.4_Model_FP32.tflite'
            )

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"BirdNET model niet gevonden: {model_path}")

        logger.info(f"Laden BirdNET model: {os.path.basename(model_path)}")

        # Laad TFLite model
        self.interpreter = tf.lite.Interpreter(model_path=model_path, num_threads=4)

        # Haal input/output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.input_index = self.input_details[0]['index']
        self.output_index = self.output_details[0]['index']

        logger.info(f"Model geladen. Output dimensie: {self.output_details[0]['shape'][-1]}")

    def load_audio(self, filepath: str) -> tuple[np.ndarray, float]:
        """Laad audio bestand en resample naar 48kHz."""
        try:
            audio, sr = librosa.load(filepath, sr=self.SAMPLE_RATE, mono=True)
            duration = len(audio) / self.SAMPLE_RATE
            return audio, duration
        except Exception as e:
            logger.error(f"Kon audio niet laden: {filepath} - {e}")
            raise

    def segment_audio(
        self,
        audio: np.ndarray,
        overlap: float = 0.0
    ) -> list[tuple[np.ndarray, float, float]]:
        """Verdeel audio in 3-seconde segmenten."""
        segments = []
        step = int(self.SEGMENT_SAMPLES * (1 - overlap))

        for start_idx in range(0, len(audio) - self.SEGMENT_SAMPLES + 1, step):
            end_idx = start_idx + self.SEGMENT_SAMPLES
            segment = audio[start_idx:end_idx]

            start_time = start_idx / self.SAMPLE_RATE
            end_time = end_idx / self.SAMPLE_RATE

            segments.append((segment, start_time, end_time))

        # Als audio korter is dan 3 seconden, pad naar 3 seconden
        if len(segments) == 0 and len(audio) > 0:
            segment = np.pad(audio, (0, max(0, self.SEGMENT_SAMPLES - len(audio))))
            segments.append((segment[:self.SEGMENT_SAMPLES], 0.0, len(audio) / self.SAMPLE_RATE))

        return segments

    def extract_features(self, audio_segment: np.ndarray) -> np.ndarray:
        """
        Extraheer features voor een enkel 3-seconde segment.

        Args:
            audio_segment: Audio array van exact 144000 samples

        Returns:
            Feature vector (6522-dim classificatie logits)
        """
        audio_segment = audio_segment.astype(np.float32)

        if len(audio_segment) != self.SEGMENT_SAMPLES:
            raise ValueError(
                f"Audio segment moet {self.SEGMENT_SAMPLES} samples zijn, "
                f"kreeg {len(audio_segment)}"
            )

        # Input data: (1, 144000)
        data = audio_segment.reshape(1, -1)

        # Resize en allocate (nodig voor dynamic batch size)
        self.interpreter.resize_tensor_input(self.input_index, [1, self.SEGMENT_SAMPLES])
        self.interpreter.allocate_tensors()

        # Set input tensor
        self.interpreter.set_tensor(self.input_index, data)

        # Run inference
        self.interpreter.invoke()

        # Haal output (classificatie logits)
        features = self.interpreter.get_tensor(self.output_index)

        return features.flatten()

    def extract_features_from_file(
        self,
        filepath: str,
        vocalization_type: str,
        overlap: float = 0.0
    ) -> list[FeatureSegment]:
        """
        Extraheer alle features uit een audio bestand.

        Args:
            filepath: Pad naar audio bestand
            vocalization_type: Label (song/call/alarm)
            overlap: Overlap tussen segmenten

        Returns:
            Lijst van FeatureSegment objecten
        """
        filename = os.path.basename(filepath)

        try:
            # Laad audio
            audio, duration = self.load_audio(filepath)

            if duration < 0.5:
                logger.warning(f"Audio te kort ({duration:.1f}s): {filename}")
                return []

            # Segment audio
            segments = self.segment_audio(audio, overlap)

            if not segments:
                logger.warning(f"Geen segmenten voor: {filename}")
                return []

            # Extraheer features
            results = []
            for audio_seg, start_time, end_time in segments:
                try:
                    features = self.extract_features(audio_seg)

                    results.append(FeatureSegment(
                        filename=filename,
                        segment_start=start_time,
                        segment_end=end_time,
                        vocalization_type=vocalization_type,
                        features=features
                    ))
                except Exception as e:
                    logger.warning(
                        f"Feature extractie mislukt voor {filename} "
                        f"segment {start_time:.1f}-{end_time:.1f}s: {e}"
                    )

            return results

        except Exception as e:
            logger.error(f"Kon bestand niet verwerken: {filename} - {e}")
            return []

    def process_directory(
        self,
        input_dir: str,
        output_file: str,
        vocalization_type: Optional[str] = None,
        overlap: float = 0.0
    ) -> int:
        """Verwerk alle audio bestanden in een directory."""
        input_path = Path(input_dir)

        audio_files = list(input_path.glob('*.mp3')) + \
                     list(input_path.glob('*.wav')) + \
                     list(input_path.glob('*.flac'))

        if not audio_files:
            logger.warning(f"Geen audio bestanden gevonden in: {input_dir}")
            return 0

        logger.info(f"Verwerken van {len(audio_files)} bestanden uit {input_dir}")

        if vocalization_type is None:
            vocalization_type = input_path.name

        all_features = []
        for audio_file in tqdm(audio_files, desc=f"Processing {vocalization_type}"):
            features = self.extract_features_from_file(
                str(audio_file),
                vocalization_type,
                overlap
            )
            all_features.extend(features)

        if all_features:
            self._write_csv(all_features, output_file, append=True)

        return len(all_features)

    def process_dataset(
        self,
        base_dir: str,
        output_file: str,
        vocalization_types: list[str] = ['song', 'call', 'alarm'],
        overlap: float = 0.0
    ) -> dict[str, int]:
        """Verwerk een complete dataset met meerdere vocalisatie types."""
        base_path = Path(base_dir)
        output_path = Path(output_file)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.exists():
            output_path.unlink()

        results = {}
        total = 0

        for voc_type in vocalization_types:
            type_dir = base_path / voc_type

            if not type_dir.exists():
                logger.warning(f"Directory niet gevonden: {type_dir}")
                results[voc_type] = 0
                continue

            count = self.process_directory(
                str(type_dir),
                output_file,
                voc_type,
                overlap
            )

            results[voc_type] = count
            total += count
            logger.info(f"  {voc_type}: {count} features")

        logger.info(f"Totaal: {total} features opgeslagen in {output_file}")

        return results

    def _write_csv(
        self,
        features: list[FeatureSegment],
        output_file: str,
        append: bool = False
    ):
        """Schrijf features naar CSV bestand."""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        mode = 'a' if append else 'w'
        write_header = not append or not output_path.exists() or output_path.stat().st_size == 0

        # Kolom namen
        header = ['filename', 'segment_start', 'segment_end', 'vocalization_type']
        header.extend([f'feature_{i}' for i in range(self.FEATURE_DIM)])

        with open(output_file, mode, newline='') as f:
            writer = csv.writer(f)

            if write_header:
                writer.writerow(header)

            for feat in features:
                row = [
                    feat.filename,
                    f"{feat.segment_start:.3f}",
                    f"{feat.segment_end:.3f}",
                    feat.vocalization_type
                ]
                row.extend(feat.features.tolist())
                writer.writerow(row)


def main():
    """Extraheer features uit Merel dataset."""
    import argparse

    parser = argparse.ArgumentParser(description='Extraheer BirdNET features uit audio')
    parser.add_argument('--input', default='data/raw/xeno-canto',
                       help='Input directory met subdirs per vocalisatie type')
    parser.add_argument('--output', default='data/embeddings/merel_features.csv',
                       help='Output CSV bestand')
    parser.add_argument('--types', nargs='+', default=['song', 'call', 'alarm'],
                       help='Vocalisatie types (= subdirectory namen)')
    parser.add_argument('--overlap', type=float, default=0.0,
                       help='Overlap tussen segmenten (0.0-1.0)')

    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("EMSN 2.0 - BirdNET Feature Extractor")
    print(f"{'='*60}")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Types: {', '.join(args.types)}")
    print(f"Overlap: {args.overlap}")
    print(f"Feature dimensie: {BirdNETFeatureExtractor.FEATURE_DIM}")
    print(f"{'='*60}\n")

    extractor = BirdNETFeatureExtractor()

    results = extractor.process_dataset(
        base_dir=args.input,
        output_file=args.output,
        vocalization_types=args.types,
        overlap=args.overlap
    )

    print(f"\n{'='*60}")
    print("SAMENVATTING")
    print(f"{'='*60}")

    total = 0
    for voc_type, count in results.items():
        print(f"  {voc_type}: {count} features")
        total += count

    print(f"{'='*60}")
    print(f"  TOTAAL: {total} features")
    print(f"  Output: {args.output}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
