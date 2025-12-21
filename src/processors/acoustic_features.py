#!/usr/bin/env python3
"""
EMSN 2.0 - Akoestische Feature Extractor

Extraheert akoestische features die onderscheid maken tussen
vocalisatietypes (song/call/alarm):
- Temporele features (duur, zero crossings, energie)
- Spectrale features (centroid, bandwidth, rolloff, flatness)
- Ritme features (onsets, tempo)
- MFCC statistieken
"""

import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class AcousticFeatures:
    """Container voor akoestische features van een audio segment."""

    # Metadata
    file_path: str
    segment_idx: int
    vocalization_type: str

    # Temporele features
    duration: float = 0.0
    zero_crossing_rate_mean: float = 0.0
    zero_crossing_rate_std: float = 0.0
    rms_energy_mean: float = 0.0
    rms_energy_std: float = 0.0
    rms_energy_max: float = 0.0

    # Spectrale features
    spectral_centroid_mean: float = 0.0
    spectral_centroid_std: float = 0.0
    spectral_bandwidth_mean: float = 0.0
    spectral_bandwidth_std: float = 0.0
    spectral_rolloff_mean: float = 0.0
    spectral_rolloff_std: float = 0.0
    spectral_flatness_mean: float = 0.0
    spectral_flatness_std: float = 0.0
    spectral_contrast_mean: float = 0.0
    spectral_contrast_std: float = 0.0

    # Ritme features
    onset_count: int = 0
    onset_rate: float = 0.0  # onsets per seconde
    tempo: float = 0.0
    beat_strength_mean: float = 0.0
    beat_strength_std: float = 0.0

    # MFCC features (13 coefficients x 4 stats = 52 features)
    mfcc_features: dict = field(default_factory=dict)

    # Frequentie features
    fundamental_freq_mean: float = 0.0
    fundamental_freq_std: float = 0.0
    freq_range: float = 0.0  # max - min frequentie

    # Harmoniciteit
    harmonic_ratio_mean: float = 0.0
    percussive_ratio_mean: float = 0.0

    def to_dict(self) -> dict:
        """Converteer naar dictionary voor DataFrame."""
        result = {
            'file_path': self.file_path,
            'segment_idx': self.segment_idx,
            'vocalization_type': self.vocalization_type,
            'duration': self.duration,
            'zcr_mean': self.zero_crossing_rate_mean,
            'zcr_std': self.zero_crossing_rate_std,
            'rms_mean': self.rms_energy_mean,
            'rms_std': self.rms_energy_std,
            'rms_max': self.rms_energy_max,
            'centroid_mean': self.spectral_centroid_mean,
            'centroid_std': self.spectral_centroid_std,
            'bandwidth_mean': self.spectral_bandwidth_mean,
            'bandwidth_std': self.spectral_bandwidth_std,
            'rolloff_mean': self.spectral_rolloff_mean,
            'rolloff_std': self.spectral_rolloff_std,
            'flatness_mean': self.spectral_flatness_mean,
            'flatness_std': self.spectral_flatness_std,
            'contrast_mean': self.spectral_contrast_mean,
            'contrast_std': self.spectral_contrast_std,
            'onset_count': self.onset_count,
            'onset_rate': self.onset_rate,
            'tempo': self.tempo,
            'beat_strength_mean': self.beat_strength_mean,
            'beat_strength_std': self.beat_strength_std,
            'f0_mean': self.fundamental_freq_mean,
            'f0_std': self.fundamental_freq_std,
            'freq_range': self.freq_range,
            'harmonic_ratio': self.harmonic_ratio_mean,
            'percussive_ratio': self.percussive_ratio_mean,
        }

        # Voeg MFCC features toe
        result.update(self.mfcc_features)

        return result


class AcousticFeatureExtractor:
    """
    Extraheert akoestische features voor vocalisatie classificatie.

    Focust op features die onderscheid kunnen maken tussen:
    - Song: Langere duur, meer melodie, lagere onset rate
    - Call: Korter, vaak herhaald, specifieke frequentie
    - Alarm: Scherp, hoge energie, snelle onsets
    """

    def __init__(
        self,
        sample_rate: int = 48000,
        segment_duration: float = 3.0,
        n_mfcc: int = 13,
        hop_length: int = 512
    ):
        """
        Initialiseer de extractor.

        Args:
            sample_rate: Sample rate voor audio processing
            segment_duration: Duur van segmenten in seconden
            n_mfcc: Aantal MFCC coefficients
            hop_length: Hop length voor spectrale analyse
        """
        self.sample_rate = sample_rate
        self.segment_duration = segment_duration
        self.n_mfcc = n_mfcc
        self.hop_length = hop_length
        self.segment_samples = int(segment_duration * sample_rate)

    def extract_from_audio(
        self,
        audio: np.ndarray,
        file_path: str,
        segment_idx: int,
        vocalization_type: str
    ) -> AcousticFeatures:
        """
        Extraheer alle akoestische features van een audio segment.

        Args:
            audio: Audio samples (mono)
            file_path: Pad naar origineel bestand
            segment_idx: Index van segment
            vocalization_type: Label (song/call/alarm)

        Returns:
            AcousticFeatures object
        """
        features = AcousticFeatures(
            file_path=file_path,
            segment_idx=segment_idx,
            vocalization_type=vocalization_type,
            duration=len(audio) / self.sample_rate
        )

        # Bereken alle features
        self._extract_temporal(audio, features)
        self._extract_spectral(audio, features)
        self._extract_rhythm(audio, features)
        self._extract_mfcc(audio, features)
        self._extract_frequency(audio, features)
        self._extract_harmonic(audio, features)

        return features

    def _extract_temporal(self, audio: np.ndarray, features: AcousticFeatures):
        """Extraheer temporele features."""
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(audio, hop_length=self.hop_length)[0]
        features.zero_crossing_rate_mean = float(np.mean(zcr))
        features.zero_crossing_rate_std = float(np.std(zcr))

        # RMS energy
        rms = librosa.feature.rms(y=audio, hop_length=self.hop_length)[0]
        features.rms_energy_mean = float(np.mean(rms))
        features.rms_energy_std = float(np.std(rms))
        features.rms_energy_max = float(np.max(rms))

    def _extract_spectral(self, audio: np.ndarray, features: AcousticFeatures):
        """Extraheer spectrale features."""
        # Spectral centroid (zwaartepunt van frequentie)
        centroid = librosa.feature.spectral_centroid(
            y=audio, sr=self.sample_rate, hop_length=self.hop_length
        )[0]
        features.spectral_centroid_mean = float(np.mean(centroid))
        features.spectral_centroid_std = float(np.std(centroid))

        # Spectral bandwidth (spreiding rond centroid)
        bandwidth = librosa.feature.spectral_bandwidth(
            y=audio, sr=self.sample_rate, hop_length=self.hop_length
        )[0]
        features.spectral_bandwidth_mean = float(np.mean(bandwidth))
        features.spectral_bandwidth_std = float(np.std(bandwidth))

        # Spectral rolloff (frequentie onder 85% van energie)
        rolloff = librosa.feature.spectral_rolloff(
            y=audio, sr=self.sample_rate, hop_length=self.hop_length
        )[0]
        features.spectral_rolloff_mean = float(np.mean(rolloff))
        features.spectral_rolloff_std = float(np.std(rolloff))

        # Spectral flatness (toontje vs ruis)
        flatness = librosa.feature.spectral_flatness(y=audio, hop_length=self.hop_length)[0]
        features.spectral_flatness_mean = float(np.mean(flatness))
        features.spectral_flatness_std = float(np.std(flatness))

        # Spectral contrast
        contrast = librosa.feature.spectral_contrast(
            y=audio, sr=self.sample_rate, hop_length=self.hop_length
        )
        features.spectral_contrast_mean = float(np.mean(contrast))
        features.spectral_contrast_std = float(np.std(contrast))

    def _extract_rhythm(self, audio: np.ndarray, features: AcousticFeatures):
        """Extraheer ritme features."""
        # Onset detection
        onset_env = librosa.onset.onset_strength(
            y=audio, sr=self.sample_rate, hop_length=self.hop_length
        )
        onsets = librosa.onset.onset_detect(
            onset_envelope=onset_env, sr=self.sample_rate, hop_length=self.hop_length
        )

        features.onset_count = len(onsets)
        features.onset_rate = len(onsets) / features.duration if features.duration > 0 else 0

        # Tempo en beat strength
        try:
            tempo, beats = librosa.beat.beat_track(
                y=audio, sr=self.sample_rate, hop_length=self.hop_length
            )
            # tempo kan een array zijn in nieuwere librosa versies
            features.tempo = float(tempo[0]) if hasattr(tempo, '__len__') else float(tempo)

            if len(beats) > 0:
                beat_strengths = onset_env[beats]
                features.beat_strength_mean = float(np.mean(beat_strengths))
                features.beat_strength_std = float(np.std(beat_strengths))
        except Exception:
            features.tempo = 0.0
            features.beat_strength_mean = 0.0
            features.beat_strength_std = 0.0

    def _extract_mfcc(self, audio: np.ndarray, features: AcousticFeatures):
        """Extraheer MFCC statistieken."""
        mfccs = librosa.feature.mfcc(
            y=audio, sr=self.sample_rate, n_mfcc=self.n_mfcc, hop_length=self.hop_length
        )

        mfcc_dict = {}
        for i in range(self.n_mfcc):
            mfcc_dict[f'mfcc_{i}_mean'] = float(np.mean(mfccs[i]))
            mfcc_dict[f'mfcc_{i}_std'] = float(np.std(mfccs[i]))
            mfcc_dict[f'mfcc_{i}_min'] = float(np.min(mfccs[i]))
            mfcc_dict[f'mfcc_{i}_max'] = float(np.max(mfccs[i]))

        # Delta MFCCs (snelheid van verandering)
        mfcc_delta = librosa.feature.delta(mfccs)
        for i in range(self.n_mfcc):
            mfcc_dict[f'mfcc_delta_{i}_mean'] = float(np.mean(mfcc_delta[i]))
            mfcc_dict[f'mfcc_delta_{i}_std'] = float(np.std(mfcc_delta[i]))

        features.mfcc_features = mfcc_dict

    def _extract_frequency(self, audio: np.ndarray, features: AcousticFeatures):
        """Extraheer frequentie features met snelle methode."""
        try:
            # Snellere methode: gebruik spectral centroid als proxy voor pitch
            # en spectral bandwidth voor frequentie range
            # pyin is te traag voor grote datasets

            # Dominant frequency via FFT
            fft = np.abs(np.fft.rfft(audio))
            freqs = np.fft.rfftfreq(len(audio), 1/self.sample_rate)

            # Filter naar vogel frequentie range (500-8000 Hz)
            bird_mask = (freqs >= 500) & (freqs <= 8000)
            if np.any(bird_mask):
                fft_bird = fft[bird_mask]
                freqs_bird = freqs[bird_mask]

                # Gewogen gemiddelde frequentie
                if np.sum(fft_bird) > 0:
                    weighted_freq = np.average(freqs_bird, weights=fft_bird)
                    features.fundamental_freq_mean = float(weighted_freq)

                    # Frequentie spreiding (gewogen std)
                    variance = np.average((freqs_bird - weighted_freq)**2, weights=fft_bird)
                    features.fundamental_freq_std = float(np.sqrt(variance))

                    # Range: 10e en 90e percentiel
                    cumsum = np.cumsum(fft_bird)
                    cumsum_norm = cumsum / cumsum[-1]
                    idx_10 = np.searchsorted(cumsum_norm, 0.1)
                    idx_90 = np.searchsorted(cumsum_norm, 0.9)
                    features.freq_range = float(freqs_bird[min(idx_90, len(freqs_bird)-1)] -
                                                freqs_bird[min(idx_10, len(freqs_bird)-1)])
        except Exception:
            features.fundamental_freq_mean = 0.0
            features.fundamental_freq_std = 0.0
            features.freq_range = 0.0

    def _extract_harmonic(self, audio: np.ndarray, features: AcousticFeatures):
        """Extraheer harmonische vs percussieve ratio."""
        try:
            harmonic, percussive = librosa.effects.hpss(audio)

            harmonic_energy = np.sum(harmonic ** 2)
            percussive_energy = np.sum(percussive ** 2)
            total_energy = harmonic_energy + percussive_energy

            if total_energy > 0:
                features.harmonic_ratio_mean = float(harmonic_energy / total_energy)
                features.percussive_ratio_mean = float(percussive_energy / total_energy)
            else:
                features.harmonic_ratio_mean = 0.5
                features.percussive_ratio_mean = 0.5
        except Exception:
            features.harmonic_ratio_mean = 0.5
            features.percussive_ratio_mean = 0.5

    def process_single_file(
        self,
        audio_path: Path,
        segment_duration: float = 3.0,
        overlap: float = 0.5
    ) -> list[dict]:
        """Verwerk een enkel audio bestand (voor parallelle processing)."""
        results = []

        try:
            # Bepaal vocalization type uit directory structuur
            voc_type = audio_path.parent.name
            if voc_type not in ['song', 'call', 'alarm']:
                return []

            # Laad audio
            audio, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)

            # Segment audio
            hop_samples = int(segment_duration * (1 - overlap) * sr)
            segment_samples = int(segment_duration * sr)

            n_segments = max(1, (len(audio) - segment_samples) // hop_samples + 1)

            for i in range(n_segments):
                start = i * hop_samples
                end = start + segment_samples

                if end > len(audio):
                    segment = np.zeros(segment_samples)
                    segment[:len(audio) - start] = audio[start:]
                else:
                    segment = audio[start:end]

                features = self.extract_from_audio(
                    segment, str(audio_path), i, voc_type
                )
                results.append(features.to_dict())

        except Exception as e:
            logger.error(f"Fout bij verwerken {audio_path}: {e}")

        return results

    def process_directory(
        self,
        audio_dir: str,
        output_csv: str,
        segment_duration: float = 3.0,
        overlap: float = 0.5,
        n_workers: int = 4
    ) -> pd.DataFrame:
        """
        Verwerk alle audio bestanden in een directory (parallel).

        Args:
            audio_dir: Directory met audio bestanden
            output_csv: Pad voor output CSV
            segment_duration: Duur van segmenten
            overlap: Overlap fractie tussen segmenten
            n_workers: Aantal parallelle workers

        Returns:
            DataFrame met alle features
        """
        audio_dir = Path(audio_dir)
        all_features = []

        # Zoek audio bestanden
        audio_files = list(audio_dir.glob('**/*.mp3')) + list(audio_dir.glob('**/*.wav'))
        logger.info(f"Gevonden audio bestanden: {len(audio_files)}")

        # Filter op geldige types
        valid_files = [f for f in audio_files if f.parent.name in ['song', 'call', 'alarm']]
        logger.info(f"Geldige bestanden: {len(valid_files)}")

        # Sequentieel maar snel door geoptimaliseerde features
        for audio_path in tqdm(valid_files, desc="Extracting features"):
            results = self.process_single_file(audio_path, segment_duration, overlap)
            all_features.extend(results)

        # Maak DataFrame
        if not all_features:
            logger.warning("Geen features geëxtraheerd!")
            return pd.DataFrame()

        df = pd.DataFrame(all_features)

        # Sla op
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)

        logger.info(f"Features opgeslagen: {output_csv}")
        logger.info(f"Totaal: {len(df)} segmenten, {len(df.columns)} features")

        # Print verdeling
        logger.info(f"Verdeling: {df['vocalization_type'].value_counts().to_dict()}")

        return df


def main():
    """Extract akoestische features van Merel dataset."""
    import argparse

    parser = argparse.ArgumentParser(description='Extract akoestische features')
    parser.add_argument('--input-dir', default='data/raw/xeno-canto/Turdus_merula',
                       help='Input directory met audio')
    parser.add_argument('--output', default='data/embeddings/merel_acoustic_features.csv',
                       help='Output CSV bestand')
    parser.add_argument('--segment-duration', type=float, default=3.0,
                       help='Segment duur in seconden')
    parser.add_argument('--overlap', type=float, default=0.5,
                       help='Overlap tussen segmenten')
    parser.add_argument('--sample-rate', type=int, default=48000,
                       help='Sample rate')

    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("EMSN 2.0 - Akoestische Feature Extractie")
    print(f"{'='*60}")
    print(f"Input: {args.input_dir}")
    print(f"Output: {args.output}")
    print(f"Segment duur: {args.segment_duration}s")
    print(f"Overlap: {args.overlap}")
    print(f"{'='*60}\n")

    extractor = AcousticFeatureExtractor(
        sample_rate=args.sample_rate,
        segment_duration=args.segment_duration
    )

    df = extractor.process_directory(
        args.input_dir,
        args.output,
        segment_duration=args.segment_duration,
        overlap=args.overlap
    )

    print(f"\n{'='*60}")
    print("RESULTAAT")
    print(f"{'='*60}")
    print(f"Segmenten: {len(df)}")
    print(f"Features: {len(df.columns) - 3}")  # minus metadata kolommen
    print(f"Output: {args.output}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
