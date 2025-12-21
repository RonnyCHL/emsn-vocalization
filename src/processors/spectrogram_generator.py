#!/usr/bin/env python3
"""
EMSN 2.0 - Spectrogram Generator

Genereert mel-spectrogrammen van audio bestanden voor CNN training.
"""

import logging
from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SpectrogramGenerator:
    """Genereer mel-spectrogrammen voor CNN training."""

    def __init__(
        self,
        sample_rate: int = 48000,
        n_mels: int = 128,
        n_fft: int = 2048,
        hop_length: int = 512,
        fmin: float = 500,
        fmax: float = 8000
    ):
        """
        Initialiseer de generator.

        Args:
            sample_rate: Sample rate voor audio
            n_mels: Aantal mel filterbanks
            n_fft: FFT window size
            hop_length: Hop length voor STFT
            fmin: Minimum frequentie (Hz)
            fmax: Maximum frequentie (Hz)
        """
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.fmin = fmin
        self.fmax = fmax

    def audio_to_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """
        Converteer audio naar mel-spectrogram.

        Args:
            audio: Audio samples

        Returns:
            Mel-spectrogram (n_mels, time_frames)
        """
        # Bereken mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            fmin=self.fmin,
            fmax=self.fmax
        )

        # Convert naar dB schaal
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        # Normaliseer naar [0, 1]
        mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)

        return mel_spec_norm

    def process_audio_file(
        self,
        audio_path: Path,
        segment_duration: float = 3.0,
        overlap: float = 0.5
    ) -> list[tuple[np.ndarray, str, int]]:
        """
        Verwerk een audio bestand naar spectrogrammen.

        Args:
            audio_path: Pad naar audio bestand
            segment_duration: Duur per segment in seconden
            overlap: Overlap fractie

        Returns:
            Lijst van (spectrogram, vocalization_type, segment_idx)
        """
        results = []

        try:
            # Bepaal type uit directory naam
            voc_type = audio_path.parent.name
            if voc_type not in ['song', 'call', 'alarm']:
                return []

            # Laad audio
            audio, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)

            # Segment parameters
            hop_samples = int(segment_duration * (1 - overlap) * sr)
            segment_samples = int(segment_duration * sr)

            n_segments = max(1, (len(audio) - segment_samples) // hop_samples + 1)

            for i in range(n_segments):
                start = i * hop_samples
                end = start + segment_samples

                if end > len(audio):
                    # Pad laatste segment
                    segment = np.zeros(segment_samples)
                    segment[:len(audio) - start] = audio[start:]
                else:
                    segment = audio[start:end]

                # Genereer spectrogram
                spec = self.audio_to_spectrogram(segment)
                results.append((spec, voc_type, i))

        except Exception as e:
            logger.error(f"Fout bij {audio_path}: {e}")

        return results

    def process_directory(
        self,
        audio_dir: str,
        output_dir: str,
        segment_duration: float = 3.0,
        overlap: float = 0.5,
        save_png: bool = False
    ) -> dict:
        """
        Verwerk alle audio bestanden en sla spectrogrammen op.

        Args:
            audio_dir: Input directory met audio
            output_dir: Output directory voor spectrogrammen
            segment_duration: Segment duur
            overlap: Overlap fractie
            save_png: Sla ook PNG visualisaties op

        Returns:
            Dict met statistieken
        """
        audio_dir = Path(audio_dir)
        output_dir = Path(output_dir)

        # Maak output directories
        for voc_type in ['song', 'call', 'alarm']:
            (output_dir / voc_type).mkdir(parents=True, exist_ok=True)

        # Zoek audio bestanden
        audio_files = list(audio_dir.glob('**/*.mp3')) + list(audio_dir.glob('**/*.wav'))
        valid_files = [f for f in audio_files if f.parent.name in ['song', 'call', 'alarm']]

        logger.info(f"Gevonden audio bestanden: {len(valid_files)}")

        stats = {'song': 0, 'call': 0, 'alarm': 0, 'total': 0}
        all_specs = []
        all_labels = []

        for audio_path in tqdm(valid_files, desc="Generating spectrograms"):
            results = self.process_audio_file(audio_path, segment_duration, overlap)

            for spec, voc_type, seg_idx in results:
                # Sla spectrogram op als .npy
                filename = f"{audio_path.stem}_seg{seg_idx:03d}"
                npy_path = output_dir / voc_type / f"{filename}.npy"
                np.save(npy_path, spec)

                # Optioneel: sla PNG op
                if save_png:
                    png_path = output_dir / voc_type / f"{filename}.png"
                    self._save_spectrogram_image(spec, png_path, voc_type)

                all_specs.append(spec)
                all_labels.append(voc_type)
                stats[voc_type] += 1
                stats['total'] += 1

        # Sla ook gecombineerde dataset op voor training
        X = np.array(all_specs)
        y = np.array(all_labels)

        np.save(output_dir / 'X_spectrograms.npy', X)
        np.save(output_dir / 'y_labels.npy', y)

        logger.info(f"Dataset shape: X={X.shape}, y={y.shape}")
        logger.info(f"Verdeling: {stats}")

        return stats

    def _save_spectrogram_image(self, spec: np.ndarray, output_path: Path, title: str = ""):
        """Sla spectrogram op als PNG."""
        fig, ax = plt.subplots(figsize=(4, 3))
        librosa.display.specshow(
            spec,
            sr=self.sample_rate,
            hop_length=self.hop_length,
            x_axis='time',
            y_axis='mel',
            fmin=self.fmin,
            fmax=self.fmax,
            ax=ax
        )
        ax.set_title(title)
        plt.tight_layout()
        plt.savefig(output_path, dpi=100)
        plt.close()


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Genereer spectrogrammen')
    parser.add_argument('--input-dir', default='data/raw/xeno-canto',
                       help='Input directory met audio')
    parser.add_argument('--output-dir', default='data/spectrograms',
                       help='Output directory voor spectrogrammen')
    parser.add_argument('--segment-duration', type=float, default=3.0,
                       help='Segment duur in seconden')
    parser.add_argument('--overlap', type=float, default=0.5,
                       help='Overlap tussen segmenten')
    parser.add_argument('--save-png', action='store_true',
                       help='Sla ook PNG visualisaties op')
    parser.add_argument('--n-mels', type=int, default=128,
                       help='Aantal mel filterbanks')

    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("EMSN 2.0 - Spectrogram Generator")
    print(f"{'='*60}")
    print(f"Input: {args.input_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Segment duur: {args.segment_duration}s")
    print(f"Overlap: {args.overlap}")
    print(f"Mel bins: {args.n_mels}")
    print(f"{'='*60}\n")

    generator = SpectrogramGenerator(n_mels=args.n_mels)

    stats = generator.process_directory(
        args.input_dir,
        args.output_dir,
        segment_duration=args.segment_duration,
        overlap=args.overlap,
        save_png=args.save_png
    )

    print(f"\n{'='*60}")
    print("RESULTAAT")
    print(f"{'='*60}")
    print(f"Totaal spectrogrammen: {stats['total']}")
    print(f"  - Song: {stats['song']}")
    print(f"  - Call: {stats['call']}")
    print(f"  - Alarm: {stats['alarm']}")
    print(f"Output: {args.output_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
