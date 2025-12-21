#!/usr/bin/env python3
"""
EMSN 2.0 - Full Automated Training Pipeline
Downloads audio, generates spectrograms, and trains CNN models for ALL Dutch bird species.
Designed to run unattended for days.

Usage:
    python full_pipeline.py                    # Train all species
    python full_pipeline.py --priority 1       # Only priority 1 species
    python full_pipeline.py --resume           # Resume from last species
    python full_pipeline.py --species "Merel"  # Train specific species
"""

import os
import sys
import time
import random
import logging
import argparse
import traceback
from pathlib import Path
from datetime import datetime
import numpy as np
import psycopg2

# PyTorch threading optimalisatie voor trage CPUs
import torch
torch.set_num_threads(4)

# Voeg src toe aan path
sys.path.insert(0, '/app')

from src.collectors.xeno_canto import XenoCantoClient
from src.processors.spectrogram_generator import SpectrogramGenerator
from src.classifiers.cnn_classifier_pytorch import CNNVocalizationClassifier, plot_training_history, plot_confusion_matrix

# Import species list
from dutch_bird_species import DUTCH_BIRD_SPECIES, get_species_by_priority, get_all_species_for_training

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/app/logs/full_pipeline.log')
    ]
)
logger = logging.getLogger(__name__)

# Directories
DATA_DIR = Path('/app/data')
RAW_DIR = DATA_DIR / 'raw'
MODELS_DIR = DATA_DIR / 'models'
LOGS_DIR = Path('/app/logs')

# Ensure directories exist
for d in [DATA_DIR, RAW_DIR, MODELS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Database config
PG_HOST = os.environ.get('PG_HOST', '192.168.1.25')
PG_PORT = os.environ.get('PG_PORT', '5433')
PG_DB = os.environ.get('PG_DB', 'emsn')
PG_USER = os.environ.get('PG_USER', 'birdpi_zolder')
PG_PASS = os.environ.get('PG_PASS', 'IwnadBon2iN')

# Training parameters
SAMPLES_PER_TYPE = 150  # Audio samples per vocalization type
MAX_SPECTROGRAMS_PER_CLASS = 1000  # Limit for training
EPOCHS = 30
BATCH_SIZE = 64
PATIENCE = 7


def get_pg():
    """Get PostgreSQL connection."""
    return psycopg2.connect(
        host=PG_HOST, port=PG_PORT, database=PG_DB,
        user=PG_USER, password=PG_PASS
    )


def save_confusion_matrix(species_name, cm, class_names):
    """Save confusion matrix to database for dashboard visualization."""
    try:
        conn = get_pg()
        cur = conn.cursor()
        cur.execute("DELETE FROM vocalization_confusion_matrix WHERE species_name = %s", (species_name,))
        for i, true_label in enumerate(class_names):
            for j, pred_label in enumerate(class_names):
                count = int(cm[i][j])
                cur.execute("""
                    INSERT INTO vocalization_confusion_matrix
                    (species_name, true_label, predicted_label, count)
                    VALUES (%s, %s, %s, %s)
                """, (species_name, true_label, pred_label, count))
        conn.commit()
        cur.close()
        conn.close()
        print(f"    Confusion matrix saved to database", flush=True)
    except Exception as e:
        print(f"    Warning: Could not save confusion matrix: {e}", flush=True)

def save_xeno_canto_metadata(species_name, recordings):
    """Save Xeno-canto recording metadata to database for world map."""
    try:
        conn = get_pg()
        cur = conn.cursor()
        cur.execute("DELETE FROM xeno_canto_recordings WHERE species_name = %s", (species_name,))
        for rec in recordings[:100]:  # Max 100 per species
            cur.execute("""
                INSERT INTO xeno_canto_recordings
                (species_name, xc_id, country, latitude, longitude, vocalization_type, quality)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (
                species_name,
                rec.get('id', ''),
                rec.get('cnt', ''),
                float(rec.get('lat', 0)) if rec.get('lat') else None,
                float(rec.get('lng', 0)) if rec.get('lng') else None,
                rec.get('type', ''),
                rec.get('q', '')
            ))
        conn.commit()
        cur.close()
        conn.close()
        print(f"    Saved {min(len(recordings), 100)} Xeno-canto locations to database", flush=True)
    except Exception as e:
        print(f"    Warning: Could not save Xeno-canto metadata: {e}", flush=True)

def update_status(species_name, scientific_name, status, phase, progress, **kwargs):
    """Update training status in database."""
    try:
        conn = get_pg()
        cur = conn.cursor()

        # Check if exists
        cur.execute("SELECT id FROM vocalization_training WHERE species_name = %s", (species_name,))
        exists = cur.fetchone()

        if exists:
            sql = """UPDATE vocalization_training SET
                     status=%s, phase=%s, progress_pct=%s, updated_at=NOW()"""
            vals = [status, phase, progress]

            # Auto-set completed_at when status is completed
            if status == 'completed':
                sql += ", completed_at=NOW()"

            for k, v in kwargs.items():
                sql += f", {k}=%s"
                vals.append(v)

            sql += " WHERE species_name=%s"
            vals.append(species_name)
            cur.execute(sql, vals)
        else:
            cur.execute("""INSERT INTO vocalization_training
                (species_name, scientific_name, status, phase, progress_pct, started_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, NOW(), NOW())""",
                (species_name, scientific_name, status, phase, progress))

        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        logger.error(f"DB update error: {e}")


def check_model_exists(dirname):
    """Check if model already exists."""
    model_path = MODELS_DIR / f'{dirname}_cnn_v1.pt'
    return model_path.exists()


def check_spectrograms_exist(dirname):
    """Check if spectrograms already exist."""
    spec_dir = DATA_DIR / f'spectrograms-{dirname}'
    if not spec_dir.exists():
        return False

    # Count spectrograms
    total = 0
    for cls in ['song', 'call', 'alarm']:
        cls_dir = spec_dir / cls
        if cls_dir.exists():
            total += len(list(cls_dir.glob('*.npy')))

    return total >= 100


XENO_CANTO_API_KEY = os.environ.get('XENO_CANTO_API_KEY', '14258afd1c8a8e055387d012f2620e20f59ef3a2')

def fetch_xeno_canto_locations(species_name, scientific_name):
    """Fetch Xeno-canto recording locations for world map (API v3)."""
    try:
        import requests
        url = f"https://xeno-canto.org/api/3/recordings?query={scientific_name.replace(' ', '+')}&key={XENO_CANTO_API_KEY}"
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            data = response.json()
            recordings = data.get('recordings', [])
            if recordings:
                save_xeno_canto_metadata(species_name, recordings)
                return len(recordings)
        else:
            logger.warning(f"Xeno-canto API error: {response.status_code}")
    except Exception as e:
        logger.warning(f"Could not fetch Xeno-canto locations: {e}")
    return 0

def download_audio(species_name, scientific_name, dirname):
    """Download audio from Xeno-canto."""
    logger.info(f"Downloading audio for {species_name} ({scientific_name})")

    output_dir = RAW_DIR / f'xeno-canto-{dirname}'

    # Fetch locations for world map
    fetch_xeno_canto_locations(species_name, scientific_name)

    client = XenoCantoClient(
        download_dir=str(output_dir),
        api_key=os.environ.get('XENO_CANTO_API_KEY')
    )

    try:
        dataset = client.download_dataset(
            species=scientific_name,
            vocalization_types=['song', 'call', 'alarm'],
            quality=['A', 'B', 'C'],  # Include C quality for more data
            samples_per_type=SAMPLES_PER_TYPE
        )

        total = sum(len(files) for files in dataset.values())
        logger.info(f"Downloaded {total} audio files for {species_name}")
        return total > 0

    except Exception as e:
        logger.error(f"Download failed for {species_name}: {e}")
        return False


def generate_spectrograms(species_name, dirname):
    """Generate spectrograms from audio files."""
    logger.info(f"Generating spectrograms for {species_name}")

    audio_dir = RAW_DIR / f'xeno-canto-{dirname}'
    spec_dir = DATA_DIR / f'spectrograms-{dirname}'

    if not audio_dir.exists():
        logger.error(f"Audio directory not found: {audio_dir}")
        return False

    generator = SpectrogramGenerator(
        sample_rate=22050,
        n_mels=128,
        hop_length=512,
        n_fft=2048,
        output_size=(128, 128)
    )

    total_generated = 0

    for voc_type in ['song', 'call', 'alarm']:
        type_dir = audio_dir / voc_type
        output_type_dir = spec_dir / voc_type
        output_type_dir.mkdir(parents=True, exist_ok=True)

        if not type_dir.exists():
            logger.warning(f"No {voc_type} directory for {species_name}")
            continue

        audio_files = list(type_dir.glob('*.mp3'))
        logger.info(f"  {voc_type}: {len(audio_files)} audio files")

        for audio_file in audio_files:
            try:
                output_path = output_type_dir / f"{audio_file.stem}.npy"
                if output_path.exists():
                    total_generated += 1
                    continue

                spec = generator.generate(str(audio_file))
                if spec is not None:
                    np.save(output_path, spec)
                    total_generated += 1
            except Exception as e:
                logger.debug(f"Failed to process {audio_file}: {e}")

    logger.info(f"Generated {total_generated} spectrograms for {species_name}")
    return total_generated >= 50


def combine_spectrograms(spec_dir, max_per_class=MAX_SPECTROGRAMS_PER_CLASS):
    """Combine spectrograms into training arrays."""
    X_list, y_list = [], []

    for cls in ['song', 'call', 'alarm']:
        cls_dir = spec_dir / cls
        if not cls_dir.exists():
            continue

        files = list(cls_dir.glob('*.npy'))
        logger.info(f"  {cls}: {len(files)} spectrograms")

        if len(files) > max_per_class:
            files = random.sample(files, max_per_class)
            logger.info(f"    Sampled to {max_per_class}")

        for f in files:
            try:
                X_list.append(np.load(f))
                y_list.append(cls)
            except:
                pass

    if len(X_list) < 100:
        logger.error(f"Not enough spectrograms: {len(X_list)}")
        return False

    X = np.stack(X_list)
    y = np.array(y_list)

    np.save(spec_dir / 'X_spectrograms.npy', X)
    np.save(spec_dir / 'y_labels.npy', y)

    logger.info(f"Combined: X={X.shape}, y={y.shape}")
    return True


def train_model(species_name, scientific_name, dirname):
    """Train CNN model."""
    logger.info(f"Training model for {species_name}")

    spec_dir = DATA_DIR / f'spectrograms-{dirname}'
    model_path = MODELS_DIR / f'{dirname}_cnn_v1.pt'

    # Combine spectrograms
    if not combine_spectrograms(spec_dir):
        update_status(species_name, scientific_name, 'failed', 'Te weinig data', 0,
                     error_message='< 100 spectrograms')
        return False

    try:
        classifier = CNNVocalizationClassifier()
        X, y, class_names = classifier.load_data(str(spec_dir))

        # Progress callback
        def on_epoch_end(current_epoch, total_epochs, val_acc):
            progress = 60 + int((current_epoch / total_epochs) * 35)
            phase = f'Epoch {current_epoch}/{total_epochs}'
            update_status(species_name, scientific_name, 'training', phase, progress, accuracy=val_acc)

        # Train
        results = classifier.train(
            X, y,
            test_size=0.2,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            patience=PATIENCE,
            progress_callback=on_epoch_end
        )

        # Save model
        classifier.save(str(model_path))

        # Save plots
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        history_path = LOGS_DIR / f"{dirname}_training_history_{timestamp}.png"
        cm_path = LOGS_DIR / f"{dirname}_confusion_matrix_{timestamp}.png"

        plot_training_history(results['history'], str(history_path))
        plot_confusion_matrix(results['confusion_matrix'], class_names, str(cm_path), results['accuracy'])

        # Save confusion matrix to database
        save_confusion_matrix(species_name, results['confusion_matrix'], class_names)

        acc = results['accuracy']
        logger.info(f"Model trained successfully: accuracy={acc:.2%}")

        if model_path.exists():
            update_status(species_name, scientific_name, 'completed', 'Voltooid', 100, accuracy=acc)
            return True
        else:
            update_status(species_name, scientific_name, 'failed', 'Model niet opgeslagen', 0)
            return False

    except Exception as e:
        error_msg = str(e)[:200]
        logger.error(f"Training failed: {error_msg}")
        traceback.print_exc()
        update_status(species_name, scientific_name, 'failed', 'Training error', 0, error_message=error_msg)
        return False


def process_species(species_name, scientific_name, dirname, skip_download=False):
    """Process a single species through the full pipeline."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing: {species_name} ({scientific_name})")
    logger.info(f"{'='*60}")

    # Check if model already exists
    if check_model_exists(dirname):
        logger.info(f"Model already exists for {species_name}, skipping")
        update_status(species_name, scientific_name, 'completed', 'Model bestaat al', 100)
        return True

    # Update status
    update_status(species_name, scientific_name, 'processing', 'Starten', 0)

    # Step 1: Download audio (or skip if spectrograms exist)
    if not skip_download and not check_spectrograms_exist(dirname):
        update_status(species_name, scientific_name, 'processing', 'Audio downloaden', 10)
        if not download_audio(species_name, scientific_name, dirname):
            logger.warning(f"Download failed for {species_name}, trying with existing data")

    # Step 2: Generate spectrograms
    if not check_spectrograms_exist(dirname):
        update_status(species_name, scientific_name, 'processing', 'Spectrogrammen genereren', 30)
        if not generate_spectrograms(species_name, dirname):
            update_status(species_name, scientific_name, 'failed', 'Geen spectrogrammen', 0,
                         error_message='Kon geen spectrogrammen genereren')
            return False

    # Step 3: Train model
    update_status(species_name, scientific_name, 'training', 'CNN training', 60)
    return train_model(species_name, scientific_name, dirname)


def main():
    parser = argparse.ArgumentParser(description='EMSN Full Training Pipeline')
    parser.add_argument('--priority', type=int, choices=[1, 2, 3],
                       help='Only train species with this priority')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from last incomplete species')
    parser.add_argument('--species', type=str,
                       help='Train specific species by Dutch name')
    parser.add_argument('--skip-download', action='store_true',
                       help='Skip download phase (use existing audio)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be done without doing it')

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("EMSN 2.0 - Full Automated Training Pipeline")
    logger.info(f"Started at: {datetime.now()}")
    logger.info(f"PyTorch threads: {torch.get_num_threads()}")
    logger.info(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    logger.info("=" * 60)

    # Get species list
    if args.species:
        species_list = [(s[0], s[1], s[2]) for s in DUTCH_BIRD_SPECIES if s[0] == args.species]
        if not species_list:
            logger.error(f"Species not found: {args.species}")
            return 1
    elif args.priority:
        species_list = [(s[0], s[1], s[2]) for s in get_species_by_priority(args.priority)]
    else:
        species_list = get_all_species_for_training()

    logger.info(f"Total species to process: {len(species_list)}")

    if args.dry_run:
        logger.info("\nDry run - would process:")
        for name, sci, dirname in species_list:
            exists = "EXISTS" if check_model_exists(dirname) else "NEW"
            logger.info(f"  [{exists}] {name} ({sci})")
        return 0

    # Process each species
    success = 0
    failed = 0
    skipped = 0

    for i, (name, scientific, dirname) in enumerate(species_list):
        logger.info(f"\nProgress: {i+1}/{len(species_list)} ({100*(i+1)/len(species_list):.1f}%)")

        if check_model_exists(dirname):
            logger.info(f"Skipping {name} - model exists")
            skipped += 1
            update_status(name, scientific, 'completed', 'Model bestaat al', 100)
            continue

        try:
            if process_species(name, scientific, dirname, skip_download=args.skip_download):
                success += 1
            else:
                failed += 1
        except KeyboardInterrupt:
            logger.info("\nInterrupted by user")
            break
        except Exception as e:
            logger.error(f"Unexpected error for {name}: {e}")
            traceback.print_exc()
            failed += 1

        # Small delay between species to be nice to xeno-canto
        time.sleep(5)

    # Final summary
    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE COMPLETED")
    logger.info(f"Ended at: {datetime.now()}")
    logger.info("-" * 60)
    logger.info(f"Success: {success}")
    logger.info(f"Failed:  {failed}")
    logger.info(f"Skipped: {skipped}")
    logger.info(f"Total:   {success + failed + skipped}/{len(species_list)}")
    logger.info("=" * 60)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
