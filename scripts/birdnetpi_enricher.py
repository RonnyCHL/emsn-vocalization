#!/usr/bin/env python3
"""
BirdNET-Pi Vocalization Enricher

Standalone service that enriches BirdNET-Pi detections with vocalization type.
Works with a standard BirdNET-Pi installation - no modifications needed.

This script:
1. Watches the BirdNET-Pi database for new detections
2. Classifies each detection as song/call/alarm
3. Updates the detection record with vocalization info

Usage:
    python3 birdnetpi_enricher.py

Environment variables:
    BIRDNET_DB: Path to birds.db (default: ~/BirdNET-Pi/scripts/birds.db)
    VOCALIZATION_MODELS_DIR: Path to model files (default: ./models)
    POLL_INTERVAL: Seconds between checks (default: 30)
"""

import os
import sys
import time
import sqlite3
import logging
from pathlib import Path
from datetime import datetime, timedelta

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from src.classifiers.cnn_inference import VocalizationClassifier
except ImportError:
    # Fallback for standalone installation
    from vocalization_classifier import VocalizationClassifier

# Configuration from environment
BIRDNET_DB = Path(os.environ.get(
    'BIRDNET_DB',
    os.path.expanduser('~/BirdNET-Pi/scripts/birds.db')
))
MODELS_DIR = Path(os.environ.get(
    'VOCALIZATION_MODELS_DIR',
    Path(__file__).parent / 'models'
))
POLL_INTERVAL = int(os.environ.get('POLL_INTERVAL', 30))
AUDIO_DIR = Path(os.environ.get(
    'BIRDNET_AUDIO_DIR',
    os.path.expanduser('~/BirdNET-Pi/Extracted/By_Date')
))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/tmp/vocalization-enricher.log')
    ]
)
logger = logging.getLogger(__name__)


def ensure_vocalization_column(db_path: Path) -> bool:
    """Add vocalization column to detections table if it doesn't exist."""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Check if column exists
        cursor.execute("PRAGMA table_info(detections)")
        columns = [row[1] for row in cursor.fetchall()]

        if 'vocalization' not in columns:
            logger.info("Adding 'vocalization' column to detections table")
            cursor.execute("ALTER TABLE detections ADD COLUMN vocalization TEXT")
            conn.commit()
            logger.info("Column added successfully")

        conn.close()
        return True

    except Exception as e:
        logger.error(f"Error adding column: {e}")
        return False


def get_unprocessed_detections(db_path: Path, limit: int = 50) -> list:
    """Get detections that haven't been processed yet."""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Get recent detections without vocalization
        cursor.execute("""
            SELECT id, Com_Name, File_Name, Date, Time
            FROM detections
            WHERE vocalization IS NULL
            ORDER BY Date DESC, Time DESC
            LIMIT ?
        """, (limit,))

        results = cursor.fetchall()
        conn.close()

        return [
            {
                'id': row[0],
                'species': row[1],
                'filename': row[2],
                'date': row[3],
                'time': row[4]
            }
            for row in results
        ]

    except Exception as e:
        logger.error(f"Error fetching detections: {e}")
        return []


def find_audio_file(detection: dict) -> Path | None:
    """Find the audio file for a detection."""
    # BirdNET-Pi stores files in By_Date/YYYY-MM-DD/filename.mp3
    date_str = detection['date']
    filename = detection['filename']

    # Try standard path
    audio_path = AUDIO_DIR / date_str / filename
    if audio_path.exists():
        return audio_path

    # Try with .mp3 extension if not present
    if not filename.endswith('.mp3'):
        audio_path = AUDIO_DIR / date_str / f"{filename}.mp3"
        if audio_path.exists():
            return audio_path

    return None


def update_detection(db_path: Path, detection_id: int, vocalization: str) -> bool:
    """Update detection with vocalization info."""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute(
            "UPDATE detections SET vocalization = ? WHERE id = ?",
            (vocalization, detection_id)
        )

        conn.commit()
        conn.close()
        return True

    except Exception as e:
        logger.error(f"Error updating detection {detection_id}: {e}")
        return False


def main():
    """Main enricher loop."""
    logger.info("=" * 60)
    logger.info("BirdNET-Pi Vocalization Enricher starting")
    logger.info(f"Database: {BIRDNET_DB}")
    logger.info(f"Models: {MODELS_DIR}")
    logger.info(f"Poll interval: {POLL_INTERVAL}s")
    logger.info("=" * 60)

    # Check database exists
    if not BIRDNET_DB.exists():
        logger.error(f"Database not found: {BIRDNET_DB}")
        logger.error("Is BirdNET-Pi installed?")
        sys.exit(1)

    # Check models directory
    if not MODELS_DIR.exists():
        logger.error(f"Models directory not found: {MODELS_DIR}")
        logger.error("Download models from: https://drive.google.com/open?id=1eUu0ECYC3vIFX5HeRg9xyd_wfYb5Zn3i")
        sys.exit(1)

    model_count = len(list(MODELS_DIR.glob("*.pt")))
    if model_count == 0:
        logger.error("No model files found in models directory")
        sys.exit(1)

    logger.info(f"Found {model_count} vocalization models")

    # Ensure database has vocalization column
    if not ensure_vocalization_column(BIRDNET_DB):
        logger.error("Failed to set up database")
        sys.exit(1)

    # Initialize classifier
    classifier = VocalizationClassifier(models_dir=MODELS_DIR)
    logger.info("Classifier initialized")

    # Stats
    total_processed = 0
    total_classified = 0
    start_time = datetime.now()

    # Main loop
    while True:
        try:
            detections = get_unprocessed_detections(BIRDNET_DB)

            if detections:
                logger.info(f"Processing {len(detections)} detections...")

                for detection in detections:
                    species = detection['species']
                    detection_id = detection['id']

                    # Find audio file
                    audio_path = find_audio_file(detection)

                    if audio_path is None:
                        # Mark as processed but without classification
                        update_detection(BIRDNET_DB, detection_id, "no_audio")
                        continue

                    # Check if we have a model for this species
                    if not classifier.has_model(species):
                        update_detection(BIRDNET_DB, detection_id, "no_model")
                        continue

                    # Classify
                    result = classifier.classify(species, audio_path)

                    if result:
                        # Format: "zang (93%)" or "roep (87%)"
                        voc_text = f"{result['type_nl']} ({result['confidence']:.0%})"
                        update_detection(BIRDNET_DB, detection_id, voc_text)
                        total_classified += 1
                        logger.debug(f"{species}: {voc_text}")
                    else:
                        update_detection(BIRDNET_DB, detection_id, "error")

                    total_processed += 1

                # Log progress
                runtime = datetime.now() - start_time
                logger.info(
                    f"Stats: {total_classified}/{total_processed} classified "
                    f"({total_classified/max(total_processed,1)*100:.0f}%) - "
                    f"Runtime: {runtime}"
                )

            # Wait before next poll
            time.sleep(POLL_INTERVAL)

        except KeyboardInterrupt:
            logger.info("Shutting down...")
            break
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
            time.sleep(60)  # Wait longer on error


if __name__ == "__main__":
    main()
