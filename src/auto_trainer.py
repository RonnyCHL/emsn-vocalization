#!/usr/bin/env python3
"""
EMSN 2.0 - Automatic Vocalization Trainer
Haalt top soorten uit BirdNET en traint automatisch CNN modellen.
Rapporteert voortgang naar PostgreSQL voor Grafana dashboard.
"""

import json
import numpy as np
import logging
import os
import sqlite3
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# PostgreSQL voor Grafana
try:
    import psycopg2
    HAS_POSTGRES = True
except ImportError:
    HAS_POSTGRES = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuratie
BIRDNET_DB = os.environ.get('BIRDNET_DB', '/data/birds.db')
DATA_DIR = Path('/app/data')
MODELS_DIR = DATA_DIR / 'models'
LOGS_DIR = Path('/app/logs')
MIN_CONFIDENCE = 0.7
MIN_DETECTIONS = 20
MAX_SPECIES = 15
SAMPLES_PER_TYPE = 200

# PostgreSQL config
PG_HOST = os.environ.get('PG_HOST', '192.168.1.25')
PG_PORT = os.environ.get('PG_PORT', '5432')
PG_DB = os.environ.get('PG_DB', 'emsn')
PG_USER = os.environ.get('PG_USER', 'emsn')
PG_PASS = os.environ.get('PG_PASS', 'emsn2024')


def get_pg_connection():
    """Maak PostgreSQL connectie voor Grafana."""
    if not HAS_POSTGRES:
        return None
    try:
        return psycopg2.connect(
            host=PG_HOST, port=PG_PORT,
            database=PG_DB, user=PG_USER, password=PG_PASS
        )
    except Exception as e:
        logger.warning(f"PostgreSQL niet beschikbaar: {e}")
        return None


def init_progress_table():
    """Maak progress tabel voor Grafana."""
    conn = get_pg_connection()
    if not conn:
        return
    
    try:
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS vocalization_training (
                id SERIAL PRIMARY KEY,
                species_name VARCHAR(100),
                scientific_name VARCHAR(100),
                status VARCHAR(20),
                phase VARCHAR(30),
                progress_pct INTEGER DEFAULT 0,
                audio_files INTEGER DEFAULT 0,
                spectrograms INTEGER DEFAULT 0,
                accuracy FLOAT,
                started_at TIMESTAMP,
                completed_at TIMESTAMP,
                error_message TEXT,
                updated_at TIMESTAMP DEFAULT NOW()
            )
        """)
        conn.commit()
        cur.close()
        conn.close()
        logger.info("PostgreSQL progress tabel gereed")
    except Exception as e:
        logger.error(f"Kon tabel niet maken: {e}")


def update_progress(species_name: str, scientific_name: str, status: str, 
                   phase: str = None, progress: int = 0, **kwargs):
    """Update voortgang in PostgreSQL."""
    conn = get_pg_connection()
    if not conn:
        # Fallback: schrijf naar JSON
        progress_file = LOGS_DIR / 'training_progress.json'
        try:
            if progress_file.exists():
                with open(progress_file) as f:
                    data = json.load(f)
            else:
                data = {}
            
            data[scientific_name] = {
                'species_name': species_name,
                'status': status,
                'phase': phase,
                'progress': progress,
                'updated_at': datetime.now().isoformat(),
                **kwargs
            }
            
            with open(progress_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Kon progress niet opslaan: {e}")
        return
    
    try:
        cur = conn.cursor()
        
        # Check of soort al bestaat
        cur.execute(
            "SELECT id FROM vocalization_training WHERE scientific_name = %s",
            (scientific_name,)
        )
        row = cur.fetchone()
        
        now = datetime.now()
        
        if row:
            # Update bestaande
            updates = ["status = %s", "phase = %s", "progress_pct = %s", "updated_at = %s"]
            values = [status, phase, progress, now]
            
            for key, val in kwargs.items():
                if key in ['audio_files', 'spectrograms', 'accuracy', 'error_message']:
                    updates.append(f"{key} = %s")
                    values.append(val)
            
            if status == 'completed':
                updates.append("completed_at = %s")
                values.append(now)
            
            values.append(scientific_name)
            cur.execute(
                f"UPDATE vocalization_training SET {', '.join(updates)} WHERE scientific_name = %s",
                values
            )
        else:
            # Insert nieuwe
            cur.execute("""
                INSERT INTO vocalization_training 
                (species_name, scientific_name, status, phase, progress_pct, started_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (species_name, scientific_name, status, phase, progress, now, now))
        
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        logger.error(f"Kon progress niet updaten: {e}")


def get_top_species(db_path: str, limit: int = MAX_SPECIES) -> list:
    """Haal top gedetecteerde soorten uit BirdNET database."""
    if not os.path.exists(db_path):
        logger.error(f"BirdNET database niet gevonden: {db_path}")
        return []
    
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    cur.execute("""
        SELECT Com_Name, Sci_Name, COUNT(*) as cnt
        FROM detections 
        WHERE Confidence > ?
        GROUP BY Sci_Name 
        HAVING cnt >= ?
        ORDER BY cnt DESC 
        LIMIT ?
    """, (MIN_CONFIDENCE, MIN_DETECTIONS, limit))
    
    species = []
    for row in cur.fetchall():
        species.append({
            'common_name': row[0],
            'scientific_name': row[1],
            'detections': row[2]
        })
    
    conn.close()
    logger.info(f"Gevonden: {len(species)} soorten met >= {MIN_DETECTIONS} detecties")
    return species


def get_trained_species() -> set:
    """Check welke soorten al getraind zijn."""
    trained = set()
    if MODELS_DIR.exists():
        for model in MODELS_DIR.glob('*_cnn_*.keras'):
            # Extract species name from filename
            name = model.stem.split('_cnn_')[0]
            trained.add(name)
    return trained


def species_to_dirname(name: str) -> str:
    """Converteer soortnaam naar directory naam."""
    return name.lower().replace(' ', '_').replace("'", "")


def download_species(species: dict) -> int:
    """Download audio voor een soort. Returns aantal bestanden."""
    dirname = species_to_dirname(species['common_name'])
    raw_dir = DATA_DIR / 'raw' / dirname
    
    update_progress(
        species['common_name'], species['scientific_name'],
        'downloading', 'Xeno-canto download', 10
    )
    
    try:
        result = subprocess.run([
            sys.executable, '-m', 'src.collectors.xeno_canto_collector',
            '--species', species['scientific_name'],
            '--output-dir', str(raw_dir),
            '--per-type', str(SAMPLES_PER_TYPE),
            '--quality', 'A', 'B',
            '--skip-existing'
        ], capture_output=True, text=True, timeout=3600)
        
        if result.returncode != 0:
            logger.warning(f"Download warning voor {species['common_name']}: {result.stderr[:200]}")
    except subprocess.TimeoutExpired:
        logger.error(f"Download timeout voor {species['common_name']}")
        return 0
    except Exception as e:
        logger.error(f"Download error: {e}")
        return 0
    
    # Tel bestanden
    count = len(list(raw_dir.rglob('*.mp3')))
    update_progress(
        species['common_name'], species['scientific_name'],
        'downloading', 'Download compleet', 30, audio_files=count
    )
    
    return count


def generate_spectrograms(species: dict) -> int:
    """Genereer spectrogrammen. Returns aantal."""
    dirname = species_to_dirname(species['common_name'])
    raw_dir = DATA_DIR / 'raw' / dirname
    spec_dir = DATA_DIR / f'spectrograms-{dirname}'
    
    update_progress(
        species['common_name'], species['scientific_name'],
        'processing', 'Spectrogrammen genereren', 40
    )
    
    try:
        result = subprocess.run([
            sys.executable, '-m', 'src.processors.spectrogram_generator',
            '--input-dir', str(raw_dir),
            '--output-dir', str(spec_dir),
            '--segment-duration', '3.0',
            '--overlap', '0.5',
            '--n-mels', '128'
        ], capture_output=True, text=True, timeout=1800)
        
        if result.returncode != 0:
            logger.warning(f"Spectrogram warning: {result.stderr[:200]}")
    except Exception as e:
        logger.error(f"Spectrogram error: {e}")
        return 0
    
    count = len(list(spec_dir.rglob('*.npy')))
    update_progress(
        species['common_name'], species['scientific_name'],
        'processing', 'Spectrogrammen klaar', 50, spectrograms=count
    )
    
    return count


def combine_spectrograms(spec_dir: Path) -> bool:
    """Combineer losse spectrogrammen tot X_spectrograms.npy en y_labels.npy."""
    X_list = []
    y_list = []

    # Verwachte klassen
    classes = ['song', 'call', 'alarm']

    for class_name in classes:
        class_dir = spec_dir / class_name
        if not class_dir.exists():
            logger.warning(f"Klasse directory niet gevonden: {class_dir}")
            continue

        npy_files = list(class_dir.glob('*.npy'))
        logger.info(f"  {class_name}: {len(npy_files)} spectrogrammen")

        for npy_file in npy_files:
            try:
                spec = np.load(npy_file)
                X_list.append(spec)
                y_list.append(class_name)
            except Exception as e:
                logger.warning(f"Kon {npy_file} niet laden: {e}")

    if len(X_list) < 100:
        logger.error(f"Te weinig spectrogrammen: {len(X_list)}")
        return False

    # Stack en sla op
    X = np.stack(X_list)
    y = np.array(y_list)

    np.save(spec_dir / 'X_spectrograms.npy', X)
    np.save(spec_dir / 'y_labels.npy', y)

    logger.info(f"Gecombineerd: X={X.shape}, y={y.shape}")
    return True


def train_model(species: dict) -> float:
    """Train CNN model. Returns accuracy."""
    dirname = species_to_dirname(species['common_name'])
    spec_dir = DATA_DIR / f'spectrograms-{dirname}'
    model_path = MODELS_DIR / f'{dirname}_cnn_v1.keras'

    update_progress(
        species['common_name'], species['scientific_name'],
        'training', 'Data voorbereiden', 55
    )

    # Combineer spectrogrammen eerst
    if not combine_spectrograms(spec_dir):
        update_progress(
            species['common_name'], species['scientific_name'],
            'failed', 'Combineren mislukt', 0,
            error_message='Kon spectrogrammen niet combineren'
        )
        return 0.0

    update_progress(
        species['common_name'], species['scientific_name'],
        'training', 'CNN training', 60
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    try:
        result = subprocess.run([
            sys.executable, '-m', 'src.classifiers.cnn_classifier',
            '--data-dir', str(spec_dir),
            '--output-model', str(model_path),
            '--output-dir', str(LOGS_DIR),
            '--epochs', '100',
            '--batch-size', '32',
            '--patience', '15'
        ], capture_output=True, text=True, timeout=14400)  # 4 uur max
        
        # Parse accuracy uit output
        accuracy = 0.0
        for line in result.stdout.split('\n'):
            if 'Test accuracy' in line or 'val_accuracy' in line:
                try:
                    accuracy = float(line.split(':')[-1].strip().rstrip('%')) / 100
                except:
                    pass
        
        if model_path.exists():
            update_progress(
                species['common_name'], species['scientific_name'],
                'completed', 'Training voltooid', 100, accuracy=accuracy
            )
            return accuracy
        else:
            update_progress(
                species['common_name'], species['scientific_name'],
                'failed', 'Model niet opgeslagen', 0, 
                error_message='Model file niet aangemaakt'
            )
            return 0.0
            
    except subprocess.TimeoutExpired:
        update_progress(
            species['common_name'], species['scientific_name'],
            'failed', 'Training timeout', 0, error_message='Timeout na 4 uur'
        )
        return 0.0
    except Exception as e:
        update_progress(
            species['common_name'], species['scientific_name'],
            'failed', 'Training error', 0, error_message=str(e)
        )
        return 0.0


def main():
    """Main training loop."""
    logger.info("=" * 60)
    logger.info("EMSN 2.0 - Automatic Vocalization Trainer")
    logger.info(f"Start: {datetime.now()}")
    logger.info("=" * 60)
    
    # Init PostgreSQL
    init_progress_table()
    
    # Haal top soorten
    species_list = get_top_species(BIRDNET_DB)
    if not species_list:
        logger.error("Geen soorten gevonden!")
        return
    
    # Check welke al getraind zijn
    trained = get_trained_species()
    logger.info(f"Al getraind: {trained}")
    
    # Filter en train
    for i, species in enumerate(species_list):
        dirname = species_to_dirname(species['common_name'])
        
        if dirname in trained:
            logger.info(f"SKIP {species['common_name']} - al getraind")
            continue
        
        logger.info("")
        logger.info("=" * 60)
        logger.info(f"[{i+1}/{len(species_list)}] {species['common_name']} ({species['scientific_name']})")
        logger.info(f"Detecties: {species['detections']}")
        logger.info("=" * 60)
        
        # Download
        audio_count = download_species(species)
        if audio_count < 30:
            logger.warning(f"Te weinig audio: {audio_count}")
            update_progress(
                species['common_name'], species['scientific_name'],
                'skipped', 'Te weinig data', 0, audio_files=audio_count
            )
            continue
        
        # Spectrogrammen
        spec_count = generate_spectrograms(species)
        if spec_count < 100:
            logger.warning(f"Te weinig spectrogrammen: {spec_count}")
            continue
        
        # Train
        accuracy = train_model(species)
        logger.info(f"Model accuracy: {accuracy:.1%}")
    
    logger.info("")
    logger.info("=" * 60)
    logger.info("TRAINING VOLTOOID")
    logger.info(f"Einde: {datetime.now()}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
