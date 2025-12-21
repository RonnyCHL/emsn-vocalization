#!/usr/bin/env python3
"""
EMSN 2.0 - Xeno-canto Collector Wrapper
CLI wrapper voor grote dataset downloads.
"""

import argparse
import logging
import sys
from pathlib import Path

# Voeg parent toe aan path voor imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.collectors.xeno_canto import XenoCantoClient

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Download grote Xeno-canto dataset')
    parser.add_argument('--species', default='Turdus merula',
                       help='Wetenschappelijke naam van de soort')
    parser.add_argument('--output-dir', default='data/raw/xeno-canto-large',
                       help='Output directory')
    parser.add_argument('--per-type', type=int, default=200,
                       help='Aantal bestanden per vocalisatie type')
    parser.add_argument('--quality', nargs='+', default=['A', 'B'],
                       help='Gewenste kwaliteiten')
    parser.add_argument('--types', nargs='+', default=['song', 'call', 'alarm'],
                       help='Vocalisatie types')
    parser.add_argument('--skip-existing', action='store_true',
                       help='Sla bestaande bestanden over')
    parser.add_argument('--api-key', default=None,
                       help='Xeno-canto API key')

    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("EMSN 2.0 - Xeno-canto Large Dataset Collector")
    print(f"{'='*60}")
    print(f"Soort: {args.species}")
    print(f"Types: {', '.join(args.types)}")
    print(f"Kwaliteit: {', '.join(args.quality)}")
    print(f"Per type: {args.per_type} bestanden")
    print(f"Output: {args.output_dir}")
    print(f"{'='*60}\n")

    client = XenoCantoClient(
        download_dir=args.output_dir,
        api_key=args.api_key
    )

    mode = "API v3" if client.api_key else "Web scraping"
    print(f"Modus: {mode}")

    if not client.api_key:
        print("\nWaarschuwing: Geen API key, gebruik web scraping (langzamer)")
        print("TIP: Vraag gratis API key aan op https://xeno-canto.org/explore/api\n")

    # Download dataset
    dataset = client.download_dataset(
        species=args.species,
        vocalization_types=args.types,
        quality=args.quality,
        samples_per_type=args.per_type
    )

    # Samenvatting
    print(f"\n{'='*60}")
    print("DOWNLOAD COMPLEET")
    print(f"{'='*60}")

    total = 0
    for voc_type, files in dataset.items():
        count = len(files)
        print(f"  {voc_type}: {count} bestanden")
        total += count

    print(f"{'='*60}")
    print(f"  TOTAAL: {total} bestanden")
    print(f"{'='*60}")

    # Return success als we minstens wat data hebben
    if total > 0:
        return 0
    else:
        logger.error("Geen bestanden gedownload!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
