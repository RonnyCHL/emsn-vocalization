#!/usr/bin/env python3
"""
EMSN 2.0 - Feature Combiner

Combineert akoestische features met BirdNET logits voor
verbeterde vocalisatie classificatie.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def combine_features(
    acoustic_csv: str,
    birdnet_csv: str,
    output_csv: str,
    birdnet_pca_components: int = 50
) -> pd.DataFrame:
    """
    Combineer akoestische features met BirdNET logits (via PCA).

    Args:
        acoustic_csv: Pad naar akoestische features CSV
        birdnet_csv: Pad naar BirdNET logits CSV
        output_csv: Output CSV pad
        birdnet_pca_components: Aantal PCA componenten voor BirdNET

    Returns:
        DataFrame met gecombineerde features
    """
    logger.info(f"Laden akoestische features: {acoustic_csv}")
    df_acoustic = pd.read_csv(acoustic_csv)

    logger.info(f"Laden BirdNET features: {birdnet_csv}")
    df_birdnet = pd.read_csv(birdnet_csv)

    logger.info(f"Akoestisch: {df_acoustic.shape}")
    logger.info(f"BirdNET: {df_birdnet.shape}")

    # Extract BirdNET features
    birdnet_feature_cols = [c for c in df_birdnet.columns if c.startswith('feature_')]
    X_birdnet = df_birdnet[birdnet_feature_cols].values

    # PCA op BirdNET features
    logger.info(f"PCA op BirdNET ({birdnet_pca_components} componenten)...")
    scaler = StandardScaler()
    X_birdnet_scaled = scaler.fit_transform(X_birdnet)

    pca = PCA(n_components=birdnet_pca_components, random_state=42)
    X_birdnet_pca = pca.fit_transform(X_birdnet_scaled)

    variance_explained = sum(pca.explained_variance_ratio_)
    logger.info(f"PCA verklaarde variantie: {variance_explained:.2%}")

    # Maak DataFrame van PCA features
    pca_cols = [f'birdnet_pca_{i}' for i in range(birdnet_pca_components)]
    df_birdnet_pca = pd.DataFrame(X_birdnet_pca, columns=pca_cols)

    # BirdNET gebruikt 'filename', bereken segment index uit segment_start
    df_birdnet_pca['filename'] = df_birdnet['filename']
    # Segment index: segment_start / 1.5 (want overlap=0.5 bij 3s segmenten)
    df_birdnet_pca['segment_idx'] = (df_birdnet['segment_start'] / 1.5).astype(int)

    # Merge op filename en segment_idx
    logger.info("Mergen van datasets...")

    # Normaliseer file paths voor matching
    df_acoustic['file_path_norm'] = df_acoustic['file_path'].apply(
        lambda x: Path(x).name
    )
    df_birdnet_pca['file_path_norm'] = df_birdnet_pca['filename'].apply(
        lambda x: Path(x).name
    )

    # Merge
    df_merged = pd.merge(
        df_acoustic,
        df_birdnet_pca.drop(columns=['filename']),
        on=['file_path_norm', 'segment_idx'],
        how='inner'
    )

    # Verwijder hulpkolom
    df_merged = df_merged.drop(columns=['file_path_norm'])

    logger.info(f"Gecombineerd: {df_merged.shape}")
    logger.info(f"Verdeling: {df_merged['vocalization_type'].value_counts().to_dict()}")

    # Sla op
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df_merged.to_csv(output_csv, index=False)
    logger.info(f"Opgeslagen: {output_csv}")

    return df_merged


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Combineer features')
    parser.add_argument('--acoustic', default='data/embeddings/merel_acoustic_features.csv')
    parser.add_argument('--birdnet', default='data/embeddings/merel_features.csv')
    parser.add_argument('--output', default='data/embeddings/merel_combined_features.csv')
    parser.add_argument('--pca-components', type=int, default=50)

    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("EMSN 2.0 - Feature Combiner")
    print(f"{'='*60}")
    print(f"Akoestisch: {args.acoustic}")
    print(f"BirdNET: {args.birdnet}")
    print(f"Output: {args.output}")
    print(f"BirdNET PCA componenten: {args.pca_components}")
    print(f"{'='*60}\n")

    df = combine_features(
        args.acoustic,
        args.birdnet,
        args.output,
        args.pca_components
    )

    # Tel features
    metadata_cols = ['file_path', 'segment_idx', 'vocalization_type']
    feature_cols = [c for c in df.columns if c not in metadata_cols]

    print(f"\n{'='*60}")
    print("RESULTAAT")
    print(f"{'='*60}")
    print(f"Samples: {len(df)}")
    print(f"Totaal features: {len(feature_cols)}")
    print(f"  - Akoestisch: {len([c for c in feature_cols if not c.startswith('birdnet_')])}")
    print(f"  - BirdNET PCA: {len([c for c in feature_cols if c.startswith('birdnet_')])}")
    print(f"Output: {args.output}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
