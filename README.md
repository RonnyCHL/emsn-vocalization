# EMSN Bird Vocalization Classifier

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18010669.svg)](https://doi.org/10.5281/zenodo.18010669)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Automated bird vocalization classifier using CNN models**

Classifies bird vocalizations into **song**, **call**, and **alarm** types for 232 Dutch bird species.

## Overview

This project trains Convolutional Neural Networks (CNN) to distinguish between different vocalization types using:
- Audio data from [Xeno-canto](https://xeno-canto.org/)
- Mel spectrograms as input features
- PyTorch for model training
- Integration with BirdNET-Pi for species identification

## Features

- Automatic download of training data from Xeno-canto
- Mel spectrogram generation from audio files
- CNN model training with PyTorch
- Support for 232 Dutch bird species
- PostgreSQL database for tracking training progress
- Docker support for isolated training environment
- Grafana dashboard for monitoring

## Requirements

- Python 3.10+
- PyTorch
- PostgreSQL (for training tracking)
- Docker (optional, for containerized training)

## Quick Start

```bash
# Clone the repository
git clone https://github.com/RonnyCHL/emsn-vocalization.git
cd emsn-vocalization

# Install dependencies
pip install -r requirements.txt

# Train a model for a species
python train_existing.py --species "Koolmees"
```

## Project Structure

```
emsn-vocalization/
├── src/
│   ├── classifiers/      # CNN model definitions
│   ├── collectors/       # Xeno-canto data collection
│   ├── processors/       # Audio processing & spectrograms
│   └── utils/            # Helper functions
├── data/
│   ├── raw/              # Downloaded audio files
│   ├── spectrograms-*/   # Generated spectrograms
│   └── models/           # Trained model files (.pt)
├── train_existing.py     # Main training script
├── full_pipeline.py      # Complete training pipeline
└── docker-compose.yml    # Docker configuration
```

## Model Architecture

The CNN classifier uses:
- 3 convolutional layers with batch normalization
- Max pooling and dropout for regularization
- Fully connected layers for classification
- Cross-entropy loss with class weighting

## Training Data

Training data is automatically collected from Xeno-canto based on:
- Dutch bird species list (232 species)
- Vocalization type labels (song/call/alarm)
- Quality ratings (A/B preferred)

## Integration with BirdNET-Pi

This classifier works alongside BirdNET-Pi:
1. BirdNET-Pi identifies the bird species
2. This classifier determines the vocalization type
3. Combined data provides richer insights

## Related Projects

- [EMSN 2.0](https://github.com/RonnyCHL/emsn2) - Main biodiversity monitoring system
- [BirdNET-Pi](https://github.com/mcguirepr89/BirdNET-Pi) - Bird species identification

## License

MIT License - see [LICENSE](LICENSE)

## Author

Ronny Hullegie - EMSN Project

## Citation

If you use this project in your research, please cite:
```
Hullegie, R. (2025). EMSN Bird Vocalization Classifier. GitHub. https://github.com/RonnyCHL/emsn-vocalization
```
