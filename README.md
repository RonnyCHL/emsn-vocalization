# BirdNET-Pi Vocalization Classifier

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

**Classify bird vocalizations as song, call, or alarm**

Works with BirdNET-Pi to add vocalization context to your bird detections.

## What does it do?

| Detection | Without | With Vocalization |
|-----------|---------|-------------------|
| Eurasian Blackbird | "Merel detected" | "Merel - **Zang** (93%)" |
| European Robin | "Roodborst detected" | "Roodborst - **Alarm** (87%)" |

### Why is this useful?

- **Song**: Bird is marking territory or attracting mate
- **Call**: Contact calls, flock communication
- **Alarm**: Predator nearby! (cat, sparrowhawk, etc.)

## Quick Start

### Use Pre-trained Models

```bash
# Download models (196 Dutch species available)
# Models are ~2MB each, download only what you need

# Example: classify a detection
from src.classifiers.cnn_inference import VocalizationClassifier

classifier = VocalizationClassifier(models_dir="./models")
result = classifier.classify("Koolmees", "/path/to/audio.wav")

if result:
    print(f"{result['type']} ({result['confidence']:.0%})")
    # Output: song (91%)
```

### Train Your Own Models

```bash
# Clone the repository
git clone https://github.com/RonnyCHL/emsn-vocalization.git
cd emsn-vocalization

# Install dependencies
pip install torch librosa numpy scikit-learn tqdm requests

# Train a model (downloads data from Xeno-canto automatically)
python train_existing.py --species "Koolmees"
```

### Train on Google Colab (Free GPU)

Open `notebooks/EMSN_Vocalization_Colab_Training.ipynb` in Google Colab for free GPU training.

## How it Works

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  BirdNET-Pi     │     │  Vocalization    │     │   Result        │
│  "Merel"        │ ──▶ │  Classifier      │ ──▶ │  "Merel - Zang" │
│                 │     │  (CNN model)     │     │  (93%)          │
└─────────────────┘     └──────────────────┘     └─────────────────┘
```

1. **BirdNET-Pi** identifies the bird species from audio
2. **This classifier** analyzes the same audio with a species-specific CNN
3. **Output** includes vocalization type (song/call/alarm) with confidence

## Project Structure

```
emsn-vocalization/
├── src/
│   ├── classifiers/      # CNN model & inference
│   ├── collectors/       # Xeno-canto data collection
│   └── processors/       # Audio → spectrogram processing
├── notebooks/            # Colab training notebooks
├── train_existing.py     # Main training script
├── full_pipeline.py      # Complete pipeline (download → train)
└── docker-compose.yml    # Docker training environment
```

## Model Details

- **Architecture**: 3-layer CNN with batch normalization
- **Input**: Mel spectrograms (128 bins, 3 seconds)
- **Output**: song / call / alarm + confidence
- **Size**: ~2MB per species model
- **Accuracy**: 85-95% for common species

## Available Models

Currently **196 trained models** for Dutch bird species, including:
- Koolmees (Great Tit)
- Merel (Eurasian Blackbird)
- Roodborst (European Robin)
- Huismus (House Sparrow)
- Vink (Common Chaffinch)
- ... and 191 more

## Integration Options

### Standalone (recommended for testing)
Run as separate service, reads BirdNET-Pi's `birds.db`.

### With BirdNET-Pi
Can be integrated to show vocalization in the web interface.

See [COMMUNITY_PITCH.md](docs/COMMUNITY_PITCH.md) for integration discussion.

## Requirements

- Python 3.10+
- PyTorch 2.0+
- librosa
- numpy
- Raspberry Pi 4/5 (for inference) or any Linux system

## Training Data

Audio data is automatically downloaded from [Xeno-canto](https://xeno-canto.org/):
- Quality A/B recordings preferred
- Balanced sampling across vocalization types
- Respects Xeno-canto API rate limits

## Contributing

- **Test the classifier**: Try it with your BirdNET-Pi setup
- **Train more species**: Use Colab notebook to train new models
- **Report issues**: Open a GitHub issue
- **Integration ideas**: See community pitch document

## Related Projects

- [BirdNET-Pi](https://github.com/mcguirepr89/BirdNET-Pi) - Bird species identification
- [Xeno-canto](https://xeno-canto.org/) - Bird sound database

## License

MIT License - free to use, modify, and distribute.

## Author

Ronny Hullegie - [EMSN Project](https://github.com/RonnyCHL/emsn2) (Ecologisch Monitoring Systeem Nijverdal)

## Citation

```bibtex
@software{hullegie2025vocalization,
  author = {Hullegie, Ronny},
  title = {BirdNET-Pi Vocalization Classifier},
  year = {2025},
  url = {https://github.com/RonnyCHL/emsn-vocalization}
}
```
