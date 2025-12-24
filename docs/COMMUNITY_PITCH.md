# BirdNET-Pi Vocalization Classifier

**Add song/call/alarm detection to your BirdNET-Pi**

## What is it?

A standalone add-on that automatically classifies bird vocalizations. When BirdNET-Pi detects a bird, this classifier tells you *how* it's vocalizing.

| Without | With Vocalization |
|---------|-------------------|
| "Merel detected" | "Merel - **Zang** (93%)" |
| "Roodborst detected" | "Roodborst - **Alarm** (87%)" |

## Why is this useful?

| Vocalization | Meaning | Example |
|--------------|---------|---------|
| **Song** | Territory marking, attracting mate | Blackbird singing at dawn |
| **Call** | Contact calls, flock communication | Great Tit calling to group |
| **Alarm** | Danger! Predator nearby | Robin warning about a cat |

### Practical applications

- **Behavior research**: When do birds sing? How does the population respond to disturbances?
- **Predator detection**: Alarm calls can indicate cats, sparrowhawks, or other predators
- **Seasonal patterns**: Track singing activity throughout the year
- **Data enrichment**: Extract more information from the same recordings

## How it works

```
BirdNET-Pi detects "Blackbird"
         ↓
Vocalization Classifier analyzes audio
         ↓
Result: "Blackbird - Song (93%)"
```

### Technical details

1. **CNN model** trained on Xeno-canto spectrograms
2. **Species-specific**: Each bird has its own trained model
3. **Lightweight**: Runs on Raspberry Pi 4/5
4. **Non-invasive**: Doesn't modify BirdNET-Pi at all

## Current status

- **197 trained models** (Dutch/European bird species)
- **80,000+ detections** processed in production
- **Running 30+ days** stable on Raspberry Pi 5
- **Open source**: MIT license

## Pre-trained models

Download ready-to-use models:

📥 **[Download from Google Drive](https://drive.google.com/open?id=1eUu0ECYC3vIFX5HeRg9xyd_wfYb5Zn3i)** (~6.9 GB for all 197 species)

Individual models are ~35 MB each - download only the species you need.

## Installation

### Quick install (recommended)

```bash
curl -sSL https://raw.githubusercontent.com/RonnyCHL/emsn-vocalization/main/install.sh | bash
```

This script:
1. Installs Python dependencies (PyTorch, librosa)
2. Downloads the classifier code
3. Optionally downloads pre-trained models
4. Sets up a systemd service

**Your BirdNET-Pi installation remains completely untouched.**

### Manual installation

See [README.md](../README.md) for manual setup instructions.

## Requirements

- Raspberry Pi 4 or 5 (Pi 3 may work but slower)
- BirdNET-Pi installed and running
- ~7 GB disk space for all models (or less for selected species)
- Python 3.9+

## Model architecture

### Ultimate Models (current)
- 4-layer CNN with batch normalization
- Trained on Google Colab A100
- 50 epochs with data augmentation
- ~35 MB per species

### Training data

Audio from [Xeno-canto](https://xeno-canto.org/):
- Quality A/B recordings
- Balanced across vocalization types
- 50 recordings per type per species

## Roadmap

- [ ] English species names support
- [ ] Web interface integration (discussion needed)
- [ ] More species (currently Dutch/European focus)
- [ ] Confidence threshold configuration

## Want to help?

1. **Test it**: Try with your BirdNET-Pi setup
2. **Train more species**: Use the Colab notebook
3. **Report issues**: Open a GitHub issue
4. **Suggest improvements**: Discussions welcome

## Links

- **GitHub**: https://github.com/RonnyCHL/emsn-vocalization
- **Models**: https://drive.google.com/open?id=1eUu0ECYC3vIFX5HeRg9xyd_wfYb5Zn3i
- **Training notebook**: Available in repository

## License

MIT License - free to use, modify, and distribute.

## Author

Ronny Hullegie
EMSN Project (Ecological Monitoring System Nijverdal, Netherlands)

*This classifier was developed with assistance from Claude (Anthropic) for code development and documentation.*

---

*Questions? Open an issue on GitHub or start a discussion.*
