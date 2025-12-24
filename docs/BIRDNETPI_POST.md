# [Feature Proposal] Vocalization Classifier Add-on: Song/Call/Alarm Detection

Hi everyone,

I'm Ronny Hullegie from the Netherlands, running BirdNET-Pi on a Raspberry Pi 5 as part of my home biodiversity monitoring project (EMSN - Ecological Monitoring System Nijverdal).

First, I want to express my appreciation for this incredible community. What Patrick McGuire (@mcguirepr89) started as a pioneer project - bringing real-time bird identification to the Raspberry Pi - has grown into something truly special. And the fact that @Nachtzuster has taken up the torch to continue development and keep the project alive and thriving is fantastic. This kind of open source collaboration is what makes projects like this possible. Thank you both, and everyone who contributes!

I've developed a **vocalization classifier** that adds extra context to BirdNET-Pi detections, and I'd like to share it with the community.

## What does it do?

When BirdNET-Pi identifies a bird, my classifier analyzes the same audio to determine the **type of vocalization**:

| Without | With Vocalization |
|---------|-------------------|
| "Blackbird detected" | "Blackbird - **Song** (93%)" |
| "European Robin detected" | "European Robin - **Alarm** (87%)" |

### Why is this useful?

- **Song** = Bird is marking territory or attracting a mate
- **Call** = Contact calls, flock communication
- **Alarm** = Predator nearby! (cat, sparrowhawk, etc.)

This gives researchers and hobbyists insight into bird *behavior*, not just presence.

## How it works

- **CNN model** per species, trained on Xeno-canto spectrograms
- **Lightweight**: Runs on Raspberry Pi 4/5 alongside BirdNET-Pi
- **Non-invasive**: Reads `birds.db`, adds a column - doesn't touch BirdNET-Pi code
- **197 trained models** for European species (focused on Dutch birds for now)

## Current status

I've been running this in production for over a month:
- 80,000+ detections classified
- Stable on Raspberry Pi 5
- Pre-trained models available (~6.9 GB total, ~35 MB per species)

## Available as standalone add-on

I've packaged this as a **standalone add-on** that works alongside any BirdNET-Pi installation:

**Repository**: https://github.com/RonnyCHL/emsn-vocalization

**Pre-trained models**: [Google Drive](https://drive.google.com/open?id=1eUu0ECYC3vIFX5HeRg9xyd_wfYb5Zn3i) (~6.9 GB)

**Quick install**:
```bash
curl -sSL https://raw.githubusercontent.com/RonnyCHL/emsn-vocalization/main/install.sh | bash
```

The installer sets up a systemd service that enriches your detections in the background. Your BirdNET-Pi installation remains completely untouched.

## Looking for feedback

I'm sharing this to:

1. **Get testers** - Does it work on your setup? Which species are you seeing?
2. **Hear ideas** - Would tighter integration with BirdNET-Pi be interesting for the community?
3. **Find collaborators** - Anyone want to help train models for other regions (North America, UK, etc.)?

## Technical notes

- MIT licensed, free to use and modify
- Training pipeline uses Google Colab (free GPU) or local Docker
- Models trained on Xeno-canto audio with data augmentation
- Species-specific models (~35 MB each) for accurate classification

## A note on development

I developed this classifier with assistance from Claude (Anthropic's AI). Claude helped with Python code, architecture decisions, and documentation. The training data comes from Xeno-canto, and all models were trained using their publicly available recordings.

---

Happy to answer any questions! And once again, huge thanks to the pioneers and maintainers who make BirdNET-Pi possible. Building on top of such a solid foundation is a privilege.

Ronny Hullegie
EMSN Project, Nijverdal, Netherlands
