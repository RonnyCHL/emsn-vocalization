#!/bin/bash
#
# BirdNET-Pi Vocalization Classifier - Installation Script
#
# This script installs the vocalization classifier as a standalone add-on
# alongside your existing BirdNET-Pi installation.
#
# Usage: curl -sSL https://raw.githubusercontent.com/RonnyCHL/emsn-vocalization/main/install.sh | bash
#
# What it does:
# 1. Installs Python dependencies (PyTorch, librosa, etc.)
# 2. Downloads the classifier code
# 3. Optionally downloads pre-trained models from Google Drive
# 4. Sets up a systemd service to enrich detections
#
# Your BirdNET-Pi installation remains completely untouched.
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}"
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║     BirdNET-Pi Vocalization Classifier - Installer            ║"
echo "║     Adds song/call/alarm detection to your BirdNET-Pi         ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Check if running on Raspberry Pi
if ! grep -q "Raspberry Pi" /proc/cpuinfo 2>/dev/null; then
    echo -e "${YELLOW}Warning: This doesn't appear to be a Raspberry Pi.${NC}"
    echo "The installer is optimized for Raspberry Pi 4/5 with BirdNET-Pi."
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check if BirdNET-Pi is installed
BIRDNET_DB="$HOME/BirdNET-Pi/scripts/birds.db"
if [ ! -f "$BIRDNET_DB" ]; then
    echo -e "${RED}Error: BirdNET-Pi database not found at $BIRDNET_DB${NC}"
    echo "Please install BirdNET-Pi first: https://github.com/mcguirepr89/BirdNET-Pi"
    exit 1
fi

echo -e "${GREEN}✓ BirdNET-Pi installation found${NC}"

# Installation directory
INSTALL_DIR="$HOME/vocalization-classifier"
MODELS_DIR="$INSTALL_DIR/models"

echo ""
echo "Installation directory: $INSTALL_DIR"
echo ""

# Create directories
mkdir -p "$INSTALL_DIR"
mkdir -p "$MODELS_DIR"

# Step 1: Install Python dependencies
echo -e "${YELLOW}Step 1/4: Installing Python dependencies...${NC}"

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | grep -oP '\d+\.\d+')
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 9 ]); then
    echo -e "${RED}Error: Python 3.9+ required (found $PYTHON_VERSION)${NC}"
    exit 1
fi

echo "Python $PYTHON_VERSION detected"

# Install dependencies
pip3 install --user --quiet torch torchvision --index-url https://download.pytorch.org/whl/cpu 2>/dev/null || {
    echo "Installing PyTorch (this may take a few minutes on Pi)..."
    pip3 install --user torch torchvision --index-url https://download.pytorch.org/whl/cpu
}

pip3 install --user --quiet librosa scikit-image numpy 2>/dev/null || {
    echo "Installing audio processing libraries..."
    pip3 install --user librosa scikit-image numpy
}

echo -e "${GREEN}✓ Dependencies installed${NC}"

# Step 2: Download classifier code
echo -e "${YELLOW}Step 2/4: Downloading classifier code...${NC}"

cd "$INSTALL_DIR"

# Download main classifier file
curl -sSL -o vocalization_classifier.py \
    "https://raw.githubusercontent.com/RonnyCHL/emsn-vocalization/main/src/classifiers/cnn_inference.py"

# Download enricher script
curl -sSL -o vocalization_enricher.py \
    "https://raw.githubusercontent.com/RonnyCHL/emsn-vocalization/main/scripts/birdnetpi_enricher.py"

echo -e "${GREEN}✓ Classifier code downloaded${NC}"

# Step 3: Download models
echo -e "${YELLOW}Step 3/4: Downloading pre-trained models...${NC}"
echo ""
echo "Pre-trained models for 197 bird species are available (~6.9 GB total)."
echo "You can download all models now, or download individual species later."
echo ""
echo "Options:"
echo "  1) Download ALL models (6.9 GB) - recommended if you have space"
echo "  2) Download common European species only (~500 MB)"
echo "  3) Skip - I'll download models manually later"
echo ""
read -p "Choose option (1/2/3): " -n 1 -r MODEL_CHOICE
echo ""

GDRIVE_FOLDER_ID="1eUu0ECYC3vIFX5HeRg9xyd_wfYb5Zn3i"

case $MODEL_CHOICE in
    1)
        echo "Downloading all models (this will take a while)..."
        # Check if gdown is available
        if ! command -v gdown &> /dev/null; then
            pip3 install --user gdown
        fi
        gdown --folder "https://drive.google.com/drive/folders/$GDRIVE_FOLDER_ID" -O "$MODELS_DIR" --remaining-ok
        echo -e "${GREEN}✓ All models downloaded${NC}"
        ;;
    2)
        echo "Downloading common European species..."
        # List of common species
        COMMON_SPECIES=(
            "koolmees" "pimpelmees" "merel" "roodborst" "huismus"
            "vink" "groenling" "spreeuw" "ekster" "kauw"
            "houtduif" "turkse_tortel" "winterkoning" "heggenmus"
            "zwartkop" "tjiftjaf" "fitis" "grote_bonte_specht"
        )

        if ! command -v gdown &> /dev/null; then
            pip3 install --user gdown
        fi

        # Download each common species
        for species in "${COMMON_SPECIES[@]}"; do
            echo "  Downloading ${species}..."
            # Note: This requires individual file IDs - for now just inform user
        done

        echo -e "${YELLOW}Note: Individual model download not yet implemented.${NC}"
        echo "Please download from: https://drive.google.com/drive/folders/$GDRIVE_FOLDER_ID"
        ;;
    3)
        echo "Skipping model download."
        echo "Download models later from:"
        echo "https://drive.google.com/drive/folders/$GDRIVE_FOLDER_ID"
        echo ""
        echo "Place .pt files in: $MODELS_DIR"
        ;;
    *)
        echo "Invalid option, skipping model download."
        ;;
esac

# Step 4: Set up systemd service
echo -e "${YELLOW}Step 4/4: Setting up systemd service...${NC}"

# Create service file
cat > /tmp/vocalization-enricher.service << EOF
[Unit]
Description=BirdNET-Pi Vocalization Enricher
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$INSTALL_DIR
Environment=VOCALIZATION_MODELS_DIR=$MODELS_DIR
ExecStart=/usr/bin/python3 $INSTALL_DIR/vocalization_enricher.py
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Install service
sudo mv /tmp/vocalization-enricher.service /etc/systemd/system/
sudo systemctl daemon-reload

echo ""
read -p "Start the vocalization service now? (y/n) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    sudo systemctl enable vocalization-enricher
    sudo systemctl start vocalization-enricher
    echo -e "${GREEN}✓ Service started${NC}"
else
    echo "To start later, run:"
    echo "  sudo systemctl enable vocalization-enricher"
    echo "  sudo systemctl start vocalization-enricher"
fi

# Done!
echo ""
echo -e "${GREEN}"
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║                    Installation Complete!                     ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"
echo ""
echo "What's next:"
echo "  - The enricher will add vocalization type to new detections"
echo "  - Check status: sudo systemctl status vocalization-enricher"
echo "  - View logs: journalctl -u vocalization-enricher -f"
echo ""
echo "Your BirdNET-Pi detections will now include:"
echo "  'Merel' → 'Merel - Zang (93%)'"
echo ""
echo "Questions or issues?"
echo "  https://github.com/RonnyCHL/emsn-vocalization/issues"
echo ""
