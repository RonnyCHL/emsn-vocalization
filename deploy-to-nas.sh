#!/bin/bash
# Deploy vocalization models to NAS
# Usage: ./deploy-to-nas.sh
# Configure your NAS settings below or via environment variables

NAS_HOST="${NAS_HOST:-192.168.1.25}"
NAS_USER="${NAS_USER:-ronny}"
NAS_PATH="${NAS_PATH:-/volume1/docker/emsn-vocalization/data/models/}"
LOCAL_PATH="${LOCAL_PATH:-./trained-models/}"

echo "Deploying models to NAS..."
echo "Source: $LOCAL_PATH"
echo "Target: $NAS_USER@$NAS_HOST:$NAS_PATH"
echo ""

# Count local models
LOCAL_COUNT=$(ls $LOCAL_PATH/*.pt 2>/dev/null | wc -l)
echo "Local models: $LOCAL_COUNT"
echo ""

# Sync to NAS
rsync -avz --progress $LOCAL_PATH/*.pt $NAS_USER@$NAS_HOST:$NAS_PATH

echo ""
echo "Done! Check NAS for deployed models."
