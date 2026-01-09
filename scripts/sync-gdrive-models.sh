#!/bin/bash
# Sync vocalization models from Google Drive
# Run: ./sync-gdrive-models.sh

echo "Syncing models from Google Drive..."
rclone sync gdrive:EMSN-Vocalization/models/ /home/ronny/emsn-vocalization/trained-models/ -P

echo ""
echo "Models synced: $(ls /home/ronny/emsn-vocalization/trained-models/ | wc -l)"
echo "Total size: $(du -sh /home/ronny/emsn-vocalization/trained-models/ | cut -f1)"
