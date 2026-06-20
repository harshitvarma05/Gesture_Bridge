#!/usr/bin/env bash
set -euo pipefail

if [[ ${EUID} -ne 0 ]]; then
  echo "Run with sudo: sudo ./deploy/install_pi.sh"
  exit 1
fi

SOURCE_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
INSTALL_DIR=/opt/gesture-bridge

install -d -o pi -g pi "$INSTALL_DIR"
rsync -a --delete \
  --exclude '.git/' \
  --exclude '.idea/' \
  --exclude 'venv/' \
  --exclude 'data/' \
  --exclude '.sequence_cache/' \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  --exclude '*.log' \
  "$SOURCE_DIR"/ "$INSTALL_DIR"/
python3 -m venv "$INSTALL_DIR/venv"
"$INSTALL_DIR/venv/bin/pip" install --upgrade pip
"$INSTALL_DIR/venv/bin/pip" install -r "$INSTALL_DIR/requirements.txt" gpiozero
chown -R pi:pi "$INSTALL_DIR"

if [[ ! -f "$INSTALL_DIR/hand_landmarker.task" ]]; then
  echo "Missing hand_landmarker.task; installation cannot continue."
  exit 1
fi

install -m 0644 "$SOURCE_DIR/deploy/gesture-bridge.service" /etc/systemd/system/gesture-bridge.service
if [[ ! -f /etc/gesture-bridge.env ]]; then
  install -m 0600 "$SOURCE_DIR/config.example.env" /etc/gesture-bridge.env
fi
systemctl daemon-reload
systemctl enable gesture-bridge.service

echo "Edit /etc/gesture-bridge.env, then run: sudo systemctl start gesture-bridge"
